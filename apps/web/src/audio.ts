/**
 * Browser audio engine.
 *
 * Two transports, one audio contract:
 *
 * * WebSocket fallback — PCM16 mono at the device capture rate up, 48 kHz down
 *   in 20 ms frames. The gateway resamples to 16 kHz for the recogniser.
 * * WebRTC — Opus both ways, with the same server-side framing guarantees.
 *
 * The playback side is deliberately strict. Every inbound frame is scheduled on
 * a monotonic audio clock; a flush stops every scheduled source and ducks the
 * gain for the server's 80 ms guard, so a barge-in can never produce overlapping
 * old and new speech.
 */

import * as api from "./api";

const TRACK_RATE = 48_000;
const WORKLET_NAME = "sarang-pcm";

interface AudioContextConstructor {
  new (options?: AudioContextOptions): AudioContext;
}

function audioContextConstructor(): AudioContextConstructor {
  const candidate =
    window.AudioContext ?? (window as unknown as { webkitAudioContext?: AudioContextConstructor }).webkitAudioContext;
  if (!candidate) throw new Error("This browser does not support the Web Audio API.");
  return candidate;
}

/** Linear resampler with state across arbitrary chunk boundaries. */
export class LinearResampler {
  private readonly step: number;
  private buffer = new Float32Array(0);
  private position = 0;

  constructor(inputRate: number, outputRate: number) {
    this.step = inputRate / outputRate;
  }

  process(input: Float32Array): Float32Array {
    if (!input.length) return new Float32Array(0);
    const merged = new Float32Array(this.buffer.length + input.length);
    merged.set(this.buffer, 0);
    merged.set(input, this.buffer.length);
    this.buffer = merged;

    const out: number[] = [];
    while (this.position + 1 < this.buffer.length) {
      const index = Math.floor(this.position);
      const fraction = this.position - index;
      const a = this.buffer[index];
      const b = this.buffer[index + 1];
      out.push(a * (1 - fraction) + b * fraction);
      this.position += this.step;
    }

    const consumed = Math.floor(this.position);
    if (consumed > 0) {
      this.buffer = this.buffer.slice(consumed);
      this.position -= consumed;
    }
    return Float32Array.from(out);
  }

  reset(): void {
    this.buffer = new Float32Array(0);
    this.position = 0;
  }
}

export function floatToInt16(samples: Float32Array): Int16Array {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    const value = Math.max(-1, Math.min(1, samples[i]));
    out[i] = value < 0 ? value * 0x8000 : value * 0x7fff;
  }
  return out;
}

export function int16ToFloat(samples: Int16Array): Float32Array {
  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) out[i] = samples[i] / 32768;
  return out;
}

export function rms(analyser: AnalyserNode): number {
  const data = new Float32Array(analyser.fftSize);
  analyser.getFloatTimeDomainData(data);
  let sum = 0;
  for (let i = 0; i < data.length; i += 1) sum += data[i] * data[i];
  return Math.sqrt(sum / data.length);
}

// ---------------------------------------------------------------- microphone

export interface MicrophoneHandle {
  stream: MediaStream;
  context: AudioContext;
  analyser: AnalyserNode;
  /** Native capture rate. Sent to the gateway so it resamples once, properly. */
  sampleRate: number;
  stop: () => Promise<void>;
}

export async function openMicrophone(
  onPcm: (pcm: Int16Array) => void,
): Promise<MicrophoneHandle> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Microphone capture is not available in this browser.");
  }

  // Device DSP is a real ASR variable: aggressive noise suppression can chew
  // up fricatives on some Windows drivers. `?dsp=off` disables all three so the
  // diagnostics can A/B the raw microphone.
  const dsp = new URLSearchParams(window.location.search).get("dsp") !== "off";
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: dsp,
      noiseSuppression: dsp,
      autoGainControl: dsp,
    },
    video: false,
  });

  // Capture at the device rate and let the gateway resample with soxr. Forcing
  // a 16 kHz AudioContext makes the browser resample the mic stream with an
  // unspecified filter, and the old hand-written linear decimator had no
  // anti-alias filter at all: sibilants folded back into the formant band and
  // the recogniser heard noise where the consonants were.
  const AudioContextClass = audioContextConstructor();
  const context = new AudioContextClass({ latencyHint: "interactive" });
  if (context.state === "suspended") {
    // Not fatal: the graph still starts, and the widget offers a resume gesture.
    await context.resume().catch(() => undefined);
  }

  const source = context.createMediaStreamSource(stream);
  const analyser = context.createAnalyser();
  analyser.fftSize = 512;
  analyser.smoothingTimeConstant = 0.6;
  const silent = context.createGain();
  silent.gain.value = 0;
  source.connect(analyser);
  analyser.connect(silent);
  silent.connect(context.destination);

  const chunk = Math.max(160, Math.round(context.sampleRate * 0.02));
  let pending: number[] = [];
  let closed = false;

  const emit = (frame: Float32Array) => {
    if (closed) return;
    for (let i = 0; i < frame.length; i += 1) pending.push(frame[i]);
    while (pending.length >= chunk) {
      const slice = pending.slice(0, chunk);
      pending = pending.slice(chunk);
      onPcm(floatToInt16(Float32Array.from(slice)));
    }
  };

  let node: AudioNode | null = null;
  let processor: ScriptProcessorNode | null = null;
  let worklet: AudioWorkletNode | null = null;

  try {
    const workletUrl = new URL("pcm-worklet.js", window.location.href).toString();
    await context.audioWorklet.addModule(workletUrl);
    worklet = new AudioWorkletNode(context, WORKLET_NAME, {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    worklet.port.onmessage = (event: MessageEvent<Float32Array>) => emit(event.data);
    node = worklet;
  } catch {
    // Older Safari and some embedded webviews have no AudioWorklet. The
    // deprecated script processor is a worse path, not a broken one.
    processor = context.createScriptProcessor(1024, 1, 1);
    processor.onaudioprocess = (event) => {
      emit(event.inputBuffer.getChannelData(0).slice(0));
    };
    node = processor;
  }

  source.connect(node);
  node.connect(silent);

  const stop = async () => {
    if (closed) return;
    closed = true;
    if (worklet) worklet.port.onmessage = null;
    if (processor) processor.onaudioprocess = null;
    try {
      source.disconnect();
      node?.disconnect();
      silent.disconnect();
      analyser.disconnect();
    } catch {
      // A disconnected graph is already the desired state.
    }
    for (const track of stream.getTracks()) track.stop();
    await context.close().catch(() => undefined);
  };

  return { stream, context, analyser, sampleRate: context.sampleRate, stop };
}

// ------------------------------------------------------------------ playback

export class Playback {
  readonly context: AudioContext;
  readonly analyser: AnalyserNode;
  private readonly gain: GainNode;
  private readonly sources = new Set<AudioBufferSourceNode>();
  private readonly resampler: LinearResampler;
  private playhead = 0;
  private closed = false;
  // Wire-level audio identity. A frame that belongs to a generation older than
  // the one already playing, or that arrives out of order, is discarded before
  // it can touch the playhead: after a barge-in the socket can still deliver
  // packets the server selected before it saw the EPOCH.
  private generation = -1;
  private lastSequence = -1;
  // When the audio scheduled in the current run first becomes audible, on the
  // performance.now() clock. Null until something is scheduled, and cleared when
  // the run is flushed. This is the only honest answer to "when did the visitor
  // hear it": a buffer that has been enqueued has not been heard yet.
  private audibleAt: number | null = null;
  // The AudioContext time at which that first buffer starts. Needed to read
  // the device clock back: getOutputTimestamp() reports where the output is,
  // not where the graph is.
  private runStartContext: number | null = null;
  private runEndContext: number | null = null;

  constructor() {
    const AudioContextClass = audioContextConstructor();
    try {
      this.context = new AudioContextClass({ latencyHint: "interactive", sampleRate: TRACK_RATE });
    } catch {
      this.context = new AudioContextClass({ latencyHint: "interactive" });
    }
    this.gain = this.context.createGain();
    this.gain.gain.value = 1;
    this.analyser = this.context.createAnalyser();
    this.analyser.fftSize = 512;
    this.analyser.smoothingTimeConstant = 0.55;
    this.gain.connect(this.analyser);
    this.analyser.connect(this.context.destination);
    this.resampler = new LinearResampler(TRACK_RATE, this.context.sampleRate);
  }

  async resume(): Promise<void> {
    if (this.closed) return;
    if (this.context.state !== "running") {
      await this.context.resume().catch(() => undefined);
    }
  }

  get running(): boolean {
    return !this.closed && this.context.state === "running";
  }

  /**
   * True while audio for the current run is still scheduled or playing.
   *
   * Barge-in detection must key off this, not off the UI state: the server can
   * legitimately report "listening" once it has handed the last frame to the
   * transport while the browser still has hundreds of milliseconds queued.
   */
  get active(): boolean {
    if (this.closed) return false;
    return this.sources.size > 0 || this.playhead > this.context.currentTime;
  }

  /** `performance.now()` at which the current run of audio becomes audible. */
  get audibleFrom(): number | null {
    return this.audibleAt;
  }

  /**
   * `performance.now()` at which the output device actually reached the first
   * scheduled sample, or null while that moment has not happened yet.
   *
   * `audibleFrom` is a prediction made when the buffer is scheduled. This is
   * the observation: `AudioContext.getOutputTimestamp()` says which point in
   * the context timeline the speakers are at, and when. Reporting the
   * prediction as "audible" is how a latency number gets flattered by the
   * audio thread's own backlog.
   */
  audibleConfirmedFrom(): number | null {
    if (this.audibleAt === null || this.runStartContext === null) return null;
    const timestamp = this.context.getOutputTimestamp?.();
    if (
      !timestamp ||
      typeof timestamp.contextTime !== "number" ||
      typeof timestamp.performanceTime !== "number"
    ) {
      return null;
    }
    if (timestamp.contextTime + 0.005 < this.runStartContext) return null;
    return timestamp.performanceTime + (this.runStartContext - timestamp.contextTime) * 1000;
  }

  enqueue(pcm: Int16Array, generation = this.generation, sequence = this.lastSequence + 1): void {
    if (this.closed || !pcm.length) return;
    if (generation < this.generation) return;
    if (generation > this.generation) {
      this.generation = generation;
      this.lastSequence = -1;
    }
    if (sequence <= this.lastSequence) return;
    this.lastSequence = sequence;
    let samples = int16ToFloat(pcm);
    if (this.context.sampleRate !== TRACK_RATE) {
      samples = this.resampler.process(samples);
    }
    if (!samples.length) return;

    const buffer = this.context.createBuffer(1, samples.length, this.context.sampleRate);
    buffer.copyToChannel(samples as Float32Array<ArrayBuffer>, 0);

    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.gain);

    const now = this.context.currentTime;
    if (this.playhead < now + 0.01) {
      // This branch only runs when nothing is queued: the first chunk of a turn,
      // or the first chunk after playback ran dry. It is therefore *not* a
      // jitter absorber for steady playback, and it is paid in full by the
      // visitor on every turn. 25 ms still clears the audio thread's own render
      // quantum with margin, and scheduling later than `now` is what stops the
      // first buffer from being clipped.
      this.playhead = now + 0.025;
      this.audibleAt = performance.now() + (this.playhead - now) * 1000;
      this.runStartContext = this.playhead;
    }
    source.start(this.playhead);
    this.playhead += buffer.duration;
    this.runEndContext = this.playhead;
    this.sources.add(source);
    source.onended = () => this.sources.delete(source);
  }

  /**
   * Fraction (0..1) of the current run that the output clock has passed.
   *
   * Used only to estimate how much of the answer's text was audible. It is
   * deliberately labelled an estimate: without word alignment there is no
   * honest character boundary, and the server records `audible_text` from the
   * character count this produces, never from the fact that audio was sent.
   */
  progress(): number {
    if (this.runStartContext === null || this.runEndContext === null) return 0;
    const total = this.runEndContext - this.runStartContext;
    if (total <= 0) return 0;
    const played = this.context.currentTime - this.runStartContext;
    return Math.max(0, Math.min(1, played / total));
  }

  /** Stop everything already scheduled and duck output for the flush guard. */
  flush(guardMs = 110): void {
    if (this.closed) return;
    for (const source of this.sources) {
      try {
        source.stop();
      } catch {
        // Already ended.
      }
    }
    this.sources.clear();
    this.playhead = 0;
    this.audibleAt = null;
    this.runStartContext = null;
    this.runEndContext = null;
    const now = this.context.currentTime;
    const gain = this.gain.gain;
    gain.cancelScheduledValues(now);
    gain.setValueAtTime(0, now);
    gain.linearRampToValueAtTime(1, now + guardMs / 1000);
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    this.audibleAt = null;
    for (const source of this.sources) {
      try {
        source.stop();
      } catch {
        // Already ended.
      }
    }
    this.sources.clear();
    await this.context.close().catch(() => undefined);
  }
}

/** Force a MediaStream through an analyser without making it audible twice. */
export function analyserForStream(
  context: AudioContext,
  stream: MediaStream,
): { analyser: AnalyserNode; stop: () => void } {
  const source = context.createMediaStreamSource(stream);
  const analyser = context.createAnalyser();
  analyser.fftSize = 512;
  analyser.smoothingTimeConstant = 0.55;
  const silent = context.createGain();
  silent.gain.value = 0;
  source.connect(analyser);
  analyser.connect(silent);
  silent.connect(context.destination);
  return {
    analyser,
    stop: () => {
      try {
        source.disconnect();
        analyser.disconnect();
        silent.disconnect();
      } catch {
        // Already disconnected.
      }
    },
  };
}

// -------------------------------------------------------------- PCM fallback

export interface PcmSocketHandlers {
  onOpen?: () => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (error: Event) => void;
  onAudio: (pcm: Int16Array, generation: number, sequence: number) => void;
  /** Called when the send queue is over the real-time budget and frames are dropped. */
  onOverload?: (queuedMs: number, droppedMs: number) => void;
}

export class PcmSocket {
  private socket: WebSocket | null = null;
  private closed = false;
  // Real-time audio must never accumulate an unbounded backlog: a slow uplink
  // would deliver yesterday's speech to a recogniser that is already answering
  // it. Frames are dropped (newest first) once the socket's own buffer holds
  // more than this much audio, and the UI is told.
  private readonly bytesPerMs: number;
  private readonly maxQueueBytes: number;
  private queuedMs = 0;
  private droppedBytes = 0;

  constructor(
    private readonly url: string,
    private readonly handlers: PcmSocketHandlers,
    options: { bytesPerMs?: number; maxQueueMs?: number } = {},
  ) {
    // 16 kHz mono PCM16 is 32 bytes/ms; the default matches the browser's
    // native capture path (48 kHz => 96 bytes/ms).
    this.bytesPerMs = options.bytesPerMs ?? 96;
    this.maxQueueBytes = Math.round(this.bytesPerMs * (options.maxQueueMs ?? 500));
  }

  /** Milliseconds of microphone audio currently waiting in the socket. */
  get backlogMs(): number {
    return this.queuedMs;
  }

  get droppedMs(): number {
    return this.droppedBytes / this.bytesPerMs;
  }

  open(): Promise<void> {
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(this.url);
      socket.binaryType = "arraybuffer";
      this.socket = socket;
      let settled = false;
      this.droppedBytes = 0;
      this.queuedMs = 0;

      const settle = (error?: Error) => {
        if (settled) return;
        settled = true;
        if (error) reject(error);
        else resolve();
      };

      socket.onopen = () => {
        settle();
        this.handlers.onOpen?.();
      };
      socket.onmessage = (event) => {
        if (typeof event.data === "string") return;
        if (event.data instanceof ArrayBuffer) {
          // pcm16-v2: u32 generation, u32 sequence, then little-endian PCM.
          // A frame shorter than the header is malformed, not audio.
          if (event.data.byteLength < 8) return;
          const header = new DataView(event.data);
          const generation = header.getUint32(0, true);
          const sequence = header.getUint32(4, true);
          this.handlers.onAudio(new Int16Array(event.data, 8), generation, sequence);
        }
      };
      socket.onerror = (event) => {
        this.handlers.onError?.(event);
        settle(new Error("The audio channel could not be opened."));
      };
      socket.onclose = (event) => {
        this.socket = null;
        settle(new Error("The audio channel closed before it was ready."));
        if (!this.closed) this.handlers.onClose?.(event);
      };
    });
  }

  send(pcm: Int16Array): void {
    const socket = this.socket;
    if (!socket || socket.readyState !== WebSocket.OPEN || !pcm.length) return;
    // `bufferedAmount` is the browser's unsent queue. Convert it to audio time,
    // not bytes: the budget is "no more than 500 ms behind live speech".
    const buffered = Math.max(0, socket.bufferedAmount || 0) + pcm.byteLength;
    this.queuedMs = buffered / this.bytesPerMs;
    if (buffered > this.maxQueueBytes) {
      this.droppedBytes += pcm.byteLength;
      this.handlers.onOverload?.(this.queuedMs, this.droppedMs);
      return;
    }
    socket.send(pcm.buffer);
  }

  /**
   * Send a JSON control message (interrupt, audible report).
   *
   * Control is a separate frame type, so it never competes for the audio
   * budget and is never mistaken for PCM by the gateway.
   */
  sendControl(message: Record<string, unknown>): void {
    const socket = this.socket;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    try {
      socket.send(JSON.stringify(message));
    } catch {
      // A closing socket loses the report, not the conversation.
    }
  }

  close(): void {
    this.closed = true;
    const socket = this.socket;
    this.socket = null;
    if (socket && socket.readyState <= WebSocket.OPEN) {
      try {
        socket.close();
      } catch {
        // Closing a closing socket is harmless.
      }
    }
  }
}

// -------------------------------------------------------------------- WebRTC

export interface RtcHandle {
  pc: RTCPeerConnection;
  stream: MediaStream;
  close: () => void;
}

export interface RtcOptions {
  sessionId: string;
  token: string;
  iceServers: RTCIceServer[];
  forceRelay: boolean;
  microphone: MediaStream | null;
  remoteAudio: HTMLAudioElement | null;
  onRemoteStream?: (stream: MediaStream) => void;
  onStateChange?: (state: RTCPeerConnectionState) => void;
}

function waitForIce(pc: RTCPeerConnection, timeoutMs: number): Promise<void> {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    const timer = window.setTimeout(done, timeoutMs);
    function done() {
      window.clearTimeout(timer);
      pc.removeEventListener("icegatheringstatechange", check);
      resolve();
    }
    function check() {
      if (pc.iceGatheringState === "complete") done();
    }
    pc.addEventListener("icegatheringstatechange", check);
  });
}

function waitForConnected(pc: RTCPeerConnection, timeoutMs: number): Promise<void> {
  if (pc.connectionState === "connected") return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => {
      cleanup();
      reject(new Error("WebRTC did not connect in time."));
    }, timeoutMs);

    function cleanup() {
      window.clearTimeout(timer);
      pc.removeEventListener("connectionstatechange", check);
    }
    function check() {
      if (pc.connectionState === "connected") {
        cleanup();
        resolve();
      } else if (pc.connectionState === "failed") {
        cleanup();
        reject(new Error("WebRTC negotiation failed."));
      }
    }
    pc.addEventListener("connectionstatechange", check);
  });
}

export async function connectWebRTC(options: RtcOptions): Promise<RtcHandle> {
  const pc = new RTCPeerConnection({
    iceServers: options.iceServers,
    iceTransportPolicy: options.forceRelay ? "relay" : "all",
  });

  const remote = new MediaStream();
  pc.ontrack = (event) => {
    const stream = event.streams[0] ?? new MediaStream([event.track]);
    for (const track of stream.getTracks()) {
      if (!remote.getTracks().includes(track)) remote.addTrack(track);
    }
    if (options.remoteAudio) {
      options.remoteAudio.srcObject = remote;
      options.remoteAudio.autoplay = true;
      void options.remoteAudio.play().catch(() => undefined);
    }
    options.onRemoteStream?.(remote);
  };
  pc.onconnectionstatechange = () => options.onStateChange?.(pc.connectionState);

  for (const track of options.microphone?.getAudioTracks() ?? []) {
    pc.addTrack(track, options.microphone as MediaStream);
  }

  const close = () => {
    try {
      pc.close();
    } catch {
      // Already closed.
    }
    if (options.remoteAudio) {
      options.remoteAudio.srcObject = null;
    }
  };

  try {
    const offer = await pc.createOffer({ offerToReceiveAudio: true });
    await pc.setLocalDescription(offer);
    await waitForIce(pc, 1_500);
    const local = pc.localDescription;
    if (!local) throw new Error("WebRTC produced no local description.");
    const answer = await api.offer(options.sessionId, options.token, local.sdp, local.type);
    await pc.setRemoteDescription({ type: "answer", sdp: answer.sdp });
    await waitForConnected(pc, 5_000);
    return { pc, stream: remote, close };
  } catch (error) {
    close();
    throw error;
  }
}

export function silenceElement(element: HTMLAudioElement | null, guardMs = 160): void {
  if (!element) return;
  element.muted = true;
  window.setTimeout(() => {
    if (element) element.muted = false;
  }, guardMs);
}
