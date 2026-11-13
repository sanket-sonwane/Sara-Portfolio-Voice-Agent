/**
 * The conversation controller.
 *
 * It owns the session token, the two sockets, the media transport and the
 * transcript. The React component is only a rendering of this state.
 *
 * Interruption is deliberately split: the browser silences locally the instant
 * it hears speech so no stale syllable is audible, and the gateway owns the
 * epoch bump that cancels providers. Either side alone is insufficient.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { RefObject } from "react";

import * as api from "./api";
import { clientEosToAudible } from "./metrics";
import {
  PcmSocket,
  Playback,
  analyserForStream,
  connectWebRTC,
  openMicrophone,
  rms,
  silenceElement,
  type MicrophoneHandle,
  type RtcHandle,
} from "./audio";

export type ViewState =
  | "idle"
  | "permission"
  | "connecting"
  | "connected"
  | "listening"
  | "processing"
  | "speaking"
  | "tool"
  | "interrupted"
  | "reconnecting"
  | "error"
  | "ended";

export interface TranscriptEntry {
  id: string;
  role: "user" | "assistant";
  text: string;
  partial: boolean;
  interrupted?: boolean;
  turnId: string;
  source: string;
  at: number;
}

export interface UiError {
  message: string;
  recoverable: boolean;
  code?: "mic" | "session" | "transport" | "expired";
}

export interface Notice {
  message: string;
  level: "info" | "warning" | "error";
  at: number;
}

export interface LogEntry {
  at: number;
  type: string;
  turn: string;
  generation: number;
  summary: string;
}

export interface EvidenceNote {
  id: string;
  kind: "fact" | "chunk" | "tool" | "lookup";
  label: string;
  detail: string;
  at: number;
}

export interface BrowserStats {
  candidateType?: string;
  rttMs?: number;
  jitterMs?: number;
  packetsLost?: number;
  bytesSent?: number;
  bytesReceived?: number;
  connectionState?: string;
}

export interface Analysers {
  input: AnalyserNode | null;
  output: AnalyserNode | null;
}

/**
 * One completed turn as the visitor experienced it. `server` is the gateway's
 * own `eos_to_answer_audio_ms`; `client` is this browser's speech-end →
 * audible. Both are absent when the turn was not a spoken question (a typed
 * turn, a greeting, a barge-in), which is why they are optional rather than
 * zero: a zero would read as a suspiciously fast turn.
 */
export interface TurnSample {
  /** `text` for a typed question, otherwise the transcript source. */
  source?: string;
  /** Server: end of speech → the first frame actually left the process. */
  first?: number;
  server?: number;
  client?: number;
}

const MAX_RECONNECTS = 6;
const MAX_TURN_SAMPLES = 8;

let transcriptSequence = 0;
function nextTranscriptId(prefix: string): string {
  transcriptSequence += 1;
  return `${prefix}-${Date.now().toString(36)}-${transcriptSequence}`;
}

function describe(event: api.ServerEvent): string {
  const payload = event.payload;
  switch (event.type) {
    case "state":
      return `${String(payload.state ?? "")}${payload.detail ? ` — ${String(payload.detail)}` : ""}`;
    case "transcript.partial":
    case "transcript.final":
    case "assistant.delta":
    case "assistant.final":
      return String(payload.text ?? "").slice(0, 100);
    case "metrics.turn":
      return typeof payload.eos_to_answer_audio_ms === "number"
        ? `${payload.eos_to_answer_audio_ms} ms to answer audio`
        : "turn recorded";
    case "evidence": {
      if (Array.isArray(payload.blocks)) return `${payload.blocks.length} fact blocks`;
      if (Array.isArray(payload.chunks)) return `${payload.chunks.length} case-study chunks`;
      if (payload.tool && typeof payload.tool === "object") {
        const tool = payload.tool as { name?: unknown };
        return `tool ${String(tool.name ?? "")}`;
      }
      return "evidence";
    }
    case "epoch":
      return `generation ${String(payload.generation ?? "")} (${String(payload.reason ?? "")})`;
    case "notice":
    case "error":
      return String(payload.message ?? "");
    default:
      return "";
  }
}

function microphoneErrorMessage(error: unknown): string {
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError" || error.name === "SecurityError") {
      return "Microphone access was blocked. Allow it in the browser, or continue with text.";
    }
    if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
      return "No microphone was found. You can still type a question.";
    }
    if (error.name === "NotReadableError") {
      return "The microphone is in use by another application. Close it and try again, or type instead.";
    }
  }
  if (error instanceof Error && error.message) return error.message;
  return "The microphone could not be opened. You can continue with text.";
}

export interface UseSarang {
  view: ViewState;
  detail: string;
  active: boolean;
  connected: boolean;
  micReady: boolean;
  muted: boolean;
  transport: "webrtc" | "ws" | null;
  serverState: api.ServerState;
  config: api.SarangConfig | null;
  transcript: TranscriptEntry[];
  sources: api.EvidenceBlock[];
  evidence: EvidenceNote[];
  lastTurn: api.TurnMetrics | null;
  turnLog: TurnSample[];
  metrics: api.MetricsResponse | null;
  browserStats: BrowserStats | null;
  connectMs: number | null;
  elapsed: number;
  notice: Notice | null;
  error: UiError | null;
  eventLog: LogEntry[];
  diagnosticsOpen: boolean;
  audioRunning: boolean;
  getAnalysers: () => Analysers;
  start: (options?: { mic?: boolean }) => Promise<void>;
  stop: () => void;
  ask: (text: string) => void;
  interrupt: (reason?: string) => void;
  toggleMute: () => void;
  dismissError: () => void;
  resumeAudio: () => void;
  clearNotice: () => void;
  setDiagnosticsOpen: (open: boolean) => void;
}

export function useSarang(remoteAudio: RefObject<HTMLAudioElement | null>): UseSarang {
  const [view, setView] = useState<ViewState>("idle");
  const [detail, setDetail] = useState("");
  const [active, setActive] = useState(false);
  const [connected, setConnected] = useState(false);
  const [micReady, setMicReady] = useState(false);
  const [muted, setMutedState] = useState(false);
  const [transport, setTransport] = useState<"webrtc" | "ws" | null>(null);
  const [serverState, setServerState] = useState<api.ServerState>("idle");
  const [config, setConfig] = useState<api.SarangConfig | null>(null);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [sources, setSources] = useState<api.EvidenceBlock[]>([]);
  const [evidence, setEvidence] = useState<EvidenceNote[]>([]);
  const [lastTurn, setLastTurn] = useState<api.TurnMetrics | null>(null);
  // A real-mic pass is read as a distribution, not a single number, and the
  // drawer used to overwrite the previous turn every time. Keeping the recent
  // samples is what makes four spoken questions reportable.
  const [turnLog, setTurnLog] = useState<TurnSample[]>([]);
  const [metrics, setMetrics] = useState<api.MetricsResponse | null>(null);
  const [browserStats, setBrowserStats] = useState<BrowserStats | null>(null);
  const [connectMs, setConnectMs] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [error, setError] = useState<UiError | null>(null);
  const [eventLog, setEventLog] = useState<LogEntry[]>([]);
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);
  const [audioRunning, setAudioRunning] = useState(false);

  const viewRef = useRef<ViewState>("idle");
  const mutedRef = useRef(false);
  const errorRef = useRef<UiError | null>(null);
  const mountedRef = useRef(true);
  const sessionRef = useRef<api.SessionResponse | null>(null);
  const eventsRef = useRef<WebSocket | null>(null);
  const audioRef = useRef<PcmSocket | null>(null);
  const rtcRef = useRef<RtcHandle | null>(null);
  const micRef = useRef<MicrophoneHandle | null>(null);
  const playbackRef = useRef<Playback | null>(null);
  const streamAnalyserRef = useRef<{ analyser: AnalyserNode; stop: () => void } | null>(null);
  const analysersRef = useRef<Analysers>({ input: null, output: null });
  const partialUserIdRef = useRef<string | null>(null);
  const assistantIdRef = useRef<{ turnId: string; id: string } | null>(null);
  const generationRef = useRef(0);
  const lifespanRef = useRef(0);
  const startingRef = useRef(false);
  const endedRef = useRef(false);
  const wantMicRef = useRef(true);
  const transportReadyRef = useRef(false);
  const connectStartedRef = useRef(0);
  const noticeTimerRef = useRef<number | null>(null);
  const bargeTimerRef = useRef<number | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const reconnectEventsRef = useRef(false);
  const reconnectAudioRef = useRef(false);
  const reconnectAttemptsRef = useRef(0);
  const lastBargeRef = useRef(0);
  const speechCandidateRef = useRef(0);
  const overloadNoticeAtRef = useRef(0);
  // Visitor-experienced latency, both stamps on the performance.now() clock so
  // no server/client clock sync is needed. `eosAt` is when this client saw the
  // server leave `user_speaking`; `audibleAt` is when the first answer chunk of
  // that turn is scheduled to be heard. Their difference is the leg the server
  // cannot measure: the return network path plus playback start.
  const eosAtRef = useRef<number | null>(null);
  const turnAudibleAtRef = useRef<number | null>(null);
  const prevServerStateRef = useRef<api.ServerState>("idle");
  const noiseFloorRef = useRef(0.01);
  const openEventsRef = useRef<(session: api.SessionResponse, lifespan: number) => void>(() => undefined);
  const startAudioRef = useRef<(session: api.SessionResponse, lifespan: number) => Promise<void>>(
    async () => undefined,
  );

  const setViewState = useCallback((next: ViewState, nextDetail = "") => {
    viewRef.current = next;
    setView(next);
    setDetail(nextDetail);
  }, []);

  const setUiError = useCallback((next: UiError | null) => {
    errorRef.current = next;
    setError(next);
  }, []);

  const pushNotice = useCallback((message: string, level: Notice["level"] = "info") => {
    if (noticeTimerRef.current) window.clearTimeout(noticeTimerRef.current);
    setNotice({ message, level, at: Date.now() });
    if (level !== "error") {
      noticeTimerRef.current = window.setTimeout(() => {
        if (mountedRef.current) setNotice(null);
      }, level === "warning" ? 9_000 : 5_500);
    }
  }, []);

  const clearNotice = useCallback(() => {
    if (noticeTimerRef.current) window.clearTimeout(noticeTimerRef.current);
    setNotice(null);
  }, []);

  const setMuted = useCallback((next: boolean) => {
    mutedRef.current = next;
    setMutedState(next);
  }, []);

  const getAnalysers = useCallback(() => analysersRef.current, []);

  // ------------------------------------------------------------- teardown

  const closeMic = useCallback(() => {
    const mic = micRef.current;
    micRef.current = null;
    analysersRef.current.input = null;
    setMicReady(false);
    if (mic) void mic.stop();
    if (bargeTimerRef.current) {
      window.clearInterval(bargeTimerRef.current);
      bargeTimerRef.current = null;
    }
  }, []);

  const closeAudio = useCallback(() => {
    audioRef.current?.close();
    audioRef.current = null;
    rtcRef.current?.close();
    rtcRef.current = null;
    streamAnalyserRef.current?.stop();
    streamAnalyserRef.current = null;
    analysersRef.current.output = null;
    const playback = playbackRef.current;
    playbackRef.current = null;
    if (playback) void playback.close();
    const element = remoteAudio.current;
    if (element) {
      element.pause();
      element.srcObject = null;
    }
    transportReadyRef.current = false;
    reconnectAudioRef.current = false;
    setTransport(null);
    setAudioRunning(false);
  }, [remoteAudio]);

  const closeEvents = useCallback(() => {
    const socket = eventsRef.current;
    eventsRef.current = null;
    if (!socket) return;
    socket.onopen = null;
    socket.onmessage = null;
    socket.onerror = null;
    socket.onclose = null;
    try {
      socket.close();
    } catch {
      // Already closing.
    }
  }, []);

  const teardown = useCallback(
    (notifyServer: boolean) => {
      endedRef.current = true;
      lifespanRef.current += 1;
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      closeMic();
      closeAudio();
      const socket = eventsRef.current;
      if (notifyServer && socket && socket.readyState === WebSocket.OPEN) {
        try {
          socket.send(JSON.stringify({ type: "end" }));
        } catch {
          // Closing anyway.
        }
        window.setTimeout(() => closeEvents(), 150);
      } else {
        closeEvents();
      }
      sessionRef.current = null;
      setActive(false);
      setConnected(false);
      setServerState("ended");
    },
    [closeAudio, closeEvents, closeMic],
  );

  // ----------------------------------------------------------- event intake

  const applyServerState = useCallback(
    (state: api.ServerState, stateDetail: string, mutedFlag?: boolean) => {
      setServerState(state);
      if (state === "user_speaking") {
        // A new utterance: the next run of playback is this turn's answer, and
        // the previous turn's stamps must not be reused.
        turnAudibleAtRef.current = null;
        eosAtRef.current = null;
      } else if (prevServerStateRef.current === "user_speaking") {
        // The server decided the turn ended. Everything after this point is the
        // leg the visitor actually waits through, measured from here.
        eosAtRef.current = performance.now();
      }
      prevServerStateRef.current = state;
      if (typeof mutedFlag === "boolean" && mutedFlag !== mutedRef.current) {
        setMuted(mutedFlag);
      }
      if (state !== "error" && errorRef.current?.recoverable) {
        setUiError(null);
      }
      switch (state) {
        case "idle":
        case "ready":
          if (viewRef.current !== "connecting" && viewRef.current !== "permission") {
            setViewState("connected", "Connected");
          } else if (transportReadyRef.current || !wantMicRef.current) {
            setViewState("connected", "Connected");
          }
          break;
        case "listening":
          setViewState("listening", mutedRef.current ? "Microphone muted" : "Listening");
          break;
        case "user_speaking":
          setViewState("listening", "Hearing you");
          break;
        case "transcribing":
          setViewState("processing", "Transcribing your question");
          break;
        case "thinking":
          setViewState("processing", stateDetail || "Thinking");
          break;
        case "reconnecting":
          setViewState("reconnecting", stateDetail || "Restoring the connection…");
          break;
        case "connecting":
          setViewState("connecting", stateDetail || "Connecting…");
          break;
        case "tool_working":
          setViewState("tool", stateDetail || "Searching the case studies");
          break;
        case "speaking":
          setViewState("speaking", "Sara is speaking");
          break;
        case "interrupted":
          setViewState("interrupted", "Interrupted — listening again");
          break;
        case "muted":
          setViewState("listening", "Microphone muted");
          break;
        case "ended":
          if (!endedRef.current) {
            endedRef.current = true;
            closeAudio();
            closeEvents();
            setActive(false);
            setConnected(false);
          }
          setViewState("ended", "Session ended");
          break;
        case "error":
          setViewState("error", stateDetail || "Something went wrong");
          break;
      }
    },
    [closeAudio, closeEvents, setMuted, setUiError, setViewState],
  );

  const handleEvent = useCallback(
    (event: api.ServerEvent) => {
      if (!mountedRef.current) return;
      if (event.type === "ping") return;
      const generation = Number(event.generation_id) || 0;
      if (generation > generationRef.current) generationRef.current = generation;
      const stale = generation > 0 && generation < generationRef.current;

      if (event.type !== "assistant.delta" || !stale) {
        setEventLog((previous) => [
          ...previous.slice(-79),
          {
            at: Date.now(),
            type: event.type,
            turn: event.turn_id,
            generation,
            summary: describe(event),
          },
        ]);
      }

      if (
        stale &&
        ["assistant.delta", "transcript.partial", "transcript.final", "assistant.final", "metrics.turn"].includes(
          event.type,
        )
      ) {
        return;
      }

      switch (event.type) {
        case "session.ready": {
          const nextConfig = event.payload.config as api.SarangConfig | undefined;
          if (nextConfig) setConfig(nextConfig);
          setConnected(true);
          if (viewRef.current === "reconnecting") {
            setViewState(transportReadyRef.current || !wantMicRef.current ? "connected" : "connecting", "Connected");
          } else if (!wantMicRef.current) {
            setViewState("connected", "Connected — text mode");
          }
          break;
        }
        case "state": {
          applyServerState(
            String(event.payload.state ?? "idle") as api.ServerState,
            String(event.payload.detail ?? ""),
            typeof event.payload.muted === "boolean" ? event.payload.muted : undefined,
          );
          break;
        }
        case "transcript.partial": {
          const text = String(event.payload.text ?? "").trim();
          if (!text) break;
          const existing = partialUserIdRef.current;
          const id = existing ?? nextTranscriptId("user");
          if (!existing) partialUserIdRef.current = id;
          setTranscript((previous) => {
            if (existing) {
              const index = previous.findIndex((entry) => entry.id === existing);
              if (index >= 0) {
                const copy = [...previous];
                copy[index] = { ...copy[index], text };
                return copy;
              }
            }
            return [
              ...previous,
              {
                id,
                role: "user",
                text,
                partial: true,
                turnId: event.turn_id,
                source: String(event.payload.source ?? "voice"),
                at: Date.now(),
              },
            ];
          });
          break;
        }
        case "transcript.final": {
          const text = String(event.payload.text ?? "").trim();
          if (!text) break;
          const partialId = partialUserIdRef.current;
          partialUserIdRef.current = null;
          const source = String(event.payload.source ?? "voice");
          const finalId = partialId ?? nextTranscriptId("user");
          setTranscript((previous) => {
            const entry: TranscriptEntry = {
              id: finalId,
              role: "user",
              text,
              partial: false,
              turnId: event.turn_id,
              source,
              at: Date.now(),
            };
            if (partialId) {
              const index = previous.findIndex((candidate) => candidate.id === partialId);
              if (index >= 0) {
                const copy = [...previous];
                copy[index] = entry;
                return copy;
              }
            }
            return [...previous, entry];
          });
          break;
        }
        case "assistant.delta": {
          const text = String(event.payload.text ?? "");
          if (!text) break;
          const turnId = event.turn_id;
          const tracked = assistantIdRef.current;
          if (!tracked || tracked.turnId !== turnId) {
            const id = nextTranscriptId("assistant");
            assistantIdRef.current = { turnId, id };
            setTranscript((previous) => [
              ...previous,
              {
                id,
                role: "assistant",
                text,
                partial: true,
                turnId,
                source: "voice",
                at: Date.now(),
              },
            ]);
          } else {
            setTranscript((previous) =>
              previous.map((entry) =>
                entry.id === tracked.id ? { ...entry, text: entry.text + text } : entry,
              ),
            );
          }
          break;
        }
        case "assistant.final": {
          const text = String(event.payload.text ?? "").trim();
          if (!text) break;
          const tracked = assistantIdRef.current;
          assistantIdRef.current = null;
          setTranscript((previous) => {
            const indexByTrack =
              tracked && tracked.turnId === event.turn_id
                ? previous.findIndex((entry) => entry.id === tracked.id)
                : -1;
            const indexByTurn = previous.findIndex(
              (entry) => entry.role === "assistant" && entry.partial && entry.turnId === event.turn_id,
            );
            const index = indexByTrack >= 0 ? indexByTrack : indexByTurn;
            if (index >= 0) {
              const copy = [...previous];
              copy[index] = {
                ...copy[index],
                text,
                partial: false,
                interrupted: false,
                at: Date.now(),
              };
              return copy;
            }
            return [
              ...previous,
              {
                id: nextTranscriptId("assistant"),
                role: "assistant",
                text,
                partial: false,
                turnId: event.turn_id,
                source: "voice",
                at: Date.now(),
              },
            ];
          });
          break;
        }
        case "evidence": {
          const blocks = event.payload.blocks;
          if (Array.isArray(blocks)) {
            setSources(blocks as api.EvidenceBlock[]);
            setEvidence((previous) => [
              ...previous.slice(-19),
              {
                id: nextTranscriptId("fact"),
                kind: "fact",
                label: `${blocks.length} fact ${blocks.length === 1 ? "block" : "blocks"}`,
                detail: (blocks as api.EvidenceBlock[]).map((block) => block.title).join(" · "),
                at: Date.now(),
              },
            ]);
          }
          const chunks = event.payload.chunks;
          if (Array.isArray(chunks)) {
            setEvidence((previous) => [
              ...previous.slice(-19),
              ...(chunks as api.EvidenceChunk[]).map((chunk) => ({
                id: nextTranscriptId("chunk"),
                kind: "chunk" as const,
                label: `Case study · ${chunk.doc_id}`,
                detail: `${chunk.title} — score ${chunk.score.toFixed(2)}`,
                at: Date.now(),
              })),
            ]);
          }
          if (event.payload.tool && typeof event.payload.tool === "object") {
            const tool = event.payload.tool as api.ToolEvidence;
            setEvidence((previous) => [
              ...previous.slice(-19),
              {
                id: nextTranscriptId("tool"),
                kind: "tool",
                label: `Tool · ${tool.name}`,
                detail: `${tool.ok ? "ok" : "failed"} · ${tool.duration_ms} ms · ${tool.chars} chars`,
                at: Date.now(),
              },
            ]);
          }
          break;
        }
        case "metrics.turn": {
          const turn = event.payload as unknown as api.TurnMetrics;
          const eos = eosAtRef.current;
          const audible = turnAudibleAtRef.current;
          // Only meaningful when both stamps belong to the same turn and the
          // answer came after the end of speech. Anything else is a text turn,
          // a barge-in, or a greeting, and must not invent a number.
          const clientMs = clientEosToAudible(eos, audible);
          setLastTurn({ ...turn, client_eos_to_audible_ms: clientMs });
          if (turn.eos_to_answer_audio_ms !== undefined || clientMs !== undefined) {
            setTurnLog((previous) => [
              ...previous.slice(-(MAX_TURN_SAMPLES - 1)),
              {
                source: turn.source,
                first: turn.eos_to_first_audio_ms,
                server: turn.eos_to_answer_audio_ms,
                client: clientMs,
              },
            ]);
          }
          break;
        }
        case "epoch": {
          const nextGeneration = Number(event.payload.generation);
          if (Number.isFinite(nextGeneration)) generationRef.current = nextGeneration;
          playbackRef.current?.flush();
          silenceElement(remoteAudio.current);
          setTranscript((previous) =>
            previous.map((entry) =>
              entry.role === "assistant" && entry.partial
                ? { ...entry, partial: false, interrupted: true }
                : entry,
            ),
          );
          assistantIdRef.current = null;
          if (viewRef.current !== "ended") setViewState("interrupted", "Interrupted — listening again");
          break;
        }
        case "notice": {
          const level = String(event.payload.level ?? "info") as Notice["level"];
          const message = String(event.payload.message ?? "");
          if (message) pushNotice(message, level);
          break;
        }
        case "error": {
          const message = String(event.payload.message ?? "Something went wrong.");
          const recoverable = event.payload.recoverable !== false;
          setUiError({ message, recoverable });
          pushNotice(message, recoverable ? "warning" : "error");
          break;
        }
        default:
          break;
      }
    },
    [applyServerState, pushNotice, remoteAudio, setMuted, setUiError, setViewState],
  );

  // --------------------------------------------------------------- network

  const scheduleEventsReconnect = useCallback(
    (session: api.SessionResponse, lifespan: number) => {
      if (lifespan !== lifespanRef.current || endedRef.current || reconnectEventsRef.current) return;
      reconnectEventsRef.current = true;
      reconnectAttemptsRef.current += 1;
      const attempt = reconnectAttemptsRef.current;
      if (attempt > MAX_RECONNECTS) {
        reconnectEventsRef.current = false;
        setUiError({
          message: "The connection to Sara was lost. Check your network and try again.",
          recoverable: true,
          code: "transport",
        });
        setViewState("error", "Connection lost");
        return;
      }
      setViewState("reconnecting", "Restoring the real-time channel…");
      const delay = Math.min(6_000, 400 * 2 ** Math.min(attempt - 1, 4));
      reconnectTimerRef.current = window.setTimeout(() => {
        reconnectTimerRef.current = null;
        if (lifespan !== lifespanRef.current || endedRef.current) return;
        openEventsRef.current(session, lifespan);
      }, delay);
    },
    [setUiError, setViewState],
  );

  const openEvents = useCallback(
    (session: api.SessionResponse, lifespan: number) => {
      if (lifespan !== lifespanRef.current || endedRef.current) return;
      const url = api.socketUrl("/ws/events", {
        session_id: session.session_id,
        token: session.token,
      });
      let socket: WebSocket;
      try {
        socket = new WebSocket(url);
      } catch {
        scheduleEventsReconnect(session, lifespan);
        return;
      }
      eventsRef.current = socket;
      socket.onopen = () => {
        if (lifespan !== lifespanRef.current) return;
        reconnectAttemptsRef.current = 0;
        reconnectEventsRef.current = false;
        if (viewRef.current === "reconnecting" && (transportReadyRef.current || !wantMicRef.current)) {
          setViewState("connected", "Connected");
        }
      };
      socket.onmessage = (message) => {
        if (lifespan !== lifespanRef.current) return;
        try {
          handleEvent(JSON.parse(String(message.data)) as api.ServerEvent);
        } catch {
          // A malformed frame must not break the conversation.
        }
      };
      socket.onerror = () => {
        // onclose follows and owns recovery.
      };
      socket.onclose = (event) => {
        if (eventsRef.current === socket) eventsRef.current = null;
        if (lifespan !== lifespanRef.current || endedRef.current) return;
        if (event.code === 4401 || event.code === 4404) {
          setUiError({
            message: "This session expired. Start a new conversation.",
            recoverable: false,
            code: "expired",
          });
          setViewState("error", "Session expired");
          return;
        }
        scheduleEventsReconnect(session, lifespan);
      };
    },
    [handleEvent, scheduleEventsReconnect, setUiError, setViewState],
  );

  useEffect(() => {
    openEventsRef.current = openEvents;
  }, [openEvents]);

  // ----------------------------------------------------------- audio setup

  const flushLocalAudio = useCallback(() => {
    playbackRef.current?.flush();
    silenceElement(remoteAudio.current);
  }, [remoteAudio]);

  const interrupt = useCallback(
    (reason = "barge-in") => {
      const session = sessionRef.current;
      if (!session) return;
      flushLocalAudio();
      const socket = eventsRef.current;
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "interrupt", reason }));
      } else {
        void api.sendInterrupt(session.session_id, session.token, reason).catch(() => undefined);
      }
      if (["speaking", "processing", "tool"].includes(viewRef.current)) {
        setViewState("interrupted", "Interrupted — listening again");
      }
    },
    [flushLocalAudio, setViewState],
  );

  const startBargeWatch = useCallback(() => {
    if (bargeTimerRef.current) return;
    bargeTimerRef.current = window.setInterval(() => {
      const mic = micRef.current;
      if (!mic || mutedRef.current || !wantMicRef.current) return;
      const level = (() => {
        try {
          return rms(mic.analyser);
        } catch {
          return 0;
        }
      })();
      if (level < 0.02) {
        noiseFloorRef.current = noiseFloorRef.current * 0.97 + level * 0.03;
      }
      const threshold = Math.max(0.045, noiseFloorRef.current * 4);
      // Playback is the ground truth: the server may already be "listening"
      // while the browser still has the tail of the answer scheduled. Watching
      // the queue as well as the view state is what keeps a late barge-in from
      // being ignored.
      const inResponse =
        ["speaking", "processing", "tool"].includes(viewRef.current) ||
        playbackRef.current?.active === true;
      if (inResponse && level > threshold) {
        if (!speechCandidateRef.current) speechCandidateRef.current = performance.now();
        // 120 ms of sustained level, checked every 30 ms: detection costs
        // 120-150 ms instead of 160-205 ms. The server VAD (Silero) fires on
        // its own at about 128 ms of speech, so either path can win; this one
        // is what stops playback before the round trip finishes.
        if (performance.now() - speechCandidateRef.current > 120 && performance.now() - lastBargeRef.current > 700) {
          speechCandidateRef.current = 0;
          lastBargeRef.current = performance.now();
          interrupt("barge-in");
        }
      } else {
        speechCandidateRef.current = 0;
      }
    }, 30);
  }, [interrupt]);

  const startPcm = useCallback(
    async (session: api.SessionResponse, lifespan: number) => {
      const socket = new PcmSocket(
        api.socketUrl("/ws/audio", {
          session_id: session.session_id,
          token: session.token,
          sr: micRef.current?.sampleRate ?? 48_000,
        }),
        {
          onAudio: (pcm) => {
            const playback = playbackRef.current;
            if (!playback) return;
            playback.enqueue(pcm);
            // Snapshot the first run of playback for this turn. Reading it later
            // would be wrong: a dry spell mid-answer starts a new run and would
            // overwrite the only stamp that answers "when did the visitor hear
            // the first word".
            if (turnAudibleAtRef.current === null && playback.audibleFrom !== null) {
              turnAudibleAtRef.current = playback.audibleFrom;
            }
          },
          onOpen: () => {
            if (lifespan !== lifespanRef.current || endedRef.current) return;
            transportReadyRef.current = true;
            reconnectAudioRef.current = false;
            setTransport("ws");
            setConnectMs(Math.round(performance.now() - connectStartedRef.current));
            if (!["speaking", "processing", "tool"].includes(viewRef.current)) {
              setViewState("connected", "Connected");
            }
            startBargeWatch();
          },
          onOverload: (queuedMs) => {
            // Dropping audio is better than transcribing yesterday's speech,
            // but the visitor still deserves to know the uplink is behind.
            const now = performance.now();
            if (now - overloadNoticeAtRef.current < 5_000) return;
            overloadNoticeAtRef.current = now;
            pushNotice(
              `Your connection is behind live audio (${Math.round(queuedMs)} ms queued); some microphone audio was dropped.`,
              "warning",
            );
          },
          onClose: () => {
            if (lifespan !== lifespanRef.current || endedRef.current) return;
            if (audioRef.current === socket) audioRef.current = null;
            transportReadyRef.current = false;
            setTransport(null);
            if (reconnectAudioRef.current) return;
            reconnectAudioRef.current = true;
            pushNotice("The audio channel dropped. Reconnecting…", "warning");
            setViewState("reconnecting", "Reconnecting audio…");
            window.setTimeout(() => {
              reconnectAudioRef.current = false;
              if (lifespan !== lifespanRef.current || endedRef.current) return;
              void startAudioRef.current(session, lifespan);
            }, 700);
          },
          onError: () => {
            // onClose owns the recovery path.
          },
        },
      );
      audioRef.current = socket;
      await socket.open();
    },
    [pushNotice, setViewState, startBargeWatch],
  );

  const startAudio = useCallback(
    async (session: api.SessionResponse, lifespan: number) => {
      if (lifespan !== lifespanRef.current || endedRef.current) return;
      const mic = micRef.current;

      let playback = playbackRef.current;
      if (!playback) {
        playback = new Playback();
        playbackRef.current = playback;
      }
      await playback.resume();
      setAudioRunning(playback.running);

      if (session.config.webrtc && mic) {
        try {
          const handle = await connectWebRTC({
            sessionId: session.session_id,
            token: session.token,
            iceServers: session.ice_servers,
            forceRelay: session.force_relay,
            microphone: mic.stream,
            remoteAudio: remoteAudio.current,
            onRemoteStream: (stream) => {
              const context = playbackRef.current?.context;
              if (!context) return;
              streamAnalyserRef.current?.stop();
              streamAnalyserRef.current = analyserForStream(context, stream);
              analysersRef.current.output = streamAnalyserRef.current.analyser;
            },
            onStateChange: (state) => {
              if (lifespan !== lifespanRef.current || endedRef.current) return;
              if (state === "connected") {
                transportReadyRef.current = true;
                setTransport("webrtc");
                if (viewRef.current === "reconnecting") setViewState("connected", "Connected");
              } else if (state === "disconnected") {
                setViewState("reconnecting", "Media connection interrupted…");
              } else if (state === "failed") {
                transportReadyRef.current = false;
                setTransport(null);
                pushNotice("The WebRTC connection failed. Falling back to WebSocket audio.", "warning");
                rtcRef.current?.close();
                rtcRef.current = null;
                void startPcm(session, lifespan).catch(() => {
                  setUiError({
                    message: "The audio channel could not be established. You can still type.",
                    recoverable: true,
                    code: "transport",
                  });
                  setViewState("error", "Audio unavailable");
                });
              }
            },
          });
          if (lifespan !== lifespanRef.current || endedRef.current) {
            handle.close();
            return;
          }
          rtcRef.current = handle;
          transportReadyRef.current = true;
          setTransport("webrtc");
          setConnectMs(Math.round(performance.now() - connectStartedRef.current));
          if (!["speaking", "processing", "tool"].includes(viewRef.current)) {
            setViewState("connected", "Connected");
          }
          startBargeWatch();
          analysersRef.current.output = playback.analyser;
          return;
        } catch (error) {
          console.warn("WebRTC unavailable; using the WebSocket audio channel.", error);
          pushNotice("WebRTC isn't available here — using the WebSocket audio channel.", "info");
        }
      }

      try {
        await startPcm(session, lifespan);
        analysersRef.current.output = playback.analyser;
        setAudioRunning(playback.running);
      } catch (error) {
        console.warn("PCM audio channel failed", error);
        setUiError({
          message:
            error instanceof Error && error.message
              ? `${error.message} You can still type a question.`
              : "The audio channel could not be established. You can still type a question.",
          recoverable: true,
          code: "transport",
        });
        setViewState("error", "Audio unavailable");
      }
    },
    [pushNotice, remoteAudio, setUiError, setViewState, startBargeWatch, startPcm],
  );

  useEffect(() => {
    startAudioRef.current = startAudio;
  }, [startAudio]);

  // --------------------------------------------------------------- commands

  const start = useCallback(
    async (options?: { mic?: boolean }) => {
      if (startingRef.current || sessionRef.current) return;
      startingRef.current = true;
      endedRef.current = false;
      reconnectEventsRef.current = false;
      reconnectAudioRef.current = false;
      reconnectAttemptsRef.current = 0;
      lifespanRef.current += 1;
      const lifespan = lifespanRef.current;
      wantMicRef.current = options?.mic !== false;

      setUiError(null);
      clearNotice();
      setTranscript([]);
      setSources([]);
      setEvidence([]);
      setLastTurn(null);
      setEventLog([]);
      setServerState("idle");
      setElapsed(0);
      setConnectMs(null);
      setTransport(null);
      transportReadyRef.current = false;
      generationRef.current = 0;
      partialUserIdRef.current = null;
      assistantIdRef.current = null;
      setMuted(false);
      setAudioRunning(false);

      // Creating the playback graph here keeps it inside the click's user
      // activation window; browsers refuse to start audio from a later tick.
      if (!playbackRef.current) {
        playbackRef.current = new Playback();
      }
      void playbackRef.current.resume();

      if (wantMicRef.current) {
        setViewState("permission", "Waiting for microphone permission");
        try {
          const mic = await openMicrophone((pcm) => {
            if (mutedRef.current) return;
            audioRef.current?.send(pcm);
          });
          if (lifespan !== lifespanRef.current) {
            void mic.stop();
            return;
          }
          micRef.current = mic;
          analysersRef.current.input = mic.analyser;
          setMicReady(true);
        } catch (error) {
          if (lifespan !== lifespanRef.current) return;
          const message = microphoneErrorMessage(error);
          setUiError({ message, recoverable: true, code: "mic" });
          setViewState("error", message);
          startingRef.current = false;
          return;
        }
      }

      setViewState("connecting", "Setting up a real-time session…");
      connectStartedRef.current = performance.now();
      try {
        const session = await api.createSession("web");
        if (lifespan !== lifespanRef.current || endedRef.current) return;
        sessionRef.current = session;
        setConfig(session.config);
        setActive(true);
        openEventsRef.current(session, lifespan);
        await startAudioRef.current(session, lifespan);
        if (!wantMicRef.current && lifespan === lifespanRef.current) {
          setViewState("connected", "Connected — text mode");
        }
      } catch (error) {
        if (lifespan !== lifespanRef.current) return;
        const message =
          error instanceof Error && error.message
            ? error.message
            : "The session could not be started. Check the gateway and try again.";
        setUiError({ message, recoverable: true, code: "session" });
        setViewState("error", message);
        closeMic();
        closeAudio();
        closeEvents();
        sessionRef.current = null;
        setActive(false);
      } finally {
        if (lifespan === lifespanRef.current) startingRef.current = false;
      }
    },
    [clearNotice, closeAudio, closeEvents, closeMic, setMuted, setUiError, setViewState],
  );

  const stop = useCallback(() => {
    teardown(true);
    setViewState("ended", "Session ended");
  }, [teardown, setViewState]);

  const ask = useCallback(
    (text: string) => {
      const clean = text.trim();
      const session = sessionRef.current;
      if (!clean || !session) return;
      const socket = eventsRef.current;
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "text", text: clean }));
      } else {
        void api
          .sendText(session.session_id, session.token, clean)
          .catch((error: unknown) =>
            pushNotice(error instanceof Error ? error.message : "The question could not be sent.", "warning"),
          );
      }
    },
    [pushNotice],
  );

  const toggleMute = useCallback(() => {
    const next = !mutedRef.current;
    setMuted(next);
    const socket = eventsRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "mute", muted: next }));
    }
    pushNotice(next ? "Microphone muted." : "Microphone on.", "info");
  }, [pushNotice, setMuted]);

  const dismissError = useCallback(() => {
    setUiError(null);
    if (viewRef.current !== "error") return;
    if (sessionRef.current) {
      setViewState(transportReadyRef.current ? "connected" : "connecting", "Ready");
    } else {
      setViewState("idle", "");
    }
  }, [setUiError, setViewState]);

  const resumeAudio = useCallback(() => {
    const playback = playbackRef.current;
    if (!playback) return;
    void playback.resume().then(() => {
      if (mountedRef.current) setAudioRunning(playback.running);
    });
    void micRef.current?.context.resume().catch(() => undefined);
  }, []);

  // ---------------------------------------------------------------- effects

  useEffect(() => {
    if (!diagnosticsOpen) return;
    let alive = true;
    const load = async () => {
      try {
        const data = await api.fetchMetrics();
        if (alive && mountedRef.current) setMetrics(data);
      } catch {
        // Diagnostics are optional; a failed poll is not a conversation error.
      }
    };
    void load();
    const timer = window.setInterval(load, 5_000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [diagnosticsOpen]);

  useEffect(() => {
    if (!diagnosticsOpen) return;
    let alive = true;
    const sample = async () => {
      const handle = rtcRef.current;
      if (!handle) return;
      try {
        const report = await handle.pc.getStats();
        const next: BrowserStats = { connectionState: handle.pc.connectionState };
        let inboundJitter = 0;
        report.forEach((stat: RTCStats) => {
          const record = stat as unknown as Record<string, unknown>;
          if (record.type === "candidate-pair" && (record.state === "succeeded" || record.nominated === true)) {
            if (typeof record.currentRoundTripTime === "number") next.rttMs = Math.round(record.currentRoundTripTime * 1000);
            if (typeof record.localCandidateId === "string") {
              const local = report.get(record.localCandidateId) as unknown as Record<string, unknown> | undefined;
              if (local && typeof local.candidateType === "string") next.candidateType = local.candidateType;
            }
          }
          if (record.type === "inbound-rtp" && record.kind === "audio") {
            if (typeof record.jitter === "number") inboundJitter = Math.max(inboundJitter, record.jitter);
            if (typeof record.packetsLost === "number") next.packetsLost = record.packetsLost;
            if (typeof record.bytesReceived === "number") next.bytesReceived = record.bytesReceived;
          }
          if (record.type === "outbound-rtp" && record.kind === "audio") {
            if (typeof record.bytesSent === "number") next.bytesSent = record.bytesSent;
          }
        });
        if (inboundJitter) next.jitterMs = Math.round(inboundJitter * 1000);
        if (alive && mountedRef.current) setBrowserStats(next);
      } catch {
        // getStats is best effort.
      }
    };
    void sample();
    const timer = window.setInterval(sample, 3_000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [diagnosticsOpen, transport]);

  useEffect(() => {
    if (!active) return;
    const started = Date.now();
    const timer = window.setInterval(() => {
      if (mountedRef.current) setElapsed(Math.floor((Date.now() - started) / 1000));
    }, 1_000);
    return () => window.clearInterval(timer);
  }, [active]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (noticeTimerRef.current) window.clearTimeout(noticeTimerRef.current);
      if (reconnectTimerRef.current) window.clearTimeout(reconnectTimerRef.current);
      if (bargeTimerRef.current) window.clearInterval(bargeTimerRef.current);
      const socket = eventsRef.current;
      if (socket && socket.readyState === WebSocket.OPEN) {
        try {
          socket.send(JSON.stringify({ type: "end" }));
        } catch {
          // Unmount is best effort.
        }
      }
      lifespanRef.current += 1;
      endedRef.current = true;
      closeMic();
      closeAudio();
      closeEvents();
    };
  }, [closeAudio, closeEvents, closeMic]);

  return {
    view,
    detail,
    active,
    connected,
    micReady,
    muted,
    transport,
    serverState,
    config,
    transcript,
    sources,
    evidence,
    lastTurn,
    turnLog,
    metrics,
    browserStats,
    connectMs,
    elapsed,
    notice,
    error,
    eventLog,
    diagnosticsOpen,
    audioRunning,
    getAnalysers,
    start,
    stop,
    ask,
    interrupt,
    toggleMute,
    dismissError,
    resumeAudio,
    clearNotice,
    setDiagnosticsOpen,
  };
}
