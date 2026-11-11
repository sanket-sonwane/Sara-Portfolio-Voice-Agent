/**
 * Sarang — the visible product.
 *
 * The controller in ``useSarang`` owns the session, transport and transcript.
 * This file only decides how one conversation state looks and how a visitor
 * drives it: talk, type, interrupt, mute, end. Every state in PRD section 6.3
 * has a distinguishing label, tone and motion so the experience is legible with
 * the sound off and with reduced motion enabled.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";

import { useSarang } from "./useSarang";
import type { Analysers, TranscriptEntry, UiError, UseSarang, ViewState } from "./useSarang";

export interface SarangWidgetProps {
  /** Floating launcher (drop-in on the portfolio) or a full inline surface. */
  mode?: "floating" | "inline";
  className?: string;
  title?: string;
  /**
   * `full` keeps the product chrome (header, hero copy, footer meta, developer
   * toggles). `embed` is the minimal surface for a host iframe: an orb, one
   * status line, the last turns, mute/end and a composer.
   */
  appearance?: "full" | "embed";
  /** Palette for the embed surface, so it can match the host page. */
  theme?: "light" | "dark";
}

type Tone = "idle" | "good" | "busy" | "warn" | "bad";

interface StateCopy {
  label: string;
  hint: string;
  tone: Tone;
}

const STATE_COPY: Record<ViewState, StateCopy> = {
  idle: { label: "Ready", hint: "A real-time conversation, not a form.", tone: "idle" },
  permission: {
    label: "Microphone permission",
    hint: "Your browser is asking whether Sara may listen. Nothing is recorded before you allow it.",
    tone: "warn",
  },
  connecting: { label: "Connecting", hint: "Opening the real-time channel.", tone: "busy" },
  connected: { label: "Connected", hint: "The session is live. Start talking whenever you like.", tone: "good" },
  listening: { label: "Listening", hint: "Speak normally. Sara stops when you pause.", tone: "good" },
  processing: { label: "Thinking", hint: "Working on your question.", tone: "busy" },
  speaking: { label: "Speaking", hint: "Speak over Sara at any time to interrupt.", tone: "good" },
  tool: {
    label: "Retrieving",
    hint: "Pulling the relevant case study — this is the one path allowed to take a moment.",
    tone: "warn",
  },
  interrupted: { label: "Interrupted", hint: "The previous answer was cancelled. Sara is listening again.", tone: "warn" },
  reconnecting: { label: "Reconnecting", hint: "The live channel dropped; re-establishing it now.", tone: "warn" },
  error: { label: "Problem", hint: "Something needs your attention. The text fallback still works.", tone: "bad" },
  ended: { label: "Ended", hint: "The session is closed and the microphone has been released.", tone: "idle" },
};

const STARTERS = [
  "What do you build?",
  "Tell me about VisionMitra.",
  "Which project was hardest technically?",
  "What are you working on now?",
];

/** One short line for the embed surface — the full hints are too long there. */
const SHORT_HINT: Record<ViewState, string> = {
  idle: "Tap the orb to start. The microphone opens only then.",
  permission: "Allow the microphone to talk.",
  connecting: "Opening the line…",
  connected: "Go ahead, I'm listening.",
  listening: "Listening… speak normally, I'll stop when you pause.",
  processing: "Thinking…",
  speaking: "Speaking — interrupt me any time.",
  tool: "Pulling the relevant detail…",
  interrupted: "Go on.",
  reconnecting: "Reconnecting…",
  error: "Something needs your attention.",
  ended: "Session ended. The microphone is off.",
};

function formatMs(value: number | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  if (value >= 1_000) return `${(value / 1_000).toFixed(2)} s`;
  return `${Math.round(value)} ms`;
}

function median(values: number[]): number | undefined {
  if (!values.length) return undefined;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function elapsedLabel(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(seconds % 60).padStart(2, "0")}`;
}

// --------------------------------------------------------------------- icons

function Icon({ path, size = 18 }: { path: string; size?: number }) {
  return (
    <svg
      aria-hidden="true"
      focusable="false"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d={path} />
    </svg>
  );
}

const ICONS = {
  mic: "M12 3a3 3 0 0 0-3 3v5a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3zM5 11a7 7 0 0 0 14 0M12 18v3",
  micOff: "M3 3l18 18M9 9v2a3 3 0 0 0 4.5 2.6M15 9V6a3 3 0 0 0-5.4-1.8M5 11a7 7 0 0 0 10.5 6.1M12 18v3",
  send: "M4 12l16-8-6 16-2.5-6.5L4 12z",
  close: "M6 6l12 12M18 6L6 18",
  end: "M4 12h16M14 6l6 6-6 6",
  wave: "M3 12h2l2-6 3 13 3-16 3 14 2-5h3",
  sliders: "M4 7h10M18 7h2M4 17h4M12 17h8M14 4v6M8 14v6",
  captions: "M4 5h16v14H4zM7 12h3M14 12h3M7 15h10",
  sound: "M4 9v6h4l5 4V5L8 9H4zM17 8a5 5 0 0 1 0 8",
  spark: "M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3z",
  arrow: "M5 12h14M13 6l6 6-6 6",
};

// ----------------------------------------------------------------- waveform

function Waveform({
  tone,
  active,
  colour,
  label,
  getAnalysers,
  level,
}: {
  tone: "input" | "output";
  active: boolean;
  colour: string;
  label: string;
  getAnalysers: () => Analysers;
  level: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    let samples: Float32Array | null = null;
    let timer = 0;

    const resize = () => {
      const ratio = Math.min(2, window.devicePixelRatio || 1);
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width * ratio));
      canvas.height = Math.max(1, Math.round(rect.height * ratio));
    };

    const draw = () => {
      const width = canvas.width;
      const height = canvas.height;
      const analyser = getAnalysers()[tone];
      if (analyser) {
        if (!samples || samples.length !== analyser.fftSize) samples = new Float32Array(analyser.fftSize);
        analyser.getFloatTimeDomainData(samples as Float32Array<ArrayBuffer>);
      } else {
        samples = null;
      }

      context.clearRect(0, 0, width, height);
      const bars = 44;
      const step = width / bars;
      const middle = height / 2;
      for (let index = 0; index < bars; index += 1) {
        const sampleIndex = samples ? Math.floor((index / bars) * samples.length) : -1;
        const value = samples && sampleIndex >= 0 ? samples[sampleIndex] : 0;
        const shaped = active ? Math.min(1, Math.abs(value) * 3.4) : 0;
        const amplitude = active ? Math.max(0.06, shaped) : 0.05 + level * 0.04;
        const barHeight = Math.max(2, amplitude * height * 0.92);
        context.globalAlpha = active ? 0.4 + amplitude * 0.6 : 0.22;
        context.fillStyle = colour;
        context.fillRect(index * step + step * 0.22, middle - barHeight / 2, Math.max(1, step * 0.56), barHeight);
      }
      context.globalAlpha = 1;
    };

    const tick = () => {
      draw();
      if (active) timer = window.setTimeout(tick, reduced ? 260 : 33);
    };
    tick();

    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(resize);
    observer?.observe(canvas);
    window.addEventListener("resize", resize);

    return () => {
      if (timer) window.clearTimeout(timer);
      observer?.disconnect();
      window.removeEventListener("resize", resize);
    };
  }, [active, colour, getAnalysers, level, tone]);

  return (
    <div className={`sarang-signal${active ? " is-active" : ""}`}>
      <span className="sarang-signal-label">{label}</span>
      <canvas ref={canvasRef} className="sarang-signal-canvas" role="img" aria-label={`${label} level`} />
    </div>
  );
}

// --------------------------------------------------------------- transcript

function Turn({ entry }: { entry: TranscriptEntry }) {
  const roleLabel = entry.role === "user" ? "You" : "Sara";
  return (
    <article className={`sarang-turn sarang-turn-${entry.role}${entry.partial ? " is-partial" : ""}`}>
      <header className="sarang-turn-head">
        <span className="sarang-turn-role">{roleLabel}</span>
        {entry.role === "user" && entry.source === "text" && <span className="sarang-tag">typed</span>}
        {entry.partial && <span className="sarang-tag sarang-tag-live">transcribing</span>}
        {entry.interrupted && <span className="sarang-tag sarang-tag-warn">interrupted</span>}
      </header>
      <p className="sarang-turn-text">
        {entry.text}
        {entry.partial && <span className="sarang-caret" aria-hidden="true" />}
      </p>
    </article>
  );
}

// -------------------------------------------------------------- diagnostics

function Diagnostics({ sarang }: { sarang: UseSarang }) {
  const { config, lastTurn, turnLog, metrics, browserStats, evidence, eventLog, transport, connectMs, audioRunning } = sarang;
  const turnStats = useMemo(() => {
    // A typed turn skips endpointing and ASR entirely, so folding it into the
    // same median would understate the number a real-mic pass is trying to
    // measure. Only turns that began with speech are compared.
    const spoken = turnLog.filter((sample) => sample.source !== "text");
    const servers = spoken
      .map((sample) => sample.server)
      .filter((value): value is number => typeof value === "number");
    const clients = spoken
      .map((sample) => sample.client)
      .filter((value): value is number => typeof value === "number");
    return {
      spoken: spoken.length,
      typed: turnLog.length - spoken.length,
      server: median(servers),
      client: median(clients),
    };
  }, [turnLog]);
  const stageRows = useMemo(() => {
    const stages = metrics?.summary.stages ?? {};
    return Object.entries(stages)
      .filter(([, stat]) => stat.n > 0)
      .sort((a, b) => a[0].localeCompare(b[0]));
  }, [metrics]);

  return (
    <div className="sarang-diagnostics" aria-label="Developer diagnostics">
      <section>
        <h4>Pipeline</h4>
        <dl>
          <dt>Profile</dt>
          <dd>{config ? `${config.profile} — ${config.profile_note}` : "no session"}</dd>
          <dt>Stages</dt>
          <dd>
            ASR {config?.asr ?? "—"} ({config?.asr_ready ? "ready" : "degraded"}) · LLM {config?.llm ?? "—"} (
            {config?.llm_ready ? "ready" : "degraded"}) · TTS {config?.tts ?? "—"} (
            {config?.tts_ready ? "ready" : "degraded"})
          </dd>
          <dt>Transport</dt>
          <dd>
            {transport ?? "none"} {audioRunning ? "· audio running" : "· audio idle"} · connect {formatMs(connectMs ?? undefined)}
          </dd>
          <dt>Retrieval</dt>
          <dd>{config?.retrieval ?? "—"}</dd>
          <dt>Endpointing</dt>
          <dd>{config ? `${config.endpoint_silence_ms} ms silence` : "—"}</dd>
        </dl>
      </section>

      <section>
        <h4>This session</h4>
        {turnLog.length ? (
          <>
            <p className="sarang-diag-lead">
              {turnStats.spoken > 0 ? (
                <>
                  {turnStats.spoken} spoken turn{turnStats.spoken === 1 ? "" : "s"} ·
                  server p50 <strong>{formatMs(turnStats.server)}</strong>
                  {turnStats.client !== undefined && (
                    <>
                      {" · "}browser p50 <strong>{formatMs(turnStats.client)}</strong>
                    </>
                  )}
                </>
              ) : (
                <>No spoken turns yet</>
              )}
              {turnStats.typed > 0 && <> · {turnStats.typed} typed</>}
            </p>
            {turnStats.spoken > 0 && turnStats.client === undefined && (
              <p className="sarang-diag-foot">
                The browser figure needs a spoken turn with the microphone enabled.
              </p>
            )}
            <ul className="sarang-diag-list">
              {turnLog.map((sample, index) => (
                <li key={index}>
                  {index + 1}. server {formatMs(sample.server)} · browser{" "}
                  {formatMs(sample.client)}
                  {sample.source === "text" && <> · typed</>}
                  {sample.first !== undefined && sample.client !== undefined && (
                    <> · +{Math.round(sample.client - sample.first)} ms network + playback</>
                  )}
                </li>
              ))}
            </ul>
          </>
        ) : (
          <p className="sarang-diag-foot">No answered turns yet.</p>
        )}
      </section>

      <section>
        <h4>Last turn</h4>
        {lastTurn ? (
          <>
            <p className="sarang-diag-lead">
              End of speech → answer audio: <strong>{formatMs(lastTurn.eos_to_answer_audio_ms)}</strong>
              {" · "}first audible (incl. egress and any filler):{" "}
              <strong>{formatMs(lastTurn.eos_to_first_audio_ms)}</strong>
            </p>
            {lastTurn.client_eos_to_audible_ms !== undefined && (
              <p className="sarang-diag-lead">
                In this browser, speech end → audible:{" "}
                <strong>{formatMs(lastTurn.client_eos_to_audible_ms)}</strong>
                {lastTurn.eos_to_first_audio_ms !== undefined && (
                  <>
                    {" · "}network + playback:{" "}
                    <strong>
                      {formatMs(
                        lastTurn.client_eos_to_audible_ms -
                          lastTurn.eos_to_first_audio_ms,
                      )}
                    </strong>
                  </>
                )}
              </p>
            )}
            <dl>
              {Object.entries(lastTurn.durations_ms ?? {}).map(([stage, value]) => (
                <div key={stage} className="sarang-diag-row">
                  <dt>{stage}</dt>
                  <dd>{formatMs(value)}</dd>
                </div>
              ))}
            </dl>
            <p className="sarang-diag-foot">
              source {lastTurn.source || "—"} · filler {lastTurn.filler_used ? "used" : "no"} ·{" "}
              {lastTurn.cancelled ? "cancelled" : "completed"}
            </p>
          </>
        ) : (
          <p className="sarang-diag-foot">No completed turn yet.</p>
        )}
      </section>

      <section>
        <h4>Rolling stages</h4>
        {stageRows.length ? (
          <dl>
            {stageRows.map(([stage, stat]) => (
              <div key={stage} className="sarang-diag-row">
                <dt>{stage}</dt>
                <dd>
                  p50 {formatMs(stat.p50)} · p95 {formatMs(stat.p95)} · n {stat.n}
                </dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="sarang-diag-foot">No samples yet.</p>
        )}
        {metrics && (
          <p className="sarang-diag-foot">
            speculation {metrics.speculation.adopted}/{metrics.speculation.prepared} adopted ·{" "}
            {metrics.speculation.wasted} wasted · tools {metrics.tools.calls} calls
          </p>
        )}
      </section>

      <section>
        <h4>Browser</h4>
        <dl>
          <dt>Connection</dt>
          <dd>{browserStats?.connectionState ?? "—"}</dd>
          <dt>Candidate / RTT</dt>
          <dd>
            {browserStats?.candidateType ?? "—"} · {browserStats?.rttMs !== undefined ? `${browserStats.rttMs} ms` : "—"}
          </dd>
          <dt>Jitter / loss</dt>
          <dd>
            {browserStats?.jitterMs !== undefined ? `${browserStats.jitterMs} ms` : "—"} ·{" "}
            {browserStats?.packetsLost ?? 0} packets
          </dd>
        </dl>
      </section>

      <section>
        <h4>Evidence</h4>
        {evidence.length ? (
          <ul className="sarang-diag-list">
            {evidence.slice(-8).map((note) => (
              <li key={note.id}>
                <strong>{note.label}</strong> — {note.detail}
              </li>
            ))}
          </ul>
        ) : (
          <p className="sarang-diag-foot">Nothing retrieved yet.</p>
        )}
      </section>

      <section>
        <h4>Event log</h4>
        {eventLog.length ? (
          <ol className="sarang-diag-list sarang-diag-events">
            {eventLog.slice(-14).map((entry, index) => (
              <li key={`${entry.at}-${index}`}>
                <time>{new Date(entry.at).toLocaleTimeString()}</time> <code>{entry.type}</code>{" "}
                {entry.generation > 0 && <span>g{entry.generation}</span>} {entry.summary}
              </li>
            ))}
          </ol>
        ) : (
          <p className="sarang-diag-foot">No events yet.</p>
        )}
      </section>
    </div>
  );
}

// -------------------------------------------------------------------- widget

export function SarangWidget({ mode = "floating", className, title = "Sara", appearance = "full", theme }: SarangWidgetProps) {
  const embed = appearance === "embed";
  const remoteAudio = useRef<HTMLAudioElement | null>(null);
  const sarang = useSarang(remoteAudio);
  const [open, setOpen] = useState(mode === "inline");
  const [draft, setDraft] = useState("");
  const [captions, setCaptions] = useState(true);
  const transcriptRef = useRef<HTMLDivElement | null>(null);
  const panelRef = useRef<HTMLElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const { view, detail, active, muted, micReady, transcript, sources, error, notice, audioRunning, config } = sarang;
  const copy = STATE_COPY[view];
  const speaking = view === "speaking";
  const listening = view === "listening" || view === "connected";
  const busy = view === "processing";

  const lastAssistant = useMemo(() => {
    for (let index = transcript.length - 1; index >= 0; index -= 1) {
      const entry = transcript[index];
      if (entry.role === "assistant") return entry;
    }
    return null;
  }, [transcript]);

  // The embed surface shows only the tail of the conversation.
  const recentTurns = useMemo(() => transcript.slice(-2), [transcript]);

  /** Talk to the host page (the portfolio) without trusting random senders. */
  const postToParent = useCallback((message: Record<string, unknown>) => {
    if (window.parent === window) return;
    let target = "*";
    try {
      if (document.referrer) target = new URL(document.referrer).origin;
    } catch {
      target = "*";
    }
    window.parent.postMessage(message, target);
  }, []);

  // The host shows its own "Connecting" overlay. Release it once a session is
  // actually live, so nobody stares at a spinner over a working orb.
  useEffect(() => {
    if (!embed || !active) return;
    postToParent({ type: "sara:ready" });
  }, [active, embed, postToParent]);

  useEffect(() => {
    if (!open) return;
    const node = transcriptRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [open, transcript]);

  useEffect(() => {
    if (!active || micReady) return;
    // Voice sessions should not steal focus from the page: a visitor who just
    // pressed "start talking" is about to speak, not type. Text mode (and a
    // failed microphone) is exactly when the composer should be ready.
    const timer = window.setTimeout(() => inputRef.current?.focus({ preventScroll: true }), 350);
    return () => window.clearTimeout(timer);
  }, [active, micReady]);

  useEffect(() => {
    if (!open || embed) return;
    const node = panelRef.current;
    if (node && typeof node.focus === "function") node.focus({ preventScroll: true });
  }, [open, embed]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      if (sarang.diagnosticsOpen) {
        sarang.setDiagnosticsOpen(false);
        return;
      }
      if (mode === "floating") {
        sarang.stop();
        setOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [mode, open, sarang.diagnosticsOpen, sarang.setDiagnosticsOpen, sarang.stop]);

  const startVoice = useCallback(() => {
    setOpen(true);
    void sarang.start();
  }, [sarang]);

  const startText = useCallback(() => {
    setOpen(true);
    void sarang.start({ mic: false });
  }, [sarang]);

  const send = useCallback(
    async (raw: string) => {
      const text = raw.trim();
      if (!text) return;
      if (!sarang.active) {
        await sarang.start({ mic: false });
      }
      // The session is registered before the sockets finish opening, so the
      // HTTP fallback in ask() covers this short window.
      window.setTimeout(() => sarang.ask(text), 220);
    },
    [sarang],
  );

  const submit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      const value = draft.trim();
      if (!value) return;
      setDraft("");
      void send(value);
    },
    [draft, send],
  );

  const closePanel = useCallback(() => {
    sarang.stop();
    if (mode === "floating") setOpen(false);
  }, [mode, sarang]);

  // In the embed surface "end" belongs to the host: stop the audio here and ask
  // the portfolio to close its panel.
  const endEmbed = useCallback(() => {
    sarang.stop();
    postToParent({ type: "sara:end" });
  }, [postToParent, sarang]);

  const errorActions = useCallback(
    (current: UiError) => {
      if (!current.recoverable) {
        return (
          <div className="sarang-error-actions">
            <button type="button" className="sarang-btn sarang-btn-primary" onClick={startText}>
              Continue with text
            </button>
          </div>
        );
      }
      return (
        <div className="sarang-error-actions">
          {current.code === "mic" ? (
            <>
              <button type="button" className="sarang-btn sarang-btn-primary" onClick={startVoice}>
                Try the microphone again
              </button>
              <button type="button" className="sarang-btn" onClick={startText}>
                Type instead
              </button>
            </>
          ) : (
            <>
              <button type="button" className="sarang-btn sarang-btn-primary" onClick={() => sarang.dismissError()}>
                Dismiss
              </button>
              <button type="button" className="sarang-btn" onClick={startText}>
                Type instead
              </button>
            </>
          )}
        </div>
      );
    },
    [sarang, startText, startVoice],
  );

  const showHero = !active && transcript.length === 0;
  const ended = view === "ended";

  const body = (
    <>
      {error && (
        <div className="sarang-error" role="alert">
          <p>{error.message}</p>
          {errorActions(error)}
        </div>
      )}

      <div className="sarang-stage">
        {showHero ? (
          <div className="sarang-hero">
            <div className="sarang-hero-orb" aria-hidden="true">
              <span />
              <span />
              <span />
            </div>
            <h2>Talk to {title}</h2>
            <p>
              A real-time voice guide through Sanket Sonwane&apos;s projects, research and experience. Ask naturally,
              interrupt any time.
            </p>
            <div className="sarang-hero-actions">
              <button type="button" className="sarang-btn sarang-btn-primary sarang-btn-lg" onClick={startVoice}>
                <Icon path={ICONS.mic} />
                Start talking
              </button>
              <button type="button" className="sarang-btn sarang-btn-lg" onClick={startText}>
                <Icon path={ICONS.arrow} />
                Type instead
              </button>
            </div>
            <p className="sarang-hero-note">
              The microphone opens only after you press start. Sara never listens in the background.
            </p>
          </div>
        ) : (
          <div className="sarang-transcript" ref={transcriptRef} role="log" aria-label="Conversation transcript">
            {transcript.map((entry) => (
              <Turn key={entry.id} entry={entry} />
            ))}
            {ended && <p className="sarang-ended-note">Session ended — the microphone is off.</p>}
          </div>
        )}
      </div>

      {active && sources.length > 0 && (
        <div className="sarang-sources" aria-label="Sources for the current answer">
          <span className="sarang-sources-label">Grounded in</span>
          <ul>
            {sources.slice(0, 4).map((source) => (
              <li key={source.id} title={source.snippet}>
                <span className="sarang-source-kind">{source.kind}</span>
                {source.title}
              </li>
            ))}
          </ul>
        </div>
      )}

      {captions && lastAssistant && (lastAssistant.partial || speaking) && (
        <div className="sarang-captions" aria-hidden="true">
          <span>{lastAssistant.partial ? "Sara is saying" : "Sara said"}</span>
          <p>{lastAssistant.text}</p>
        </div>
      )}

      <div className="sarang-signals" aria-hidden="true">
        <Waveform
          tone="input"
          active={active && listening && !muted && micReady}
          colour="#7dd3fc"
          label="You"
          getAnalysers={sarang.getAnalysers}
          level={muted ? 0 : 1}
        />
        <Waveform
          tone="output"
          active={active && speaking}
          colour="#a78bfa"
          label="Sara"
          getAnalysers={sarang.getAnalysers}
          level={speaking ? 1 : 0}
        />
      </div>

      {busy && (
        <div className="sarang-thinking" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
      )}

      {notice && (
        <div className={`sarang-notice sarang-notice-${notice.level}`} role="status">
          <p>{notice.message}</p>
          <button type="button" className="sarang-icon-btn" onClick={() => sarang.clearNotice()} aria-label="Dismiss notice">
            <Icon path={ICONS.close} size={14} />
          </button>
        </div>
      )}

      {active && !audioRunning && (
        <button type="button" className="sarang-sound-hint" onClick={() => sarang.resumeAudio()}>
          <Icon path={ICONS.sound} size={15} />
          Enable sound
        </button>
      )}

      <form className="sarang-composer" onSubmit={submit}>
        <input
          ref={inputRef}
          type="text"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder={active ? "Type a question — or just speak" : "Type instead of speaking"}
          aria-label="Type a question for Sara"
          autoComplete="off"
          enterKeyHint="send"
        />
        <button
          type="submit"
          className="sarang-send"
          aria-label="Send question"
          disabled={!draft.trim()}
        >
          <Icon path={ICONS.send} />
        </button>
      </form>

      {(!active || transcript.length === 0) && (
        <div className="sarang-starters" aria-label="Suggested questions">
          {STARTERS.map((question) => (
            <button
              key={question}
              type="button"
              className="sarang-chip"
              onClick={() => void send(question)}
            >
              {question}
            </button>
          ))}
        </div>
      )}
    </>
  );

  const fullPanel = (
    <section
      ref={panelRef}
      tabIndex={-1}
      className={`sarang-panel sarang-panel-${mode}${open ? " is-open" : ""}`}
      aria-label={`${title} voice assistant`}
    >
      <header className="sarang-header">
        <div className={`sarang-orb sarang-tone-${copy.tone}${speaking ? " is-speaking" : ""}${busy ? " is-busy" : ""}`}>
          <Icon path={ICONS.wave} size={19} />
        </div>
        <div className="sarang-heading">
          <div className="sarang-heading-row">
            <h1>{title}</h1>
            <span className={`sarang-status sarang-tone-${copy.tone}`}>
              <span className="sarang-status-dot" aria-hidden="true" />
              {copy.label}
            </span>
          </div>
          <p className="sarang-heading-hint" title={detail || copy.hint}>
            {active ? detail || copy.hint : copy.hint}
          </p>
        </div>
        <div className="sarang-head-actions">
          {active && micReady && (
            <button
              type="button"
              className={`sarang-icon-btn${muted ? " is-on" : ""}`}
              aria-pressed={muted}
              aria-label={muted ? "Unmute microphone" : "Mute microphone"}
              title={muted ? "Unmute microphone" : "Mute microphone"}
              onClick={() => sarang.toggleMute()}
            >
              <Icon path={muted ? ICONS.micOff : ICONS.mic} />
            </button>
          )}
          {active && (
            <button
              type="button"
              className="sarang-icon-btn sarang-icon-danger"
              aria-label="End session"
              title="End session"
              onClick={closePanel}
            >
              <Icon path={ICONS.end} />
            </button>
          )}
          {mode === "floating" && (
            <button
              type="button"
              className="sarang-icon-btn"
              aria-label="Close Sara"
              title="Close"
              onClick={closePanel}
            >
              <Icon path={ICONS.close} />
            </button>
          )}
        </div>
      </header>

      <div className="sarang-scroll">
        {body}
        {sarang.diagnosticsOpen && <Diagnostics sarang={sarang} />}
      </div>

      <footer className="sarang-footer">
        <div className="sarang-meta">
          <span className={`sarang-dot sarang-tone-${copy.tone}`} aria-hidden="true" />
          <span>{active ? `Live · ${elapsedLabel(sarang.elapsed)}` : "Idle"}</span>
          {config && <span className="sarang-meta-sep">{config.profile}</span>}
          {sarang.transport && <span className="sarang-meta-sep">{sarang.transport === "webrtc" ? "WebRTC" : "WebSocket"}</span>}
        </div>
        <div className="sarang-footer-actions">
          <button
            type="button"
            className={`sarang-text-btn${captions ? " is-on" : ""}`}
            aria-pressed={captions}
            onClick={() => setCaptions((value) => !value)}
            title="Toggle live captions"
          >
            <Icon path={ICONS.captions} size={15} />
            Captions
          </button>
          <button
            type="button"
            className={`sarang-text-btn${sarang.diagnosticsOpen ? " is-on" : ""}`}
            aria-pressed={sarang.diagnosticsOpen}
            onClick={() => sarang.setDiagnosticsOpen(!sarang.diagnosticsOpen)}
            title="Toggle developer diagnostics"
          >
            <Icon path={ICONS.sliders} size={15} />
            Diagnostics
          </button>
          {ended && (
            <button type="button" className="sarang-text-btn is-accent" onClick={startVoice}>
              <Icon path={ICONS.spark} size={15} />
              Restart
            </button>
          )}
        </div>
      </footer>

      <div className="sarang-sr-only" role="status" aria-live="polite">
        {`${copy.label}. ${error ? error.message : lastAssistant?.partial === false ? lastAssistant.text : copy.hint}`}
      </div>

      <audio ref={remoteAudio} className="sarang-remote-audio" />
    </section>
  );

  const embedPanel = (
    <section
      ref={panelRef}
      tabIndex={-1}
      className="sarang-panel sarang-panel-inline sarang-panel-embed is-open"
      aria-label={`${title} voice assistant`}
    >
      <div className="sarang-embed-scroll">
        {error && (
          <div className="sarang-embed-error" role="alert">
            <p>{error.message}</p>
            {errorActions(error)}
          </div>
        )}

        <div className="sarang-embed-status">
          <span className={`sarang-embed-dot sarang-tone-${copy.tone}${active ? " is-live" : ""}`} aria-hidden="true" />
          <span aria-live="polite">{copy.label}</span>
        </div>

        <div className="sarang-embed-center">
          {active ? (
            <div
              className={`sarang-embed-orb sarang-tone-${copy.tone}${speaking ? " is-speaking" : ""}${busy ? " is-busy" : ""}`}
              aria-hidden="true"
            >
              <span className="sarang-embed-orb-core">
                <Icon path={ICONS.wave} size={28} />
              </span>
            </div>
          ) : (
            <button
              type="button"
              className="sarang-embed-orb is-button"
              onClick={startVoice}
              aria-label={`Start talking to ${title}`}
            >
              <span className="sarang-embed-orb-core">
                <Icon path={ICONS.mic} size={28} />
              </span>
            </button>
          )}
          {!active && <p className="sarang-embed-hint">{SHORT_HINT[view]}</p>}
        </div>

        {recentTurns.length > 0 && (
          <div className="sarang-embed-log" role="log" aria-label="Recent conversation">
            {recentTurns.map((entry) => (
              <p
                key={entry.id}
                className={`sarang-embed-line sarang-embed-line-${entry.role}${entry.partial ? " is-partial" : ""}`}
              >
                {entry.text}
                {entry.partial && <span className="sarang-caret" aria-hidden="true" />}
              </p>
            ))}
            {ended && <p className="sarang-embed-line is-note">Session ended — the microphone is off.</p>}
          </div>
        )}

        {speaking && (
          <div className="sarang-embed-meter">
            <Waveform
              tone="output"
              active
              colour={theme === "light" ? "#d9482a" : "#ffb199"}
              label="Sara"
              getAnalysers={sarang.getAnalysers}
              level={1}
            />
          </div>
        )}
      </div>

      <div className="sarang-embed-bottom">
        {notice && (
          <div className={`sarang-notice sarang-notice-${notice.level}`} role="status">
            <p>{notice.message}</p>
            <button
              type="button"
              className="sarang-icon-btn"
              onClick={() => sarang.clearNotice()}
              aria-label="Dismiss notice"
            >
              <Icon path={ICONS.close} size={14} />
            </button>
          </div>
        )}

        {active && !audioRunning && (
          <button type="button" className="sarang-sound-hint" onClick={() => sarang.resumeAudio()}>
            <Icon path={ICONS.sound} size={15} />
            Enable sound
          </button>
        )}

        {(active || ended) && (
          <div className="sarang-embed-controls">
            {active && micReady && (
              <button
                type="button"
                className={`sarang-embed-control${muted ? " is-on" : ""}`}
                aria-pressed={muted}
                aria-label={muted ? "Unmute microphone" : "Mute microphone"}
                title={muted ? "Unmute microphone" : "Mute microphone"}
                onClick={() => sarang.toggleMute()}
              >
                <Icon path={muted ? ICONS.micOff : ICONS.mic} size={18} />
                {muted ? "Unmute" : "Mute"}
              </button>
            )}
            {active && (
              <button
                type="button"
                className="sarang-embed-control is-end"
                aria-label="End voice session"
                title="End session"
                onClick={endEmbed}
              >
                <Icon path={ICONS.end} size={18} />
                End
              </button>
            )}
            {ended && (
              <button type="button" className="sarang-embed-control is-restart" onClick={startVoice}>
                <Icon path={ICONS.spark} size={16} />
                Start again
              </button>
            )}
          </div>
        )}

        <form className="sarang-composer" onSubmit={submit}>
          <input
            ref={inputRef}
            type="text"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder={active ? "Ask anything" : "Type instead of speaking"}
            aria-label="Type a question for Sara"
            autoComplete="off"
            enterKeyHint="send"
          />
          <button type="submit" className="sarang-send" aria-label="Send question" disabled={!draft.trim()}>
            <Icon path={ICONS.send} />
          </button>
        </form>

        {!active && transcript.length === 0 && (
          <div className="sarang-starters" aria-label="Suggested questions">
            {STARTERS.slice(0, 3).map((question) => (
              <button key={question} type="button" className="sarang-chip" onClick={() => void send(question)}>
                {question}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="sarang-sr-only" role="status" aria-live="polite">
        {`${copy.label}. ${error ? error.message : copy.hint}`}
      </div>

      <audio ref={remoteAudio} className="sarang-remote-audio" />
    </section>
  );

  const panel = embed ? embedPanel : fullPanel;

  if (mode === "inline") {
    return (
      <div
        className={`sarang-widget sarang-widget-inline${embed ? " sarang-embed" : ""} ${className ?? ""}`}
        data-theme={theme}
      >
        {panel}
      </div>
    );
  }

  return (
    <div className={`sarang-widget sarang-widget-floating ${className ?? ""}`}>
      {open ? (
        panel
      ) : (
        <button
          type="button"
          className={`sarang-launcher${active ? " is-live" : ""}`}
          onClick={() => setOpen(true)}
          aria-label={`Open ${title} voice assistant`}
        >
          <span className="sarang-launcher-orb" aria-hidden="true">
            <Icon path={ICONS.wave} size={20} />
          </span>
          <span className="sarang-launcher-copy">
            <strong>Talk to {title}</strong>
            <small>{active ? "Session live — tap to return" : "Real-time voice portfolio guide"}</small>
          </span>
        </button>
      )}
    </div>
  );
}

export default SarangWidget;
