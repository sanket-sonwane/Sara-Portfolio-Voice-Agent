/**
 * Typed client for the Sarang gateway.
 *
 * Two rules the rest of the frontend relies on:
 *
 * 1. Nothing here knows a provider or a credential. The gateway returns a
 *    short-lived session token and a public description of the pipeline.
 * 2. Every event carries a generation. The UI may drop stale events locally,
 *    but the server is still the authority that cancels the old generation.
 */

export type ServerState =
  | "idle"
  | "connecting"
  | "ready"
  | "listening"
  | "user_speaking"
  | "transcribing"
  | "thinking"
  | "tool_working"
  | "speaking"
  | "interrupted"
  | "reconnecting"
  | "muted"
  | "error"
  | "ended";

export interface SarangConfig {
  profile: string;
  profile_note: string;
  asr: string;
  asr_ready: boolean;
  llm: string;
  llm_ready: boolean;
  tts: string;
  tts_ready: boolean;
  retrieval: string;
  webrtc: boolean;
  endpoint_silence_ms: number;
  max_answer_tokens: number;
  states: string[];
  broadcast_sample_rate: number;
  input_sample_rate: number;
  speculation_ms: number;
  filler: boolean;
  version: string;
}

export interface SessionResponse {
  session_id: string;
  token: string;
  expires_in: number;
  ice_servers: RTCIceServer[];
  force_relay: boolean;
  config: SarangConfig;
}

export interface ServerEvent {
  type: string;
  session_id: string;
  turn_id: string;
  generation_id: number;
  payload: Record<string, unknown>;
}

export interface EvidenceBlock {
  id: string;
  kind: string;
  title: string;
  snippet: string;
}

export interface EvidenceChunk {
  doc_id: string;
  title: string;
  text: string;
  score: number;
}

export interface ToolEvidence {
  name: string;
  call_id: string;
  ok: boolean;
  duration_ms: number;
  chars: number;
}

export interface TurnMetrics {
  turn_id: string;
  durations_ms: Record<string, number>;
  offsets_ms: Record<string, number>;
  cancelled: boolean;
  filler_used: boolean;
  source: string;
  first_audio_ms?: number;
  eos_to_first_audio_ms?: number;
  answer_first_audio_ms?: number;
  eos_to_answer_audio_ms?: number;
}

export interface StageStat {
  n: number;
  p50: number;
  p95: number;
  mean: number;
  max: number;
}

export interface MetricsResponse {
  summary: {
    stages: Record<string, StageStat>;
    turns: TurnMetrics[];
  };
  profile: string;
  providers: Record<string, unknown>;
  webrtc: boolean;
  filler_cached: number;
  speculation: { prepared: number; adopted: number; wasted: number };
  tools: { ready: boolean; calls: number; timeouts: number };
}

const API_BASE = (import.meta.env.VITE_SARANG_API ?? "").replace(/\/+$/, "");

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

export function socketUrl(path: string, params: Record<string, string | number> = {}): string {
  const url = new URL(`${API_BASE}${path}`, window.location.href);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  for (const [key, value] of Object.entries(params)) {
    url.searchParams.set(key, String(value));
  }
  return url.toString();
}

async function parse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // Non-JSON errors are reported by status; the body is not useful.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return parse<T>(response);
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(apiUrl(path), { headers: { Accept: "application/json" } });
  return parse<T>(response);
}

export function createSession(client = "web"): Promise<SessionResponse> {
  return post<SessionResponse>("/api/session", { client });
}

export function sendText(
  sessionId: string,
  token: string,
  text: string,
): Promise<{ turn_id: string; accepted: boolean }> {
  return post("/api/turn/text", { session_id: sessionId, token, text });
}

export function sendInterrupt(
  sessionId: string,
  token: string,
  reason = "barge-in",
): Promise<{ accepted: boolean; generation: number }> {
  return post("/api/interrupt", { session_id: sessionId, token, reason });
}

export function offer(
  sessionId: string,
  token: string,
  sdp: string,
  type: string,
): Promise<{ sdp: string; type: string }> {
  return post("/api/offer", { session_id: sessionId, token, sdp, type });
}

export function fetchMetrics(): Promise<MetricsResponse> {
  return get<MetricsResponse>("/api/metrics");
}

export function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}
