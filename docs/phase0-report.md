# Phase 0 — Provider Validation Report

**Status:** local baseline complete — Fly region matrix still pending
**Owner:** Sanket
**Started:** 2026-11-09
**Updated:** 2026-11-10

Exit criteria: the provider integrations are proven reachable, their latencies are
measured rather than assumed, the ASR protocol is identified, and the media
gateway region is chosen from data. The numbers below are from the real adapters
with real keys, measured from a consumer connection in India. The one gate still
open is the Fly region matrix (§5), which needs Fly machines.

---

## 1. Environment

| Item | Value |
|---|---|
| Probe host | local — Windows 11, Python 3.12, consumer connection in India |
| Public location at probe time | IN, colo `MAA` |
| Provider hosts resolved | Sarvam, Gemini, Groq, Cartesia, NVIDIA |
| API keys available at last run | all five |
| Default profile | `bom-india` (ADR 0002, amended) |

Every figure in this report is a **local client** measurement. It includes the
consumer last mile to the provider, so it is an upper bound on what a Fly `bom`
gateway would see, not a substitute for §5.

## 2. ASR — Sarvam Saaras v4 (default), Nemotron (comparison)

**Question:** which streaming surface does each account actually reach, and how
fast is the first partial?

Sarvam is the default because it is in-region and streams. The adapter uses the
realtime WebSocket at `/speech-to-text-realtime/ws` with `api-subscription-key`,
`model=saaras:v4`, `stream_type=fast`, `endpointing=manual`, base64 `audio_input`
frames and `speech_start` / `speech_end` / `flush` control messages.

| Candidate surface | Reachable | Notes |
|---|---:|---|
| Sarvam WS `/speech-to-text-realtime/ws` | **yes** | default; streaming partials, manual endpointing |
| Sarvam REST `/speech-to-text` | yes | used by `sarvam_probe.py`; no partials |
| NVIDIA OpenAI-compatible (`/v1/audio/transcriptions`) | yes | non-streaming |
| NVIDIA SSE (`stream=true`) | yes | input is not streamed |
| NVIDIA WebSocket | no | protocol unconfirmed |
| NVIDIA self-hosted NIM | not attempted | only if it removes the hop |

Measured (fill from `bench/results/sarvam-*.json` and `nemotron-*.json`):

| Metric | Sarvam Saaras v4 | Nemotron 80 ms | Nemotron 160 ms | Nemotron 320 ms |
|---|---:|---:|---:|---:|
| Connect | 275 ms (REST) | | | |
| Time to first partial | ~1.2 s (`stream_type=fast`) | | | |
| Finalization after last frame | 592 ms p50 (`asr_final`, §6) | | | |
| WER on bench corpus | not scored; fixture clip correct 6/6 with 1-3 keyterms | | | |

The `ARAI` → `ARII` alias miss is handled twice: keyterm biasing reduces it at
the source, and `lexicon.misheard` repairs the transcript deterministically. The
KB aliases (`ARAI`, `ARII`, `ARI`) mean the transcript error never changed the
answer. See ADR 0003 for the measured keyterm behaviour: a large keyterm list
made the same clip *worse* ("AAI"), so the list is rationed to three acronyms.

## 3. LLM — Sarvam 105B Conversations (default), Gemini and Groq (comparison)

`llm_matrix.py`, 9 timed runs per arm, identical short prompt, streaming:

| Provider / model | Prompt tokens | TTFT p50 | TTFT p95 | Total p50 |
|---|---:|---:|---:|---:|
| Sarvam `sarvam-105b-conversations` | ~180 | **206 ms** | 304 ms | 234 ms |
| Gemini `gemini-3.5-flash-lite` | ~180 | 980 ms | 1 258 ms | 986 ms |
| Groq `openai/gpt-oss-20b` | ~180 | n/a (no streamed TTFT) | n/a | 588 ms |

**Prefill cost** (TTFT at 900 vs 150 prompt tokens): not isolated. The prompt is
kept at 972 characters by policy (§ADR 0002 §5), so the question only matters if
that policy is ever relaxed.

**Decision:** Sarvam 105B is the default. It is the in-region arm *and* the
fastest measured TTFT — roughly 4.8× faster to first token than Gemini Flash-Lite
from this client, and Groq's total is ~2.5× slower despite no streamed TTFT to
compare. The earlier plan to default to Gemini on the strength of `asia-south1`
is retired: the measurement did not support it, and the free tier also throttles
(see §7).

## 4. TTS — Cartesia Sonic (default), Sarvam Bulbul (comparison)

Voice pinned: `2a12b36c-7f9b-4c3a-9f7a-72731b15323a` ("Ella — Caring Scout",
feminine, en-AU), model `sonic-latest`, 24 kHz.

Cartesia is a WebSocket, so the turn engine gets audio as it is generated.
`cartesia_ttfa.py`, 3 runs / 5 segments:

| Metric | Target | Cartesia p50 | Cartesia p95 | Sarvam Bulbul |
|---|---:|---:|---:|---|
| TTFA, first segment | < 120 ms | **148 ms** | 204 ms | request/response, not comparable |
| Subsequent segment TTFA | ≈ first | **5.6 ms** | 56 ms | n/a |
| RTF | < 1.0 | **0.09** | 0.09 | n/a |
| Cancel-to-silence | < 100 ms | **65 ms** | 72 ms | n/a (cannot cancel mid-request) |

Bulbul is request/response, not a socket. The turn engine compensates by
synthesising one short segment at a time, so the meaningful number is the first
segment's time-to-audio, not the whole paragraph — but cancellation is still
coarse, which is why Cartesia holds the default despite the non-Indian accent.

**Accepted trade-off:** "Ella" is en-AU, not Indian English. It was chosen for the
measured TTFA and the clean cancel behaviour; en-IN alternatives were offered and
remain one env var away (`CARTESIA_VOICE_ID`).

## 5. Region matrix

### Local baseline (informational only — India, laptop)

Two runs of `regions.py --rtt`, 3 and 2 samples per host:

| Host | Run 1 p50 | Run 2 p50 | Note |
|---|---:|---:|---|
| api.groq.com | 210.2 ms | 244.1 ms | no India edge; every call crosses to the US |
| api.cartesia.ai | 94.0 ms | 283.9 ms | 3× run-to-run variance |
| integrate.api.nvidia.com | 102.7 ms | 209.2 ms | 2× run-to-run variance |

Consumer-link variance is large enough that single runs are worthless. The
surviving conclusion is the one that drove ADR 0002: Groq has no nearby point of
presence from India, so a `bom` gateway talking to Groq spends ~210 ms of the
500 ms budget before inference starts.

### Fly measurements (authoritative — **pending**)

Run `regions.py --full` inside Fly machines, one per candidate region. This has
not been done: the Fly account is not available in this environment, so the
matrix below is intentionally empty rather than estimated.

| Region | Sarvam RTT p50 | Cartesia RTT p50 | Groq RTT p50 | Groq TTFT p50 | Public loc |
|---|---:|---:|---:|---:|---|
| `bom` | | | | | |
| `sin` | | | | | |
| `fra` | | | | | |
| `iad` | | | | | |

**Decision:** provisional `bom`. Final only when this table is filled and `bom`
wins end-to-end for an Indian client (ADR 0002 §4).

## 6. Budget reconciliation

End-to-end voice turn, measured through the real gateway over the same
HTTP/WebSocket surface the browser uses (`e2e_turn.py`, 16 kHz PCM in at
real-time pace, 1 500 ms trailing silence, loopback target so no browser network
leg is included). These numbers are **after** the turn-taking and recognition
work in ADR 0003:

| Total | p50 | p95 |
|---|---:|---:|
| Client: last speech → first audible | 1 961 ms | 1 974 ms |
| Server: end-of-speech → first audible | 1 133 ms | 1 142 ms |
| Server: end-of-speech → first answer audio | 1 123 ms | 1 141 ms |

Client total includes the **800 ms** endpointing window; server totals start
after it. The arithmetic closes: 800 ms endpointing + ~1 160 ms of server work
≈ 1 960 ms, so on loopback the transport adds nothing measurable.

Server-side components, against the plan's estimates:

| Stage | Target (ms) | Measured p50 | Delta |
|---|---:|---:|---|
| Endpointing (`ENDPOINT_SILENCE_MS`) | 100 | 800 (config) | +700, deliberate |
| ASR finalization | ≤ 20 | 592 | +572 |
| Fact lookup | ≤ 10 | 0.6 | ok |
| Context build | ≤ 5 | 0.0 | ok |
| LLM TTFT | 130–180 | 418 | +238 |
| Segmentation | ≤ 5 | 125 | +120 |
| TTS TTFA | 70–100 | 93 | ok |
| Egress reframe | 20–30 | 10 | ok |
| **Server subtotal (excl. endpointing)** | **340–385** | **~1 160** | +775 |
| Browser network (one-way) | 120–200 | not measured (loopback) | — |
| Playback start | 20–30 | not measured (loopback) | — |
| **End-to-end** | **480–615** | **1 961** | +1 350 |

The two totals are 10 ms apart, so no filler was needed on this question. Report
two numbers, as the PRD requires: first audible (may include a filler) and first
answer audio. Text turns leave the first-audio figure null.

Where the budget is lost, in order of expected gain:

1. **Endpointing, 800 ms** — deliberately larger than the 500 ms it replaced. At
   500 ms with a 140 ms hangover the gateway closed turns mid-sentence, which is
   what produced fragment transcripts and answers that talked over the visitor.
   The 800 ms pre-roll buffer means the extra wait costs only the wait itself.
2. **ASR finalization, 592 ms** — this is the cost of waiting for a real
   `transcript.final` instead of accepting a partial. The previous 224 ms figure
   was partly a *timeout* path: 1.8 s after which a low-accuracy partial was
   used. Accuracy was the reported problem, so this is the correct trade; a
   REST hybrid finalizer is the next lever (ADR 0003 §9).
3. **LLM TTFT, 418 ms** — higher than the 206 ms standalone figure in §3 and the
   noisiest component (p95 586 ms). Provider variance across a 2-turn sample;
   the 53-turn aggregate is the better estimate (p50 234 ms).
4. **Segmentation, 125 ms** — first-segment boundary detection before TTS can
   start; CPU work on a laptop, so a Fly machine should improve it.
5. **Browser network and playback** — genuinely unmeasured. This run is loopback,
   so the India last mile and the playback start remain the unknown that a
   real-mic browser pass has to close.

Run-to-run spread is large, and a 2-turn sample cannot separate p50 from p95, so
treat any single-run comparison as noise. The JSON is
`bench/results/e2e-turn-local-india-20261110-121439.json`.

## 7. Risks confirmed or retired

| Risk | Status | Evidence |
|---|---|---|
| Sub-500 ms impossible from India due to provider region | **confirmed for Groq**, mitigated | no Asia region; ADR 0002 moves the default in-region |
| Gemini free tier is a viable default | **retired — it is not** | 980 ms TTFT p50 and quota throttling; Sarvam is 206 ms |
| Cartesia leaks audio after cancel | **retired** | 65 ms cancel-to-silence, 18 late chunks measured and dropped |
| Nemotron streaming access unavailable | open | comparison arm only |
| aiortc cannot obtain reachable ICE candidates on Fly | open | `/ws/audio` fallback keeps the product working |
| Prompt length materially increases TTFT | open | `llm_matrix.py` + `groq_ttft.py` measure it |
| Bulbul request/response adds first-segment latency | **retired by not shipping it** | Cartesia is the default; Bulbul stays an arm |
| pgvector unavailable on the managed plan | mitigated | BM25 is the default; pg is opt-in |

## 8. Decisions required before production

1. ~~Sarvam ASR surface and finalization behaviour~~ — done: realtime WS,
   `saaras:v4`, manual endpointing, one warm socket per connection, 592 ms
   finalization (ADR 0003).
2. ~~Default LLM~~ — done: Sarvam 105B, 206 ms TTFT p50.
3. ~~TTS voice~~ — done: Cartesia `sonic-latest`, voice `2a12b36c…`, en-AU.
4. **Media-gateway region** — `bom` provisional; the Fly matrix (§5) decides.
5. **Whether TURN relay is mandatory or host candidates suffice** — diagnostics
   `candidateType` from real sessions.
6. **Real human-mic browser pass** — headless Chrome cannot grant microphone
   access, so every browser check so far is transport-level only.
