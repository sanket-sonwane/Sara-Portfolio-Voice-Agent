# Phase 0 — Provider Validation Report

**Status:** complete for `sin`; `bom` unavailable on this Fly account
**Owner:** Sanket
**Started:** 2026-11-09
**Updated:** 2026-11-10

Exit criteria: the provider integrations are proven reachable, their latencies are
measured rather than assumed, the ASR protocol is identified, and the media
gateway region is chosen from data. The numbers below are from the real adapters
with real keys, measured from a consumer connection in India and, for §6, from the
deployed Fly machine. The one gate still open is the full region matrix (§5):
`bom` is not available on this account.

---

## 1. Environment

| Item | Value |
|---|---|
| Probe host | local — Windows 11, Python 3.12, consumer connection in India |
| Public location at probe time | IN, colo `MAA` |
| Provider hosts resolved | Sarvam, Gemini, Groq, Cartesia, Smallest, NVIDIA |
| API keys available at last run | all six |
| Default profile | `bom-india` (ADR 0002, amended; ADR 0004 for TTS) |

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

**No smaller in-region arm exists.** The obvious "is a smaller model faster"
experiment is closed by the catalogue: this account's `GET /v1/models` lists only
`sarvam-105b` and `sarvam-105b-conversations`, and `sarvam-30b` returns HTTP 400.
The remaining candidates are US-hosted, which ADR 0002 rules out for the
India-first profile, and both measured slower anyway.

## 4. TTS — Smallest Lightning (default), Cartesia Sonic (comparison)

Voice pinned: `rhea`, model `lightning_v3.1_pro`, `language: "hi"` — Indian-accented
English with mid-utterance Hindi code-switching — at 24 kHz PCM16, resampled to the
48 kHz egress track. `language: "en"` selects the UK/US accents, which is the thing
this choice is moving away from.

Both providers are WebSockets, so the turn engine gets audio while it is generated.
`cartesia_ttfa.py` and `smallest_ttfa.py` share the same passage, the same
segmentation and the same metric names, so the two runs compare row for row. Five
runs of five segments each, one laptop, one session (India, consumer link):

| Metric | Target | Cartesia p50 | Cartesia p95 | Smallest p50 | Smallest p95 |
|---|---:|---:|---:|---:|---:|
| TTFA, first segment | < 120 ms | **156 ms** | 243 ms | **246 ms** | 293 ms |
| Subsequent segment TTFA | ≈ first | **4.9 ms** | 62 ms | 259 ms | 314 ms |
| RTF | < 1.0 | **0.10** | 0.11 | 0.58 | 0.62 |
| Cancel-to-silence | < 100 ms | 137 ms | 273 ms | **89 ms** | 109 ms |

From the production region (`sin`), single-sentence probe, 5 samples each:
Cartesia **186 ms** p50 (121–187), Smallest **216 ms** p50 (210–240). TCP RTT
measured inside a Fly machine is useless here — `api.smallest.ai`,
`api.cartesia.ai` and Google all answer in 1–2 ms because Fly proxies egress
locally — so only the full TLS+WebSocket number is trustworthy.

### Why Smallest is the default anyway

The 60 ms production gap is smaller than the turn's own variance (§6) and is not
the deciding factor:

| | Cartesia Sonic | Smallest Lightning |
|---|---|---|
| List price | $0.50 / 10K chars | **$0.195 / 10K chars** (2.6× cheaper) |
| Free allowance | 20K credits/month, concurrency 2 | $10 ≈ 51K chars, concurrency 10–15 |
| Accent | en-AU (`Ella`) | **en-IN** (`rhea`, `language: "hi"`) |
| RTF | 0.10 | 0.58 |
| Barge-in | in-band cancel, 137 ms | socket close, 89 ms |

The concurrency ceiling is what actually broke: the orchestrator fans out one task
per answer segment, Cartesia's free tier allows two, and the third segment's audio
was dropped. That is the origin of `TTS_MAX_CONCURRENCY=2`, and it is a structural
limit, not a tuning knob. Smallest has room to speak every segment of a 35-word
answer at once, costs 2.6× less per character, and speaks Indian English — which is
the audience. Cartesia remains one env var away (`TTS_PROVIDER=cartesia`) and keeps
the `iad-global` profile, where its lower latency and RTF are worth more than accent.

Two Lightning quirks are worth recording because they are easy to reintroduce:

* **`complete_backoff_ms` defaults to 4 s.** The provider holds `complete` for four
  seconds after the last audio chunk unless the request asks for `0`. Every request
  in the adapter and the probe asks for `0`; with the default, RTF measures 2.38
  instead of 0.58 and every turn looks four seconds slower than it is.
* **`context_close: true` does not stop audio.** It kept delivering chunks for
  9.3 s p50 (143–226 late chunks) after the call. Closing the socket stops it in
  89 ms, so a cancelled context closes its socket instead of returning it to the
  pool. `continue: true` continuations were also rejected: they buy prosody across
  segments for +90–160 ms to first audio.

Bulbul is request/response, not a socket. The turn engine compensates by
synthesising one short segment at a time, so the meaningful number is the first
segment's time-to-audio — but cancellation is still coarse, which is why it is not
the default.

Full comparison: ADR 0004. Raw JSON: `bench/results/smallest-ttfa-india-en-rhea-20261110-183801.json`
and `bench/results/cartesia-ttfa-compare-2026-11-10-20261110-183824.json`.

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

### Fly measurements (partial — `sin` only)

`bom` is not available on this Fly account, so the matrix cannot be run as
designed. What was measured inside the deployed machine (`sin`, version 6):

| Region | Sarvam RTT p50 | Cartesia TTFA p50 | Smallest TTFA p50 | Public loc |
|---|---:|---:|---:|---|
| `bom` | unavailable | — | — | Mumbai |
| `sin` | **55.9 ms** | 186 ms | 216 ms | Singapore |
| `fra` | not measured | — | — | Frankfurt |
| `iad` | not measured | — | — | Virginia |

`api.sarvam.ai` shows a real 55.9 ms RTT from `sin`, which is consistent with the
ASR finalisation times in §6. The other hosts cannot be measured by TCP RTT from
Fly at all (see §4), so their rows hold the WebSocket TTFA instead.

**Decision:** `sin` is the deployed region, provisional. `bom` wins on paper for an
Indian audience and should be re-tested if the account gains access (ADR 0002 §4).

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

### Production, same probe against `https://sarang-gateway.fly.dev` (`sin`, Smallest)

| Total | p50 | p95 |
|---|---:|---:|
| Client: last speech → first audible | 2 230 ms | 2 296 ms |
| Server: end-of-speech → first audible | 984 ms | 1 037 ms |
| Server: end-of-speech → first answer audio | **926 ms** | 978 ms |

| Stage | Measured p50 |
|---|---:|
| ASR finalization | 258 ms |
| Fact lookup | 0.9 ms |
| Context build | 0.0 ms |
| LLM TTFT | 395 ms |
| Segmentation | 128 ms |
| TTS TTFA | 234 ms |
| Egress reframe | 58 ms |

The server total improves on the laptop run (926 ms vs 1 123 ms) because the
machine is closer to the providers than the laptop is, and it is the number the
product owns. The client total is dominated by the 800 ms endpointing window plus
the round trip to `sin`; that is the figure a real-mic browser pass has to move.
JSON: `bench/results/e2e-turn-fly-smallest-20261110-185158.json`.

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

### Latency pass 2 — the endpointing clock was charging 1 050 ms, not 800

Four defects were found while re-reading the budget, all of them structural
rather than provider variance:

1. **Endpointing ran the hangover and the silence window in series** in
   `audio/vad.py`. `ENDPOINT_SILENCE_MS=800` therefore ended turns 1 050 ms after
   the last speech frame. Both clocks now start at the last frame above
   threshold and the hangover survives as a floor. This is why the client total
   and the server EOS figure never reconciled: the missing 250 ms sat in front of
   the server's clock, where no stage could see it.
2. **`segment` was measured after the whole model stream finished**, so it
   reported the rest of the generation (128 ms) under the name of the stage that
   decides when speech can start, and `llm_ttft + segment ≈ llm_total` hid it. It
   now closes when the first segment reaches TTS. Not a speed-up (the old 128 ms
   was double-counted inside `llm_total`), but the budget adds up now.
3. **Egress held every generation for a 40 ms pre-buffer.** The first frame of a
   turn is the one the visitor is waiting for and one 20 ms frame is already
   playable, so it now leaves immediately; later frames keep the pre-buffer.
4. **The first TTS span is dead air.** Smallest's first span measured 34.6 ms
   with a peak of 0–6 out of 32 768, and the next span carried another ~16 ms
   below threshold. Egress counts only frames that are audible, so all of it was
   charged as latency (measured 79 ms with a handoff that costs under 1 ms).
   Inaudible leading samples are now dropped before the first audible frame, with
   10 ms of pre-roll so a breathy onset is never clipped (`bench/probe/tts_leading_silence.py`
   reproduces the provider number).

Same probe, same machine (`sin`), 8 runs warm, measuring the shipped image:

| Total | Before | After |
|---|---:|---:|
| Client: last speech → first audible | 2 230 ms | 2 143 ms |
| Server: end-of-speech → first audible | 984 ms | 976 ms |
| Server: end-of-speech → first answer audio | 926 ms | **887 ms** |

| Stage | Before | After | Note |
|---|---:|---:|---|
| ASR finalization | 258 | 175 | provider variance, not a code change |
| Fact lookup | 0.9 | 0.7 | |
| Context build | 0.0 | 0.0 | |
| LLM TTFT | 395 | 398 | provider variance |
| Segmentation | 128 | 32 | same work, measured honestly |
| TTS TTFA | 234 | 245 | provider variance |
| Egress reframe | 58 | 74 | see below |

The 250 ms is not visible as a 250 ms drop in the remote client total, for two
reasons worth stating rather than hiding: the endpointing window is *client-side*
of end-of-speech, so it can only appear in the client delta (2 230 − 984 = 1 246
before, 2 143 − 976 = 1 167 after, with ~330 ms of India→`sin` network folded in
that varies run to run), and LLM TTFT plus TTS TTFA were 45 ms slower in the
after window. On loopback, where the transport adds nothing, the client total
measured 1 652 ms p50 in the same session. The change itself is pinned by
`tests/test_vad.py`, which asserts an 800 ms window ends a turn at 800 ms and not
at 1 050 ms.

Egress did not improve on Fly in that window (58 → 74 ms) even though the same
code measures 79 → 31 ms on loopback, and the cause is not the event loop. The
first synthesized span is entirely inaudible and is now discarded, so the egress
cannot emit a frame until the span carrying the first phoneme arrives: `egress`
measures the provider's gap between its first and second chunk, not work the
gateway does. The gateway's own handoff is bounded by
`test_first_frame_of_a_generation_is_not_paced` (< 10 ms), and a generation's
first frame is explicitly not paced. The split between `tts_ttfa` and `egress` is
an artifact of when the provider chunks; what the visitor experiences is their
sum, 258 ms on loopback and 293–319 ms from `sin`. There is nothing to reclaim
there, so the item is closed rather than deferred.

A related instrumentation defect is fixed at the same time. Silent frames emitted
while waiting for the first voiced samples were counted as `underruns`, so after
the dead-air trim every turn reported a stutter it did not have. An underrun now
only counts once a generation has played something, which is the only case where
the visitor can hear playback stop.

What is left, in order: endpointing (800 ms, deliberate — ADR 0003 §1), LLM TTFT
(~400 ms, provider), TTS request → first audible sample (~260–320 ms, provider
chunk cadence), ASR finalization (~175 ms, provider). Nothing in that list is
ours to optimise without either changing a provider or accepting a quality trade
the ADR rejected. The one remaining idea that is not provider-bound is
finalizing the recogniser before the endpointing window expires; it is only safe
if a resumed utterance can be replayed to the recogniser in full, which is a
project rather than an optimisation, and it is not started without evidence that
its ~175 ms is worth the accuracy risk.

## 7. Risks confirmed or retired

| Risk | Status | Evidence |
|---|---|---|
| Sub-500 ms impossible from India due to provider region | **confirmed for Groq**, mitigated | no Asia region; ADR 0002 moves the default in-region |
| Gemini free tier is a viable default | **retired — it is not** | 980 ms TTFT p50 and quota throttling; Sarvam is 206 ms |
| Cartesia leaks audio after cancel | **retired** | 137 ms cancel-to-silence, 18–46 late chunks measured and dropped |
| Smallest `context_close` stops audio | **retired — it does not** | 9.3 s p50, 143–226 late chunks; the adapter closes the socket instead (89 ms) |
| Smallest's 4 s `complete` backoff inflates every turn | **retired** | `complete_backoff_ms: 0` on every request; RTF 0.58, not 2.38 |
| Cartesia concurrency ceiling truncates multi-segment answers | **confirmed, retired by switching** | free tier allows 2 contexts; one task per segment exceeded it |
| Nemotron streaming access unavailable | open | comparison arm only |
| aiortc cannot obtain reachable ICE candidates on Fly | open | `/ws/audio` fallback keeps the product working |
| Prompt length materially increases TTFT | open | `llm_matrix.py` + `groq_ttft.py` measure it |
| Bulbul request/response adds first-segment latency | **retired by not shipping it** | Smallest is the default; Bulbul stays an arm |
| pgvector unavailable on the managed plan | mitigated | BM25 is the default; pg is opt-in |

## 8. Decisions required before production

1. ~~Sarvam ASR surface and finalization behaviour~~ — done: realtime WS,
   `saaras:v4`, manual endpointing, one warm socket per connection, 592 ms
   finalization (ADR 0003).
2. ~~Default LLM~~ — done: Sarvam 105B, 206 ms TTFT p50.
3. ~~TTS voice~~ — done: Smallest Lightning `lightning_v3.1_pro`, voice `rhea`,
   `language: "hi"` (ADR 0004). Cartesia remains the `iad-global` arm.
4. **Media-gateway region** — `sin` deployed; `bom` unavailable on this account,
   so the matrix in §5 stays partial.
5. **Whether TURN relay is mandatory or host candidates suffice** — diagnostics
   `candidateType` from real sessions.
6. **Real human-mic browser pass** — headless Chrome cannot grant microphone
   access, so every browser check so far is transport-level only.
