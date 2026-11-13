# Voice pipeline fix pass — change log

Date: 2026-11-13
Scope: end-to-end voice path (browser capture → `/ws/audio` → VAD → ASR →
turn engine → TTS → egress → browser playback), plus the instrumentation that
measures it.

Target under test: **600 ms from the last speech frame**. That target is not
met; this file records exactly what was fixed, what was measured, and the
arithmetic that makes 600 ms unreachable on the current provider mix without a
pre-final audio path.

Companion documents:

- `docs/pipeline.md` — component/audio-flow map and the current measured budget.
- `docs/adr/0003-turn-taking-and-asr-quality.md` — decisions on turn taking.
- `bench/results/e2e-turn-local-india-20261113-173631.json` — steady baseline.

---

## 1. Findings fixed

Nine source-level findings plus two instrumentation defects were confirmed and
fixed. Each entry says what was wrong, what changed, and where.

### 1.1 The PCM worklet was TypeScript in a `.js` file

- **File:** `apps/web/public/pcm-worklet.js`
- **Symptom:** `audioWorklet.addModule()` parses the file as plain JavaScript.
  The `process(inputs: Float32Array[][]): boolean` annotation was a parse error,
  so `addModule()` rejected; the old code swallowed that rejection and silently
  fell back to the deprecated `ScriptProcessorNode`. Every measurement was taken
  on the fallback path.
- **Fix:** the worklet is plain JavaScript (`process(inputs)`). The comment in
  the file explains why it must stay that way.
- **Guard:** `apps/web/package.json` now runs `node --check public/pcm-worklet.js`
  before `tsc`/`vite` in `npm run build`, and `tests/test_web_assets.py` parses
  the file, rejects type annotations, and checks the processor registration.

### 1.2 ASR ingress blocked the microphone path

- **Files:** `apps/api/sarang/asr/sarvam.py`, `apps/api/sarang/asr/base.py`
- **Symptom:** `push()` awaited a WebSocket send per chunk and `begin_utterance()`
  could wait up to `session_begin_timeout` (2.5 s) for the provider. Both run
  inside the frame handler, so a slow uplink or an unarmed socket stalled audio
  capture at the exact moment the visitor was speaking.
- **Fix:**
  - Every outbound frame goes through a bounded, ordered `deque` (`_outbox`).
  - One `_sender_loop()` task drains it; `push()` and `begin_utterance()` only
    enqueue and return.
  - When the outbox exceeds `max_queue_ms` (default 500 ms), the **oldest audio**
    is dropped and `dropped_ms` is counted — real time has already moved past it.
  - Sends have their own `send_timeout`; a timeout/failure marks the session
    dead and clears the queue.
  - `ASRSession.begin_utterance()` now returns `bool`, and `SarvamASR` returns
    `False` instead of waiting when the session is not armed.
- **Behaviour kept:** audio is still only sent after `speech_start`, in order,
  and `speech_end` still closes the provider turn (`flush` is not sent; it could
  truncate the final).

### 1.3 A live but unarmed ASR socket looked ready

- **Files:** `apps/api/sarang/asr/sarvam.py`, `apps/api/sarang/voice.py`
- **Symptom:** a socket that opened but never received `session.begin` was
  treated as warm. Audio sent before arming can lose the first word.
- **Fix:**
  - New `ready` property = `session.begin` seen **and** socket alive.
  - `connect()` now raises `ASRError` when `session.begin` does not arrive
    within `session_begin_timeout` and closes the socket.
  - `push()`/`finalize()`/`begin_utterance()` no-op or return `False` when not
    ready.
  - `voice.py` only treats an explicit `False` as “unready”; legacy adapters
    returning `None` keep working.

### 1.4 Finalizing a superseded turn blocked audio and leaked state

- **File:** `apps/api/sarang/voice.py`
- **Symptom:** on barge-in, the loop waited up to 250 ms for the previous
  finalize, and on cancellation it reset `_finalizing` without closing the
  utterance. The utterance stayed in `_open` / `_by_provider`, where a late
  provider event could be routed to a turn that would never run.
- **Fix:**
  - `_abandon_finalize()` cancels the previous finalize task synchronously; it
    is never awaited on the microphone path.
  - The turn is marked `superseded` first, so even a completed final cannot
    start an answer.
  - `_close_utterance()` moved into an unconditional `finally` in `_finalize`.
  - The failed-`begin_utterance` path now resets the VAD and state; otherwise
    the detector stays `active` and cannot emit `speech_started` again.
  - `USER_SPEAKING` is published when capture starts.

### 1.5 The server declared LISTENING while the answer was still playing

- **Files:** `apps/api/sarang/audio/shaper.py`, `apps/api/sarang/orchestrator.py`
- **Symptom:** TTS synthesis finishing was treated as playback finishing. The
  egress can still hold hundreds of milliseconds of the answer, so both
  barge-in paths stopped listening while the visitor was still hearing speech.
- **Fix:**
  - `AudioEgress` tracks one drain event per generation. `wait_drained()` is
    satisfied only when the chunk queue is empty **and** the residual has no
    whole frame left.
  - `put()` clears the completion flag when new audio for that generation
    arrives; `interrupt()` releases waiters immediately (the old generation is
    cancelled, not played out).
  - `Orchestrator._watch_playback()` waits for the drain in its own task, then
    flips to `LISTENING` only if the session/generation is still current. This
    keeps `turn_lock` free for the duration of playback.
  - The old `_deadline`-less first frame still sends immediately; drain tracking
    does not add pre-buffer latency.

### 1.6 Cancellation held the turn lock

- **File:** `apps/api/sarang/orchestrator.py`
- **Symptom:** on `CancelledError`, the turn awaited `speech.cancel()`, including
  provider teardown, while holding `turn_lock`. The next utterance paid for the
  old provider's teardown.
- **Fix:**
  - The cancelled turn schedules cleanup as a background task and re-raises, so
    the lock is released immediately.
  - `OrderedSpeech.cancel()` is idempotent (`_cancelled` guard), so a late
    explicit cancel cannot cancel provider tasks twice or count twice.
  - `_active_speech` entries are still removed in `finally`.

### 1.7 WebSocket uplink had no backpressure policy

- **Files:** `apps/web/src/audio.ts`, `apps/web/src/useSarang.ts`
- **Symptom:** a slow uplink accumulated an unbounded `bufferedAmount`, so the
  recogniser could receive speech seconds late.
- **Fix:**
  - `PcmSocket` tracks queued audio time from `bufferedAmount`; once it exceeds
    `maxQueueMs` (500 ms default), the newest frame is dropped and `droppedMs`
    is counted.
  - An `onOverload` handler surfaces a throttled (5 s) warning to the visitor.
  - `Playback.active` reports whether audio is still scheduled/playing, and the
    client barge-in watch keys off playback as well as the server view state.

### 1.8 Speculation armed on noise

- **File:** `apps/api/sarang/speculative.py`
- **Symptom:** `stable_hits=1` armed the first eligible partial immediately, so
  every recogniser revision bought a full provider stream.
- **Fix:**
  - Defaults are now `stable_hits=2`, `min_age_ms=180`, `max_arms=12`,
    `max_arms_per_turn=3`.
  - A hypothesis must be repeated **and** have been on screen for the dwell.
  - Changing the hypothesis discards the previous arm (stops paying for it).
  - `reset()` clears the hit/dwell state and reopens the per-turn arm budget.
- **Unchanged safety rule:** a speculative answer is buffered, never spoken
  before the final transcript passes the token/evidence/prompt gates.

### 1.9 Instrumentation reported the wrong event as “first audio”

- **Files:** `apps/api/sarang/metrics.py`, `apps/api/sarang/app.py`,
  `apps/api/sarang/webrtc.py`, `bench/probe/e2e_turn.py`
- **Symptom:** `first_audio` meant three different things depending on where it
  was written: first synthesized span, first frame the egress selected, or first
  frame on the wire.
- **Fix:** three explicit fields and payload keys:
  - `tts_first_span_ms` — first PCM received from the provider.
  - `egress_first_frame_ms` — first frame the shaper selected.
  - `transport_first_frame_ms` — first frame written to `/ws/audio` or the
    WebRTC RTP track.
  - `answer_first_audio_ms` now aliases `tts_first_span_ms` (it is documented as
    the provider span), and `first_audio_ms` / `eos_to_first_audio_ms` remain
    deprecated aliases for the egress frame for one release.
  - `STAGE_ORDER` and the metrics aggregator understand the new sample names.
  - The probe now prints all four EOS→event totals so a run cannot silently
    compare unlike stamps.

### 1.10 Additional instrumentation/robustness fixes

- `apps/api/sarang/app.py`: `/healthz` reports the detector actually in use
  (`silero`/`energy`) and the endpoint window; the WS sender marks
  `transport_first_frame` only after `send_bytes` returns.
- `apps/api/sarang/webrtc.py`: same transport stamp on the RTP track.
- `docs/pipeline.md`: budget and metric names updated to match the code.
- Probe JSON now records the split totals.

---

## 2. Tests added / updated

All tests run with zero API keys (local profile + synthetic TTS).

| File | Coverage |
|---|---|
| `tests/test_asr_sarvam.py` | Sender task wiring; non-blocking `push` under a slow socket; overflow drops oldest and counts; unarmed `begin_utterance` returns `False`; `connect()` raises when `session.begin` never arrives; ordering of `speech_start`/audio/`speech_end`; no `flush`. |
| `tests/test_speculation.py` | Stability dwell and two-hit gate; per-turn arm budget; `reset()` reopens the budget; all existing adoption/refusal gates. |
| `tests/test_voice.py` | New stability/settle expectations; abandoned finalize leaves `_open`/`_by_provider`; existing stale-final, punctuation and endpoint-timing regressions. |
| `tests/test_shaper.py` | `wait_drained` blocks until the last frame; immediate for no audio/stale generation; `interrupt` releases a waiter; new audio clears the completion flag. |
| `tests/test_orchestrator.py` | State stays `SPEAKING` until the egress drains; a cancelled generation is not resurrected; a cancelled turn releases `turn_lock` before slow provider cleanup. |
| `tests/test_metrics.py` | Split first-audio fields, aliases, and transport-only payload. |
| `tests/test_web_assets.py` | Worklet registers `sarang-pcm`, has no type annotations, parses with `node --check`; client `clientEosToAudible()` returns `undefined` for unpaired/reversed stamps. |
| `apps/web/src/metrics.ts` | Extracted pure `clientEosToAudible()` used by `useSarang.ts`; testable without a browser. |

Result: **251 passed, 1 warning** (the Starlette/httpx deprecation warning).

---

## 3. Verification

```powershell
$env:PYTHONPATH="apps/api"
.venv\Scripts\python.exe -m pytest -q          # 251 passed
.venv\Scripts\ruff.exe check apps tests bench  # clean
cd apps/web; npm run build                     # node --check + tsc + vite
$env:PYTHONPATH="apps/api;."
.venv\Scripts\python.exe -m bench.probe.e2e_turn --runs 3
```

Probe turn: “What did Sanket build during the ARAI internship?” — transcript
correct 3/3.

### Steady baseline (3 runs, `bench/results/e2e-turn-local-india-20261113-173631.json`)

| Stage | p50 |
|---|---|
| endpoint | 560 ms |
| asr_finalize | 0.0 ms |
| asr_final | 344 ms |
| lookup | 0.6 ms |
| context | 0.0 ms |
| llm_ttft | 255 ms |
| segment | 60 ms |
| tts_ttfa | 189 ms |
| egress | 63 ms |
| **last speech → first frame on the wire** | **~1,470 ms** |
| last speech → first audible (browser clock) | 1,357 ms |

Run-to-run variance is real: a second 3-run set immediately after the
instrumentation change measured 1,571 ms client / 1,701 ms server p50, driven by
a cold first run (ASR final 795 ms, LLM TTFT 466 ms). The stage ordering was the
same, and the new stamps ordered correctly: `tts_first_span 1,629 ms ≤
egress_first_frame 1,670 ms ≤ transport_first_frame 1,701 ms`.

---

## 4. The 600 ms target — honest arithmetic

600 ms is measured **from the last speech frame**. The current pipeline spends:

```
endpoint          560 ms   (measured; configured 500, 32 ms windows + 20 ms pushes)
ASR final         344 ms   (provider emits the final after speech_end)
--------------------------------
subtotal          904 ms   before the model or TTS is asked to do anything
LLM TTFT          255 ms
segment            60 ms
TTS first span    189 ms
egress             63 ms
--------------------------------
total            ~1,471 ms   (server clock, last speech → first frame)
```

Even if lookup, context and egress were free, endpoint + ASR final alone are
~900 ms. 600 ms is therefore **arithmetically unreachable** without starting the
answer before the final transcript exists.

What actually has to change to get near 600 ms:

1. **Shorten the endpoint window.** 500 ms is the measured safe floor for this
   question clip (its longest internal pause is 460 ms). A smaller window
   truncates real questions. This is a product/accuracy trade-off, not a bug.
2. **Overlap ASR finalisation with model work.** The speculative LLM arm already
   does this when a stable partial arms; it removes the post-final LLM TTFT on
   adopted turns. Adoption is gated and often refused, so it is not a guarantee.
3. **Pre-final audio.** To remove the 344 ms ASR-final wait you must synthesize
   (and, for the full win, play) audio for a hypothesis that the final transcript
   has not validated. This pass deliberately does **not** do that.

### Safety rule kept

**No speculative audio is ever sent to the transport before transcript
validation.** The gates in `Speculation.consume()` (token overlap, evidence
identity, prompt equality) run first; an arm that fails is discarded and counted
as wasted. Pre-final playback would mean the visitor hears an answer to a
question they did not ask, and unlike text it cannot be edited after the fact.

### P2 decision: speculative TTS

**Deferred, and no no-op flag was added.** The only safe form is “synthesize for
the armed hypothesis into a private buffer, release it only if the final
transcript adopts the arm.” That needs:

- a second, validated PCM buffer between `OrderedSpeech` and `AudioEgress`;
- a cancel path that stops provider synthesis without charging the egress;
- an adoption check that compares the synthesized text, not just the queue;
- accounting for paid-but-discarded synthesis.

The speculative LLM arm already captures most of the model-side win. Adding TTS
speculation without the buffer/gate plumbing would violate the rule above, so it
is a follow-up with its own design and benchmark.

---

## 5. Known residual risks

- **Run-to-run variance** in provider latency (ASR final 262–795 ms, LLM TTFT
  230–466 ms) is larger than several of the fixes. Compare p50s over ≥3 runs and
  keep the raw JSON.
- **Egress drain is transport-level.** It means “the process handed every frame
  to the socket”, not “the speaker finished”. Browser buffering is reported
  separately by the client clock.
- **Queue shedding is lossy by design.** On a slow uplink the oldest microphone
  audio is dropped; the visitor is warned, and the transcript may miss a word.
- **WebRTC is still secondary.** The baseline above is `/ws/audio`; migrating
  transports invalidates it.
- **Credential rotation is still required** if the distributed ZIP contained
  `.env`; it is not part of this code change.
