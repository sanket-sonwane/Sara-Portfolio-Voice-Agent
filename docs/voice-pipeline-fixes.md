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

---

## 6. Second pass — isolation, egress lifecycle, audio protocol

Date: 2026-11-13. Everything below is implemented and covered by tests
(`pytest -q` 271 passed; `ruff` clean; `npm run build` passes).

### 6.1 Session-scoped events

- `Bus.subscribe(session_id)` keeps session queues separate from global ones;
  `publish` routes by the envelope's `session_id`.
- `/ws/events` filters again before writing, so a mis-stamped envelope cannot
  cross sessions even if publishing is wrong somewhere else.
- The browser rejects any `event.session_id` that is not the current session
  **before** it can advance `generationRef`. A foreign epoch can no longer make
  legitimate events look stale.
- Tests: `tests/test_isolation.py`, `test_api.py::test_events_do_not_cross_session_boundaries`.

### 6.2 Egress lifecycle

`AudioEgress` now has `begin_generation`, `put`, `finish_generation`,
`interrupt`, `ack_transport` and `wait_transport_drained`:

- `begin_generation` clears the residual, so a previous turn's last partial
  frame can no longer prepend itself to the next answer.
- `finish_generation` switches the pre-buffer off and flushes/pads the final
  partial frame. The "one frame stuck forever, session stays SPEAKING" failure
  is gone.
- `next_frame()` returns an `EgressFrame` carrying `generation` and `sequence`,
  and re-checks the generation after the pacing sleep. A frame selected before
  an interrupt is returned as silence, never as old audio.
- Drain is decided on `ack_transport`, not on the queue looking empty.

### 6.3 Audio protocol

- `session.ready` advertises `audio_format: "pcm16-v2"`.
- `/ws/audio` frames are `u32 generation | u32 sequence | pcm16`.
- `Playback.enqueue` drops frames from an older generation or with a
  non-increasing sequence, which closes the post-EPOCH window.
- A client that receives an unknown `audio_format` refuses to parse binary
  frames instead of guessing.

### 6.4 ASR lifecycles

- Batch Nemotron uses a persistent event queue: `events()` waits for
  `finalize()` instead of exiting before a final exists.
- Streaming Nemotron exposes `alive`, reports dead when its reader dies, and
  tags every event with the turn from `begin_utterance()` rather than the
  default index 0 that every later turn rejected as closed.

### 6.5 Warm-up and shutdown

- The audio route retains the warm-up task and cancels it on close.
- `UtteranceLoop.warm()` closes an adapter it created after the loop closed, and
  the adapter is never installed once `closed` is set.
- `_begin_utterance` no longer waits on the ASR lock; an unready adapter drops
  the onset and lets the detector fire again.
- `Session.close()` closes the voice loop.
- The audio route clears `session.voice` only when it still owns the slot.

### 6.6 One turn owner

- `Session.input_seq` gives each text submission an owner. `run_turn` returns
  immediately when a newer submission exists, at entry and again after
  acquiring `turn_lock`.
- A typed question interrupts `user_speaking` as well as the producing states.
- `orchestrator._interrupt_now` calls `voice.abandon_pending()`, which cancels
  `_final_task` and marks the utterance superseded, so a transcript that arrives
  after the typed question cannot start a second answer.

### 6.7 generated / committed / audible

`TurnText` (`apps/api/sarang/text_state.py`) records three separate values:

- `generated` — the model's full output;
- `committed` — segments whose synthesis completed and whose audio reached the
  egress, in playback order;
- `audible` — the prefix the browser reports playing (server hook:
  `{"type": "audible", "turn_id": ..., "chars": n}`).

`assistant.final` carries `text` (the honest transcript = committed, falling
back to generated) plus `generated_text`, `committed_text`, `audible_text` and
`audible_chars`. A partial TTS segment is not committed even if some of its
audio was queued.

### 6.8 Smaller corrections

- Sarvam creates one resampler per audio blob; Smallest and Cartesia flush the
  resampler's filter tail at completion. `StreamResampler.flush()` added.
- English release configuration: `SMALLEST_LANGUAGE` defaults to `en` in
  `config.py`, `fly.toml` and `.env.example`; code-switching remains available
  explicitly with `SMALLEST_LANGUAGE=hi`.
- Startup owns a `startSeq` token; `teardown()` clears it and `startingRef`, so
  a stop during permission/session creation cannot permanently block restart.
- WebRTC fallback is single-flight: the `failed` callback and the enclosing
  `catch` share one `fallbackToPcm` promise, so one connection cannot open two
  PCM sockets.

### 6.9 Still open from the audit

LLM mid-stream honest stop, TTS partial-tail recovery with alignment offsets,
dropped-audio accounting in the transcript, follow-up entity resolution, the
pronunciation lexicon, and the measurement split (browser-confirmed audible
time vs scheduled time) remain for the next pass.

---

## 7. Third pass — honest degradation and a client-confirmed measurement

Date: 2026-11-13. Implemented and covered by tests (`pytest -q` 291 passed;
`ruff check apps tests bench` clean; `npm run build` passes). Everything here
is about not claiming more than actually happened.

### 7.1 LLM drops after the answer has started

The old handler ran the local answerer for *any* provider failure. If the
visitor had already heard two sentences, that appended a second, unrelated
answer after them, and the transcript claimed a response the model never
finished.

`_run_turn` now separates the two cases:

- nothing spoken yet (`not runtime.first_token and not runtime.segments`): the
  local fallback runs as before;
- speech started: `runtime.truncated = True`,
  `marks.degraded = "llm_midstream"`, the answer is the prefix already emitted,
  and `_stream_fallback` is **not** called. A `notice` tells the visitor the
  model dropped.

Tests: `tests/test_orchestrator.py::
test_llm_failure_after_speech_keeps_the_prefix_and_does_not_fall_back`.

### 7.2 TTS fails mid-segment

`TTSPartialError` (in `tts/base.py`) means audio was emitted but the segment did
not complete. `FallbackTTS` now has three outcomes:

- **nothing emitted:** the fallback speaks the whole segment, as before;
- **aligned partial:** `AudioSpan.text_range` gives the covered character
  range, so the fallback speaks only `text[covered:].lstrip()` — never the
  prefix the visitor already heard;
- **unaligned partial:** the boundary is unknown, so it raises
  `TTSPartialError` instead of replaying or pretending.

`OrderedSpeech._synthesize` catches it, sets `degraded = "tts_partial"`,
records the segment in `partial_segments`, and `submit()` refuses every later
segment. The turn ends on the prefix it actually spoke. If nothing was
committed, the transcript is the cut sentence with a `" …"` marker; the
`generated_text` remains the full model output.

Tests: `tests/test_tts_partial.py`, and
`test_tts_partial_stops_the_turn_and_claims_only_a_cut_sentence` for the turn
level.

### 7.3 ASR audio that was never heard

Sarvam sheds the oldest queued audio on outbox overflow. The recogniser never
heard it, yet the transcript looked complete.

- `_Utterance` records `raw_latest`, `raw_final`, `dropped_baseline` and
  `dropped_ms`; the delta is taken from `begin_utterance()`, so one turn is not
  charged for another's loss.
- `transcript.final` carries `degraded` (`asr_audio_dropped` / `asr_timeout` /
  `""`), `dropped_ms` and `raw_text`.
- The ASR dump JSON carries the same fields.

Tests: `test_dropped_asr_audio_is_reported_honestly` (dump + event) and
`test_a_clean_utterance_reports_no_degradation`.

### 7.4 Follow-ups resolve against the previous turn, without burying new subjects

`is_followup(text, previous, current)` in `kb/lookup.py` replaced the
orchestrator's bare regex. A pronoun (`it`, `that`, `this`, `the numbers`) or a
continuation phrase (`tell me more`, `more about`, `and the`) means the question
refers back. The one subject-introducing shape, `what about X?` / `how about X?`,
is a follow-up only when X did not resolve a new confident block — so
“what about QA-Forge?” keeps its own evidence.

Tests: the follow-up block in `tests/test_lookup.py`.

### 7.5 Pronunciation stays at the TTS boundary

`tts/lexicon.py` maps identifiers and acronyms (`ARAI`, `MEC`, `VisionMitra`,
`ClassNode`, `sched_ext`, …) to a spoken form. It is applied in
`OrderedSpeech.submit` only: display text, the transcript and the evidence card
keep the original spelling.

### 7.6 The latency number is not the browser's “last chunk sent”

The primary number is **server-side**: last detected speech frame →
`transport_first_frame` (the first answer frame written to the wire).

The browser cannot share the server's monotonic clock, so it reports deltas
measured from the first voiced binary frame it received:

- `scheduled_ms` — until the first buffer is scheduled (`Playback.audibleFrom`);
- `audible_ms` — until `getOutputTimestamp()` confirms the output device reached
  it (`Playback.audibleConfirmedFrom()`).

`TurnMarks.client_payload()` adds them to `eos_to_transport_first_frame_ms`,
producing `eos_to_client_scheduled_ms` and `eos_to_client_audible_ms`. The late
report arrives as a `metrics.client` event and is merged into the turn by
`MetricsStore.observe_client`.

`bench/eval.py` now reports `eos_to_transport_first_frame` as the primary
metric, `eos_to_client_audible` as the confirmed figure when a browser sent one,
and labels `answer_first_audio_provider_output` as provider output rather than
something the visitor heard.

`assistant.final` text is the committed prefix (or generated text when nothing
was committed); `audible_text` is only ever set from the browser report.

### 7.7 Deployed endpoint experiment (2026-11-13)

Target: `sarang-gateway.fly.dev` (`sin`, `SARANG_PROFILE=bom-india`, Sarvam ASR +
Sarvam LLM, Smallest Lightning TTS). Five turns per window, same fixture
(`bench/audio/question-16k.wav`, one 460 ms internal pause). Primary number is
the server's last-speech-frame → first frame on the wire.

| `ENDPOINT_SILENCE_MS` | endpoint stage | EOS→wire p50 | p90 | p95 | client p50 |
| --- | --- | --- | --- | --- | --- |
| 800 (baseline) | 860 ms | 1676 ms | 1952 ms | 1967 ms | 1918 ms |
| 650 | 760 ms | 1718 ms | 1972 ms | 2054 ms | 2052 ms |
| 500 (shipped) | 560 ms | 1586 ms | 1828 ms | 1897 ms | 1855 ms |

All 15 transcripts were identical and correct, including the 500 ms window: the
460 ms pause did not split a turn. The endpoint stage moved by exactly the
configured amount each time (860 → 760 → 560; the extra 60 ms is the detector's
frame granularity).

EOS→wire did **not** fall by 300 ms across the experiment because the legs after
endpointing vary more than the window being removed. At 500 ms the p50 breaks
down as 560 endpoint + 240 ASR final + 414 LLM TTFT + 32 segmentation + 253 TTS
first audio + 82 egress. Even with endpointing at zero, that cascade is ~1.0 s.
**600 ms is not reachable without producing audio before the final transcript
is validated**; shortening the silence window only moves a fraction of it.

Results:

- `bench/results/e2e-turn-deployed-sin-800ms-20261113-190821.json`
- `bench/results/e2e-turn-deployed-sin-650ms-20261113-191240.json`
- `bench/results/e2e-turn-deployed-sin-500ms-20261113-191658.json`

The deployed value was **500 ms** when this work shipped. The 800 ms row is the
baseline it replaced; 650 was rejected as noise within the p90 spread. The probe
was fixed first: it used to parse the pcm16-v2 header as samples, which makes
every frame look voiced (the generation word is non-zero) and would have
reported a flattering first-frame time. It now strips and validates
`u32 generation | u32 sequence`, and reports p50/p90/p95.

Five turns per window is enough to show the endpoint stage moved by exactly the
configured amount, but not enough to separate the post-endpoint stages from
provider jitter. Treat the 500→800 p50 gap as directional until a 20–30 turn run
confirms it on the shipped build.

### 7.8 Still open

- The `metrics.client` event and the `audible` control message are not yet in
  the README or the public docs.
- `wait_transport_drained` can wait up to 30 s when a transport never acks.
  Correct, but the ceiling should be configurable before the next long soak.
- `eos_to_transport_first_frame` includes a cached filler when one is played, so
  it is "first audio on the wire", not always "first answer frame". The
  distinction is visible in `eos_to_answer_audio_ms`; keep both when reporting.



