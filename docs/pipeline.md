# Sarang voice pipeline: components, triggers, and the audio path

This is the current, measured map of the pipeline after the turn-taking pass in
ADR 0003 (§8, §11, §12). It describes what actually runs, in order, with the
timings the probes report on the reference machine. Where a number appears it is
p50 from a real run through `/ws/audio`, not an estimate.

## 1. The shape of one turn

```
browser mic (48 kHz)
  └─ AudioWorklet → 20 ms frames → downsample to 16 kHz PCM16
      └─ /ws/audio (binary)  ── or ── /api/offer (WebRTC track)
          └─ StreamResampler(16 kHz)            (only if sr != 16 kHz)
              └─ UtteranceLoop.push(320 samples)
                  ├─ ring buffer (last 300 ms)          ← pre-roll
                  ├─ SileroVad.push()  (512-sample / 32 ms windows)
                  │    ├─ speech_started ─→ _begin_utterance()
                  │    │     ├─ barge-in gate (if SPEAKING/THINKING/TOOL_WORKING)
                  │    │     ├─ ASR.begin_utterance() + pre-roll replay
                  │    │     └─ capture frames → ASR.push()
                  │    ├─ partial_due ────→ transcript.partial (+ speculation note)
                  │    └─ speech_ended ──→ _start_finalize()  (own task)
                  │                          └─ ASR.finalize() → transcript.final
                  │                              └─ Orchestrator.run_turn()
                  │                                   ├─ lookup / tier-2
                  │                                   ├─ LLM stream → segmenter
                  │                                   ├─ TTS (per segment, concurrent)
                  │                                   └─ egress.put(pcm, generation)
                  └─ egress.next_frame() → 960-sample 20 ms frames @ 48 kHz
                        └─ sender task → WS binary frame / WebRTC audio track
                            └─ browser AudioContext (jitter buffer) → speaker
```

Two transports, one loop. `UtteranceLoop` owns endpointing, barge-in and the ASR
lifecycle for both `/ws/audio` and WebRTC; the transports differ only in how PCM
arrives and how outbound frames leave.

## 2. Components and what each one owns

| Component | File | Owns |
|---|---|---|
| `UtteranceLoop` | `voice.py` | VAD feeding, pre-roll, utterance state, ASR routing, finalize |
| `SileroVad` / `Vad` | `audio/vad.py` | Speech probability per 32 ms window; endpoint window |
| `SarvamASR` | `asr/sarvam.py` | One recogniser socket per connection, 100 ms batches |
| `Speculator` | `speculation.py` | Stable-partial lookup + speculative LLM arm |
| `Orchestrator` | `orchestrator.py` | Turn lifecycle, retrieval, LLM, segmentation, barge-in |
| `TextSegmenter` | `segmenter.py` | First speakable chunk, then natural phrasing |
| `OrderedSpeech` | `orchestrator.py` | Concurrent synthesis, strict playback order |
| `TTS` (Cartesia/Smallest) | `tts/*` | Streaming PCM at 48 kHz |
| `AudioEgress` | `audio/shaper.py` | Fixed framing, generation gating, pacing, lead trim |
| `TurnMarks` | `metrics.py` | One timeline per turn; p50/p90/p95 aggregation |

### Detector choice and the window

`VAD_MODE=silero` is the default. It scores a 512-sample (32 ms) window on the
signal itself and keeps 64 samples of context between windows (v5 framing), so it
is level-invariant and needs no modulation gate. `build_vad` loads the vendored
model (`audio/models/silero_vad.onnx`, SHA256 pinned), caches one ONNX session
per file, and **restores `ENDPOINT_SILENCE_MS=800` if it has to fall back to the
energy detector** — 500 ms was only ever measured safe for Silero.

| knob | value | why |
|---|---|---|
| `ENDPOINT_SILENCE_MS` | 500 | longest internal pause in the reference question is 460 ms |
| `ENDPOINT_MIN_SPEECH_MS` | 96 | commit after ~3 windows; pre-roll covers the rest |
| `ENDPOINT_PREROLL_MS` | 300 | commit + first syllable; 800 ms was sized for the energy VAD |
| `ENDPOINT_HANGOVER_MS` | 180 | inert under Silero (`max(silence, hangover)`) |
| `ENDPOINT_EARLY_SILENCE_MS` | −1 | content-based early close removed; it truncated real questions |

## 3. The three clocks

Latency bugs in this pipeline came from mixing clocks, so the model is explicit.

1. **Media time** (`_media_ms`, 16 samples/ms). Used for the endpoint window and
   silence onset. It advances with the audio, so a test that pushes frames faster
   than real time still measures 500 ms for a 500 ms window.
2. **Monotonic wall clock** (`time.perf_counter`). Used for stage marks, egress
   pacing, and provider timeouts.
3. **Browser clock** (`performance.now`). Used only for the client-side
   last-speech → audible measurement; never compared with server stamps.

The turn origin is the **last speech frame**, recorded when `speech_ended` fires
as `decision_at − endpoint_ms`. The `endpoint` stage therefore runs forward from
the origin, and every later stage is disjoint from it.

## 4. Per-turn phase map

| Phase | Trigger | Work | Exit condition |
|---|---|---|---|
| Pre-roll | every frame while not capturing | keep last 300 ms in a ring | ring limit |
| Onset | `speech_started` | barge-in gate, ASR begin, replay ring in one send | ASR armed |
| Capture | each frame while capturing | `ASR.push(20 ms)`, collect dump PCM | `speech_ended` |
| Endpoint | `speech_ended` | stamp origin, start finalize task | detector says 500 ms of silence |
| Finalize | finalize task | `ASR.finalize()`, await final (≤3.5 s), normalize, dump | final or timeout |
| Retrieval | `run_turn` | speculation consume + tier-1, optional tier-2 | evidence published |
| Prompt | — | history (6 msgs) + context + lexicon | messages built |
| Model | LLM stream | first token closes `llm_ttft` | stream ends |
| Segment | first delta | `TextSegmenter.push/due` | first segment submitted |
| Synthesis | per segment | TTS streams 48 kHz PCM | first span closes `tts_ttfa` |
| Egress | first chunk | `put`, generation check, lead trim | first audible frame |
| Playback | sender loop | 960 samples / 20 ms, paced | queue drained |

Barge-in is a state transition, not a phase: `_interrupt_now` bumps the
generation, gates egress, cancels the turn task and publishes `EPOCH`
synchronously; provider cancellation runs in a background task bounded at 1 s.
The browser flushes on `EPOCH` instead of waiting for the server round trip.

## 5. Trigger matrix

| Event | Emitted by | Published / action | Consumers |
|---|---|---|---|
| `speech_started` | VAD | `_begin_utterance`, maybe `interrupt_soon` | ASR, orchestrator |
| `partial_due` | VAD (320 ms of speech) | `transcript.partial` if one has arrived | UI |
| ASR partial | Sarvam | `transcript.partial` + `Speculator.note` | UI, lookup/arm |
| `speech_ended` | VAD | `_start_finalize` | ASR finalize |
| ASR final | Sarvam | `_Utterance.event`, `transcript.final` | finalize task |
| Turn start | `run_turn` | `state=thinking`, `evidence` | UI |
| First token | LLM | `assistant.delta`, `llm_ttft` exit | segmenter, UI |
| Segment ready | segmenter | `speech.submit` | TTS |
| First span | TTS | `tts_ttfa` exit | `OrderedSpeech` |
| First chunk | egress put | `egress` enter, answer audio mark | shaper |
| First audible | egress | `state=speaking` | UI |
| `EPOCH` | interrupt | generation bump, flush | browser, egress |
| Turn end | `run_turn` | `assistant.final`, `metrics.turn` | UI, metrics ring |

## 6. Measured budget (reference machine, `/ws/audio`, 3 runs)

Turn: “What did Sanket build during the ARAI internship?” — transcript correct 3/3.
Baseline file: `bench/results/e2e-turn-local-india-20261113-173631.json`.

| Stage | p50 | Note |
|---|---|---|
| endpoint | 560 ms | configured 500; observed 512–560 from 32 ms windows + 20 ms pushes |
| asr_finalize | 0.0 ms | `speech_end` is queued, not awaited |
| asr_final | 344 ms | wait for provider final |
| lookup | 0.6 ms | tier-1, in-process |
| context | 0.0 ms | prompt assembly |
| llm_ttft | 255 ms | first delta |
| segment | 60 ms | `first_hold_ms=80`; released on 3-word floor |
| tts_ttfa | 189 ms | first PCM span |
| egress | 63 ms | egress-selected frame |
| **last speech → first frame on the wire** | **~1,470 ms** | server clock, `eos_to_transport_first_frame_ms` |
| **last speech → first audible (browser)** | **~1,357 ms** | client clock, measured from the last uploaded chunk |

Two clocks, intentionally not reconciled: the server origin is the last frame the
detector *heard as speech*; the browser origin is when it finished *sending* the
last chunk. The browser number is therefore slightly shorter and is not the
number the 600 ms target is judged against. The honest number is the server's
last-speech → first frame on the wire, and the stages sum to it:
`560 + 344 + 1 + 0 + 255 + 60 + 189 + 63 ≈ 1,472 ms`.

Instrumentation now separates three events that used to share `first_audio`:

| Payload key | Meaning |
|---|---|
| `tts_first_span_ms` | first synthesized PCM received from the provider |
| `egress_first_frame_ms` | first frame the shaper selected for transmission |
| `transport_first_frame_ms` | first frame actually written to the socket / RTP track |

The ordering is asserted by the probe (`1629 ≤ 1670 ≤ 1701 ms` on the
instrumentation run); a fast provider is not a fast egress and neither proves
the visitor heard anything. `first_audio_ms` remains as a deprecated alias for
the egress frame for one release.

## 7. Invariants

- **One recogniser socket per connection.** The handshake is warmed when the
  transport opens, not on the first question. A dead socket is rewarmed in the
  background; never inside `push`.
- **The recogniser hears the whole utterance.** Pre-roll is replayed once, in
  one send, and in order.
- **A final is only applied to the turn that produced it.** Provider indices are
  mapped as events arrive; an unplaceable event is dropped, not applied to
  `current`.
- **A superseded finalize never speaks.** If new speech starts before the final
  lands, the transcript is dumped but no turn runs.
- **Audio from a cancelled generation is dropped at the egress boundary**, not
  merely stopped upstream.
- **Every frame is 960 samples at 48 kHz**, paced on a monotonic deadline.
- **A stage that never ran is not measured** (`exit_open`), and the endpoint
  window is always on the timeline before ASR.

## 8. Known limits

- **600 ms from last speech is not reachable on the current provider mix.**
  The endpoint window (560 ms measured) plus the provider's ASR final (344 ms)
  already exceed the budget by ~300 ms before the model or TTS run. Meeting it
  requires a pre-final path: the answer must begin synthesizing before the
  final transcript lands. That is not implemented for audio, by design — see
  `docs/voice-pipeline-fixes.md` for the arithmetic and the safety rule.
- ASR finalisation (~344 ms) is the largest controllable stage; the provider
  emits no partial within 10% WER, so speculation is armed from later partials.
- Endpoint cannot go below ~512 ms observed without a shorter model window; 500
  is the safe floor for this clip, and 250 ms was refuted by measurement.
- WebRTC is not the primary transport yet; migrating would invalidate the
  baseline in this document.
