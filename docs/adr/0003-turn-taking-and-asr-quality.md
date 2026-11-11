# ADR 0003 — Turn-taking and recognition quality

**Date:** 2026-11-10
**Status:** accepted
**Context:** the first live user test of `bom-india` produced transcripts that
were a *fragment* of the sentence, with the first few words and some middle
words garbled, and Sara began answering while the visitor was still talking. The
target is WER < 6%. This ADR records what was actually wrong and why the fixes
are shaped the way they are.

---

## 1. Endpointing is owned by the gateway, and it was too eager

**Decision.** Silence to end a turn is **800 ms**, minimum speech **120 ms**,
hangover **250 ms**, and the last **800 ms** of audio is kept as a pre-roll ring
buffer and replayed when speech starts.

**Why.** The previous 500 ms window with a 140 ms hangover treated the tail of a
word, or a breath, as the end of the turn. That produces exactly the reported
symptom: a transcript that is a fragment, and an answer that starts over the
visitor. Indian-accented English has longer closure and more breath than the
corpora the default was tuned on.

The 120 ms minimum used to *discard* the audio that triggered the VAD, so the
first syllable of every question was never sent to the recogniser. The pre-roll
buffer is what makes a longer silence window free: nothing is lost at the start
of the next utterance, so the extra wait only costs the 800 ms itself.

**Trade-off.** Every turn pays 800 ms before the recogniser is told the turn
ended. That is the correct place to spend latency: answering a fragment costs a
whole extra turn.

**Correction (same day).** The implementation made the hangover and the silence
window run in series, so the configured 800 ms behaved like 1 050 ms and every
turn paid 250 ms nobody had budgeted. Both clocks now start at the last frame
above threshold; the hangover remains a floor so a very short `silence_ms`
cannot clip soft word endings. The decision above is unchanged — the point is
that the number in the config is now the number charged, which is what the plan
and the budget both assume. See Phase 0 §6, "Latency pass 2".

**Why it cannot be made adaptive (measured, not assumed).** Endpointing is the
largest remaining term, so it is the obvious target, and two ways of shortening
it were tried and rejected on evidence:

1. *A shorter window.* `bench/probe/pause_profile.py` measures the silences
   inside a normally-spoken question with the gateway's own VAD. The reference
   question — "What did Sanket build during the ARAI internship?" — has a **460 ms**
   pause between "Sanket" and "build". Any window at or below 450 ms therefore
   cuts a question that was spoken correctly and completely. Confirmed end to
   end: closing the recogniser 450 ms into a pause returned the final
   `"What did Sanket"`. 500 ms is the first value that survives this one sample,
   which is consistent with the real cut-offs that produced this ADR at 500 ms.
2. *Trusting the partial.* `bench/probe/endpoint_evidence.py` records what the
   live partial says at each candidate cutoff. During a mid-sentence pause the
   partial was `"What did Sanket build during the ARII internship?"` — complete,
   punctuated, and stable for over a second — while the visitor was still
   talking. Re-running the identical audio produced `"What did Sanket"` instead,
   so the signal is not reproducible either. A partial ending in punctuation is
   not evidence that a sentence ended, and a partial that looks stable may be a
   hallucinated completion.

Closing the recogniser early and reopening it on resume fails the same way: the
join is ungrammatical (`"What did Sanket Built during the ARAI internship."`)
because the provider starts a new segment with a capital, and a finalize on
trailing silence produced a spurious `"I"`. The 800 ms stays.

**The speculative recogniser is blocked by the provider plan.** The remaining
safe shape was a second recogniser that never touches the live turn: feed it the
same audio in parallel, close it at ~500 ms of silence, and use its final when
the window expires. `bench/probe/asr_speculation.py` measured two things:

1. It does not fit. The speculative finalize took 274 ms and 676 ms in
   consecutive runs, against a ~300 ms budget, so it would sometimes arrive too
   late to be used — the worst of both worlds, paying for a second recogniser
   and still waiting.
2. It cannot be opened at all. A second concurrent realtime ASR socket is
   refused with `rate_limit_exceeded`, and the refusal sometimes takes the live
   socket down with it; a single socket connects immediately afterwards. This
   account's plan supports one realtime ASR connection, so any design that
   doubles them — parallel feed or burst replay — is unavailable, and would also
   multiply with session count in production.

The REST endpoint is a different pool, so it is not blocked by that concurrency
limit — `bench/probe/asr_realtime.py` completes a `/speech-to-text` call while the
realtime socket is still being refused. It is still unusable, for two
independent reasons measured in the same run:

* **Latency.** 1 335 ms for a 4.16 s clip, against the ~300 ms left in the
  window after closing at 500 ms of silence. The window would have to grow, which
  defeats the purpose.
* **Accuracy.** The REST path has no keyterm or prompt biasing, so the same
  audio transcribed as `"…during the ARII internship?"` where the realtime
  socket, given the keyterms, produced `"ARAI"`. Trading a wrong proper noun for
  latency is exactly what this ADR exists to prevent.

The one single-socket lever was also measured and is not one.
`bench/probe/asr_final_tail.py` compares the adapter's chunking: 20 ms chunks
give 297 ms p50 from `speech_end` to final against 221 ms at 50 ms, because the
smaller chunk sends five times as many messages without reducing what the
provider has to process. Production already runs 100 ms at ~175 ms p50, so the
differences are provider noise and the tail is not ours to shorten.

Endpointing and ASR finalization are therefore both closed: 800 ms because the
measured pause floor demands it, and `asr_final` because every way to overlap it
is refused by the plan, slower than waiting, or less accurate.


## 2. One recogniser socket per connection, not one per utterance

**Decision.** The ASR WebSocket is opened when the transport opens
(`UtteranceLoop.warm()`) and reused for every turn on that connection. Turns are
delimited with `speech_start` / `speech_end`.

**Why.** Opening a socket on speech start spends a 200–400 ms handshake at the
exact moment the visitor begins talking, and forces a second buffer to guess
what was missed. A warm socket also means the provider's `session.begin` has
already been validated before audio arrives.

`session.begin` echoes the resolved `sample_rate`, `encoding` and
`turn_detection`. We assert them: "we asked for 16 kHz" and "the server is
decoding 16 kHz" are different claims, and only the second one keeps audio
intelligible.

## 3. The answer is built from `transcript.final` only

**Decision.** `transcript.partial` drives the live transcript. The turn engine
runs on `transcript.final`. A partial is used only after
`ASR_FINAL_TIMEOUT_S` (3.5 s) and the fallback is logged.

**Why.** Partials are unstable by construction: on the same fixture, one partial
came back in Devanagari for an English session while the final was correct
English. Answering from a partial is answering a sentence the visitor did not
finish saying. `flush` was also removed: `speech_end` already finalises, and
sending both could truncate the final.

Partial events whose script does not match the session language are dropped
before they reach the UI — a stray partial only ever costs credibility on
screen.

## 4. Resample once, on the server, with a stateful filter

**Decision.** The browser captures at the device's native rate and sends that
rate as `sr`. The gateway resamples once per connection with `StreamResampler`
(soxr HQ). The browser-side `LinearResampler` is gone from the microphone path.

**Why.** The old path decimated 48 kHz → 16 kHz with no anti-alias filter, so
sibilants folded back into the formant band; the recogniser heard noise where
the consonants were. It also duplicated one sample at every block boundary. A
stateful resampler is required because per-chunk resampling discards filter
state at each boundary — fifty clicks a second.

## 5. Keyterms are rationed; the prompt is not

**Decision.** At most **12** keyterms: three priority acronyms (`ARAI`, `AEUE`,
`ZEISS INSPECT`) plus project names. Spelled-out sequences (`"A E U E"`) and any
term containing a comma or semicolon are excluded. The domain `prompt` is always
sent.

**Why.** Measured against the live endpoint on the fixture clip "what did Sanket
build during the ARAI internship":

| keyterms sent | transcript |
|---|---|
| none | "ARII" (3/3) |
| 1–3 acronyms (+ projects) | "ARAI" (6/6) |
| 7 acronyms | "ARII" (2/2) |
| 20 terms, acronym-heavy | "AAI" (4/4) |

Keyterms help only while the acronym set stays small: past a handful the
candidates compete and the decoder lands on a *different* plausible acronym.
Project names are free because they are not confusable with each other, so only
the acronyms are rationed. A comma in one term is worse still — the provider
rejects the whole session with a fatal `invalid_request`.

## 6. Known mis-hearings are repaired in the transcript, deterministically

**Decision.** `lexicon.misheard` in `kb/fact.yaml` maps observed variants
(`ARII`, `AAI`, `AII`, `ARI`, `A R A I`) to the canonical spelling, applied to
every transcript by `asr/normalize.py`.

**Why.** Biasing is probabilistic; a visitor reading "the AAI internship"
concludes the system is broken even when the answerer resolved the entity
correctly. The table is small and curated, not a spell checker: rewriting words
that are not in the table is worse than a visible mis-transcription.

## 7. Cartesia contexts are bounded to the account's concurrency

**Decision.** `TTS_MAX_CONCURRENCY` (default 2) caps concurrent Cartesia
contexts.

**Why.** The orchestrator submits every answer segment at once, so a
three-sentence answer opened three contexts against an account limit of two and
silently fell back to the synthetic voice mid-answer. Synthesis runs ~11x faster
than playback, so the bound costs no latency: the next segment is ready long
before it is heard.

**Amended 2026-11-10.** The bound was a workaround for Cartesia's two-context
free tier, not a property worth keeping. `bom-india` now uses Smallest Lightning,
which allows 10–15 concurrent streams, so `SMALLEST_MAX_CONCURRENCY` defaults to 4
and every segment of an answer can be in flight at once (ADR 0004). The Cartesia
adapter still honours `TTS_MAX_CONCURRENCY`.

## 8. Measured result

Through the real gateway over `/ws/audio`, 2 turns on the fixture clip:

| metric | value |
|---|---|
| transcript | correct (2/2) |
| ASR finalise (p50) | 592 ms |
| LLM TTFT (p50) | 418 ms |
| TTS TTFA (p50) | 93 ms |
| EOS → first answer audio (p50) | 1,123 ms |

ASR finalisation is now the largest single stage, which is the expected
consequence of waiting for a real final instead of accepting a partial.

## 9. Not done, deliberately

* **Silero VAD.** The energy VAD is adequate once endpointing is generous, and
  its detector is now measured rather than assumed (§10). A learned VAD would
  let the silence window drop back towards 500 ms without clipping, and is the
  next quality step.
* **REST hybrid finalizer.** Measured and rejected: 1 335 ms for a 4.16 s clip,
  and no keyterm biasing, so it transcribed `ARAI` as `ARII` where the realtime
  socket got it right. It runs on a separate concurrency pool, so the realtime
  limit is not what rules it out — the latency and the accuracy are. See §1.
* **TURN for WebRTC.** With STUN only, Fly gives private `fdaa:` candidates and
  ICE fails, so `ENABLE_WEBRTC=false` in `fly.toml` until a TURN server exists.
  WebRTC's Opus and jitter buffer are worth having, but not at the cost of every
  visitor paying a failed ICE before the WebSocket fallback connects.

## 10. What counts as speech at all

**Decision.** The energy detector's absolute floor is **−55 dBFS** (was −42),
and a frame is speech only if it is *both* above the threshold **and** not
steady: level spread over a rolling **300 ms** window must be at least
**1.5 dB**. The zero-crossing hum guard stays as it was.

**Why the floor moved (measured).** `bench/probe/vad_accuracy.py` measures the
reference clip against ground truth it cannot see. Real speech sits at RMS
p05/p50/p95 = −47/−27/−20 dBFS — **11% of speech frames were below the old
−42 floor**, and a speaker scaled to −50 dBFS was not detected at all. The
floor only binds in a *quiet* room (`noise_floor + 9 dB` is higher in a noisy
one), so lowering it costs nothing where it used to work and buys hearing a
quiet speaker.

**Why the steadiness gate is required (the worst failure found so far).**
Steady ambient sound at or above the floor — a fan at −35 dBFS, mains hum —
passed the level test after ~120 ms, because `min_speech_ms` commits before the
adaptive noise floor (780 ms to climb) ever learns the room. Once active, every
following frame was also "speech", so **the turn never endpointed**: the
utterance loop captured forever. The gate kills the whole class because speech
is amplitude-modulated by syllables and steady sound is not:

| signal | level spread over 300 ms | zero-crossing rate |
|---|---|---|
| speech | p05 **3.1 dB** | 0.000 – 0.389 |
| 50 Hz hum −35 | **0.0 dB** | 0.003 |
| white noise −35 | **0.2 – 0.4 dB** | 0.455 |

The separation is wide (10× either side of the 1.5 dB gate), and a zero-crossing
test can never catch the noise case at all — noise crosses zeros like speech
does. The gate also makes the lower floor safe: room hiss at −50 dBFS is
level-speech and steady-not-speech.

**Evidence of effect.** After both changes the probe reports **0 problem
scenarios** out of 27, where the shipped configuration had 5: quiet speech
(−50 dBFS) is detected (start delay 160 ms), all ten steady hum/noise-only
streams are rejected, speech endpoints *through* continuous −45/−40 dBFS
room noise instead of hanging, endpoint placement is unchanged (745–785 ms =
the configured window), and speech starting at frame 0 opens in 140 ms. The
gate only ever delays a *transition*, never steady sound: at a normal onset the
silence-to-speech jump spikes the window wide open.

**Trade-off.** After an abrupt level jump in a steady environment (a door
slam next to a running fan), the transition window stays open for up to 300 ms,
so ≥120 ms of loud ambient right after a transient can still commit
`min_speech_ms`. The turn then endpointes within the silence window instead of
hanging, and the transcript comes back empty or noise — bounded, not silent.
This is detection, not recognition: the WER gate (FR-025) is still unscored
on real recordings, so nothing here claims anything about transcript accuracy.

## 11. Stream type and chunk size are not accuracy levers

**Decision.** Keep `stream_type=fast` and `chunk_ms=100` in production. Both
were suspects — the stream type because it is a latency knob with no accuracy
measurement behind it, the chunk size because only finalise latency had ever
been measured for it.

**Measured.** `bench/probe/asr_matrix.py` sweeps
`{fast, balanced, simulated} x {20, 50, 100} ms` on the reference clip and
scores WER against the known transcript. The API corrected the axis:
`accurate` does not exist — the values are `fast`, `balanced`, `simulated`.

| finding | evidence |
|---|---|
| WER is **0%** on all 9 cells | every stream type × chunk size transcribes the clip word-perfect |
| chunk size changes nothing | first-partial and finalise vary more *between runs of the same cell* (656–1 443 ms) than between chunk sizes |
| `simulated` emits **zero partials** | no live transcript at all; finalise 155–256 ms, i.e. faster than `fast`, but the drawer and partial UX need partials |
| first partial lands 0.65–2.0 s into speech | provider-side: partials start once enough words accumulate; not a client lever |

**Consequences.** The production config is validated, not merely assumed.
"ASR is not performing well" is **not** raw transcription on clean audio —
remaining suspects are the ones no fixture covers yet: real microphone noise,
accents, and endpoint truncation (§1), and the answer layer `bench/eval.py`
already grades. The sweep also produces the first WER numbers this project has
ever had (FR-025 was open with no scorer); the caveat is fixture count: one
10-word sentence moves WER in 10% steps, so here anything non-zero is a failure
rather than a rate. Recorded reference clips (see `bench/audio/README.md`)
are what turn this into a rate gate.

**Not done:** `simulated` is not worth its missing partials for ~100–200 ms of
finalise — it would trade every visitor's live transcript for a gain smaller
than run-to-run variance.

