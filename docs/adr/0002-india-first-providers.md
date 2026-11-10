# ADR 0002 — India-first provider profile; Groq leaves the product path

**Date:** 2026-11-09
**Status:** accepted; §1 and §3 amended 2026-11-10 (§6)
**Supersedes:** the implicit Groq/Cartesia/NVIDIA default in ADR 0001 §3 and §6.
**Context:** the audience is Indian. PRD §17 sets a sub-500 ms target that counts
both network legs, and ADR 0001 §6 requires the region to be chosen from
measurements rather than assumed.

---

## 1. The default profile is `bom-india`

**Decision.** The default deployment is:

| Stage | Provider | Region |
|---|---|---|
| Media gateway | Fly.io machine | `bom` (Mumbai) |
| TURN | coturn | `bom` |
| ASR | Sarvam Saarika (`saarika:v2.5`) | India |
| LLM | Gemini Flash-Lite (`gemini-2.5-flash-lite`) | `asia-south1` |
| TTS | Sarvam Bulbul (`bulbul:v4-flash`) | India |
| Tier-2 retrieval | pgvector / tsvector | `bom` |

Any individual provider can still be pinned with `ASR_PROVIDER`, `LLM_PROVIDER`
or `TTS_PROVIDER`, so this is a default, not a cage.

**Why.** A visitor in India should not pay a US round trip twice per turn. Keeping
audio and inference in-region removes roughly 180–250 ms from the end-to-end
budget before any model work happens, and it keeps voice data in the country,
which is the safer default for an Indian audience.

## 2. Groq is demoted to a comparison arm

**Decision.** `GROQ_API_KEY` remains supported and the `iad-global` profile still
exists, but neither is the product path. Groq is used only by
`bench/probe/llm_matrix.py --providers groq` and by the region matrix.

**Why.** Groq has no Asia data centre (its regions are US, Canada, Australia,
Finland and Saudi Arabia at the time of writing). A gateway in `bom` talking to
Groq pays a full transatlantic or transpacific leg per token stream. That is
incompatible with the target, and no amount of prompt tuning fixes a network
distance.

**Consequence.** The US stack is still measurable, so the decision can be
revisited with data if Groq opens an Indian region. Until then, the default
answer path never depends on it.

## 3. Cartesia and NVIDIA are also comparison arms

**Decision.** Cartesia Sonic and NVIDIA Nemotron stay implemented and probeable,
but the default profile uses Sarvam for both ASR and TTS.

**Why.** The same region argument applies. Cartesia's Indian presence was not
guaranteed on the available plan, and NVIDIA's ASR endpoint is US-hosted. Both
remain useful as quality/latency baselines.

## 4. Region is still decided by measurement

**Decision.** `bom` is the provisional region. It becomes final when the Fly
matrix in `docs/phase0-report.md` is filled by running `regions.py --full` in
`bom`, `sin`, `fra` and `iad`.

**Why.** ADR 0001 §6 is not weakened by this ADR: a nearby provider region is an
argument, not evidence. The matrix must still show that `bom` wins end-to-end for
an Indian client.

**Revisit when.** The Fly matrix shows a different region wins, a provider opens
an Indian region, or the measured India-client figure exceeds 500 ms even with
the in-region stack. In the last case, speculative tier-1/tier-2 prefetch
(PRD §17.6) is promoted from optional to required, and LLM speculation remains
forbidden because its adoption rate is unproven.

## 5. What this does not change

- One LLM call per turn; retrieval races it.
- Short system prompt plus `kb/fact.yaml` lookup; no portfolio digest in the prompt.
- Generation-scoped cancellation and split-authority barge-in.
- The `/ws/audio` PCM fallback, which keeps the product working where WebRTC or
  TURN is unavailable.

---

## 6. Amendment (2026-11-10) — measured stack replaces the provisional one

**Status:** accepted. This amends §1 and §3; §2 and §4 stand.

The provisional table in §1 was chosen on region arguments before keys existed.
With keys and measurements (`docs/phase0-report.md` §3–§4), three of its rows are
now different:

| Stage | §1 (provisional) | Now (measured) | Why |
|---|---|---|---|
| ASR | Sarvam `saarika:v2.5`, REST | Sarvam **`saaras:v4`**, realtime WS, `endpointing=manual` | streaming partials and manual EOS; 224 ms finalization |
| LLM | Gemini `gemini-2.5-flash-lite` | Sarvam **`sarvam-105b-conversations`** | 206 ms TTFT p50 vs Gemini's 980 ms; also in-region |
| TTS | Sarvam `bulbul:v4-flash` | **Cartesia `sonic-latest`**, voice `2a12b36c-7f9b-4c3a-9f7a-72731b15323a` | socket streaming, 148 ms TTFA p50, 65 ms cancel-to-silence |

**What survives from §3.** Cartesia is not "a comparison arm only" any more; the
region argument was outweighed by the measurement, because TTS is the stage where
cancellation latency is user-visible. The trade-off is accent: the pinned voice is
en-AU, not Indian English. That is a product judgement, recorded here rather than
hidden, and it is one env var (`CARTESIA_VOICE_ID`) away from being reversed.

**What does not change.** Sarvam stays the default for ASR and LLM, so the
in-region argument in §1 still holds for the two stages that carry the most data.
Groq remains demoted (§2) and NVIDIA remains a comparison arm. §4 still governs
the region: `bom` is provisional until the Fly matrix is filled.

**Revisit when.** An en-IN Cartesia voice measures within ~20 ms of the current
one, Sarvam ships a streaming Bulbul, or the Fly matrix shows a non-`bom` region
wins for an Indian client.

---

## 7. Amendment (2026-11-10) — Smallest Lightning replaces Cartesia

**Status:** accepted. This amends §3 and the TTS row of §6. ADR 0004 carries the
measurements.

Cartesia's free tier allows **two** concurrent contexts, and the turn engine submits
one request per answer segment. The third segment was therefore dropped mid-answer,
which is a structural ceiling, not a tuning problem. Smallest Lightning speaks every
segment, costs 2.6× less per character, and offers an en-IN voice (`rhea`,
`language: "hi"`). It is ~30–60 ms slower to first audio from `sin` (216 ms vs
186 ms), which is smaller than the turn's own run-to-run variance.

The region argument in §3 is now settled the other way for TTS: the accent and the
concurrency ceiling both point at an Indian provider, so `bom-india` uses Smallest.
Cartesia is retained for `iad-global` and as a one-env-var comparison arm.

**What does not change.** §1's in-region default, §2's demotion of Groq, and §4's
rule that the region is decided by measurement.
