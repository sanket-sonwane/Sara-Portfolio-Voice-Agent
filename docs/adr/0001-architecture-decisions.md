# ADR 0001 — Architecture decisions locked before implementation

**Date:** 2026-11-09
**Status:** accepted
**Context:** `prd.md` defines a production-oriented cascade voice agent. Several
implementation-shaping decisions are not resolved by the PRD and were settled in
review. This ADR records them so they are not re-litigated later.

---

## 1. Single FastAPI process with an in-process event bus

**Decision.** One deployable backend. Stages communicate through typed asyncio
events (`event_type`, `session_id`, `turn_id`, `generation_id`, monotonic
timestamps). No Redis, Kafka, Kubernetes or microservices in v1.

**Why.** PRD §20.3 forbids premature distribution. Session isolation and
generation-scoped cancellation are easier to prove in a single process, and the
event bus gives us the same seams a message broker would.

**Revisit when.** Concurrent-session measurements show CPU-bound interference, or
media processing needs independent scaling (PRD §26.3 stages 3–5).

---

## 2. Cancellation is split-authority

**Decision.** The browser stops audio locally and immediately (gain to zero, flush
scheduled sources). The server owns conversation state and bumps the generation
id. Neither waits for the other.

**Why.** Waiting for a server round trip before silencing puts 150–250 ms between
a detected interruption and actual silence, which fails the barge-in requirement.

**Consequence.** The browser must tolerate a state that is briefly ahead of the
server, and reconciling those states on reconnect is an explicit Sprint 4 task.

---

## 3. One LLM call per turn, retrieval races it

**Decision.** On `transcript.final`, run the fact lookup and the Groq call
concurrently. Tools remain available to the model. Routing (which model tier,
whether retrieval is mandatory) is decided by cheap local rules, never by an LLM
call.

**Why.** A separate classification call costs a full TTFT — 200–300 ms of a
500 ms budget. Retrieval at ~10 ms finishes long before the first token.

**Rejected.** Two-call routing as drawn in PRD §13.2, and a large always-in-prompt
portfolio digest: for a knowledge base this small, an in-process lookup is cheaper
than the prompt tokens it would replace.

---

## 4. Knowledge is fact-first; RAG is the exception

**Decision.** Three tiers:

| Tier | Content | Mechanism | Budget |
|---|---|---|---|
| 0 | Short system prompt | identity, style, grounding rules, tool schemas | 150–200 tokens |
| 1 | `kb/fact.yaml` | in-process hybrid match, injected top 1–3 blocks | ≤ 10 ms |
| 2 | case studies, READMEs, paper | pgvector + tsvector behind tools | slow path only |

**Why.** The corpus is small and curated, so tier-1 lookup is faster and cheaper
than prompt bloat, and it keeps every answer attributable. Tier 2 is reserved for
genuine depth questions, where a short spoken acknowledgement is acceptable.

**Consequence.** `kb/fact.yaml` becomes release-critical and needs validation in
CI (`kb/validate.py`).

---

## 5. Acknowledgements are pre-rendered audio

**Decision.** "Give me a second" style fillers are WAV assets played by the egress
shaper, never synthesized on demand. They fire only on the tier-2 path.

**Why.** Synthesizing the filler costs the same 70–100 ms it is meant to hide.
Firing it routinely makes the agent feel slower than silence.

---

## 6. Media gateway is co-located with providers, and region is measured

**Decision.** Deploy the backend to the region that minimises end-to-end latency,
measured by `bench/probe/regions.py` running inside candidate Fly regions. Media
egress is direct where possible, TURN-relayed where ICE requires it.

**Why.** The 500 ms target counts both network legs. A laptop measurement cannot
answer this.

**Consequence.** Sub-500 ms will be reported with declared conditions (PRD §17.4),
including a co-located client figure and an India-client figure. If the India
figure exceeds target, speculative generation (PRD §17.6) is promoted from
optional optimisation to a required stage.

---

## 7. The UI is the PRD interaction surface, not a portfolio

**Decision.** Ship the complete interaction surface — twelve states, partial vs
final transcript, input/output visualisation, accessibility, diagnostics overlay —
as a standalone app (`apps/web`). Do not build portfolio content pages. The
existing portfolio site keeps its own text assistant; Sarang sits alongside it and
must never contradict it.

**Why.** The PRD's state machine and diagnostics are deliverables, not decoration.
Keeping Sarang as a self-contained surface makes independent testing possible now
and integration a packaging task later.

**Consequence.** Transcript and evidence display must be dense enough to be useful
without becoming a second chat product.
