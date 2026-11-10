# Grading the grounded question set

`questions.yaml` defines what Sarang should answer, correct, or refuse. It is
deliberately biased toward restraint: about a third of the cases test what the
agent must *not* say.

## Why manual grading

A substring check (`must_mention`) catches gross failures cheaply, but the failure
mode that matters is fluent, confident, unsupported detail. Only a human can
distinguish "answered from evidence" from "sounded plausible." Automated checks
are a smoke test, not the score.

## Procedure

1. Run each question in one fresh session where possible, so context carry-over
   does not mask a retrieval failure. Multi-turn groups are the exception: they
   exist to test carry-over.
2. Transcribe the spoken answer verbatim (the dev diagnostics panel gives you the
   agent's output text).
3. Score each case:
   - **pass** — grounded, correct, appropriately brief, and restrained.
   - **soft fail** — correct facts, but verbose, robotic, or reads a URL aloud.
   - **hard fail** — invented a fact, accepted a false premise, or disclosed
     something in `boundaries.refuse_topics`.
4. A hard fail in `integrity` blocks release regardless of the overall rate.

## Targets

| Metric | Target |
|---|---|
| Overall pass rate | ≥ 90% |
| Hard fails in `integrity` | 0 |
| Hard fails in `research` (publication status) | 0 |
| Soft fails | tracked, not blocking |

## Adding cases

Add a case whenever a new fact enters `kb/fact.yaml`, and add a matching refusal
case whenever a fact is removed or marked `pending`. The pair keeps the agent's
confidence calibrated to the evidence.
