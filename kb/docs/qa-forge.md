# QA-Forge

An AI-native multi-agent auditing platform. QA-Forge points a team of
specialised agents at an API and asks them to find what is broken before a user
does.

## Structure

An orchestrator coordinates specialised agents rather than letting one model do
everything. Each agent has a narrow job: explore the API surface, generate
requests, validate responses, attempt adversarial input, or write up findings.
The orchestrator owns task ordering and decides when a finding is worth
escalating.

That structure is not decoration. A single agent asked to test an API tends to
invent plausible endpoints and then validate its own invention. Splitting
exploration from validation makes the second agent a real check on the first.

## Adversarial testing

Adversarial API testing is a first-class task, not a prompt variant. The
platform generates malformed payloads, boundary values, missing fields and
unexpected types, then compares the response against the contract. The point is
to find the cases where the API returns a success-shaped response to a request
it did not actually satisfy.

## Reporting

Findings are compiled into an automated engineering report: what was attempted,
what failed, the request that triggered it, and a minimal reproduction. A report
that a developer can act on is the deliverable — a list of agent impressions is
not.

## Stack

LangGraph for the agent graph and FastAPI for the service surface.

## Limitations

Agent roles are kept deliberately generic in public descriptions. Only roles
that are backed by the implementation should be named, so the portfolio does not
claim a specialist agent that does not exist. The project is a platform
demonstration, not a hosted product with published coverage numbers.

## The interesting problem

Multi-agent systems fail in coordination, not in intelligence. Two agents that
both think they own a task will duplicate work or contradict each other. QA-Forge
treats the orchestrator as the authority on who is doing what, which is the same
lesson that shows up in commerce: decide where authority lives, then enforce it.
