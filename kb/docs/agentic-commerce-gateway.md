# Agentic Commerce Gateway

An agentic-commerce infrastructure layer that makes merchants transactable by
unknown AI buyers. The project asks a concrete question: what has to be true
before an autonomous agent is allowed to spend a user's money?

## Problem

Agents can already browse, compare and decide. Commerce infrastructure still
assumes a human is holding the authority to spend, clicking the button and
reading the confirmation. Handing an agent a payment credential is the obvious
wrong answer: it gives the model both the intent and the money.

## Mechanism

The gateway separates three things that commerce normally fuses together.

First, pricing is server-authoritative. The agent can request a product, but the
totals are computed by the server from its own price table, so a model cannot
negotiate its own discount or rewrite a line item.

Second, spend approval is an explicit, signed user intent. The approval is an
Ed25519-signed statement covering what is being bought and for how much. The
agent relays it; the gateway verifies it. A model that invents an approval
produces a signature that does not verify.

Third, order finalisation is idempotent and reservation-backed. A retried
request, a duplicated tool call or a crash between payment and fulfilment does
not create a second charge or a second shipment.

## Provenance

Every state change is written to a SHA-256 hash-chained audit trail. Each entry
commits to the previous one, so a record cannot be silently edited after the
fact. The result is transaction provenance that a third party can verify rather
than trust.

## Stack

Python and FastAPI for the gateway, Razorpay in test mode for payment
integration, SQLite for the local store, Ed25519 for signatures, and React with
Vite for the demo client. Tool calling against Groq was used in the agent loop.

## Limitations

Payments ran in Razorpay test mode. No production money movement is claimed and
there is no revenue figure to report. The flow diagram on the portfolio is
illustrative rather than production data. Protocol interoperability and failure
recovery are implemented in the demo, not operated at merchant scale.

## Why it matters

The interesting engineering is not the checkout. It is where authority lives.
The model may propose; the server prices; the user signs; the ledger proves. That
split is what makes an agent safe to point at a merchant.
