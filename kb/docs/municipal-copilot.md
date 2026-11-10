# Municipal Asset Intelligence Copilot

An agentic copilot built during the EPIC Technology Research internship. It
answers multi-step questions over millions of municipal asset records: what
exists, where it is, what condition it is in, and what should happen next.

## The shape of the problem

Municipal asset data is broad and uneven. A single question can require several
steps — find the asset class, filter by ward, join inspection history, then
reason over the result. A dense-only retriever does badly here because the
question is often about structured constraints, not semantic similarity. A pure
SQL agent does badly because the question arrives in natural language with
implicit scope.

## Approach

The copilot routes before it retrieves. An intent-detection step decides what
kind of question is being asked and which path it should take: a direct record
lookup, a structured query over PostgreSQL, a lexical search, or a vector
search. Routing first keeps the common question cheap and keeps the model from
improvising a query shape it has not seen.

Retrieval is hybrid. Lexical search handles identifiers, names and exact
municipal vocabulary; pgvector handles paraphrases and fuzzy descriptions. The
two result sets are combined before the model reasons over them, so a question
that mixes a proper noun with a description still lands.

Model orchestration runs on-premises. That was a data-residency constraint, not
a preference: municipal records cannot be shipped to an arbitrary third-party
endpoint. The orchestration layer therefore had to treat the model as a local
resource with its own queueing and failure behaviour.

## Stack

Python, PostgreSQL with pgvector, hybrid lexical-plus-vector retrieval,
intent-aware query routing and on-premises LLM orchestration.

## Limitations

There are no published accuracy or latency figures. The scale claim is
qualitative: the system was built to reason across millions of records, but the
portfolio does not document a benchmark number for retrieval quality. That
honesty matters more than a tidy metric.

## What transfers

The routing idea is the reusable part. A retrieval system that asks "what kind
of question is this?" before it embeds the query avoids a large class of
confident wrong answers, and it keeps structured data in the query engine where
it belongs.
