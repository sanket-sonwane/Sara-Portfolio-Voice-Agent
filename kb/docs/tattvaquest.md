# TattvaQuest

An autonomous knowledge-retrieval agent for multi-document question answering.
The project is about the unglamorous part of RAG: what happens before the
embedding and after the first retrieval.

## Chunking

Chunking is heading-aware. The documents are structured with headings, and those
headings carry meaning that a fixed character window destroys. Splitting on
structure first — then packing sections up to a target size — keeps a chunk
about one thing. That directly improves retrieval, because a chunk that mixes
two topics matches neither cleanly.

## Retrieval

Retrieval is hybrid rather than dense-only. Embeddings are good at paraphrase
and bad at exact identifiers, version strings and rare proper nouns. Lexical
matching is the reverse. Running both and combining the ranked lists covers the
questions a single retriever drops.

A query router decides which documents and which retrieval mode a question
should hit. That avoids embedding every query against every document when the
question clearly names one source.

## Tool use and reasoning

The agent uses tool calling to fetch evidence and to decide whether it has
enough to answer. Contextual reasoning then resolves follow-ups: a pronoun or a
terse second question is expanded against the conversation rather than treated
as a fresh, underspecified query. That is the difference between a search box
and an agent.

## Stack

LlamaIndex for the document and retrieval abstractions, embeddings for dense
search, hybrid retrieval, query routing and tool calling.

## Limitations

Retrieval-quality work on this project is described qualitatively. There are no
published retrieval scores, no ablation table and no benchmark numbers in the
portfolio. The claims here are about mechanism, not measured lift.

## What was learned

Most RAG failures are not embedding failures. They are chunking failures,
routing failures and follow-up failures. Fixing those three is cheaper and more
effective than swapping to a larger embedding model.
