"""Tier-2 retrieval over the long-form evidence corpus."""

from __future__ import annotations

from sarang.kb.retrieval import LocalRetriever, chunk_markdown


def test_corpus_is_indexed():
    retriever = LocalRetriever()
    assert retriever.ready
    assert len(retriever.chunks) >= 20


def test_heading_aware_chunking_labels_sections():
    chunks = chunk_markdown(
        "# Doc Title\n\nIntro line.\n\n## Mechanism\n\nA body paragraph.\n\n## Limits\n\nShort.",
        "sample",
    )
    titles = [chunk.title for chunk in chunks]
    assert "Doc Title — Mechanism" in titles
    assert "Doc Title — Limits" in titles
    assert all(chunk.text for chunk in chunks)


def test_signed_spend_approval_finds_the_commerce_case_study():
    results = LocalRetriever().search("signed spend approval and Ed25519", 3)
    assert results
    assert results[0].doc_id == "agentic-commerce-gateway"


def test_heading_aware_chunking_finds_tattvaquest():
    results = LocalRetriever().search("heading aware chunking hybrid retrieval", 3)
    assert results
    assert results[0].doc_id == "tattvaquest"


def test_simulation_limits_find_the_paper():
    results = LocalRetriever().search("rogue node detection paper AEUE", 3)
    assert results
    assert results[0].doc_id == "research-rogue-nodes"


def test_empty_query_returns_nothing():
    assert LocalRetriever().search("") == []


def test_scores_are_sorted_descending():
    results = LocalRetriever().search("multi agent orchestrator adversarial API testing", 4)
    scores = [chunk.score for chunk in results]
    assert scores == sorted(scores, reverse=True)
