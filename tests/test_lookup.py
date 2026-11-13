"""Tier-1 lookup: routing, confidence and the refusal path."""

from __future__ import annotations

from sarang.kb.lookup import is_followup, lookup, strip_discourse, tokenize


def test_cgpa_routes_to_education():
    result = lookup("What is Sanket's CGPA?")
    assert result.top is not None
    assert result.top.kind == "education"
    assert result.confident


def test_named_project_routes_to_that_project():
    result = lookup("Tell me about TattvaQuest")
    assert result.top is not None
    assert result.top.id == "project:tattvaquest"
    assert result.confident


def test_named_internship_routes_to_experience():
    result = lookup("What did he do at ARAI?")
    assert result.top is not None
    assert result.top.kind == "experience"
    assert "arai" in result.top.id


def test_owner_name_does_not_outrank_the_topic_of_the_question():
    result = lookup("What did Sanket build during the ARAI internship?")
    assert result.top is not None
    assert result.top.id == "experience:arai"
    assert result.confident


def test_owner_name_only_generic_project_question_returns_the_overview():
    result = lookup("What did Sanket build?")
    assert result.reason == "generic project overview"
    assert result.top is not None
    assert result.top.kind == "project"


def test_salary_is_blocked_by_boundaries():
    result = lookup("What is his salary expectation?")
    assert result.blocked
    assert result.top is not None
    assert result.top.kind == "boundaries"


def test_phone_number_is_blocked_by_boundaries():
    result = lookup("Can I get his phone number?")
    assert result.blocked
    assert result.top.kind == "boundaries"


def test_pending_project_is_answered_with_pending_block():
    result = lookup("Tell me about the CPU scheduling project")
    assert result.top is not None
    assert result.top.kind == "pending"


def test_deep_question_without_a_named_block_needs_retrieval():
    result = lookup("How did he handle the internal tradeoffs and why?")
    assert result.needs_depth
    assert not result.confident


def test_named_block_does_not_trigger_retrieval():
    result = lookup("Explain the Agentic Commerce Gateway")
    assert result.top is not None
    assert result.top.id == "project:agentic-commerce-gateway"
    assert not result.needs_depth


def test_unrelated_question_is_not_confident_and_offers_boundaries():
    result = lookup("What is the airspeed velocity of an unladen swallow?")
    assert not result.confident
    assert result.blocks
    assert result.top.kind == "boundaries"


def test_tokenize_strips_stopwords_and_plurals():
    assert tokenize("What are the projects?") == ["project"]


def test_generic_project_question_returns_a_confident_overview():
    result = lookup("What do you build?")
    assert result.confident
    assert result.reason == "generic project overview"
    assert any(block.kind == "project" for block in result.blocks)
    assert len(result.blocks) > 1


def test_current_work_routes_to_the_owner_block():
    result = lookup("What are you currently working on?")
    assert result.confident
    assert result.top is not None
    assert result.top.kind == "owner"


def test_payload_is_json_safe():
    payload = lookup("Tell me about QA-Forge").to_payload()
    assert payload["ids"]
    assert isinstance(payload["scores"], dict)
    assert all(isinstance(value, float) for value in payload["scores"].values())


# ---------------------------------------------------------------- discourse
# Regressions for the measured filler damage: "however i mean" turned the
# portfolio overview into profile + three arbitrary blocks, because discourse
# markers survived tokenization as topical keywords (df = 0 in the corpus, so
# they could never help a match - only poison the topical set).


def test_discourse_markers_are_not_keywords():
    tokens = tokenize("However, I mean, what did he build during the ARAI internship")
    assert "however" not in tokens
    assert "mean" not in tokens
    assert "arai" in tokens


def test_filler_words_do_not_change_routing():
    clean = lookup("what projects did he build")
    noisy = lookup("however i mean what projects did he build")

    assert clean.reason == "generic project overview"
    assert noisy.reason == clean.reason
    assert [block.id for block in noisy.blocks] == [block.id for block in clean.blocks]

    clean = lookup("what has he built")
    noisy = lookup("i mean like what has he built")
    assert noisy.reason == clean.reason
    assert [block.id for block in noisy.blocks] == [block.id for block in clean.blocks]


def test_strip_discourse_removes_only_the_wrapping():
    assert strip_discourse("well, i mean, however what projects did he build") == (
        "what projects did he build"
    )
    assert strip_discourse("what did he build, you know, during the arai internship") == (
        "what did he build during the arai internship"
    )
    assert strip_discourse("tell me about his salary") == "tell me about his salary"
    # A marker may never eat into a content word, and a marker-only utterance
    # may never become an empty query.
    assert strip_discourse("solomon says hi") == "solomon says hi"
    assert strip_discourse("however") == "however"


def test_intent_survives_the_strip():
    """The strip runs before routing; intent patterns match on content words."""
    assert lookup(strip_discourse("well however what is his salary")).blocked
    assert lookup(strip_discourse("i mean like what did he do at arai")).top is not None


# ---------------------------------------------------------------- follow-ups
# A continuation is only a continuation when the question has no subject of its
# own. These cases come from the transcript that prompted the split: pronouns
# and "the numbers" refer back, "what about QA-Forge" does not, and "this
# system" must not be lost just because the router scored "system" somewhere.


def test_a_pronoun_continuation_carries_the_previous_evidence():
    previous = lookup("What did he do at ARAI?")
    current = lookup("How does this system work?")
    assert is_followup("How does this system work?", previous, current)


def test_a_bare_number_question_refers_back():
    previous = lookup("What did he do at ARAI?")
    current = lookup("The numbers?")
    assert is_followup("The numbers?", previous, current)


def test_a_subject_inside_a_follow_up_shape_keeps_its_own_evidence():
    """"what about" introduces a subject; the router already found that one."""
    previous = lookup("What did he do at ARAI?")
    current = lookup("What about QA-Forge?")
    assert current.confident and current.top is not None
    assert current.top.id == "project:qa-forge"
    assert not is_followup("What about QA-Forge?", previous, current)


def test_a_follow_up_phrase_about_the_same_block_still_merges():
    previous = lookup("What did he do at ARAI?")
    current = lookup("What about this at ARAI?")
    assert is_followup("What about this at ARAI?", previous, current)


def test_tell_me_more_is_a_follow_up_even_when_the_router_scores_something():
    previous = lookup("What did he do at ARAI?")
    current = lookup("Tell me more about that internship.")
    assert is_followup("Tell me more about that internship.", previous, current)


def test_a_named_new_question_is_not_a_follow_up():
    previous = lookup("What did he do at ARAI?")
    assert not is_followup("Tell me about TattvaQuest", previous, lookup("Tell me about TattvaQuest"))
    assert not is_followup("What did he build at ARAI?", previous, lookup("What did he build at ARAI?"))


def test_no_previous_evidence_means_no_follow_up_to_merge():
    current = lookup("The numbers?")
    assert not is_followup("The numbers?", None, current)
