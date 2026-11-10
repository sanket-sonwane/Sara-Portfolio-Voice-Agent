"""The deterministic fallback must answer ordinary questions without guessing.

These are the zero-key path: with no LLM configured, every answer comes from
``answer_from``. A refusal on "what do you build?" is a product failure, and an
invented claim is worse, so both shapes are pinned here.
"""

from __future__ import annotations

from sarang.kb.lookup import lookup
from sarang.llm.local import NOT_DOCUMENTED, answer_from


def answer(question: str) -> str:
    return answer_from(question, lookup(question))


def test_generic_project_question_gets_an_overview():
    text = answer("What do you build?")
    assert text != NOT_DOCUMENTED
    assert "Agentic Commerce Gateway" in text
    assert "TattvaQuest" in text


def test_superlative_project_question_gets_an_overview_not_a_refusal():
    for question in (
        "What are your strongest projects?",
        "What is the most interesting project?",
        "Tell me about your portfolio.",
    ):
        text = answer(question)
        assert text != NOT_DOCUMENTED, question
        assert "portfolio includes" in text


def test_current_work_is_answered_from_the_owner_block():
    for question in ("What are you currently working on?", "What do you do?", "Tell me about yourself."):
        text = answer(question)
        assert "AI Engineer" in text, question
        assert "Agentic AI" in text, question


def test_internship_answer_is_grounded_in_the_experience_block():
    text = answer("Where did you intern?")
    assert "EPIC" in text or "ARAI" in text or "COEP" in text


def test_named_internship_question_is_not_answered_with_the_owner_intro():
    text = answer("What did Sanket build during the ARAI internship?")
    assert "Python Developer Intern" in text
    assert "based in Pune" not in text


def test_contact_question_answers_with_the_email_not_the_whole_profile():
    text = answer("What is Sanket's email?")
    assert "sanketsonwane512@gmail.com" in text
    assert "Agentic AI" not in text


def test_cgpa_is_quoted_from_education():
    text = answer("What is your CGPA?")
    assert "9.14" in text


def test_years_of_experience_is_not_invented():
    text = answer("How many years of experience do you have?")
    assert "years-of-experience" in text
    assert "AI Engineer" not in text


def test_12th_marks_are_refused_even_though_education_matches():
    text = answer("What is your 12th percentage?")
    assert "12th or HSC" in text
    assert "9.14" not in text


def test_pending_project_is_refused():
    text = answer("Tell me about the CPU scheduling project.")
    assert "isn't documented" in text
    assert "sched_ext" not in text


def test_unrelated_question_is_refused():
    text = answer("What is the airspeed velocity of an unladen swallow?")
    assert text == NOT_DOCUMENTED


def test_metrics_are_quoted_without_grammar_guessing():
    text = answer("Tell me about VisionMitra.")
    assert "transit stops processed: 1,300+" in text
