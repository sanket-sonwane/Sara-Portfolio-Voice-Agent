"""Recognition biasing and transcript repair.

Both are driven by the knowledge base, and both have provider constraints that
used to fail silently (a comma in one keyterm killed the whole ASR session).
"""

from __future__ import annotations

from sarang.asr.keyterms import MAX_KEYTERMS, build_keyterms, domain_prompt
from sarang.asr.normalize import normalize_transcript


def test_keyterms_never_contain_a_provider_separator():
    # Sarvam rejects the entire session (invalid_request, fatal) if one term has
    # a comma or semicolon: "AISSMS Polytechnic, Pune" used to do exactly that.
    for term in build_keyterms():
        assert "," not in term
        assert ";" not in term


def test_keyterms_never_teach_the_decoder_to_spell():
    # "A E U E" made the recogniser return "A E U E" for "ARAI".
    for term in build_keyterms():
        tokens = term.split()
        assert not (len(tokens) > 1 and all(len(token) <= 2 for token in tokens)), term


def test_acronyms_are_rationed():
    # Measured: 1-3 acronyms keep "ARAI" correct, 7 make it "ARII", 20 make it
    # "AAI". Project names are free; acronyms compete with each other.
    terms = build_keyterms()
    acronyms = [term for term in terms if term.isupper() and " " not in term]
    assert 0 < len(acronyms) <= 3
    assert "ARAI" in terms


def test_the_list_stays_short_and_deduplicated():
    terms = build_keyterms()
    assert len(terms) <= MAX_KEYTERMS
    assert len(terms) == len({term.lower() for term in terms})


def test_the_domain_prompt_names_the_projects():
    prompt = domain_prompt()
    assert "TattvaQuest" in prompt
    assert "ARAI" in prompt


def test_observed_mishearings_are_repaired():
    assert normalize_transcript("during the ARII internship") == "during the ARAI internship"
    assert normalize_transcript("during the AAI internship") == "during the ARAI internship"
    assert normalize_transcript("during the AII internship") == "during the ARAI internship"
    # A letter-spaced variant is matched as one unit.
    assert normalize_transcript("at A R A I") == "at ARAI"


def test_unrelated_words_are_left_alone():
    # "ARI" must not rewrite the middle of a word, and ordinary text passes
    # through untouched: a transcript is the visitor's words, not ours.
    assert normalize_transcript("arise and shine") == "arise and shine"
    assert normalize_transcript("what did you build") == "what did you build"
    assert normalize_transcript("") == ""
