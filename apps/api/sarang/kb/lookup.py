"""Tier-1 fact lookup: the router.

Runs in-process in a few milliseconds and decides three things:

* which fact blocks the model should see (top 1-3),
* whether the question is answerable at all,
* whether the turn needs tier-2 depth retrieval.

Deliberately not an LLM call. A classifier hop would cost a full TTFT, which is a
third of the budget. Lexical scoring over a curated corpus this small is both
faster and more predictable.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .loader import FactBlock, KnowledgeBase

# Stopwords for *scoring*: what survives tokenization is the topical set of a
# query. It includes discourse markers and filled pauses - "however", "mean",
# "well" - because they otherwise score like ordinary keywords. Measured:
# none of these tokens appear in any fact block (df = 0), so they can never
# help a match - but they poison the topical set (the generic-project subset
# test failed: "however i mean what projects" was no longer a subset of
# GENERIC_PROJECT_TOKENS, and the router picked profile + three arbitrary
# blocks instead of the portfolio overview) and they travel into the tier-2
# query and the model prompt as if they carried meaning.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "will", "would", "can", "could",
    "should", "may", "might", "must", "shall", "to", "of", "in", "on", "at",
    "for", "with", "by", "from", "as", "into", "about", "than", "then", "that",
    "this", "these", "those", "it", "its", "he", "she", "they", "them", "his",
    "her", "their", "you", "your", "i", "me", "my", "we", "our", "and", "or",
    "but", "if", "so", "not", "no", "yes", "please", "tell", "say", "give",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "much", "many", "more", "most", "some", "any", "there", "here", "also",
    "just", "very", "really", "actually", "kind", "sort", "thing", "things",
    "know", "think", "want", "like", "get", "got", "make", "made", "use",
    "used", "using", "work", "works", "worked", "working", "done",
    # Discourse markers and filled pauses.
    "however", "mean", "well", "anyway", "basically", "maybe", "probably",
    "honestly", "literally", "yeah", "okay", "ok", "alright", "hmm", "um",
    "uh", "eh", "huh", "kinda", "sorta",
}

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.#_-]*")

# Words said *around* a question, not part of it. The prefix pattern is
# repeated on purpose: visitors stack markers ("well, i mean, however...").
# The separator is mandatory ([,\s]+): an optional one would let "so" eat the
# head of "solomon".
DISCOURSE_PREFIX_RE = re.compile(
    r"^(?:(?:however|well|okay|ok|yeah|alright|hmm+|um+|uh+|eh+|huh|"
    r"basically|actually|anyway|maybe|i mean|you know|like|so|right)"
    r"[,\s]+)+",
    re.IGNORECASE,
)
# Filled pauses in the middle of a sentence, with the separator that
# surrounds them: removing "you know" from "build, you know, during" must not
# leave "build, during" behind.
FILLED_PAUSE_RE = re.compile(
    r"(?:\s*,\s*|\s+)\b(?:i mean|you know|hmm+|um+|uh+|eh+|huh)\b[\s,]*",
    re.IGNORECASE,
)


def strip_discourse(text: str) -> str:
    """Return ``text`` without the words wrapped around the question.

    The transcript card keeps what the visitor actually said; the router,
    the retriever and the model get the question itself. Without this the
    measured damage was concrete: ``lookup("what projects did he build")``
    returned the portfolio overview while ``lookup("however i mean what
    projects did he build")`` returned profile + three arbitrary blocks -
    the fillers broke the generic-project subset test and reached the model
    as query text. Routing patterns match on content words that survive the
    strip, and the guard keeps a marker-only utterance from becoming an
    empty query.
    """
    cleaned = text.strip()
    for _ in range(2):
        cleaned = DISCOURSE_PREFIX_RE.sub("", cleaned)
        cleaned = " ".join(FILLED_PAUSE_RE.sub(" ", cleaned).split())
    return cleaned or text.strip()


def tokenize(text: str) -> list[str]:
    lowered = text.lower().replace("’", "'")
    raw = TOKEN_RE.findall(lowered)
    out: list[str] = []
    for token in raw:
        token = token.strip("-_.+")
        if len(token) <= 1 or token in STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        out.append(token)
    return out


# Query intent → (block kind, boost). The patterns are deliberately narrow: they
# only fire on vocabulary a recruiter actually uses.
INTENT_RULES: tuple[tuple[re.Pattern[str], str, float], ...] = (
    (re.compile(r"\b(cgpa|sgpa|gpa|percentage|marks|grades?)\b"), "education", 10.0),
    (
        re.compile(
            r"\b(internships?|work experience|professional experience|employment|"
            r"where has he worked|worked (at|for)|jobs?)\b"
        ),
        "experience",
        5.0,
    ),
    (re.compile(r"\b(education|college|university|degree|diploma|study|studied|school)\b"), "education", 4.0),
    (re.compile(r"\b(intern|internship|interned|company|companies|employer)\b"), "experience", 4.0),
    (re.compile(r"\b(paper|journal|publication|published|submitted|review|aeue|research)\b"), "research", 4.0),
    (re.compile(r"\b(contact|email|mail|reach|linkedin|github|hire)\b"), "owner", 5.0),
    # Naming the subject is a routing hint, not a topic. Without this, "Sanket"
    # outscores the actual subject of the question ("... during the ARAI
    # internship?"), because the owner block is the only block that repeats it.
    (re.compile(r"\b(sanket|sonwane)\b"), "owner", 5.0),
    (
        re.compile(
            r"\b(who are you|about yourself|about sanket|what do you do|what does he do|"
            r"background|introduce|introduction|currently|current work|working on|focus|"
            r"availability|available|open to)\b"
        ),
        "owner",
        5.0,
    ),
    (re.compile(r"\b(skill|skills|stack|technolog|language|framework|tool)\b"), "skills", 4.0),
    (re.compile(r"\b(project|projects|built|build|building|portfolio)\b"), "project", 2.5),
    (re.compile(r"\b(achiev|award|workshop|nvidia|presented|certificat)\b"), "achievements", 5.0),
    (re.compile(r"\b(lead|leader|leadership|coordinator|volunteer|captain|crew)\b"), "leadership", 5.0),
    (re.compile(r"\b(salary|compensation|ctc|phone|mobile|whatsapp|contact number|address|dob|birth|patent|weakness|google|microsoft)\b"), "boundaries", 6.0),
    (re.compile(r"\b(cpu scheduling|sched_ext|kernel scheduling)\b"), "pending", 7.0),
    # Fit is a judgement about the whole profile, not a topic lookup: it gets its
    # own path that injects the capability summary. Negative framings ("gaps",
    # "why not hire him") are fit questions too, and must not fall through to the
    # salary refusal.
    (
        re.compile(
            r"\b(good fit|fit for|suitable for|right for|hire him|hiring him|"
            r"should we hire|why should we|why not hire|good match|strong candidate|"
            r"this role|the role|role fit|interview him|right candidate|worth a call|"
            r"recommend him|reject him|not hire him|weakness\w*|gaps? in his|"
            r"any gaps?|concerns? about him)\b"
        ),
        "role_fit",
        9.0,
    ),
    (
        re.compile(
            r"\b(hobb\w*|interests?|free time|outside work|chess|travel|"
            r"motivat\w*|passion\w*|driven|what drives|what makes him|"
            r"care about|cares about)\b"
        ),
        "profile",
        7.0,
    ),
    (
        re.compile(
            r"\b(vision|mission|values?|ambition|long term|five years|future plans|"
            r"where does he see|beliefs?|principles?)\b"
        ),
        "profile",
        7.0,
    ),
    (
        re.compile(
            r"\b(community|volunteer|meetup|meetups|developer tech society|"
            r"dts pune|mentor|mentoring|give back)\b"
        ),
        "profile",
        7.0,
    ),
    (
        re.compile(r"\b(hobb\w*|free time|outside work|chess|travel|for fun)\b"),
        "profile",
        7.0,
    ),
)

# Questions with nothing to do with Sanket. Deliberately narrow, and only applied
# when the question does not mention him: "did he play cricket?" is a question
# about him and belongs on the not-documented path, not the apology path.
OFF_TOPIC_RE = re.compile(
    r"\b(weather|temperature|forecast|humidity|cricket|football|soccer|match score|"
    r"movie|film|lyrics|recipe|cook|news|headline|stock price|bitcoin|"
    r"politics|election|president|prime minister|capital of|"
    r"joke|poem|story|sing|horoscope|astrology|"
    r"write (me )?(a |an )?(\w+ ){0,3}(poem|code|program|script|email|essay)|"
    r"debug|homework|assignment|translate|summari[sz]e this|solve this)\b"
)
SUBJECT_RE = re.compile(r"\b(he|him|his|sanket|sonwane)\b")

# Requests to do a task for the visitor. These are out of scope even when they
# name Sanket ("write a cover letter for Sanket"), so they are checked before the
# subject guard that protects questions *about* him.
TASK_RE = re.compile(
    r"\b(write|draft|compose|generate) (me |us )?(a |an |the )?(\w+ ){0,4}"
    r"(cover letter|email|e-mail|essay|poem|story|code|program|script|resume|cv|"
    r"report|blog|post|tweet|speech)\b"
    r"|\b(book|reserve) (me |us )?(a |an )?(flight|ticket|hotel|table|cab|taxi|train)\b"
    r"|\bignore (all )?(the )?(previous|prior|above) (instructions|prompts?)\b"
    r"|\b(system prompt|jailbreak|act as (a|an|my)|you are now)\b"
    r"|\b(translate|summari[sz]e) (this|the following|this text)\b"
)

DEPTH_RE = re.compile(
    r"\b(architect|architecture|design|internal|internals|tradeoff|trade-off|"
    r"explain|deep|detail|details|implement|implementation|why|compare|approach|"
    r"how does|how did|walk me|walk through)\b"
)

ALLOWED_KINDS = {"owner", "experience", "project", "research", "education", "skills",
                 "leadership", "achievements", "pending", "boundaries", "profile",
                 "capabilities", "role_fit", "off_topic"}

# "What do you build?" has no discriminating token: every project scores the same
# intent boost. Recognising the generic shape lets the fallback answer with an
# overview of the portfolio instead of refusing a perfectly ordinary question.
GENERIC_PROJECT_TOKENS = {
    "build",
    "built",
    "building",
    "project",
    "portfolio",
    "strongest",
    "best",
    "favourite",
    "favorite",
    "interesting",
    "impressive",
    "coolest",
    "main",
    "top",
}


@dataclass
class LookupResult:
    query: str
    blocks: list[FactBlock] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    confident: bool = False
    needs_depth: bool = False
    intent: str = ""
    blocked: bool = False
    reason: str = ""

    @property
    def top(self) -> FactBlock | None:
        return self.blocks[0] if self.blocks else None

    def to_payload(self) -> dict:
        return {
            "query": self.query,
            "confident": self.confident,
            "needs_depth": self.needs_depth,
            "intent": self.intent,
            "blocked": self.blocked,
            "ids": [block.id for block in self.blocks],
            "scores": {key: round(value, 2) for key, value in self.scores.items()},
            "reason": self.reason,
        }


class FactIndex:
    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.docs: list[tuple[FactBlock, dict[str, float]]] = []
        self.df: dict[str, int] = {}
        owner = kb.raw.get("owner") or {}
        self.subject_tokens = set(
            tokenize(f"{owner.get('name', '')} {owner.get('full_name', '')}")
        )
        self._build()

    def _build(self) -> None:
        for block in self.kb.blocks:
            weights: dict[str, float] = {}
            title_tokens = set(tokenize(block.title))
            alias_tokens: set[str] = set()
            for alias in block.aliases:
                alias_tokens |= set(tokenize(alias))
            for token in tokenize(block.text):
                weights[token] = weights.get(token, 0.0) + 1.0
            for token in alias_tokens:
                weights[token] = weights.get(token, 0.0) + 3.0
            for token in title_tokens:
                weights[token] = weights.get(token, 0.0) + 4.0
            self.docs.append((block, weights))
            for token in weights:
                self.df[token] = self.df.get(token, 0) + 1

        total = max(1, len(self.docs))
        self.idf = {
            token: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for token, freq in self.df.items()
        }

    def score(self, query: str) -> tuple[dict[str, float], str]:
        query_tokens = tokenize(query)
        # The owner's name identifies who the corpus is about, so it must not
        # act as a topical keyword. Route name-only questions with the owner
        # intent rule instead.
        topical_tokens = [token for token in query_tokens if token not in self.subject_tokens]
        lowered = query.lower()
        scores: dict[str, float] = {block.id: 0.0 for block in self.kb.blocks}
        intent_hit = ""
        intent_boosts: dict[str, float] = {}

        for pattern, kind, boost in INTENT_RULES:
            if pattern.search(lowered):
                intent_hit = kind
                intent_boosts[kind] = max(intent_boosts.get(kind, 0.0), boost)

        phrase = " ".join(query_tokens)
        for block, weights in self.docs:
            score = 0.0
            for token in topical_tokens:
                weight = weights.get(token)
                if not weight:
                    continue
                score += min(weight, 6.0) * self.idf.get(token, 0.3)
            # Exact multi-word title or alias match is a strong signal.
            for alias in [block.title, *block.aliases]:
                alias_norm = " ".join(tokenize(alias))
                if alias_norm and len(alias_norm) > 3 and alias_norm in f" {phrase} ":
                    score += 7.0
            if block.kind in intent_boosts:
                score += intent_boosts[block.kind]
            scores[block.id] = score

        return scores, intent_hit

    def lookup(self, query: str, limit: int = 3) -> LookupResult:
        scores, intent = self.score(query)
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        result = LookupResult(query=query, scores=dict(ordered[:6]), intent=intent)
        lowered = query.lower()

        # A request to perform a task is out of scope regardless of who it names.
        if TASK_RE.search(lowered):
            block = self.kb.by_id.get("off_topic")
            if block is not None:
                result.blocks = [block]
                result.confident = True
                result.intent = "off_topic"
                result.reason = "task request"
                return result

        if intent == "boundaries":
            boundary = self.kb.by_id.get("boundaries")
            if boundary:
                result.blocks = [boundary]
                result.confident = True
                result.blocked = True
                result.reason = "boundary topic"
                return result

        if intent == "experience":
            blocks = [
                self.kb.by_id[block_id]
                for block_id, score in ordered
                if score > 0.9 and self.kb.by_id.get(block_id) is not None
                and self.kb.by_id[block_id].kind == "experience"
            ][:3]
            if blocks:
                result.blocks = blocks
                result.confident = True
                result.reason = "experience"
                return result

        if intent == "role_fit":
            blocks = [
                self.kb.by_id[key]
                for key in ("capabilities", "role_fit", "profile")
                if key in self.kb.by_id
            ]
            if blocks:
                result.blocks = blocks
                result.confident = True
                result.reason = "role fit"
                return result

        if intent == "profile":
            if re.search(r"\b(hobb\w*|interests?|free time|outside work|chess|travel)\b", lowered):
                block = self.kb.by_id.get("interests")
                if block is not None:
                    result.blocks = [block]
                    result.confident = True
                    result.reason = "interests"
                    return result
            block = self.kb.by_id.get("profile")
            if block is not None:
                result.blocks = [block]
                result.confident = True
                result.reason = "profile"
                return result

        if not intent and OFF_TOPIC_RE.search(lowered) and not SUBJECT_RE.search(lowered):
            block = self.kb.by_id.get("off_topic")
            if block is not None:
                result.blocks = [block]
                result.confident = True
                result.reason = "off topic"
                return result

        query_tokens = set(tokenize(query))
        topical = query_tokens - self.subject_tokens
        generic_project = bool(topical) and topical <= GENERIC_PROJECT_TOKENS
        if generic_project:
            projects = [
                self.kb.by_id[block_id]
                for block_id, _ in ordered
                if block_id in self.kb.by_id and self.kb.by_id[block_id].kind == "project"
            ][:6]
            if projects:
                profile = self.kb.by_id.get("profile")
                result.blocks = projects + ([profile] if profile else [])
                result.confident = True
                result.intent = "project"
                result.reason = "generic project overview"
                return result

        top_score = ordered[0][1] if ordered else 0.0
        second = ordered[1][1] if len(ordered) > 1 else 0.0

        # Refusal-shaped questions must reach the model even with no match: the
        # model has to say "not documented", and that requires the boundaries.
        refusal_shape = bool(
            re.search(r"\b(how many years|salary|phone|patent|revenue|cluster|"
                      r"google|microsoft|12th|hsc|ssc)\b", query.lower())
        )

        # Fit and off-topic blocks are only injected by their own intents. Letting
        # them score lexically would pull fit guidance into an ordinary answer.
        special = {"capabilities", "role_fit", "off_topic"}
        selected = [
            self.kb.by_id[block_id]
            for block_id, score in ordered
            if score > 0.9
            and block_id in self.kb.by_id
            and self.kb.by_id[block_id].kind not in special
        ][:limit]

        if refusal_shape:
            for extra in ("boundaries", "pending"):
                block = self.kb.by_id.get(extra)
                if block and block not in selected:
                    selected.append(block)
            result.confident = True
            result.reason = "refusal-shaped question"
        else:
            result.confident = top_score >= 3.0 and (top_score - second) >= 0.4 or top_score >= 6.0
            result.reason = f"top={top_score:.2f} second={second:.2f}"

        result.blocks = selected
        result.needs_depth = bool(DEPTH_RE.search(query)) and (
            not result.confident or top_score < 8.0
        )
        if not result.blocks and not result.confident:
            boundary = self.kb.by_id.get("boundaries")
            if boundary:
                result.blocks = [boundary]
        return result


_INDEX: FactIndex | None = None


def get_index(kb: KnowledgeBase | None = None) -> FactIndex:
    global _INDEX
    if _INDEX is None or kb is not None:
        from .loader import load_kb

        _INDEX = FactIndex(kb or load_kb())
    return _INDEX


def lookup(query: str, kb: KnowledgeBase | None = None, limit: int = 3) -> LookupResult:
    return get_index(kb).lookup(query, limit=limit)
