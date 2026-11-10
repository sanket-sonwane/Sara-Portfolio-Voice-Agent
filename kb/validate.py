"""Validate kb/fact.yaml and bench/questions/questions.yaml.

This is the mechanical half of the grounding guarantee in PRD section 14.6. It does
not judge truth; it enforces that every claim is attributable and every reference
resolves. If a metric has no `source`, or a skill points at a project that does not
exist, the build stops here rather than in front of a recruiter.

Usage::

    uv run python kb/validate.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
FACT = ROOT / "fact.yaml"
QUESTIONS = ROOT.parent / "bench" / "questions" / "questions.yaml"

ALLOWED_ORIGINS = {"resume", "portfolio", "repo", "stated"}
ALLOWED_STATUS = {"submitted", "accepted", "published", "in_preparation", "unknown"}
IGNORED_VOICE_KEYS = {
    "meta",
    "owner",
    "profile",
    "role_fit",
    "off_topic",
    "voice_persona",
    "pending",
    "boundaries",
    "lexicon",
}


class Validator:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.checks = 0

    def check(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.errors.append(message)

    def warn(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.warnings.append(message)

    def origins(self, node: dict[str, Any], where: str) -> None:
        origin = node.get("origin")
        self.check(isinstance(origin, list) and bool(origin), f"{where}: missing `origin` list")
        if isinstance(origin, list):
            unknown = set(origin) - ALLOWED_ORIGINS
            self.check(not unknown, f"{where}: unknown origin values {sorted(unknown)}")

    def metrics(self, node: dict[str, Any], where: str) -> None:
        for index, metric in enumerate(node.get("metrics") or []):
            label = metric.get("label", f"#{index}")
            self.check(bool(metric.get("label")), f"{where}: metric {label} missing label")
            self.check(
                metric.get("value") not in (None, ""),
                f"{where}: metric {label} missing value",
            )
            self.check(
                bool(metric.get("source")),
                f"{where}: metric {label} has no `source` (PRD 14.6 requires provenance)",
            )


def validate_facts(data: dict[str, Any], v: Validator) -> set[str]:
    v.check("meta" in data, "missing `meta`")
    v.check(data.get("meta", {}).get("version") is not None, "meta.version missing")
    v.check("owner" in data, "missing `owner`")

    owner = data.get("owner") or {}
    for field in ("name", "headline", "location", "email"):
        v.check(bool(owner.get(field)), f"owner.{field} missing")
    v.check(isinstance(owner.get("links"), dict), "owner.links missing")

    project_ids: set[str] = set()
    for project in data.get("projects") or []:
        identifier = project.get("id") or "<missing id>"
        where = f"project[{identifier}]"
        v.check(bool(project.get("id")), f"{where}: missing id")
        v.check(identifier not in project_ids, f"{where}: duplicate id")
        project_ids.add(identifier)
        v.check(bool(project.get("name")), f"{where}: missing name")
        v.check(bool(project.get("summary")), f"{where}: missing summary")
        v.check(isinstance(project.get("limitations"), list), f"{where}: missing limitations list")
        v.check(bool(project.get("stack")), f"{where}: missing stack")
        v.origins(project, where)
        v.metrics(project, where)

    experience_ids: set[str] = set()
    for role in data.get("experience") or []:
        identifier = role.get("id") or "<missing id>"
        where = f"experience[{identifier}]"
        v.check(identifier not in experience_ids, f"{where}: duplicate id")
        experience_ids.add(identifier)
        for field in ("org", "role", "start", "end", "summary"):
            v.check(bool(role.get(field)), f"{where}: missing {field}")
        v.origins(role, where)
        v.metrics(role, where)

    for paper in data.get("research") or []:
        where = f"research[{paper.get('id', '?')}]"
        status = paper.get("status")
        v.check(status in ALLOWED_STATUS, f"{where}: invalid status {status!r}")
        if status in {"submitted", "in_preparation"}:
            v.warn(
                any("accepted" in item.lower() for item in (data.get("boundaries") or {}).get("never_claim", [])),
                "a paper is not published but boundaries.never_claim does not forbid claiming acceptance",
            )
        v.origins(paper, where)
        v.metrics(paper, where)

    # Skill evidence must resolve to a real project.
    skills = data.get("skills") or {}
    v.check(bool(skills), "skills section missing")
    seen_skills: set[tuple[str, str]] = set()
    for category, entries in skills.items():
        v.check(isinstance(entries, list), f"skills.{category} must be a list")
        for entry in entries or []:
            name = entry.get("name", "?")
            where = f"skills.{category}[{name}]"
            v.check(bool(entry.get("name")), f"{where}: missing name")
            key = (category, name.lower())
            v.warn(key not in seen_skills, f"{where}: duplicate skill in category")
            seen_skills.add(key)
            evidence = entry.get("evidence")
            v.check(
                evidence is None or evidence in project_ids,
                f"{where}: evidence {evidence!r} is not a known project id",
            )
            v.origins(entry, where)

    for item in data.get("education") or []:
        where = f"education[{item.get('id', '?')}]"
        for field in ("qualification", "institution", "score", "score_label"):
            v.check(bool(item.get(field)), f"{where}: missing {field}")

    for section in ("leadership", "achievements"):
        for item in data.get(section) or []:
            where = f"{section}[{item.get('id', '?')}]"
            v.check(bool(item.get("summary")), f"{where}: missing summary")
            v.origins(item, where)

    for item in data.get("pending") or []:
        where = f"pending[{item.get('id', '?')}]"
        v.check(bool(item.get("topic")), f"{where}: missing topic")
        v.check(bool(item.get("action")), f"{where}: missing action")

    profile = data.get("profile") or {}
    v.check(bool(profile), "profile section missing")
    for field in ("motivation", "vision", "mission", "community"):
        v.check(bool(profile.get(field)), f"profile.{field} missing")
    v.check(isinstance(profile.get("interests"), list), "profile.interests must be a list")
    v.origins(profile, "profile")

    role_fit = data.get("role_fit") or {}
    v.check(bool(role_fit), "role_fit section missing")
    v.check(bool(role_fit.get("headline")), "role_fit.headline missing")
    v.check(bool(role_fit.get("guidance")), "role_fit.guidance missing")
    v.check(bool(role_fit.get("target_roles")), "role_fit.target_roles missing")
    known_evidence = project_ids | experience_ids
    for index, item in enumerate(role_fit.get("strengths") or []):
        where = f"role_fit.strengths[{index}]"
        v.check(bool(item.get("claim")), f"{where}: missing claim")
        v.check(
            item.get("evidence") in known_evidence,
            f"{where}: evidence {item.get('evidence')!r} is not a known project or experience id",
        )
    v.origins(role_fit, "role_fit")

    off_topic = data.get("off_topic") or {}
    v.check(bool(str(off_topic.get("response", "")).strip()), "off_topic.response missing")

    boundaries = data.get("boundaries") or {}
    v.check(bool(boundaries.get("never_claim")), "boundaries.never_claim missing")
    v.check(bool(boundaries.get("refuse_topics")), "boundaries.refuse_topics missing")

    lexicon = data.get("lexicon") or {}
    v.check(bool(lexicon.get("proper_nouns")), "lexicon.proper_nouns missing")

    return project_ids


def validate_questions(data: dict[str, Any], v: Validator) -> int:
    seen: set[str] = set()
    total = 0

    def walk(items: list[dict[str, Any]]) -> None:
        nonlocal total
        for item in items:
            identifier = item.get("id")
            if identifier:
                v.check(identifier not in seen, f"questions: duplicate id {identifier}")
                seen.add(identifier)
            total += 1
            if "question" in item:
                v.check(bool(item.get("expect")), f"question[{identifier}] missing expect")
            elif "turns" in item:
                v.check(len(item["turns"]) >= 2, f"turns[{identifier}] needs at least two turns")
                v.check(bool(item.get("expect")), f"turns[{identifier}] missing expect")
            else:
                v.check(False, f"question[{identifier}] has neither question nor turns")

    for group in data.get("groups") or []:
        walk(group.get("questions") or [])
        walk(group.get("turns") or [])

    return total


def main() -> int:
    if not FACT.exists():
        print(f"missing {FACT}", file=sys.stderr)
        return 2

    facts = yaml.safe_load(FACT.read_text(encoding="utf-8"))
    v = Validator()
    validate_facts(facts, v)

    question_count = 0
    if QUESTIONS.exists():
        questions = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))
        question_count = validate_questions(questions, v)
    else:
        v.warn(False, f"question set not found at {QUESTIONS}")

    skills = sum(len(entries or []) for entries in (facts.get("skills") or {}).values())
    summary = {
        "projects": len(facts.get("projects") or []),
        "experience": len(facts.get("experience") or []),
        "research": len(facts.get("research") or []),
        "skills": skills,
        "questions": question_count,
    }

    print("fact.yaml summary: " + ", ".join(f"{k}={val}" for k, val in summary.items()))
    for warning in v.warnings:
        print(f"WARN  {warning}")
    for error in v.errors:
        print(f"ERROR {error}")

    print(f"\n{v.checks} checks, {len(v.errors)} errors, {len(v.warnings)} warnings")
    if v.errors:
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
