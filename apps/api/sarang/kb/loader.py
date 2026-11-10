"""Knowledge base loading.

``kb/fact.yaml`` is the only thing the agent may assert. This module turns it into
addressable blocks once at startup; nothing is re-read per turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import KB_DIR


@dataclass
class FactBlock:
    id: str
    kind: str
    title: str
    text: str
    aliases: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    tokens: set[str] = field(default_factory=set)


def _join(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if item)
    if isinstance(value, dict):
        return "; ".join(f"{key}: {val}" for key, val in value.items() if val)
    return str(value)


def _trim_sentence(text: str, limit: int) -> str:
    """Cut to ``limit`` characters without leaving a half word or half sentence."""
    if len(text) <= limit:
        return text
    window = text[:limit]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if cut >= max(200, limit // 2):
        return window[: cut + 1].strip()
    space = window.rfind(" ")
    return (window[:space] if space > 0 else window).strip()


def _metric_text(metrics: list[dict] | None) -> str:
    if not metrics:
        return ""
    parts = []
    for metric in metrics:
        label = metric.get("label", "").strip()
        value = str(metric.get("value", "")).strip()
        if label and value:
            parts.append(f"{label}: {value}")
    return "; ".join(parts)


class KnowledgeBase:
    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.blocks: list[FactBlock] = []
        self.by_id: dict[str, FactBlock] = {}
        self._build()

    # ------------------------------------------------------------------ build

    def _add(self, block: FactBlock) -> None:
        self.blocks.append(block)
        self.by_id[block.id] = block

    def _build(self) -> None:
        owner = self.raw.get("owner") or {}
        links = owner.get("links") or {}
        self._add(
            FactBlock(
                id="owner",
                kind="owner",
                title="Sanket Sonwane — profile",
                aliases=["sanket", "who is he", "about him", "contact", "email", "linkedin",
                         "currently", "current", "focus", "availability", "open to", "background"],
                text=(
                    f"{owner.get('name', '')}, {owner.get('headline', '')}. "
                    f"{owner.get('tagline', '')} Based in {owner.get('location', '')}. "
                    f"Email: {owner.get('email', '')}. "
                    f"Summary: {owner.get('summary', '')} "
                    f"Focus: {owner.get('focus', '')} "
                    f"Availability: {owner.get('availability', '')} "
                    f"Portfolio: {links.get('portfolio', '')}. "
                    f"GitHub: {links.get('github', '')}. "
                    f"LinkedIn: {links.get('linkedin', '')}."
                ),
                data=owner,
            )
        )

        profile = self.raw.get("profile") or {}
        if profile:
            self._add(
                FactBlock(
                    id="profile",
                    kind="profile",
                    title="Motivation, vision and values",
                    aliases=[
                        "motivation",
                        "vision",
                        "mission",
                        "values",
                        "working style",
                        "community",
                        "volunteer",
                        "meetup",
                        "developer tech society",
                        "what drives him",
                        "what makes him tick",
                        "career goals",
                    ],
                    text=(
                        f"Motivation: {profile.get('motivation', '')} "
                        f"Vision: {profile.get('vision', '')} "
                        f"Mission: {profile.get('mission', '')} "
                        f"Values: {_join(profile.get('values'))}. "
                        f"Working style: {profile.get('working_style', '')} "
                        f"Community: {profile.get('community', '')}"
                    ),
                    data=profile,
                )
            )
            # Interests are their own block: the profile block is trimmed to the
            # context budget, and hobbies at its tail were being cut away.
            interests = profile.get("interests") or []
            if interests:
                self._add(
                    FactBlock(
                        id="interests",
                        kind="profile",
                        title="Interests",
                        aliases=["hobbies", "free time", "outside work", "chess", "travel"],
                        text=f"Interests outside work: {_join(interests)}.",
                        data={"interests": interests},
                    )
                )

        role_fit = self.raw.get("role_fit") or {}
        if role_fit:
            roles = _join(role_fit.get("target_roles"))
            strengths = "; ".join(
                f"{item.get('claim', '')} ({item.get('evidence', '')})"
                for item in role_fit.get("strengths") or []
            )
            # Capabilities is the block injected for fit questions: a compact,
            # pre-rendered summary so the turn does not pay for the whole KB.
            experience_line = "; ".join(
                f"{role.get('role', '')} at {role.get('org', '')}"
                for role in self.raw.get("experience") or []
            )
            project_line = "; ".join(
                f"{project.get('name', '')} — {project.get('tagline', '')}"
                for project in self.raw.get("projects") or []
            )
            skill_line = "; ".join(
                f"{category.replace('_', ' ')}: "
                + ", ".join(
                    entry.get("name", "") for entry in entries or [] if entry.get("name")
                )
                for category, entries in (self.raw.get("skills") or {}).items()
            )
            # 680 characters keeps capabilities + role_fit inside the context
            # budget, so the fit guidance is never the block that gets dropped.
            capabilities = _trim_sentence(
                f"Positioning: {role_fit.get('headline', '')} "
                f"Open to roles: {roles}. "
                f"Proof: {role_fit.get('proof', '')} "
                f"Strengths: {strengths}. "
                f"Experience: {experience_line}. "
                f"Projects: {project_line}. "
                f"Skills: {skill_line}.",
                680,
            )
            self._add(
                FactBlock(
                    id="capabilities",
                    kind="capabilities",
                    title="Capabilities",
                    aliases=["fit", "suitable", "good match", "candidate"],
                    text=capabilities,
                    data={"role_fit": role_fit},
                )
            )
            self._add(
                FactBlock(
                    id="role_fit",
                    kind="role_fit",
                    title="Role fit",
                    aliases=["good fit", "fit for", "suitable for", "hire him", "role fit"],
                    text=(
                        f"{role_fit.get('headline', '')} {role_fit.get('proof', '')} "
                        f"Open to roles: {roles}. {role_fit.get('guidance', '')}"
                    ),
                    data=role_fit,
                )
            )

        off_topic = self.raw.get("off_topic") or {}
        if off_topic:
            self._add(
                FactBlock(
                    id="off_topic",
                    kind="off_topic",
                    title="Out of scope",
                    aliases=["off topic", "unrelated question"],
                    text=str(off_topic.get("response", "")).strip(),
                    data=off_topic,
                )
            )

        for role in self.raw.get("experience") or []:
            metrics = _metric_text(role.get("metrics"))
            text = (
                f"{role.get('role', '')} at {role.get('org', '')} ({role.get('start', '')} to "
                f"{role.get('end', '')}, {role.get('location', '')}). {role.get('summary', '')} "
                f"Work: {_join(role.get('highlights'))}. Stack: {_join(role.get('stack'))}. "
                f"{('Numbers: ' + metrics + '. ') if metrics else ''}"
                f"{('Caveats: ' + _join(role.get('limitations')) + '. ') if role.get('limitations') else ''}"
            )
            aliases = [
                role.get("org", ""),
                role.get("role", ""),
                role.get("id", ""),
                *(role.get("aliases") or []),
            ]
            self._add(
                FactBlock(
                    id=f"experience:{role.get('id', '')}",
                    kind="experience",
                    title=f"{role.get('role', '')} — {role.get('org', '')}",
                    aliases=[a for a in aliases if a],
                    text=text,
                    data=role,
                )
            )

        for project in self.raw.get("projects") or []:
            metrics = _metric_text(project.get("metrics"))
            text = (
                f"{project.get('name', '')} ({project.get('tagline', '')}). "
                f"{project.get('summary', '')} "
                f"Problem: {project.get('problem', '')} "
                f"Approach: {_join(project.get('approach'))}. "
                f"Stack: {_join(project.get('stack'))}. "
                f"Period: {project.get('period', '')}. "
                f"{('Numbers: ' + metrics + '. ') if metrics else ''}"
                f"{('Caveats: ' + _join(project.get('limitations')) + '. ') if project.get('limitations') else ''}"
            )
            aliases = [
                project.get("name", ""),
                project.get("short_name", ""),
                project.get("id", ""),
                *(project.get("aliases") or []),
            ]
            self._add(
                FactBlock(
                    id=f"project:{project.get('id', '')}",
                    kind="project",
                    title=project.get("name", ""),
                    aliases=[a for a in aliases if a],
                    text=text,
                    data=project,
                )
            )

        for paper in self.raw.get("research") or []:
            metrics = _metric_text(paper.get("metrics"))
            aliases = [
                "paper",
                "research",
                "publication",
                paper.get("venue", ""),
                paper.get("id", ""),
                *(paper.get("aliases") or []),
            ]
            self._add(
                FactBlock(
                    id=f"research:{paper.get('id', '')}",
                    kind="research",
                    title=paper.get("title", ""),
                    aliases=[a for a in aliases if a],
                    text=(
                        f"Research: {paper.get('title', '')}. Venue: {paper.get('venue', '')}. "
                        f"Status: {paper.get('status', '')} ({paper.get('year', '')}). "
                        f"{paper.get('authors_note', '')} {paper.get('summary', '')} "
                        f"{('Numbers: ' + metrics + '. ') if metrics else ''}"
                        f"Caveats: {_join(paper.get('limitations'))}."
                    ),
                    data=paper,
                )
            )

        for item in self.raw.get("education") or []:
            self._add(
                FactBlock(
                    id=f"education:{item.get('id', '')}",
                    kind="education",
                    title=item.get("qualification", ""),
                    aliases=["education", "college", "degree", "cgpa", "sgpa", "percentage"],
                    text=(
                        f"{item.get('qualification', '')} at {item.get('institution', '')}, "
                        f"{item.get('start', '')} to {item.get('end', '')}. "
                        f"{item.get('score_label', '')}: {item.get('score', '')}. "
                        f"Status: {item.get('status', '')}."
                    ),
                    data=item,
                )
            )

        for category, entries in (self.raw.get("skills") or {}).items():
            names = [entry.get("name", "") for entry in entries or [] if entry.get("name")]
            evidenced = [
                f"{entry.get('name')} ({entry.get('evidence')})"
                for entry in entries or []
                if entry.get("evidence")
            ]
            self._add(
                FactBlock(
                    id=f"skills:{category}",
                    kind="skills",
                    title=f"Skills — {category.replace('_', ' ')}",
                    aliases=["skills", "stack", "technologies", category.replace("_", " ")],
                    text=(
                        f"{category.replace('_', ' ').title()}: {', '.join(names)}. "
                        f"Demonstrated in a public project: {', '.join(evidenced) if evidenced else 'not itemised'}."
                    ),
                    data={"category": category, "entries": entries},
                )
            )

        leadership = self.raw.get("leadership") or []
        if leadership:
            self._add(
                FactBlock(
                    id="leadership",
                    kind="leadership",
                    title="Leadership",
                    aliases=["leadership", "coordinator", "team", "volunteer"],
                    text="Leadership: " + " ".join(
                        f"{item.get('role', '')} — {item.get('summary', '')}" for item in leadership
                    ),
                    data={"items": leadership},
                )
            )

        achievements = self.raw.get("achievements") or []
        if achievements:
            self._add(
                FactBlock(
                    id="achievements",
                    kind="achievements",
                    title="Achievements",
                    aliases=["achievements", "award", "workshop", "nvidia", "presented"],
                    text="Achievements: " + " ".join(
                        f"{item.get('title', '')}. {item.get('summary', '')} "
                        f"{item.get('caveat', '')}"
                        for item in achievements
                    ),
                    data={"items": achievements},
                )
            )

        pending = self.raw.get("pending") or []
        if pending:
            self._add(
                FactBlock(
                    id="pending",
                    kind="pending",
                    title="Not documented",
                    aliases=["cpu scheduling", "sched_ext", "kernel scheduling"],
                    text=(
                        "Not documented in this portfolio: "
                        + "; ".join(f"{item.get('topic', '')}" for item in pending)
                        + ". If asked, say it is not documented rather than guessing."
                    ),
                    data={"items": pending},
                )
            )

        boundaries = self.raw.get("boundaries") or {}
        if boundaries:
            self._add(
                FactBlock(
                    id="boundaries",
                    kind="boundaries",
                    title="Boundaries",
                    aliases=["salary", "compensation", "phone", "address", "patent", "opinion"],
                    text=(
                        "Never claim: " + _join(boundaries.get("never_claim")) + ". "
                        "Refuse: " + _join(boundaries.get("refuse_topics")) + ". "
                        "Deflect: " + _join(boundaries.get("deflect_topics")) + "."
                    ),
                    data=boundaries,
                )
            )

    # ------------------------------------------------------------- accessors

    @property
    def lexicon(self) -> list[str]:
        lexicon = self.raw.get("lexicon") or {}
        terms = list(lexicon.get("proper_nouns") or []) + list(lexicon.get("technical") or [])
        return [term for term in terms if term]

    @property
    def persona(self) -> dict[str, Any]:
        return self.raw.get("voice_persona") or {}

    def project_ids(self) -> set[str]:
        return {block.data.get("id", "") for block in self.blocks if block.kind == "project"}


_CACHE: KnowledgeBase | None = None


def load_kb(path: Path | None = None) -> KnowledgeBase:
    global _CACHE
    if _CACHE is not None and path is None:
        return _CACHE
    target = path or (KB_DIR / "fact.yaml")
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    kb = KnowledgeBase(raw)
    if path is None:
        _CACHE = kb
    return kb
