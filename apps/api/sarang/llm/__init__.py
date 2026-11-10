"""LLM provider selection."""

from __future__ import annotations

from ..config import Settings
from .base import LLM
from .gemini import GeminiLLM
from .openai_compat import OpenAICompatLLM


def build_llm(settings: Settings) -> LLM | None:
    """Return a streaming LLM, or ``None`` for the deterministic local answerer."""
    provider = settings.llm_provider

    if provider == "groq" and settings.groq_key:
        return OpenAICompatLLM(
            name="groq",
            base_url=settings.groq_base_url,
            api_key=settings.groq_key,
            model=settings.groq_model,
            timeout=20.0,
        )

    if provider == "sarvam" and settings.sarvam_key:
        return OpenAICompatLLM(
            name="sarvam",
            base_url=f"{settings.sarvam_base_url.rstrip('/')}/v1",
            api_key=settings.sarvam_key,
            model=settings.sarvam_llm_model,
            auth_header="api-subscription-key",
            auth_prefix="",
            timeout=25.0,
        )

    if provider == "gemini" and settings.gemini_key:
        return GeminiLLM(
            api_key=settings.gemini_key,
            model=settings.gemini_model,
            base_url=settings.gemini_base_url,
        )

    return None
