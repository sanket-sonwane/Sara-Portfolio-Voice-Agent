"""ASR selection."""

from __future__ import annotations

from ..config import Settings
from .base import ASRError, ASRSession, FixtureASR, NullASR
from .keyterms import build_keyterms, domain_prompt
from .nemotron import NemotronASR, NemotronStreamASR
from .sarvam import SarvamASR


def build_asr(settings: Settings) -> ASRSession:
    provider = settings.asr_provider
    if provider == "sarvam" and settings.sarvam_key:
        # Keyterm biasing and the domain prompt are ``saaras:v4`` features; on
        # the older realtime model they are ignored by the server, so only send
        # them when they can help.
        v4 = settings.sarvam_asr_model.startswith("saaras:v4")
        return SarvamASR(
            api_key=settings.sarvam_key,
            base_url=settings.sarvam_base_url,
            model=settings.sarvam_asr_model,
            language=settings.sarvam_asr_language,
            stream_type=settings.sarvam_asr_stream_type,
            mode=settings.sarvam_asr_mode,
            chunk_ms=settings.sarvam_asr_chunk_ms,
            keyterms=list(build_keyterms()) if v4 else [],
            prompt=domain_prompt() if v4 else "",
        )
    if provider == "nemotron" and settings.nvidia_key:
        import os

        stream_url = os.environ.get("NVIDIA_ASR_WS_URL", "").strip()
        if stream_url:
            return NemotronStreamASR(url=stream_url, api_key=settings.nvidia_key)
        return NemotronASR(
            api_key=settings.nvidia_key,
            url=settings.nvidia_asr_url,
            model=settings.nvidia_model,
        )
    if provider == "fixture":
        return FixtureASR()
    return NullASR()


__all__ = ["ASRError", "ASRSession", "build_asr"]
