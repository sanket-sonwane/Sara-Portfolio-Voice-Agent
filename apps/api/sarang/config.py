"""Configuration for the Sarang gateway.

Everything provider-specific lives here so a region can be switched by changing
one environment variable. No module hardcodes a base URL, a model id or a voice.

Profiles
--------
``bom-india``  (default) India-hosted providers. Lowest last-mile latency for an
               Indian visitor, which is the audience this portfolio is for.
``iad-global`` All providers in the US. Fastest per-component stack, ~200 ms of
               extra last-mile from India. Kept for a future global deploy.
``local``      No network providers at all: deterministic local answerer and a
               synthetic voice. The whole pipeline runs with zero API keys.

Any individual provider can be pinned with an environment variable, so a profile
is a starting point rather than a cage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
KB_DIR = REPO_ROOT / "kb"
DOCS_DIR = KB_DIR / "docs"


@dataclass(frozen=True)
class ProviderProfile:
    name: str
    asr: str
    llm: str
    tts: str
    note: str


PROFILES: dict[str, ProviderProfile] = {
    "bom-india": ProviderProfile(
        name="bom-india",
        asr="sarvam",
        llm="sarvam",
        tts="cartesia",
        note=(
            "India-first: Sarvam ASR and LLM in-region; Cartesia TTS for the "
            "lowest measured time-to-first-audio."
        ),
    ),
    "iad-global": ProviderProfile(
        name="iad-global",
        asr="nemotron",
        llm="groq",
        tts="cartesia",
        note="All-US stack. Best per-component latency, worst last mile from India.",
    ),
    "local": ProviderProfile(
        name="local",
        asr="none",
        llm="local",
        tts="synthetic",
        note="No network providers; deterministic local pipeline.",
    ),
}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


@dataclass
class Settings:
    """Resolved runtime settings.

    ``resolve_settings`` is the only constructor the application uses, so tests can
    build a settings object from a dict of environment variables without touching
    process state.
    """

    profile: ProviderProfile
    env: str = "dev"

    # Auth
    session_secret: str = "dev-only-secret"
    session_ttl_s: int = 3600
    allow_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173"])

    # Media
    enable_webrtc: bool = True
    ice_servers: list[dict] = field(default_factory=list)
    force_relay: bool = False

    # Providers
    asr_provider: str = "sarvam"
    llm_provider: str = "gemini"
    tts_provider: str = "sarvam"

    sarvam_key: str = ""
    sarvam_llm_model: str = "sarvam-105b-conversations"
    sarvam_asr_model: str = "saaras:v4"
    # Realtime ASR tuning. ``fast`` emits the first partial roughly a second
    # earlier than ``balanced`` and produced the same final transcript on the
    # fixture set, so the live transcript (which is what the visitor reads while
    # still speaking) wins. The answer is only ever built from the final.
    sarvam_asr_language: str = "en-IN"
    sarvam_asr_stream_type: str = "fast"
    sarvam_asr_mode: str = "transcribe"
    # Sarvam's own guidance is 100-500 ms chunks. 20 ms frames are aggregated in
    # the adapter so the recogniser sees a stable block size.
    sarvam_asr_chunk_ms: int = 100
    sarvam_tts_model: str = "bulbul:v4-flash"
    sarvam_voice: str = "dev_en_conversational"
    sarvam_base_url: str = "https://api.sarvam.ai"

    gemini_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    groq_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    nvidia_key: str = ""
    nvidia_asr_url: str = "https://integrate.api.nvidia.com/v1/audio/transcriptions"
    nvidia_model: str = "nvidia/nemotron-speech-streaming-en-0.6b"

    cartesia_key: str = ""
    cartesia_model: str = "sonic-latest"
    cartesia_voice: str = ""
    # Cartesia counts concurrent contexts against the account. The lowest paid
    # tier allows 2; submitting every answer segment at once exceeded it and the
    # fallback voice leaked into the middle of answers.
    tts_max_concurrency: int = 2
    cartesia_ws_url: str = "wss://api.cartesia.ai/tts/websocket"
    cartesia_http_url: str = "https://api.cartesia.ai"

    # Retrieval
    database_url: str = ""
    retrieval_backend: str = ""

    # Writable cache (pre-rendered acknowledgements). Empty means the repository
    # default, ``<repo>/cache``.
    cache_dir: str = ""

    # Behaviour
    # Endpointing is the single biggest cause of "it talks over me": an
    # energy VAD hears the tail of a word as silence, so a 500 ms window closes
    # the turn mid-sentence. 800 ms is the conversational floor for accented
    # speech; the pre-roll buffer in the utterance loop makes the extra wait
    # invisible (no audio is lost at the start of the next utterance).
    endpoint_silence_ms: int = 800
    endpoint_min_speech_ms: int = 120
    endpoint_hangover_ms: int = 250
    # Audio sent to the recogniser, in seconds, kept from before the VAD fires.
    endpoint_preroll_ms: int = 800
    # How long to wait for ``transcript.final`` before falling back to the last
    # partial. Answering from a partial is the last resort, not the default.
    asr_final_timeout_s: float = 3.5
    # Debug aid: write every utterance's exact 16 kHz PCM plus the transcript to
    # this directory. Empty disables it.
    asr_dump_dir: str = ""
    speculative_ms: int = 0
    max_answer_tokens: int = 160
    # Hard ceiling on what is actually spoken. The prompt asks for 35 words; this
    # is the enforcement, because a model that ignores the request would otherwise
    # monologue for twenty seconds. Whole sentences only: the segmenter emits them.
    max_spoken_words: int = 50
    filler_enabled: bool = True

    # Abuse limits. A public demo URL is an open door to the provider bill.
    max_sessions: int = 200
    session_rate_limit: int = 12  # new sessions per minute per client IP
    turn_rate_limit: int = 20  # turns per minute per session

    def provider_ready(self, name: str | None = None) -> bool:
        which = name or self.llm_provider
        return {
            "local": True,
            "synthetic": True,
            "none": False,
            "sarvam": bool(self.sarvam_key),
            "gemini": bool(self.gemini_key),
            "groq": bool(self.groq_key),
            "nemotron": bool(self.nvidia_key),
            "cartesia": bool(self.cartesia_key),
        }.get(which, False)

    def describe(self) -> dict:
        return {
            "profile": self.profile.name,
            "profile_note": self.profile.note,
            "asr": self.asr_provider,
            "asr_ready": self.provider_ready(self.asr_provider),
            "llm": self.llm_provider,
            "llm_ready": self.provider_ready(self.llm_provider),
            "tts": self.tts_provider,
            "tts_ready": self.provider_ready(self.tts_provider),
            "webrtc": self.enable_webrtc,
            "retrieval": self.retrieval_backend
            or ("pgvector" if self.database_url else "bm25-local"),
        }


def resolve_settings(env: dict[str, str] | None = None) -> Settings:
    """Build settings from ``env`` (defaults to ``os.environ``).

    A provider whose key is missing degrades to the local adapter instead of
    crashing, so the application is always startable. The profile name is still
    reported, and ``/healthz`` says which stages are real.
    """
    source = os.environ if env is None else env

    def get(name: str, default: str = "") -> str:
        return source.get(name, default).strip()

    def flag(name: str, default: bool) -> bool:
        raw = get(name)
        if not raw:
            return default
        return raw.lower() in {"1", "true", "yes", "on"}

    def integer(name: str, default: int) -> int:
        raw = get(name)
        try:
            return int(raw) if raw else default
        except ValueError:
            return default

    def number(name: str, default: float) -> float:
        raw = get(name)
        try:
            return float(raw) if raw else default
        except ValueError:
            return default

    profile_name = get("SARANG_PROFILE") or ("local" if get("SARANG_OFFLINE") else "bom-india")
    profile = PROFILES.get(profile_name, PROFILES["bom-india"])

    origins = [item.strip() for item in get("ALLOWED_ORIGINS").split(",") if item.strip()]
    if not origins:
        origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "https://sanket--sonwane.vercel.app",
        ]

    ice_raw = get("ICE_SERVERS")
    ice_servers: list[dict] = []
    if ice_raw:
        for entry in ice_raw.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            if "|" in entry:
                urls, credential = entry.split("|", 1)
                username, _, secret = credential.partition(":")
                ice_servers.append(
                    {"urls": [u.strip() for u in urls.split(",") if u.strip()],
                     "username": username, "credential": secret}
                )
            else:
                ice_servers.append({"urls": [u.strip() for u in entry.split(",") if u.strip()]})
    if not ice_servers:
        ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

    settings = Settings(
        profile=profile,
        env=get("SARANG_ENV", "dev"),
        session_secret=get("SESSION_SECRET", "dev-only-secret"),
        session_ttl_s=integer("SESSION_TTL_S", 3600),
        allow_origins=origins,
        enable_webrtc=flag("ENABLE_WEBRTC", True),
        ice_servers=ice_servers,
        force_relay=flag("ICE_FORCE_RELAY", False),
        asr_provider=get("ASR_PROVIDER") or profile.asr,
        llm_provider=get("LLM_PROVIDER") or profile.llm,
        tts_provider=get("TTS_PROVIDER") or profile.tts,
        sarvam_key=get("SARVAM_API_KEY"),
        sarvam_llm_model=get("SARVAM_LLM_MODEL", "sarvam-105b-conversations"),
        sarvam_asr_model=get("SARVAM_ASR_MODEL", "saaras:v4"),
        sarvam_asr_language=get("SARVAM_ASR_LANGUAGE", "en-IN"),
        sarvam_asr_stream_type=get("SARVAM_ASR_STREAM_TYPE", "fast"),
        sarvam_asr_mode=get("SARVAM_ASR_MODE", "transcribe"),
        sarvam_asr_chunk_ms=integer("SARVAM_ASR_CHUNK_MS", 100),
        sarvam_tts_model=get("SARVAM_TTS_MODEL", "bulbul:v4-flash"),
        sarvam_voice=get("SARVAM_VOICE", "dev_en_conversational"),
        sarvam_base_url=get("SARVAM_BASE_URL", "https://api.sarvam.ai"),
        gemini_key=get("GEMINI_API_KEY") or get("GOOGLE_API_KEY"),
        gemini_model=get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        gemini_base_url=get(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
        ),
        groq_key=get("GROQ_API_KEY"),
        groq_model=get("GROQ_MODEL", "openai/gpt-oss-20b"),
        groq_base_url=get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        nvidia_key=get("NVIDIA_API_KEY"),
        nvidia_asr_url=get(
            "NVIDIA_ASR_URL", "https://integrate.api.nvidia.com/v1/audio/transcriptions"
        ),
        nvidia_model=get("NVIDIA_ASR_MODEL", "nvidia/nemotron-speech-streaming-en-0.6b"),
        cartesia_key=get("CARTESIA_API_KEY"),
        cartesia_model=get("CARTESIA_MODEL", "sonic-latest"),
        cartesia_voice=get("CARTESIA_VOICE_ID"),
        tts_max_concurrency=integer("TTS_MAX_CONCURRENCY", 2),
        cartesia_ws_url=get("CARTESIA_WS_URL", "wss://api.cartesia.ai/tts/websocket"),
        cartesia_http_url=get("CARTESIA_BASE_URL", "https://api.cartesia.ai"),
        database_url=get("DATABASE_URL"),
        cache_dir=get("SARANG_CACHE_DIR"),
        endpoint_silence_ms=integer("ENDPOINT_SILENCE_MS", 800),
        endpoint_min_speech_ms=integer("ENDPOINT_MIN_SPEECH_MS", 120),
        endpoint_hangover_ms=integer("ENDPOINT_HANGOVER_MS", 250),
        endpoint_preroll_ms=integer("ENDPOINT_PREROLL_MS", 800),
        asr_final_timeout_s=number("ASR_FINAL_TIMEOUT_S", 3.5),
        asr_dump_dir=get("ASR_DUMP_DIR"),
        speculative_ms=integer("SPECULATIVE_MS", 0),
        max_answer_tokens=integer("MAX_ANSWER_TOKENS", 160),
        max_spoken_words=integer("MAX_SPOKEN_WORDS", 50),
        filler_enabled=flag("FILLER_ENABLED", True),
        max_sessions=integer("MAX_SESSIONS", 200),
        session_rate_limit=integer("SESSION_RATE_LIMIT", 12),
        turn_rate_limit=integer("TURN_RATE_LIMIT", 20),
    )

    # Missing credentials degrade the stage, never the process.
    if not settings.provider_ready(settings.llm_provider):
        settings.llm_provider = "local"
    if not settings.provider_ready(settings.tts_provider):
        settings.tts_provider = "synthetic"
    if not settings.provider_ready(settings.asr_provider):
        settings.asr_provider = "none"

    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except ModuleNotFoundError:  # pragma: no cover
        pass
    return resolve_settings()
