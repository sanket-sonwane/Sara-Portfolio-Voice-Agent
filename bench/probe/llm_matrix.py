"""Compare the LLM candidates for the India-first profile.

The decision this probe informs is ADR 0002: which model answers a turn with the
lowest time-to-first-token from Mumbai. It measures the same short, grounded
question against every configured provider and reports distributions, not a
single lucky run.

Providers:

* ``gemini``  Vertex/AI Studio ``gemini-2.5-flash-lite`` (native SSE)
* ``sarvam``  ``sarvam-30b`` via the OpenAI-compatible chat endpoint
* ``groq``    kept as the US comparison arm, never the default

Usage::

    uv run python -m bench.probe.llm_matrix --providers gemini sarvam --runs 7
    uv run python -m bench.probe.llm_matrix --list gemini
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
from functools import partial
from typing import Any

import httpx

from bench.probe.common import Result, format_table, load_env, now, optional_env, require_env

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
SARVAM_BASE = "https://api.sarvam.ai"
GROQ_BASE = "https://api.groq.com/openai/v1"

SYSTEM_PROMPT = (
    "You are Sara, the voice agent for Sanket Sonwane's engineering portfolio. "
    "Answer only from the provided facts, in at most forty spoken words, and say "
    "plainly when something is not documented."
)

QUESTION = "What did Sanket build during the ARAI internship, and what changed?"

FACTS = (
    "FACT: ARAI internship. Role: Python Developer Intern. Work: built data "
    "pipelines and automation for automotive test workflows."
)

MAX_TOKENS = 120


def _payload() -> dict[str, Any]:
    return {
        "model": "",
        "messages": [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{FACTS}"},
            {"role": "user", "content": QUESTION},
        ],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }


async def _check(response: httpx.Response) -> None:
    """Fail with the body attached, which a streaming response has not read yet."""
    if response.status_code < 400:
        return
    body = (await response.aread()).decode("utf-8", "replace")[:200]
    raise httpx.HTTPStatusError(
        f"HTTP {response.status_code}: {body}",
        request=response.request,
        response=response,
    )


async def measure_openai(
    *,
    url: str,
    headers: dict[str, str],
    model: str,
    client: httpx.AsyncClient,
) -> dict[str, Any]:
    """OpenAI-compatible streaming (Sarvam, Groq)."""
    payload = _payload()
    payload["model"] = model
    started = now()
    first: float | None = None
    text: list[str] = []
    async with client.stream("POST", url, json=payload, headers=headers) as response:
        await _check(response)
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            body = line[6:].strip()
            if body == "[DONE]":
                break
            event = json.loads(body)
            choices = event.get("choices") or []
            if not choices:
                continue
            content = (choices[0].get("delta") or {}).get("content")
            if content:
                if first is None:
                    first = now()
                text.append(content)
    finished = now()
    return {
        "ttft_ms": (first - started) * 1000 if first else float("nan"),
        "total_ms": (finished - started) * 1000,
        "chars": len("".join(text)),
        "text": "".join(text)[:160],
    }


async def measure_gemini(
    *,
    api_key: str,
    model: str,
    base_url: str,
    client: httpx.AsyncClient,
) -> dict[str, Any]:
    """Gemini native SSE stream."""
    url = f"{base_url.rstrip('/')}/models/{model}:streamGenerateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": f"{SYSTEM_PROMPT}\n\n{FACTS}"}]},
        "contents": [{"role": "user", "parts": [{"text": QUESTION}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": MAX_TOKENS},
    }
    started = now()
    first: float | None = None
    text: list[str] = []
    async with client.stream(
        "POST",
        url,
        params={"alt": "sse", "key": api_key},
        json=payload,
        headers={"Content-Type": "application/json"},
    ) as response:
        await _check(response)
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            body = line[6:].strip()
            if not body or body == "[DONE]":
                continue
            event = json.loads(body)
            for candidate in event.get("candidates") or []:
                for part in (candidate.get("content") or {}).get("parts") or []:
                    chunk = part.get("text")
                    if chunk:
                        if first is None:
                            first = now()
                        text.append(chunk)
    finished = now()
    return {
        "ttft_ms": (first - started) * 1000 if first else float("nan"),
        "total_ms": (finished - started) * 1000,
        "chars": len("".join(text)),
        "text": "".join(text)[:160],
    }


async def list_gemini(api_key: str, base_url: str) -> int:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{base_url.rstrip('/')}/models", params={"key": api_key})
        await _check(response)
        data = response.json().get("models", [])
    rows = [
        {
            "id": model.get("name", "?").removeprefix("models/"),
            "input_tokens": model.get("inputTokenLimit", "?"),
            "methods": ",".join(model.get("supportedGenerationMethods") or []),
        }
        for model in data
    ]
    print(format_table(rows, ["id", "input_tokens", "methods"]))
    return 0


async def run(args: argparse.Namespace) -> int:
    result = Result(probe="llm-matrix", target="gemini+sarvam+groq", label=args.label)
    result.metrics["runs"] = args.runs
    rows: list[dict[str, Any]] = []

    gemini_key = optional_env("GEMINI_API_KEY") or optional_env("GOOGLE_API_KEY")
    sarvam_key = optional_env("SARVAM_API_KEY")
    groq_key = optional_env("GROQ_API_KEY")

    async with httpx.AsyncClient(timeout=60) as client:
        for provider in args.providers:
            if provider == "gemini":
                if not gemini_key:
                    result.note("gemini skipped: GEMINI_API_KEY is not set")
                    continue
                model = args.gemini_model or optional_env("GEMINI_MODEL", "gemini-2.5-flash-lite")
                base = optional_env("GEMINI_BASE_URL", GEMINI_BASE)
                runner = partial(
                    measure_gemini,
                    api_key=gemini_key,
                    model=model,
                    base_url=base,
                    client=client,
                )
                label = f"gemini/{model}"
            elif provider == "sarvam":
                if not sarvam_key:
                    result.note("sarvam skipped: SARVAM_API_KEY is not set")
                    continue
                model = args.sarvam_model or optional_env("SARVAM_LLM_MODEL", "sarvam-30b")
                base = optional_env("SARVAM_BASE_URL", SARVAM_BASE)
                headers = {
                    "api-subscription-key": sarvam_key,
                    "Authorization": f"Bearer {sarvam_key}",
                    "Content-Type": "application/json",
                }
                runner = partial(
                    measure_openai,
                    url=f"{base.rstrip('/')}/v1/chat/completions",
                    headers=headers,
                    model=model,
                    client=client,
                )
                label = f"sarvam/{model}"
            elif provider == "groq":
                if not groq_key:
                    result.note("groq skipped: GROQ_API_KEY is not set")
                    continue
                model = args.groq_model or optional_env("GROQ_MODEL", "llama-3.1-8b-instant")
                base = optional_env("GROQ_BASE_URL", GROQ_BASE)
                headers = {
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json",
                }
                runner = partial(
                    measure_openai,
                    url=f"{base.rstrip('/')}/chat/completions",
                    headers=headers,
                    model=model,
                    client=client,
                )
                label = f"groq/{model}"
            else:
                result.note(f"unknown provider {provider!r}")
                continue

            ttfts: list[float] = []
            totals: list[float] = []
            for index in range(args.runs + 1):
                try:
                    run = await runner()
                except httpx.HTTPStatusError as error:
                    with contextlib.suppress(Exception):
                        await error.response.aread()
                    body = error.response.text[:200]
                    result.note(f"{label}: HTTP {error.response.status_code} {body}")
                    print(f"skip {label}: HTTP {error.response.status_code}")
                    break
                except Exception as error:  # noqa: BLE001 - a probe reports, never crashes
                    result.note(f"{label}: {type(error).__name__}: {error}")
                    print(f"skip {label}: {type(error).__name__}")
                    break
                if index == 0:
                    # Warm-up: keep connection setup out of the distribution.
                    continue
                ttfts.append(run["ttft_ms"])
                totals.append(run["total_ms"])
                result.add(f"{label} ttft", run["ttft_ms"])
                result.add(f"{label} total", run["total_ms"])

            if ttfts:
                ordered = sorted(ttfts)
                rows.append(
                    {
                        "provider": label,
                        "ttft_p50": round(ordered[len(ordered) // 2], 1),
                        "ttft_p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 1),
                        "total_p50": round(sorted(totals)[len(totals) // 2], 1),
                        "n": len(ttfts),
                    }
                )

    print()
    print(format_table(rows, ["provider", "ttft_p50", "ttft_p95", "total_p50", "n"]))
    path = result.write()
    print(f"\nJSON: {path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--providers", nargs="+", default=["gemini", "sarvam"])
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--gemini-model", default="")
    parser.add_argument("--sarvam-model", default="")
    parser.add_argument("--groq-model", default="")
    parser.add_argument("--label", default="local")
    parser.add_argument("--list", choices=["gemini"], help="List models and exit")
    return parser.parse_args()


def main() -> int:
    load_env()
    args = parse_args()
    if args.list == "gemini":
        key = require_env("GEMINI_API_KEY")
        return asyncio.run(list_gemini(key, optional_env("GEMINI_BASE_URL", GEMINI_BASE)))
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
