"""Measure Groq streaming latency: TTFT, inter-token pacing and tool-call decisions.

Phase 0 questions this probe answers:

* What TTFT does the fast model tier actually give us from this machine/region?
* How much does a longer system prompt really cost? (the "keep the prompt short"
  decision should be a measurement, not an opinion)
* How quickly does a tool-call decision arrive, versus plain content?

Usage::

    uv run python -m bench.probe.groq_ttft --list-models
    uv run python -m bench.probe.groq_ttft --runs 7
    uv run python -m bench.probe.groq_ttft --models llama-3.1-8b-instant --sizes 150 900
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import httpx

from bench.probe.common import (
    Result,
    format_table,
    load_env,
    now,
    require_env,
)

API_ROOT = "https://api.groq.com/openai/v1"
CHAT_URL = f"{API_ROOT}/chat/completions"
MODELS_URL = f"{API_ROOT}/models"

# Default fast tier plus one stronger tier. Override with --models; use
# --list-models first, because availability changes per account and month.
DEFAULT_MODELS = ["llama-3.1-8b-instant", "openai/gpt-oss-20b"]
DEFAULT_SIZES = [150, 450, 900]

FILLER = (
    "Sara is a browser based voice agent. It streams microphone audio to a speech "
    "recognizer, retrieves approved portfolio evidence, generates a short conversational "
    "answer and synthesizes speech incrementally. "
)

BASE_PROMPT = (
    "You are Sara, the voice agent for Sanket Sonwane's engineering portfolio. "
    "Answer only from provided evidence, keep replies under forty spoken words, and say "
    "plainly when something is not documented. "
)

QUESTION = "What did Sanket actually build during the ARAI internship, and what changed?"

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_portfolio",
            "description": (
                "Search approved portfolio evidence about projects, internships, "
                "research, education and skills."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for"},
                },
                "required": ["query"],
            },
        },
    }
]


def build_system_prompt(target_tokens: int) -> str:
    """Approximate a prompt of ``target_tokens`` (words ~ 1.35 tokens)."""
    target_words = max(20, int(target_tokens / 1.35))
    words = BASE_PROMPT.split()
    filler = FILLER.split()
    while len(words) < target_words:
        words.extend(filler)
    return " ".join(words[:target_words])


async def list_models(api_key: str) -> int:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            MODELS_URL, headers={"Authorization": f"Bearer {api_key}"}
        )
        response.raise_for_status()
        data = response.json().get("data", [])
    rows = [
        {
            "id": model.get("id", "?"),
            "context": model.get("context_window") or model.get("context_length") or "?",
            "owned_by": model.get("owned_by", "?"),
        }
        for model in sorted(data, key=lambda item: item.get("id", ""))
    ]
    print(format_table(rows, ["id", "context", "owned_by"]))
    print(f"\n{len(rows)} models visible to this account")
    return 0


async def measure_once(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    question: str = QUESTION,
    use_tools: bool = False,
    max_tokens: int = 120,
    cold_start: bool = False,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """One streaming request. Returns latency marks in milliseconds."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if use_tools:
        payload["tools"] = TOOLS
        payload["tool_choice"] = "auto"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    owned_client = client is None or cold_start
    active = httpx.AsyncClient(timeout=60) if owned_client else client

    marks: dict[str, float] = {}
    text: list[str] = []
    usage: dict[str, Any] = {}
    first_delta_is_tool = False
    status = 0
    try:
        started = now()
        assert active is not None
        async with active.stream("POST", CHAT_URL, json=payload, headers=headers) as response:
            status = response.status_code
            response.raise_for_status()
            marks["headers"] = now()
            first_any: float | None = None
            first_content: float | None = None
            last_delta: float | None = None

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                body = line[6:].strip()
                if body == "[DONE]":
                    break
                event = json.loads(body)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                has_tool = bool(delta.get("tool_calls"))
                content = delta.get("content")
                if has_tool or content:
                    stamp = now()
                    if first_any is None:
                        first_any = stamp
                        first_delta_is_tool = has_tool
                    if content:
                        if first_content is None:
                            first_content = stamp
                        text.append(content)
                    last_delta = stamp
        finished = now()
    finally:
        if owned_client:
            await active.aclose()  # type: ignore[union-attr]

    completion_tokens = usage.get("completion_tokens") or len(text)
    generation_window = (finished - first_any) if first_any else 0.0
    return {
        "status": status,
        "ttft_ms": (first_any - started) * 1000 if first_any else float("nan"),
        "headers_ms": (marks.get("headers", started) - started) * 1000,
        "first_content_ms": (first_content - started) * 1000 if first_content else None,
        "total_ms": (finished - started) * 1000,
        "generation_ms": (last_delta - first_any) * 1000 if first_any and last_delta else None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "tokens_per_second": (
            completion_tokens / generation_window if generation_window > 0 else None
        ),
        "first_delta_is_tool": first_delta_is_tool,
        "text": "".join(text)[:200],
    }


async def run_matrix(args: argparse.Namespace) -> int:
    api_key = require_env(
        "GROQ_API_KEY", "Create a key at https://console.groq.com/keys"
    )
    result = Result(probe="groq-ttft", target=CHAT_URL, label=args.label)
    result.metrics["models"] = args.models
    result.metrics["prompt_token_targets"] = args.sizes
    result.metrics["runs_per_config"] = args.runs

    rows: list[dict[str, Any]] = []
    per_config: dict[str, dict[str, Any]] = {}

    async with httpx.AsyncClient(timeout=60) as client:
        for model in args.models:
            for size in args.sizes:
                for use_tools in (False, True) if args.tools else (False,):
                    config = f"{model} | p{size} | {'tools' if use_tools else 'plain'}"
                    system_prompt = build_system_prompt(size)
                    # One warm-up run keeps connection setup out of the samples.
                    try:
                        await measure_once(
                            api_key=api_key,
                            model=model,
                            system_prompt=system_prompt,
                            use_tools=use_tools,
                            client=client,
                        )
                    except httpx.HTTPStatusError as error:
                        result.note(f"{config}: warm-up failed {error.response.status_code}")
                        print(f"skip {config}: HTTP {error.response.status_code}")
                        continue

                    ttfts: list[float] = []
                    tps: list[float] = []
                    prompt_tokens = None
                    tool_first = 0
                    for _ in range(args.runs):
                        try:
                            run = await measure_once(
                                api_key=api_key,
                                model=model,
                                system_prompt=system_prompt,
                                use_tools=use_tools,
                                client=client,
                            )
                        except httpx.HTTPStatusError as error:
                            result.note(f"{config}: HTTP {error.response.status_code}")
                            continue
                        result.add(config, run["ttft_ms"])
                        ttfts.append(run["ttft_ms"])
                        if run["tokens_per_second"]:
                            tps.append(run["tokens_per_second"])
                        prompt_tokens = run["prompt_tokens"] or prompt_tokens
                        tool_first += int(run["first_delta_is_tool"])

                    if not ttfts:
                        continue
                    ordered = sorted(ttfts)
                    per_config[config] = {
                        "prompt_tokens": prompt_tokens,
                        "ttft_p50": ordered[len(ordered) // 2],
                    }
                    rows.append(
                        {
                            "config": config,
                            "prompt_tok": prompt_tokens or "?",
                            "ttft_p50": round(ordered[len(ordered) // 2], 1),
                            "ttft_max": round(ordered[-1], 1),
                            "tps_p50": round(sorted(tps)[len(tps) // 2], 1) if tps else "?",
                            "tool_first": f"{tool_first}/{len(ttfts)}" if use_tools else "-",
                        }
                    )

        # Cold start: fresh client per run, so DNS + TCP + TLS are included.
        if args.cold:
            try:
                cold = await measure_once(
                    api_key=api_key,
                    model=args.models[0],
                    system_prompt=build_system_prompt(450),
                    cold_start=True,
                )
                result.metrics["cold_start_ttft_ms"] = round(cold["ttft_ms"], 1)
                result.note("cold_start includes DNS, TCP and TLS setup in TTFT")
            except httpx.HTTPStatusError as error:
                result.note(f"cold start failed: HTTP {error.response.status_code}")

    # Prefill cost: TTFT growth across prompt sizes, per model.
    for model in args.models:
        low = per_config.get(f"{model} | p{args.sizes[0]} | plain")
        high = per_config.get(f"{model} | p{args.sizes[-1]} | plain")
        if low and high:
            delta = high["ttft_p50"] - low["ttft_p50"]
            result.metrics[f"prefill_delta_ms[{model}]"] = round(delta, 1)
            result.note(
                f"{model}: {args.sizes[-1]} vs {args.sizes[0]} prompt tokens cost "
                f"{delta:+.1f} ms TTFT"
            )

    print()
    print(format_table(rows, ["config", "prompt_tok", "ttft_p50", "ttft_max", "tps_p50", "tool_first"]))
    path = result.write()
    print(f"\nJSON: {path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=DEFAULT_SIZES,
        help="Approximate system-prompt sizes in tokens",
    )
    parser.add_argument("--runs", type=int, default=5, help="Samples per configuration")
    parser.add_argument("--tools", action="store_true", help="Also measure the tool-call path")
    parser.add_argument("--cold", action="store_true", help="Also measure a cold connection")
    parser.add_argument("--label", default="local", help="Region/machine label for the JSON")
    parser.add_argument("--list-models", action="store_true")
    return parser.parse_args()


def main() -> int:
    load_env()
    args = parse_args()
    if args.list_models:
        return asyncio.run(list_models(require_env("GROQ_API_KEY")))
    return asyncio.run(run_matrix(args))


if __name__ == "__main__":
    raise SystemExit(main())
