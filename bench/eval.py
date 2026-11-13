"""Grounded evaluation runner for Sara.

Runs ``bench/questions/questions.yaml`` against a live gateway over HTTP + the
events WebSocket, grades each answer with the same substring rules the manual
pass uses, and writes a JSON report.

    python bench/eval.py --base http://127.0.0.1:8080
    python bench/eval.py --group integrity --only i02 --verbose

This is an honesty tool, not a marketing one. It reports refusals and false
premises separately from ordinary answers, and it stores the raw answer next to
the grade so a human can overrule the substring check.

Start the server with ``SESSION_RATE_LIMIT=0`` for a full run; otherwise the
runner waits for the 429 window like any other client.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

try:  # websockets >= 13
    from websockets.asyncio.client import connect as ws_connect
except ImportError:  # pragma: no cover
    from websockets import connect as ws_connect  # type: ignore

BENCH_DIR = Path(__file__).resolve().parent
QUESTIONS = BENCH_DIR / "questions" / "questions.yaml"
RESULTS = BENCH_DIR / "results"

# A refusal must say that the portfolio does not cover the thing, not merely sound
# evasive. These are the phrases the prompt and the local answerer are designed to
# produce.
REFUSAL_MARKERS = (
    "don't have that detail",
    "do not have that detail",
    "don't have any detail",
    "don't have any information",
    "don't have information",
    "don't have details",
    "do not have details",
    "isn't documented",
    "is not documented",
    "not documented",
    "doesn't document",
    "does not document",
    "not in his portfolio",
    "not in the portfolio",
    "his portfolio does not",
    "portfolio does not",
    "can't quote",
    "cannot quote",
    "don't discuss",
    "won't share",
    "rather not speculate",
    "no google",
    "no patents",
    "outside what i can cover",
    "portfolio does not cover",
    "portfolio documents",
    "not something i can",
)

DECLINE_MARKERS = REFUSAL_MARKERS + (
    "sorry",
    "i can only speak",
    "only speak to him",
    "happy to talk about",
    "steer",
)


@dataclass
class Case:
    id: str
    group: str
    expect: str
    turns: list[str]
    must_mention: list[str] = field(default_factory=list)
    must_not_mention: list[str] = field(default_factory=list)
    note: str = ""


def load_cases(path: Path = QUESTIONS) -> list[Case]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases: list[Case] = []
    for group in raw.get("groups", []):
        group_id = str(group.get("id", "?"))
        for item in group.get("questions", []):
            cases.append(_case(group_id, item))
        for item in group.get("turns", []):
            cases.append(_case(group_id, item))
    return cases


def _case(group_id: str, item: dict) -> Case:
    turns = item.get("turns") or [item.get("question", "")]
    return Case(
        id=str(item.get("id", "?")),
        group=group_id,
        expect=str(item.get("expect", "answer")),
        turns=[str(turn) for turn in turns if str(turn).strip()],
        must_mention=[str(term).lower() for term in item.get("must_mention") or []],
        must_not_mention=[str(term).lower() for term in item.get("must_not_mention") or []],
        note=str(item.get("note", "")),
    )


def _normalize_for_match(text: str) -> str:
    # Treat hyphens as spaces and collapse whitespace so "rogue-node" matches
    # "rogue node" and number formatting is a bit more forgiving.
    return " ".join(text.lower().replace("-", " ").replace("_", " ").split())


def _mentions(text: str, term: str) -> bool:
    norm = _normalize_for_match(text)
    needle = _normalize_for_match(term)
    if needle in norm:
        return True
    # Accept common spoken/number variants the eval uses as digit strings.
    aliases = {
        "1300": ("1,300", "one thousand three hundred", "thousand three hundred"),
        "1,300": ("1300", "one thousand three hundred"),
        "100": ("one hundred", "hundred weekly"),
        "10 hours": ("ten hours",),
        "10 seconds": ("ten seconds",),
        "98": ("ninety eight", "ninety-eight", "98 percent", "98%"),
        "50": ("fifty", "50 millisecond", "50ms"),
        "9.14": ("nine point one four", "9 point 14"),
        "85.13": ("eighty five point one three", "eight five point one three", "85 point 13"),
        "rag": ("retrieval augmented", "retrieval-augmented"),
    }
    for alt in aliases.get(term.lower(), ()):
        if _normalize_for_match(alt) in norm:
            return True
    return False


def grade(case: Case, answer: str) -> tuple[bool, list[str]]:
    text = answer.lower()
    reasons: list[str] = []
    for term in case.must_mention:
        if not _mentions(text, term):
            reasons.append(f"missing: {term}")
    for term in case.must_not_mention:
        if _mentions(text, term):
            reasons.append(f"forbidden: {term}")

    has_refusal = any(_normalize_for_match(marker) in _normalize_for_match(text)
                      for marker in REFUSAL_MARKERS)
    has_decline = any(_normalize_for_match(marker) in _normalize_for_match(text)
                      for marker in DECLINE_MARKERS)

    if case.expect == "answer" and has_refusal and not case.must_mention:
        # A bare refusal to an answerable question. If the answer still names the
        # required facts, treat the "I don't have more" tail as honest hedging.
        reasons.append("refused an answerable question")
    elif case.expect == "refuse" and not has_refusal:
        reasons.append("did not say the portfolio lacks this")
    elif case.expect == "correct":
        if not answer.strip():
            reasons.append("empty answer")
    elif case.expect == "decline" and not has_decline:
        reasons.append("did not decline out of scope")
    # "partial" accepts either an answer or a named gap.

    return not reasons, reasons


@dataclass
class TurnResult:
    answer: str
    metrics: dict
    elapsed_ms: float


async def _wait_for_turn(ws, turn_id: str, timeout: float) -> TurnResult:
    deadline = time.monotonic() + timeout
    answer = ""
    metrics: dict = {}
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, remaining))
        try:
            event = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(event, dict) or "type" not in event:
            continue
        if event.get("turn_id") != turn_id:
            continue
        if event["type"] == "assistant.final":
            answer = str(event.get("payload", {}).get("text", ""))
        elif event["type"] == "metrics.turn":
            metrics = dict(event.get("payload", {}))
            return TurnResult(answer=answer, metrics=metrics, elapsed_ms=0.0)
    raise TimeoutError(f"no metrics.turn for {turn_id} within {timeout:.0f}s")


async def run_case(
    client: httpx.AsyncClient,
    base: str,
    ws_base: str,
    case: Case,
    *,
    timeout: float,
    gap: float,
) -> TurnResult:
    session = await _post_session(client, base)
    session_id = session["session_id"]
    token = session["token"]
    ws_url = f"{ws_base}/ws/events?session_id={session_id}&token={token}"
    started = time.monotonic()
    result: TurnResult | None = None
    async with ws_connect(ws_url, open_timeout=10, max_size=None) as ws:
        for question in case.turns:
            turn_id = await _post_turn(client, base, session_id, token, question)
            result = await _wait_for_turn(ws, turn_id, timeout)
            await asyncio.sleep(gap)
    assert result is not None
    result.elapsed_ms = round((time.monotonic() - started) * 1000.0, 1)
    await client.post(f"{base}/api/interrupt", json={"session_id": session_id, "token": token})
    return result


async def _post_session(client: httpx.AsyncClient, base: str) -> dict:
    for _attempt in range(8):
        response = await client.post(f"{base}/api/session", json={"client": "bench"})
        if response.status_code == 429:
            wait = float(response.headers.get("retry-after", "10"))
            print(f"  rate limited; waiting {wait:.0f}s", file=sys.stderr)
            await asyncio.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("session creation kept returning 429")


async def _post_turn(
    client: httpx.AsyncClient, base: str, session_id: str, token: str, text: str
) -> str:
    response = await client.post(
        f"{base}/api/turn/text",
        json={"session_id": session_id, "token": token, "text": text},
    )
    response.raise_for_status()
    return response.json()["turn_id"]


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return round(ordered[index], 1)


def _collect(records: list[dict], key: str) -> list[float]:
    values = [float(record["metrics"].get(key) or 0.0) for record in records]
    return [value for value in values if value > 0]


def _latency(values: list[float]) -> dict[str, float]:
    return {
        "p50": _percentile(values, 0.5),
        "p90": _percentile(values, 0.9),
        "p95": _percentile(values, 0.95),
    }


def summarise(records: list[dict]) -> dict:
    passed = [record for record in records if record["passed"]]
    by_group: dict[str, dict[str, int]] = {}
    for record in records:
        bucket = by_group.setdefault(record["group"], {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(record["passed"])
    return {
        "total": len(records),
        "passed": len(passed),
        "pass_rate": round(len(passed) / len(records), 3) if records else 0.0,
        "by_group": {
            group: {**counts, "rate": round(counts["passed"] / counts["total"], 3)}
            for group, counts in sorted(by_group.items())
        },
        "latency_ms": {
            # Primary: last speech frame -> the first answer frame actually
            # written to the wire. Text turns have no browser leg.
            "eos_to_transport_first_frame": _latency(
                _collect(records, "eos_to_transport_first_frame_ms")
            ),
            # Browser-confirmed, when a client sent the audible report. Absent
            # for the HTTP bench, which has no playback device.
            "eos_to_client_audible": _latency(
                _collect(records, "eos_to_client_audible_ms")
            ),
            # Provider output, kept for comparison and labelled as such: this
            # is when the first synthesized span arrived, not when the visitor
            # heard anything.
            "answer_first_audio_provider_output": _latency(
                _collect(records, "answer_first_audio_ms")
            ),
            "llm_ttft_median": round(
                statistics.median(
                    [
                        float(
                            record["metrics"].get("durations_ms", {}).get("llm_ttft", 0)
                            or 0
                        )
                        for record in records
                    ]
                ),
                1,
            )
            if records
            else 0.0,
            "degraded_turns": sum(
                1 for record in records if record["metrics"].get("degraded")
            ),
        },
    }


async def main_async(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.questions))
    if args.group:
        cases = [case for case in cases if case.group in set(args.group)]
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        cases = [case for case in cases if case.id in wanted]
    if not cases:
        print("no cases matched", file=sys.stderr)
        return 2

    base = args.base.rstrip("/")
    ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
    records: list[dict] = []
    failures = 0

    async with httpx.AsyncClient(timeout=30) as client:
        health = await client.get(f"{base}/healthz")
        health.raise_for_status()
        stages = health.json().get("stages", {})
        print(
            f"Sara eval against {base} | profile={health.json().get('profile')} "
            f"llm={stages.get('llm')} tts={stages.get('tts')} asr={stages.get('asr')}"
        )
        for case in cases:
            try:
                result = await run_case(
                    client,
                    base,
                    ws_base,
                    case,
                    timeout=args.timeout,
                    gap=args.gap,
                )
                passed, reasons = grade(case, result.answer)
            except Exception as error:  # noqa: BLE001 - one bad case must not stop the run
                passed, reasons = False, [f"error: {error}"]
                result = TurnResult(answer="", metrics={}, elapsed_ms=0.0)
            failures += int(not passed)
            record = {
                "id": case.id,
                "group": case.group,
                "expect": case.expect,
                "turns": case.turns,
                "passed": passed,
                "reasons": reasons,
                "answer": result.answer,
                "metrics": result.metrics,
                "elapsed_ms": result.elapsed_ms,
            }
            records.append(record)
            mark = "PASS" if passed else "FAIL"
            print(f"  [{mark}] {case.id:<5} {case.turns[-1][:58]}")
            if args.verbose or not passed:
                print(f"         -> {result.answer[:220]!r}")
                if reasons:
                    print(f"         -> {', '.join(reasons)}")

    summary = summarise(records)
    primary = summary["latency_ms"]["eos_to_transport_first_frame"]
    confirmed = summary["latency_ms"]["eos_to_client_audible"]
    provider = summary["latency_ms"]["answer_first_audio_provider_output"]
    print(
        f"\n{summary['passed']}/{summary['total']} passed "
        f"({summary['pass_rate'] * 100:.1f}%) | "
        f"EOS→wire p50 {primary['p50']:.0f} ms, p90 {primary['p90']:.0f}, "
        f"p95 {primary['p95']:.0f}"
    )
    if confirmed["p50"]:
        print(
            f"  browser-confirmed EOS→audible p50 {confirmed['p50']:.0f} ms, "
            f"p90 {confirmed['p90']:.0f}, p95 {confirmed['p95']:.0f}"
        )
    print(
        f"  provider first audio (output, not audible) p50 {provider['p50']:.0f} ms, "
        f"p95 {provider['p95']:.0f} | degraded turns "
        f"{summary['latency_ms']['degraded_turns']}"
    )
    for group, counts in summary["by_group"].items():
        print(f"  {group:<14} {counts['passed']}/{counts['total']} ({counts['rate'] * 100:.0f}%)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else RESULTS / f"eval-{stamp}.json"
    payload = {
        "base": base,
        "generated_at": stamp,
        "health": health.json(),
        "summary": summary,
        "records": records,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 1 if failures and args.fail_on_error else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Sara's grounded eval set.")
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    parser.add_argument("--questions", default=str(QUESTIONS))
    parser.add_argument("--group", action="append", help="group id (repeatable)")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--out", default="")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--gap", type=float, default=0.3)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="exit non-zero when any case fails (for CI)",
    )
    return asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
