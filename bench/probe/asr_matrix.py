"""Sweep stream type and chunk size on one recogniser, and score what comes out.

Production runs ``stream_type=fast`` with 100 ms chunks - a latency knob that
has never been measured against accuracy (WER has never been scored anywhere;
FR-025 / PRD G-04 are open), and chunk size has only ever been measured for
*finalise* latency (``asr_final_tail.py``), never for how partials behave while
the visitor is still talking. This sweeps both axes on the same audio:

    {fast, balanced, simulated} x {20, 50, 100} ms

Per cell, three numbers and a transcript:

* ``connect_ms``   - socket handshake (amortised over the whole session).
* ``partial_ms``   - first ``transcript.partial`` relative to the first audio
                     push: how stale the partials are *during* speech.
* ``finalize_ms``  - ``speech_end`` to ``transcript.final``: the ``asr_final``
                     term of the post-speech budget.
* **WER** against the reference transcript, punctuation-insensitive.

Runs are sequential on purpose: Sarvam allows one realtime socket at a time.

WER granularity caveat: the default fixture is one 10-word sentence, so WER
moves in 10% steps and the PRD's <6% target means *word-perfect*. Treat a
non-zero WER here as a failure, not as a rate; score a rate once several
reference clips exist (see ``bench/audio/README.md``).

Usage::

    .venv\\Scripts\\python.exe -m bench.probe.asr_matrix
    .venv\\Scripts\\python.exe -m bench.probe.asr_matrix --runs 3 --stream-types fast balanced
    .venv\\Scripts\\python.exe -m bench.probe.asr_matrix --chunk-ms 100 --no-bias
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path

from sarang.asr.keyterms import build_keyterms, domain_prompt

from bench.probe.asr_realtime import read_pcm, run_realtime
from bench.probe.common import (
    AUDIO_DIR,
    format_table,
    load_env,
    optional_env,
    require_env,
    word_error_rate,
)

DEFAULT_REFERENCE = "What did Sanket build during the ARAI internship?"


def median(values: list[float]) -> str:
    if not values:
        return "-"
    return f"{statistics.median(values):.0f}"


async def run_cell(
    pcm: bytes, args: argparse.Namespace, keyterms: list[str], prompt: str,
    stream_type: str, chunk_ms: int,
) -> dict:
    """One sweep cell: a realtime run plus the three metrics we score."""
    row = {
        "stream": stream_type,
        "chunk": chunk_ms,
        "partial_ms": None,
        "finalize_ms": None,
        "partials": None,
        "wer": None,
        "error": "",
        "text": "",
    }
    try:
        result = await run_realtime(
            pcm,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            language=args.language,
            stream_type=stream_type,
            chunk_ms=chunk_ms,
            keyterms=keyterms,
            prompt=prompt,
        )
    except Exception as exc:  # noqa: BLE001 - a rejected config is a result
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        return row
    row["error"] = result.get("error") or ""
    row["text"] = result["text"]
    if row["error"]:
        return row
    row["wer"] = word_error_rate(args.reference, result["text"])
    row["partials"] = result.get("partials")
    first = result.get("first_partial_ms")
    if first is not None and result.get("connect_ms") is not None:
        row["partial_ms"] = max(0.0, first - result["connect_ms"])
    if result.get("final_ms") is not None and result.get("pump_ms") is not None:
        row["finalize_ms"] = max(0.0, result["final_ms"] - result["pump_ms"])
    return row


async def sweep(args: argparse.Namespace) -> list[dict]:
    load_env()
    args.api_key = require_env("SARVAM_API_KEY", "Create a key at https://dashboard.sarvam.ai")
    args.base_url = optional_env("SARVAM_BASE_URL", "https://api.sarvam.ai")

    if args.keyterms:
        keyterms = [term.strip() for term in args.keyterms.split(";") if term.strip()]
    else:
        keyterms = [] if args.no_bias else list(build_keyterms())
    prompt = "" if (args.no_bias or args.no_prompt) else domain_prompt()

    pcm = read_pcm(Path(args.audio))
    rows: list[dict] = []
    total = len(args.stream_types) * len(args.chunk_ms) * max(1, args.runs)
    index = 0
    for stream_type in args.stream_types:
        for chunk_ms in args.chunk_ms:
            for run in range(1, max(1, args.runs) + 1):
                index += 1
                print(f"[{index}/{total}] stream={stream_type} chunk={chunk_ms}ms run={run} ...",
                      end="", flush=True)
                row = await run_cell(pcm, args, keyterms, prompt, stream_type, chunk_ms)
                row["run"] = run
                if row["error"]:
                    print(f" {row['error'][:60]}")
                else:
                    print(f" WER {row['wer']:.0%} partials={row['partials']}")
                rows.append(row)
                # The plan limits how often a socket may be opened, not just how
                # many are open at once; wait it out and take the cell again.
                if row["error"] and "rate limit" in row["error"].lower():
                    print("    rate limited; waiting 25s and retrying once ...", flush=True)
                    await asyncio.sleep(25.0)
                    rows.pop()
                    retry = await run_cell(pcm, args, keyterms, prompt, stream_type, chunk_ms)
                    if retry["error"]:
                        print(f"    retry failed: {retry['error'][:60]}")
                    else:
                        print(f"    retry ok: WER {retry['wer']:.0%} partials={retry['partials']}")
                    rows.append(retry)
                await asyncio.sleep(max(0.0, args.pause))
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    cells: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        cells.setdefault((row["stream"], row["chunk"]), []).append(row)
    out = []
    for (stream, chunk), group in cells.items():
        good = [r for r in group if not r["error"]]
        out.append({
            "stream": stream,
            "chunk": chunk,
            "runs": len(group),
            "ok": len(good),
            "partial_ms": median([r["partial_ms"] for r in good if r["partial_ms"] is not None]),
            "finalize_ms": median([r["finalize_ms"] for r in good if r["finalize_ms"] is not None]),
            "wer": "-" if not good or any(r["wer"] is None for r in good)
                   else f"{statistics.median([r['wer'] for r in good]):.0%}",
            "text": (good[-1]["text"][:60] if good else ""),
            "error": (group[0]["error"][:60] if len(good) < len(group) else ""),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", default=str(AUDIO_DIR / "question-16k.wav"))
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--stream-types", nargs="+", default=["fast", "balanced", "simulated"])
    parser.add_argument("--chunk-ms", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--pause", type=float, default=2.0,
                        help="Seconds between runs; the plan rate-limits socket opens")
    parser.add_argument("--model", default=optional_env("SARVAM_ASR_MODEL", "saaras:v4"))
    parser.add_argument("--language", default=optional_env("SARVAM_ASR_LANGUAGE", "en-IN"))
    parser.add_argument("--no-bias", action="store_true", help="Disable keyterms/prompt")
    parser.add_argument("--no-prompt", action="store_true", help="Send keyterms but not the prompt")
    parser.add_argument("--keyterms", default="",
                        help="Semicolon-separated override instead of the KB-derived list")
    args = parser.parse_args()

    print(f"audio={args.audio} model={args.model} "
          f"reference={args.reference!r}")
    print(f"sweep: {args.stream_types} x {args.chunk_ms} ms x {args.runs} run(s)\n")

    started = time.perf_counter()
    rows = asyncio.run(sweep(args))
    print(f"\nsweep took {time.perf_counter() - started:.0f}s\n")

    print("per run:")
    print(format_table(
        [
            {
                "stream": r["stream"],
                "chunk": f"{r['chunk']}ms",
                "run": r.get("run", "-"),
                "partial": f"{r['partial_ms']:.0f}" if r["partial_ms"] is not None else "-",
                "finalize": f"{r['finalize_ms']:.0f}" if r["finalize_ms"] is not None else "-",
                "parts": r["partials"] if r["partials"] is not None else "-",
                "wer": f"{r['wer']:.0%}" if r["wer"] is not None else "-",
                "text": (r["error"] or r["text"])[:64],
            }
            for r in rows
        ],
        ["stream", "chunk", "run", "partial", "finalize", "parts", "wer", "text"],
    ))

    summary = summarise(rows)
    print("\nper cell (medians; production is fast/100ms):")
    print(format_table(
        [
            {
                "stream": c["stream"],
                "chunk": f"{c['chunk']}ms",
                "runs": f"{c['ok']}/{c['runs']}",
                "partial": c["partial_ms"],
                "finalize": c["finalize_ms"],
                "wer": c["wer"],
                "note": c["error"] or ("PROD" if c["stream"] == "fast" and c["chunk"] == 100 else ""),
                "text": c["text"],
            }
            for c in summary
        ],
        ["stream", "chunk", "runs", "partial", "finalize", "wer", "note", "text"],
    ))

    if any(c["wer"] not in ("-", "0%") for c in summary):
        print("\nnon-zero WER somewhere: on this fixture the <6% target means")
        print("word-perfect, so any cell with WER > 0 misses it.")
    errored = [c for c in summary if c["ok"] < c["runs"]]
    if errored and len(errored) == len(summary):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
