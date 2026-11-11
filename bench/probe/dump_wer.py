"""Score WER on real recordings captured by the gateway's ASR dump.

Every spoken turn can be written to disk as ``utt-NNN-HHMMSS.wav`` plus a JSON
sidecar (final transcript, ASR config, ``asr_final_ms``) by setting
``ASR_DUMP_DIR``. The sidecar holds what the recogniser *heard*; WER needs what
the visitor actually *said*, which only they can supply. Hence the workflow:

1. Record a session with ``ASR_DUMP_DIR`` set (see runbook §6 for the reading
   script - read known sentences so the references are known too).
2. ``--template`` writes a fillable ``refs.yaml`` with every dump's recognised
   text pre-filled as a comment. Replace each empty value with what was really
   said; for read sentences, paste the script line.
3. ``--refs`` scores: per-file WER, corpus WER (total edits / total reference
   words), and coverage in both directions.

``--max-wer`` turns the corpus number into a gate (the PRD target is 6%,
FR-025); without it the run reports and exits 0.

Usage::

    .venv\\Scripts\\python.exe -m bench.probe.dump_wer --dump bench/audio/recorded
    .venv\\Scripts\\python.exe -m bench.probe.dump_wer --dump bench/audio/recorded --template
    .venv\\Scripts\\python.exe -m bench.probe.dump_wer --dump bench/audio/recorded \\
        --refs bench/audio/recorded/refs.yaml --max-wer 0.06
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from bench.probe.common import edit_distance, format_table, word_error_rate, words


def load_dumps(directory: Path) -> list[dict]:
    dumps = []
    for sidecar in sorted(directory.glob("utt-*.json")):
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        wav = sidecar.with_suffix(".wav")
        dumps.append({
            "name": sidecar.stem,
            "wav": wav if wav.exists() else None,
            "recorded_at": payload.get("recorded_at", ""),
            "source": payload.get("source", ""),
            "asr_final_ms": payload.get("asr_final_ms"),
            "heard": payload.get("final") or payload.get("text") or "",
            "error": payload.get("asr_error", ""),
        })
    return dumps


def write_template(dumps: list[dict], target: Path, force: bool) -> int:
    if target.exists() and not force:
        print(f"{target} exists; refusing to overwrite without --force")
        return 1
    lines = [
        "# Reference transcripts for bench/probe/dump_wer.py.",
        "# Fill each value with what the visitor ACTUALLY said (for read",
        "# sentences: the script line). The comment shows what the recogniser",
        "# heard - leave it alone as a cross-check.",
    ]
    for dump in dumps:
        lines.append(f"# heard: {dump['heard']!r}")
        lines.append(f"{dump['name']}: \"\"")
        lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {target} with {len(dumps)} entries to fill in")
    return 0


def score(dumps: list[dict], refs: dict[str, str], max_wer: float | None) -> int:
    rows = []
    total_edits = 0
    total_words = 0
    missing = []
    for dump in dumps:
        reference = refs.get(dump["name"], "")
        if not reference:
            missing.append(dump["name"])
            rows.append({
                "file": dump["name"],
                "wer": "-",
                "ref": "(no reference)",
                "heard": dump["heard"][:60],
            })
            continue
        wer = word_error_rate(reference, dump["heard"])
        total_edits += edit_distance(words(reference), words(dump["heard"]))
        total_words += max(1, len(words(reference)))
        rows.append({
            "file": dump["name"],
            "wer": f"{wer:.0%}",
            "ref": reference[:60],
            "heard": dump["heard"][:60],
        })
    print(format_table(rows, ["file", "wer", "ref", "heard"]))
    unused = [name for name in refs if name not in {d["name"] for d in dumps}]
    if missing:
        print(f"\nno reference for: {', '.join(missing)}")
    if unused:
        print(f"references with no dump: {', '.join(unused)}")
    if total_words == 0:
        print("\nnothing scored: no references matched a dump")
        return 1
    corpus = total_edits / total_words
    print(f"\ncorpus WER: {corpus:.1%} ({total_edits} edits / {total_words} words)")
    if max_wer is not None:
        if corpus > max_wer:
            print(f"FAIL: corpus WER exceeds --max-wer {max_wer:.0%} (FR-025)")
            return 1
        print(f"pass: corpus WER within --max-wer {max_wer:.0%}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", required=True, help="Directory with utt-*.wav/json dumps")
    parser.add_argument("--refs", default="", help="refs.yaml: dump name -> reference text")
    parser.add_argument("--template", action="store_true", help="Write a fillable refs.yaml")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing template")
    parser.add_argument("--max-wer", type=float, default=None,
                        help="Gate: fail if corpus WER exceeds this (0.06 = FR-025)")
    args = parser.parse_args()

    directory = Path(args.dump)
    if not directory.is_dir():
        print(f"no such dump directory: {directory}")
        return 1
    dumps = load_dumps(directory)
    if not dumps:
        print(f"no utt-*.json sidecars in {directory}")
        return 1

    if args.template:
        target = Path(args.refs) if args.refs else directory / "refs.yaml"
        return write_template(dumps, target, args.force)

    if not args.refs:
        print(format_table(
            [
                {
                    "file": d["name"],
                    "at": d["recorded_at"],
                    "src": d["source"],
                    "final_ms": d["asr_final_ms"] if d["asr_final_ms"] is not None else "-",
                    "heard": (d["error"] or d["heard"])[:70],
                }
                for d in dumps
            ],
            ["file", "at", "src", "final_ms", "heard"],
        ))
        print(f"\n{len(dumps)} dumps; run again with --template to write refs, "
              "then --refs to score.")
        return 0

    refs_path = Path(args.refs)
    if not refs_path.exists():
        print(f"no such refs file: {refs_path}")
        return 1
    refs = yaml.safe_load(refs_path.read_text(encoding="utf-8")) or {}
    return score(dumps, {str(k): str(v) for k, v in refs.items()}, args.max_wer)


if __name__ == "__main__":
    raise SystemExit(main())
