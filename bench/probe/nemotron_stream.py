"""Discover and measure the NVIDIA Nemotron streaming-ASR surface available to this account.

We deliberately do not assume the protocol. The PRD names
``nvidia/nemotron-speech-streaming-en-0.6b``, but the reachable surface depends on
the account: an OpenAI-compatible NIM endpoint, a streaming SSE endpoint, a
WebSocket endpoint, a gRPC Riva endpoint, or a self-hosted NIM container. The
last option is architecturally the most attractive (no network hop at all), so it
is worth knowing which one we actually have before writing an adapter.

Modes::

    --discover                     list models visible on integrate.api.nvidia.com
    --transcribe FILE --model ID   one non-streaming round trip
    --stream-sse FILE --model ID   POST with stream=true, read SSE events
    --stream-ws URL --audio FILE   generic WebSocket streaming probe

If discovery returns nothing useful, the next step is to check the model card on
build.nvidia.com for a NIM container image (``nvcr.io/nim/nvidia/...``), which can
run on the media gateway itself.

Usage::

    uv run python -m bench.probe.nemotron_stream --discover
    uv run python -m bench.probe.nemotron_stream --transcribe bench/audio/tts-sample.wav \
        --model nvidia/nemotron-speech-streaming-en-0.6b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
from pathlib import Path

import httpx

from bench.probe.common import (
    Result,
    chunks,
    format_table,
    load_env,
    load_wav_16k,
    now,
    require_env,
)

MODELS_URL = "https://integrate.api.nvidia.com/v1/models"
TRANSCRIBE_URL = "https://integrate.api.nvidia.com/v1/audio/transcriptions"

KEYWORDS = ("asr", "speech", "nemotron", "parakeet", "riva", "whisper", "stt", "audio")


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


async def discover(api_key: str, label: str) -> int:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(MODELS_URL, headers=_auth_headers(api_key))
        response.raise_for_status()
        models = response.json().get("data", [])

    interesting = sorted(
        (
            model
            for model in models
            if any(keyword in model.get("id", "").lower() for keyword in KEYWORDS)
        ),
        key=lambda model: model.get("id", ""),
    )
    rows = [
        {
            "id": model.get("id", "?"),
            "owned_by": model.get("owned_by", "?"),
            "created": model.get("created", "?"),
        }
        for model in interesting
    ]
    print(f"{len(models)} models visible; {len(rows)} match ASR keywords\n")
    print(format_table(rows, ["id", "owned_by", "created"]))

    result = Result(probe="nemotron-discover", target=MODELS_URL, label=label)
    result.metrics["model_count"] = len(models)
    result.metrics["asr_candidates"] = [row["id"] for row in rows]
    if not rows:
        result.note(
            "no ASR model visible via the OpenAI-compatible model list. Check "
            "build.nvidia.com for a NIM container image for "
            "nvidia/nemotron-speech-streaming-en-0.6b; self-hosting it on the media "
            "gateway removes the ASR network hop entirely."
        )
    path = result.write()
    print(f"\nJSON: {path}")
    print(
        "\nNext steps, in order of preference:\n"
        "  1. If an ASR model id is listed: try --transcribe, then --stream-sse.\n"
        "  2. If the model card offers a NIM container: run it next to the gateway "
        "(best latency, no egress).\n"
        "  3. If only gRPC/Riva is exposed: the adapter will need nvidia-riva-client "
        "and a streaming recognize call.\n"
        "  4. If access is gated: stop and pick an alternative before writing Sprint 2."
    )
    return 0


async def transcribe(api_key: str, wav: Path, model: str, label: str) -> int:
    data, rate = load_wav_16k(wav)
    # Re-encode at 16k so we send exactly what the pipeline would send.
    from bench.probe.common import pcm_bytes

    payload = pcm_bytes(data)
    files = {
        "file": (wav.name, payload, mimetypes.guess_type(wav.name)[0] or "audio/wav"),
    }
    form = {"model": model, "language": "en", "response_format": "json"}

    result = Result(probe="nemotron-transcribe", target=TRANSCRIBE_URL, label=label)
    result.metrics.update({"model": model, "audio_ms": round(len(data) / rate * 1000, 1)})

    async with httpx.AsyncClient(timeout=120) as client:
        started = now()
        response = await client.post(
            TRANSCRIBE_URL, headers=_auth_headers(api_key), data=form, files=files
        )
        elapsed = (now() - started) * 1000

    result.metrics["status"] = response.status_code
    result.metrics["total_ms"] = round(elapsed, 1)
    if response.status_code >= 400:
        result.note(response.text[:500])
        print(f"HTTP {response.status_code}: {response.text[:500]}")
    else:
        try:
            body = response.json()
        except ValueError:
            body = {"raw": response.text[:1000]}
        result.metrics["response_keys"] = sorted(body.keys())
        result.metrics["text"] = str(body.get("text", ""))[:400]
        print(json.dumps(body, indent=2)[:1200])

    path = result.write()
    print(f"\nJSON: {path}")
    return 0 if response.status_code < 400 else 1


async def stream_sse(api_key: str, wav: Path, model: str, label: str) -> int:
    """POST the clip with stream=true and timestamp each SSE event."""
    data, rate = load_wav_16k(wav)
    from bench.probe.common import pcm_bytes

    payload = pcm_bytes(data)
    files = {
        "file": (wav.name, payload, mimetypes.guess_type(wav.name)[0] or "audio/wav"),
    }
    form = {"model": model, "language": "en", "stream": "true"}

    result = Result(probe="nemotron-sse", target=TRANSCRIBE_URL, label=label)
    result.metrics.update({"model": model, "audio_ms": round(len(data) / rate * 1000, 1)})

    async with httpx.AsyncClient(timeout=120) as client:
        started = now()
        async with client.stream(
            "POST", TRANSCRIBE_URL, headers=_auth_headers(api_key), data=form, files=files
        ) as response:
            result.metrics["status"] = response.status_code
            if response.status_code >= 400:
                body = await response.aread()
                result.note(body.decode("utf-8", "ignore")[:500])
                print(f"HTTP {response.status_code}: {body[:500]!r}")
                path = result.write()
                print(f"JSON: {path}")
                return 1
            first_event: float | None = None
            events = 0
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                if first_event is None:
                    first_event = now()
                    result.metrics["first_event_ms"] = round((first_event - started) * 1000, 1)
                events += 1
                print(f"{(now() - started) * 1000:8.1f} ms  {line[:200]}")
            result.metrics["events"] = events
            result.metrics["total_ms"] = round((now() - started) * 1000, 1)

    path = result.write()
    print(f"\nJSON: {path}")
    return 0


async def _connect(url: str, headers: dict[str, str]):
    try:
        from websockets.asyncio.client import connect as ws_connect
    except ImportError:  # pragma: no cover
        from websockets import connect as ws_connect
    try:
        return ws_connect(url, additional_headers=headers, max_size=None, open_timeout=20)
    except TypeError:  # websockets < 13
        return ws_connect(url, extra_headers=headers, max_size=None, open_timeout=20)


async def stream_ws(
    api_key: str,
    url: str,
    wav: Path,
    label: str,
    *,
    config_json: str,
    chunk_ms: int,
    header: list[str],
) -> int:
    """Protocol-agnostic probe: optional JSON config frame, then paced PCM frames."""
    data, rate = load_wav_16k(wav)
    headers = _auth_headers(api_key)
    for item in header:
        if "=" in item:
            key, value = item.split("=", 1)
            headers[key.strip()] = value.strip()

    result = Result(probe="nemotron-ws", target=url, label=label)
    result.metrics.update(
        {"audio_ms": round(len(data) / rate * 1000, 1), "chunk_ms": chunk_ms}
    )

    async with await _connect(url, headers) as socket:
        if config_json:
            await socket.send(config_json)
            print(f"sent config: {config_json[:160]}")
        started = now()
        first_message: float | None = None
        audio_finished = 0.0

        async def reader() -> None:
            nonlocal first_message
            while True:
                try:
                    message = await socket.recv()
                except Exception:
                    return
                stamp = (now() - started) * 1000
                if first_message is None:
                    first_message = now()
                    result.metrics["first_partial_ms"] = round(stamp, 1)
                text = message if isinstance(message, str) else f"<{len(message)} bytes>"
                print(f"{stamp:8.1f} ms  {text[:220]}")
                try:
                    payload = json.loads(text)
                except (ValueError, TypeError):
                    continue
                if payload.get("done") or payload.get("is_final") or payload.get("final"):
                    if audio_finished:
                        result.metrics["finalization_ms"] = round(stamp - audio_finished, 1)
                    return

        reader_task = asyncio.create_task(reader())
        for frame in chunks(data, chunk_ms, rate):
            await socket.send(frame)
            await asyncio.sleep(chunk_ms / 1000.0)
        audio_finished = (now() - started) * 1000

        try:
            await asyncio.wait_for(reader_task, timeout=10)
        except TimeoutError:
            reader_task.cancel()
            result.note("no final event within 10 s of the last audio frame")

    result.metrics["audio_end_ms"] = round(audio_finished, 1)
    path = result.write()
    print(f"\nJSON: {path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--discover", action="store_true")
    mode.add_argument("--transcribe", metavar="WAV")
    mode.add_argument("--stream-sse", metavar="WAV")
    mode.add_argument("--stream-ws", metavar="URL")
    parser.add_argument("--audio", metavar="WAV", help="Audio for --stream-ws")
    parser.add_argument(
        "--model", default="nvidia/nemotron-speech-streaming-en-0.6b"
    )
    parser.add_argument("--chunk-ms", type=int, default=80)
    parser.add_argument("--config-json", default="", help="Optional JSON config frame for WS")
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="K=V",
        help="Extra header for --stream-ws (repeatable)",
    )
    parser.add_argument("--label", default="local")
    return parser.parse_args()


def main() -> int:
    load_env()
    args = parse_args()
    api_key = require_env("NVIDIA_API_KEY", "Key from build.nvidia.com or your NGC account")
    if args.transcribe:
        return asyncio.run(transcribe(api_key, Path(args.transcribe), args.model, args.label))
    if args.stream_sse:
        return asyncio.run(stream_sse(api_key, Path(args.stream_sse), args.model, args.label))
    if args.stream_ws:
        if not args.audio:
            raise SystemExit("--stream-ws requires --audio WAV")
        return asyncio.run(
            stream_ws(
                api_key,
                args.stream_ws,
                Path(args.audio),
                args.label,
                config_json=args.config_json,
                chunk_ms=args.chunk_ms,
                header=args.header,
            )
        )
    return asyncio.run(discover(api_key, args.label))


if __name__ == "__main__":
    raise SystemExit(main())
