"""Network and provider-latency measurements for choosing the media-gateway region.

The 500 ms budget counts both network legs: visitor audio in, synthesized audio out.
Running this inside a Fly machine in each candidate region is the only honest way to
compare, because a laptop measurement says nothing about a machine in ``bom``.

Modes::

    --rtt     DNS / TCP / TLS / time-to-first-byte to each provider (no API key needed)
    --full    --rtt plus one Groq TTFT and one Cartesia TTFA measurement
Usage::

    uv run python -m bench.probe.regions --rtt --runs 7

    # inside a Fly machine, one region at a time
    fly machine run . --region bom -- python -m bench.probe.regions --full --label bom
    fly machine run . --region iad -- python -m bench.probe.regions --full --label iad
    fly machine run . --region fra -- python -m bench.probe.regions --full --label fra
    fly machine run . --region sin -- python -m bench.probe.regions --full --label sin
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import socket
import ssl

import httpx

from bench.probe.common import Result, now, optional_env

HOSTS = {
    "sarvam": "api.sarvam.ai",
    "gemini": "generativelanguage.googleapis.com",
    "groq": "api.groq.com",
    "cartesia": "api.cartesia.ai",
    "nvidia": "integrate.api.nvidia.com",
}

TRACE_URL = "https://www.cloudflare.com/cdn-cgi/trace"


async def public_identity() -> dict[str, str]:
    """Ask Cloudflare where this process appears to be coming from."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(TRACE_URL)
            response.raise_for_status()
            pairs = {}
            for line in response.text.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    pairs[key] = value
            return {
                "ip": pairs.get("ip", ""),
                "loc": pairs.get("loc", ""),
                "colo": pairs.get("colo", ""),
            }
    except Exception as error:  # network probe should never crash the run
        return {"error": str(error)}


async def network_rtt(host: str, port: int = 443) -> dict[str, float]:
    """DNS, TCP, TLS and TTFB for one host, in milliseconds."""
    loop = asyncio.get_running_loop()
    started = now()
    await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    dns = (now() - started) * 1000

    started = now()
    reader, writer = await asyncio.open_connection(host, port)
    tcp = (now() - started) * 1000

    started = now()
    context = ssl.create_default_context()
    await writer.start_tls(context, server_hostname=host)
    tls = (now() - started) * 1000

    started = now()
    writer.write(
        f"HEAD / HTTP/1.1\r\nHost: {host}\r\nUser-Agent: sarang-probe\r\n"
        f"Connection: close\r\n\r\n".encode()
    )
    await writer.drain()
    await reader.read(1)
    ttfb = (now() - started) * 1000

    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass

    return {"dns": dns, "tcp": tcp, "tls": tls, "ttfb": ttfb, "rtt_est": tcp + tls}


async def run_rtt(result: Result, runs: int) -> None:
    for name, host in HOSTS.items():
        for _ in range(runs):
            try:
                marks = await network_rtt(host)
            except Exception as error:
                result.note(f"{name}: {error}")
                continue
            for key, value in marks.items():
                result.add(f"{name}.{key}", value)


async def run_providers(result: Result, runs: int) -> None:
    from bench.probe import cartesia_ttfa, groq_ttft

    groq_key = optional_env("GROQ_API_KEY")
    cartesia_key = optional_env("CARTESIA_API_KEY")

    if groq_key:
        for _ in range(runs):
            try:
                run = await groq_ttft.measure_once(
                    api_key=groq_key,
                    model=groq_ttft.DEFAULT_MODELS[0],
                    system_prompt=groq_ttft.build_system_prompt(450),
                )
            except Exception as error:
                result.note(f"groq: {error}")
                break
            result.add("groq.ttft", run["ttft_ms"])
            if run["tokens_per_second"]:
                result.add("groq.tps", run["tokens_per_second"])
    else:
        result.note("GROQ_API_KEY not set, skipping Groq")

    if cartesia_key:
        try:
            voice = await cartesia_ttfa.resolve_voice(cartesia_key)
            model = optional_env("CARTESIA_MODEL", "sonic-latest")
            for _ in range(runs):
                run = await cartesia_ttfa.stream_passage(
                    api_key=cartesia_key, voice=voice, model=model, sample_rate=24000
                )
                result.add("cartesia.ttfa_first", run["ttfa_first_ms"])
                result.add("cartesia.rtf", run["rtf"])
        except Exception as error:
            result.note(f"cartesia: {error}")
    else:
        result.note("CARTESIA_API_KEY not set, skipping Cartesia")


async def main(args: argparse.Namespace) -> int:
    result = Result(probe="regions", target=",".join(HOSTS.values()), label=args.label)
    result.metrics["python"] = platform.python_version()

    identity = await public_identity()
    result.metrics["public_identity"] = identity
    print(f"public identity: {json.dumps(identity)}")

    await run_rtt(result, args.runs)
    if args.full:
        await run_providers(result, max(1, min(args.runs, 3)))

    path = result.write()
    print()
    for name, host in HOSTS.items():
        rtt = result.distributions().get(f"{name}.rtt_est")
        if rtt:
            print(f"{name:10} {host:32} rtt p50={rtt['p50']:7.1f} ms  p95={rtt['p95']:7.1f} ms")
    for key in ("groq.ttft", "cartesia.ttfa_first"):
        distribution = result.distributions().get(key)
        if distribution:
            print(
                f"{key:22} p50={distribution['p50']:7.1f} ms  "
                f"p95={distribution['p95']:7.1f} ms"
            )
    print(f"\nJSON: {path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rtt", action="store_true", help="Network-only measurement")
    parser.add_argument("--full", action="store_true", help="Network plus provider calls")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--label",
        default=os.environ.get("FLY_REGION", "local"),
        help="Region label for the JSON (defaults to FLY_REGION)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(parse_args())))
