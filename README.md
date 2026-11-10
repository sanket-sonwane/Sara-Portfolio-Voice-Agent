# Sara

Sanket Sonwane's real-time, browser-based voice agent: his work, background,
motivation and fit for a role. (The repository, Python package and CSS namespace
remain `sarang`.)

Cascaded streaming pipeline: **WebRTC audio → VAD/endpointing → streaming ASR →
fact lookup + LLM + tools → text segmenter → streaming TTS → WebRTC audio**, with
generation-scoped cancellation so a visitor can interrupt at any time.

The product spec lives in [`prd.md`](./prd.md). Architectural decisions are in
[`docs/adr/`](./docs/adr); operations are in [`docs/runbook.md`](./docs/runbook.md).

## What is implemented

- **Gateway** (`apps/api/sarang`): FastAPI, one process, in-process asyncio bus,
  typed envelopes, epoch-scoped cancellation at LLM tokens, tool results, TTS
  chunks, egress and browser playback.
- **Knowledge**: `kb/fact.yaml` is the only thing the agent may assert; tier-1
  hybrid lookup runs in-process, tier-2 BM25/pgvector sits behind tools. No
  portfolio digest in the prompt.
- **Voice**: Sarvam `saaras:v4` streaming ASR and Sarvam 105B generation in the
  India-first profile, with Smallest Lightning TTS (`rhea`, en-IN) for an Indian
  accent and enough concurrency to speak every answer segment. Cartesia Sonic,
  Gemini, Groq and NVIDIA remain comparison arms.
- **Transport**: WebRTC when `aiortc` is installed; otherwise `/ws/audio` PCM16 at
  16 kHz up / 48 kHz down. Text-only sessions still open the audio transport for
  output.
- **Client** (`apps/web`): the full PRD §6 interaction surface — 12 states, live
  partial/final transcript, input/output waveforms, captions, diagnostics drawer,
  text fallback, keyboard and reduced-motion support.
- **Zero-key mode**: the `local` profile runs the entire pipeline with a
  deterministic grounded answerer and a synthetic voice.

## Quickstart

```bash
uv sync --extra api
cp .env.example .env          # optional: leave keys blank to run in local mode
uv run python -m sarang       # http://127.0.0.1:8000

cd apps/web
npm install
npm run dev                   # http://127.0.0.1:5173, proxies /api and /ws
```

For the single-origin production surface:

```bash
cd apps/web && npm run build && cd ../..
uv run python -m sarang
```

The gateway serves `apps/web/dist` at `/`; restart it after a rebuild because the
static root is resolved at import time.

## Profiles

| Profile | ASR | LLM | TTS | Use |
|---|---|---|---|---|
| `bom-india` (default) | Sarvam `saaras:v4` | Sarvam 105B | Smallest Lightning | the product |
| `iad-global` | NVIDIA | Groq | Cartesia | comparison only |
| `local` | none | deterministic | synthetic | tests, zero keys |

See [ADR 0002](./docs/adr/0002-india-first-providers.md) for why Groq is not the
default despite its per-component speed, and §6 for the measured stack that
replaced the provisional one.

## Tests and gates

```bash
pytest -q
ruff check .
python kb/validate.py
cd apps/web && npm run typecheck && npm run build
```

`kb/validate.py` checks provenance, pending items and refusal boundaries in the
canonical fact base. The test suite covers routing, refusals, epoch cancellation,
the turn engine and the gateway contract.

## Probes

All probes run from the repository root and write JSON to `bench/results/`.
Durations use `perf_counter` and are reported as distributions, not single runs.

```bash
# Network legs to every provider host
uv run python -m bench.probe.regions --rtt --runs 7

# Gemini vs Sarvam TTFT for the India-first stack
uv run python -m bench.probe.llm_matrix --providers gemini sarvam --runs 7

# Sarvam TTS first-audio and ASR first-partial
uv run python -m bench.probe.sarvam_probe --tts --save-audio
uv run python -m bench.probe.sarvam_probe --asr --audio bench/audio/sarvam_probe.wav

# Comparison arms
uv run python -m bench.probe.groq_ttft --runs 7
uv run python -m bench.probe.cartesia_ttfa --runs 5
uv run python -m bench.probe.nemotron_stream --discover
```

Region comparison must run inside Fly machines, not a laptop:

```bash
cd infra/prober
fly machine run . --region bom -- python -m bench.probe.regions --full --label bom
fly machine run . --region iad -- python -m bench.probe.regions --full --label iad
```

## Deployment

```bash
fly launch --no-deploy --copy-config --name sarang-gateway
fly secrets set SESSION_SECRET=... SARVAM_API_KEY=... GEMINI_API_KEY=...
fly deploy
```

`fly.toml` targets `bom`; `apps/api/Dockerfile` builds the client and the Python
runtime in one image. A local production-shaped stack (gateway + pgvector +
coturn) is in `infra/compose/docker-compose.yml`. Full procedures, TURN setup,
health checks and rollback are in [`docs/runbook.md`](./docs/runbook.md).

## Layout

```
apps/api/         FastAPI gateway (sarang package, Dockerfile)
apps/web/         React widget + standalone surface
bench/probe/      provider and network probes
bench/questions/  grounded evaluation set, including refusals
infra/prober/     Fly app used to measure provider latency in-region
infra/compose/    local gateway + pgvector + coturn stack
infra/coturn/     production TURN template
kb/               canonical facts, docs, validator
docs/             ADRs, Phase 0 report, runbook
tests/            gateway, engine and grounding tests
```
