# Sara runbook

Operational notes for the gateway. The product decisions behind these steps are
in `docs/adr/0001-architecture-decisions.md` and `docs/adr/0002-india-first-providers.md`.

## 1. What runs where

| Piece | Default | Notes |
|---|---|---|
| Gateway | Fly app `sarang-gateway`, region `bom` | FastAPI, WebRTC + WebSocket PCM |
| Client | static `apps/web/dist`, served by the gateway | also embeddable as `<SarangWidget />` |
| TURN | coturn in `bom` | only needed when ICE cannot go direct |
| Tier-2 store | pgvector Postgres | optional; BM25 is the fallback |

The gateway is deliberately one process. Do not split stages until concurrent
sessions prove CPU interference (ADR 0001 §1).

## 2. First deploy

```bash
fly launch --no-deploy --copy-config --name sarang-gateway
fly secrets set \
  SESSION_SECRET="$(openssl rand -hex 32)" \
  SARVAM_API_KEY=... \
  GEMINI_API_KEY=...
fly deploy
fly logs
```

`SESSION_SECRET` must not be the development default. Rotating it invalidates
every live session token, which is the intended blast radius.

Verify:

```bash
curl -s https://sarang-gateway.fly.dev/healthz | jq
curl -s https://sarang-gateway.fly.dev/api/config | jq '{profile, asr, llm, tts, retrieval}'
```

`/healthz` must report `ok: true`, the expected profile, and `ready: true`. A
provider with no key reports `*_ready: false` and degrades; it does not fail the
process.

## 3. Secrets and configuration

All credentials are backend-only. The browser receives a short-lived
`session_id.expiry.signature` token and nothing else.

| Variable | Purpose |
|---|---|
| `SESSION_SECRET` | HMAC key for session tokens (required in production) |
| `SARVAM_API_KEY` | ASR + LLM (default `bom-india` path) |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Gemini Flash-Lite; comparison arm only |
| `GROQ_API_KEY` | comparison arm only; never required |
| `CARTESIA_API_KEY` | TTS on the default `bom-india` path |
| `CARTESIA_VOICE_ID` | spoken voice; defaults to `2a12b36c-7f9b-4c3a-9f7a-72731b15323a` (Sara) |
| `NVIDIA_API_KEY` | comparison ASR arm only |
| `ALLOWED_ORIGINS` | comma-separated origin allowlist |
| `DATABASE_URL` | enables pgvector/tsvector tier-2 retrieval |
| `ICE_SERVERS` | `urls|user:secret;urls|user:secret` |
| `ICE_FORCE_RELAY` | `true` forces TURN, useful when diagnosing ICE |
| `SARANG_CACHE_DIR` | writable filler cache; defaults to `<repo>/cache` |

## 4. TURN

TURN is required when a visitor is behind symmetric NAT or a restrictive
corporate network. Without it, the browser falls back to `/ws/audio`, which still
works but loses WebRTC's jitter handling.

Local stack:

```bash
docker compose -f infra/compose/docker-compose.yml --env-file .env up --build
```

Production on Fly:

1. Create a second app in `bom` with a dedicated IPv4.
2. Render `infra/coturn/turnserver.conf` with `__TURN_EXTERNAL_IP__` and
   `__TURN_SHARED_SECRET__`, or pass the same values as flags.
3. Open 3478/tcp, 3478/udp and 49160–49200/udp.
4. Set the gateway secret:
   `fly secrets set ICE_SERVERS="turn:<ip>:3478?transport=udp,turn:<ip>:3478?transport=tcp|sarang:<secret>"`
5. Confirm with the diagnostics drawer: `candidateType` should become `relay`
   only when `ICE_FORCE_RELAY=true`; otherwise a direct candidate is preferred.

Do not reuse the session secret as the TURN secret.

## 5. Deploying a change

```bash
cd apps/web && npm ci && npm run build && cd ../..
pytest -q
ruff check .
python kb/validate.py
fly deploy
```

The image builds the client itself, so a local `dist` is only needed when running
the gateway outside Docker. `WEB_DIST` is resolved at import time; restart the
process after rebuilding.

## 6. Reading health and metrics

- `/healthz` — profile, provider readiness, KB block count, uptime.
- `/api/config` — public, credential-free configuration.
- `/api/metrics` — rolling stage distributions, tool calls, speculation adoption.
- Diagnostics drawer — last turn's `eos_to_answer_audio_ms` (the honest number),
  `eos_to_first_audio_ms` (may include a filler), transport, ICE candidate type.

If `eos_to_answer_audio_ms` regresses:

1. Check `endpoint_silence_ms` first; it is the hard floor.
2. Check provider RTT from the region (`bench/probe/regions.py --rtt`).
3. Check whether tier-2 retrieval is firing for questions that should be tier-1.
4. Check whether the filler is firing on fast turns; it must not.

## 7. Failure modes

| Symptom | Likely cause | Action |
|---|---|---|
| Voice input never transcribes | ASR key missing or WS blocked | text still works; check `/healthz` `asr_ready` |
| No audio at all | browser autoplay policy | press "Enable sound"; playback graph is created in the click handler |
| `/api/offer` returns 503 | `aiortc` not installed | expected; browser uses `/ws/audio` |
| Answers are generic/refusals | LLM unavailable, local answerer active | check `llm_ready`; local answers are grounded but extractive |
| `retrieval` says `bm25-local (database unreachable)` | bad `DATABASE_URL` | fix the DSN or unset it; BM25 is fine |
| High `wasted` speculation | prefetch misses | lower `SPECULATIVE_MS` or disable it |

## 8. Rollback

```bash
fly releases
fly deploy --image <previous-image>
```

The gateway is stateless apart from the filler cache; rolling back does not
migrate data. The pgvector table is additive and safe to leave in place.

## 9. Local development without keys

```bash
uv sync --extra api
uv run python -m sarang
cd apps/web && npm run dev
```

The `local` profile uses the deterministic answerer and synthetic voice. Speech
input is unavailable (ASR `none`) and the server says so once instead of failing
silently. Text turns exercise the whole turn engine, including streaming TTS and
the egress shaper.
