# Sara runbook

Operational notes for the gateway. The product decisions behind these steps are
in `docs/adr/0001-architecture-decisions.md`,
`docs/adr/0002-india-first-providers.md` and
`docs/adr/0004-tts-provider-selection.md`.

## 1. What runs where

| Piece | Default | Notes |
|---|---|---|
| Gateway | Fly app `sarang-gateway`, region `sin` | FastAPI, WebRTC + WebSocket PCM |
| Client | static `apps/web/dist`, served by the gateway | also embeddable as `<SarangWidget />` |
| TURN | coturn | only needed when ICE cannot go direct |
| Tier-2 store | pgvector Postgres | optional; BM25 is the fallback |

`bom` (Mumbai) is the intended region and is not available on this Fly account;
`sin` is the nearest substitute (ADR 0002 §4).

The gateway is deliberately one process. Do not split stages until concurrent
sessions prove CPU interference (ADR 0001 §1).

## 2. First deploy

```bash
fly launch --no-deploy --copy-config --name sarang-gateway
fly secrets set \
  SESSION_SECRET="$(openssl rand -hex 32)" \
  SARVAM_API_KEY=... \
  SMALLEST_API_KEY=...
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
| `SMALLEST_API_KEY` | TTS on the default `bom-india` path (ADR 0004) |
| `SMALLEST_VOICE` | spoken voice; defaults to `rhea` (en-IN) |
| `SMALLEST_LANGUAGE` | `hi` = Indian-accented English + Hindi code-switching; `en` = UK/US |
| `SMALLEST_MODEL` | defaults to `lightning_v3.1_pro` |
| `SMALLEST_MAX_CONCURRENCY` | in-flight segments; defaults to `4` |
| `TTS_PROVIDER` | pins the arm (`smallest`, `cartesia`, `sarvam`, `synthetic`) |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Gemini Flash-Lite; comparison arm only |
| `GROQ_API_KEY` | comparison arm only; never required |
| `CARTESIA_API_KEY` | TTS on the `iad-global` path and the comparison arm |
| `CARTESIA_VOICE_ID` | Cartesia voice; defaults to `2a12b36c-7f9b-4c3a-9f7a-72731b15323a` |
| `NVIDIA_API_KEY` | comparison ASR arm only |
| `ALLOWED_ORIGINS` | comma-separated origin allowlist |
| `DATABASE_URL` | enables pgvector/tsvector tier-2 retrieval |
| `ICE_SERVERS` | `urls|user:secret;urls|user:secret` |
| `ICE_FORCE_RELAY` | `true` forces TURN, useful when diagnosing ICE |
| `SARANG_CACHE_DIR` | writable filler cache; defaults to `<repo>/cache` |

`TTS_PROVIDER` is pinned in `fly.toml` `[env]`, not in a secret, so the deployed
arm is reviewable in the repository. Filler clips are cached under a directory
named after the provider, model, voice and language, so changing any of them
re-renders instead of replaying the old voice.

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
- Diagnostics drawer - last turn's `eos_to_answer_audio_ms` (the honest number),
  `eos_to_first_audio_ms` (may include a filler), transport, ICE candidate type.

### Measuring the visitor's own number

`eos_to_answer_audio_ms` is measured on the server, so it cannot contain the
return network path or the browser's playback start. The diagnostics drawer
reports both sides, and the difference between them is that leg:

- **in this browser, speech end → audible** — the client's own stamps: when it
  saw the server leave `user_speaking`, and when its `AudioContext` scheduled the
  first answer buffer to be *heard*. Both are on `performance.now()`, so no clock
  sync is involved and a stalled tab cannot shift the answer.
- **network + playback** — the difference from `eos_to_answer_audio_ms`.

A headless browser cannot be granted a microphone, so this needs a real one:

1. Open `https://sarang-gateway.fly.dev/` in Chrome, allow the microphone, and
   press "Enable sound" (the playback graph is created in the click handler, so
   without the gesture the browser's autoplay policy blocks it).
2. Ask three or four questions, leaving a natural pause at the end of each, and
   read the diagnostics drawer after every turn.
3. Compare the two figures. A server number that is high with a small gap means
   the provider was slow on that turn; a small server number with a large gap
   means the network leg or playback start is the problem, and no amount of
   pipeline work will move it.

Expect the gap to sit near the round trip to `sin` plus 25 ms of playback lead.


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
| Answer audio stops mid-sentence | TTS concurrency exhausted | Cartesia's free tier allows 2 contexts; `smallest` allows 10–15. Check `/healthz` `tts` |
| Every turn is ~4 s slower than the audio | Smallest `complete_backoff_ms` left at its 4 s default | the adapter must send `0`; `bench/probe/smallest_ttfa.py` reports it as RTF 2.38 |
| Fillers speak in the previous voice | filler cache reused across a provider change | clear `$SARANG_CACHE_DIR/fillers`; the tag should name the current provider |
| `/api/offer` returns 503 | `aiortc` not installed | expected; browser uses `/ws/audio` |
| Answers are generic/refusals | LLM unavailable, local answerer active | check `llm_ready`; local answers are grounded but extractive |
| `retrieval` says `bm25-local (database unreachable)` | bad `DATABASE_URL` | fix the DSN or unset it; BM25 is fine |
| High `wasted` speculation | prefetch misses | lower `SPECULATIVE_MS` or disable it |
| `fly deploy` fails with "insufficient CPUs available … on the current host" | the `sin` host has no room for an in-place machine update | `fly deploy --strategy immediate` replaces the machine instead of updating it; the volume and secrets are untouched |

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
