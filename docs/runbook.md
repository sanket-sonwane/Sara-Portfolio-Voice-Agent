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
  `eos_to_first_audio_ms` (includes egress, and any filler), transport, ICE
  candidate type. The two differ by `egress` on a turn without a filler: the
  first is stamped when the answer's first chunk reaches the egress queue, the
  second when a frame actually leaves the process. Neither is wrong; they answer
  different questions, and "network + playback" is subtracted from the second
  because that is the moment the browser can begin to hear anything.

### Measuring the visitor's own number

`eos_to_answer_audio_ms` is measured on the server, so it cannot contain the
return network path or the browser's playback start. The diagnostics drawer
reports both sides, and the difference between them is that leg:

- **in this browser, speech end → audible** — the client's own stamps: when it
  saw the server leave `user_speaking`, and when its `AudioContext` scheduled the
  first answer buffer to be *heard*. Both are on `performance.now()`, so no clock
  sync is involved and a stalled tab cannot shift the answer.
- **network + playback** — the difference from `eos_to_first_audio_ms`, the
  moment a frame left the gateway. Using `eos_to_answer_audio_ms` here would
  fold `egress` into a leg that is supposed to be pure network and playback.

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

The drawer now keeps the last eight answered turns, so the pass produces a
distribution rather than one number. Typed turns appear in the list but are
excluded from the median, because a typed question skips endpointing and ASR
entirely and would drag the figure down.

Before spending a session on the manual pass, check the drawer still works:

```bash
cd apps/web
node scripts/smoke.mjs --url https://sarang-gateway.fly.dev/ \
  --question "What did Sanket build during the ARAI internship?"
```

(Not `npm run smoke --`: npm parses `--url`/`--question` as npm config and
drops them, so the script silently falls back to `127.0.0.1:8080`.)

It drives the built page in headless Chrome, asks a typed question, opens the
drawer, and exits non-zero on any console error. It cannot test the microphone —
that is what the manual pass is for — but it catches a broken panel first.

### Capturing reference recordings (the WER corpus)

FR-025 wants WER < 6% as a *rate*, and one fixture cannot produce a rate. The
ASR dump turns a normal voice session into fixtures: every spoken turn is
written as `utt-NNN-HHMMSS.wav` plus a JSON sidecar (what the recogniser
heard, its config, `asr_final_ms`). WER needs what the visitor really said, so
this pass reads a fixed script — the script *is* the reference.

Record locally so the files land in the repo instead of a container (the local
profile still speaks real Sarvam ASR: `.env` pins `SARANG_PROFILE=bom-india`):

```bash
ASR_DUMP_DIR=bench/audio/recorded PYTHONPATH=apps/api \
  .venv/Scripts/python.exe -m sarang          # port 8000, loads .env
cd apps/web && npm run dev                    # 5173, proxies /ws to :8000
```

Open `http://localhost:5173`, allow the microphone, enable sound, and read
these six sentences **once each, in order, at natural pace** (if a retake is
unavoidable, note which number — alignment is by dump timestamp):

1. What did Sanket build during the ARAI internship?
2. Tell me how the voice pipeline decides when a turn is over.
3. Why does retrieval sometimes give a long answer?
4. Sanket and Prisha reviewed OSMR in Pune last March.
5. Does the scheduler accept SCHED_EXT workloads here?
6. How many milliseconds does endpointing add to every answer?

Then score:

```bash
.venv/Scripts/python.exe -m bench.probe.dump_wer --dump bench/audio/recorded --template
# fill the generated refs.yaml (the script lines; "heard" is a comment to check against)
.venv/Scripts/python.exe -m bench.probe.dump_wer --dump bench/audio/recorded \
  --refs bench/audio/recorded/refs.yaml --max-wer 0.06
```

The scorer reports per-file and corpus WER, exits non-zero past `--max-wer`,
and lists dumps with no reference (own questions from the latency pass score
nothing until their text is pasted in). The recordings double as real-mic
input for `bench/probe/vad_accuracy.py`, which until now only ever saw
synthetic speech.


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
| High `wasted` / `arm_wasted` speculation | partials don't match finals, or evidence/prompt gates refuse the arm | check `/api/metrics` `speculation` (`prepared`/`adopted`/`wasted` for lookup, `armed`/`arm_adopted`/`arm_wasted` for LLM); disable with `SPECULATIVE_LLM=false` or `SPECULATION_ENABLED=false` |
| Voice turns wait the full silence window | the neural detector did not load | the gateway logs `silero VAD unavailable` and restores `ENDPOINT_SILENCE_MS=800` automatically; check that `onnxruntime` is installed (`uv sync --extra api`) and the model hash matches |
| Turns close mid-sentence | `ENDPOINT_SILENCE_MS` set below the speaker's pause length | the reference clip's longest internal pause is 460 ms; the default 500 leaves 40 ms of margin. Re-run `bench/probe/pause_profile.py` on your own audio before lowering it |
| Barge-in feels late | client detection (120 ms) or provider cancellation | EPOCH is published before cleanup now; if it is still slow, check `tts` cancel latency, not the VAD |
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
