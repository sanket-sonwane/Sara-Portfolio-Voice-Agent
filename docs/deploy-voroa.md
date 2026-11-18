# Deploying Sarang on Voroa

Voroa is the primary host: a web service in **Mumbai (IN)**, built from
`apps/api/Dockerfile`, which puts the gateway in the same country as Sarvam and
Smallest. Fly (`sin`) stays configured as a fallback.

## 1. Service settings

Create the service in the Voroa dashboard (**New → Web Service**) with:

| Field | Value |
|---|---|
| Repository | `sanket-sonwane/Sara-Portfolio-Voice-Agent` |
| Branch | `main` (or the deploy branch being tested) |
| Root directory | `/` |
| Runtime | **Docker (Dockerfile)** |
| Dockerfile path | `/apps/api/Dockerfile` |
| Health check path | `/healthz` |
| Instance | Free (Nano) to start — see §4 |

The Dockerfile path is fixed when the service is created and cannot be changed
from Settings. If the file moves, create a new service.

The image builds the widget (`npm ci && npm run build`), the Python runtime and
the knowledge base into one container, so there is no second service and no
build command to configure.

### Port contract

Voroa injects `PORT` (3000 on their platform). The image runs:

```sh
exec python -m sarang --host 0.0.0.0 --port ${PORT:-8000}
```

so the same image works locally with `-p 8000:8000` and on Voroa without a
change. `HEALTHCHECK` reads the same variable. Do not hard-code a port back into
the Dockerfile — a container that reports healthy on a port nobody serves is the
one failure mode Voroa cannot detect for us.

## 2. Environment

Set these as **runtime** variables (never build arguments — build arguments are
readable from the image):

| Variable | Value | Why |
|---|---|---|
| `SARANG_PROFILE` | `bom-india` | Sarvam ASR + LLM, Smallest TTS |
| `TTS_PROVIDER` | `smallest` | pins the arm (ADR 0004) |
| `SMALLEST_LANGUAGE` | `en` | English release configuration |
| `ENABLE_WEBRTC` | `false` | no TURN yet; go straight to the PCM path |
| `ENDPOINT_SILENCE_MS` | `800` | first Voroa deploy; see §5 |
| `ALLOWED_ORIGINS` | `https://sanket--sonwane.vercel.app` | the embedder, if used |
| `LOG_LEVEL` | `info` | |

Secrets:

- `SARVAM_API_KEY`
- `SMALLEST_API_KEY`
- `SESSION_SECRET` — generate with `openssl rand -hex 32`; rotating it
  invalidates every live session token, which is the intended blast radius.

`DATABASE_URL` is optional; retrieval falls back to local BM25 and the free tier
does not include a database in the hour budget below.

## 3. Build and deploy

Push to the configured branch (auto-deploy on) or select **Deploy**. Voroa
streams the build log, starts the new version beside the old one, and moves
traffic only after `/healthz` answers below 400.

First build is slower than later ones: `npm ci`, `uv sync` and two image stages.
Later builds reuse the dependency layers as long as `package-lock.json`,
`uv.lock` and `pyproject.toml` are unchanged.

## 4. Sizing on the free tier

Free (Nano) is 0.1 vCPU shared, 512 MB, 750 running hours per workspace per
month, 5 GB bandwidth, 500 build minutes.

Two facts decide whether it is enough:

**A Dockerfile-built service never sleeps.** Voroa's idle-sleep rules exclude
images built from a Dockerfile, even with a health check path set. So there is
**no wake-on-click cold start** and no "starting" page for a visitor. The trade
is that the service runs continuously: one always-on service uses ~744 of the
750 monthly hours, leaving no room for a second free service or a free database
in the same workspace.

**Cold start only happens on deploy or restart.** Measured locally on the
built image: imports and FastAPI ready in **2.75 s**, `/healthz` answering 200
with `ready: true` in **4.0 s**, and the running container settles at
**92.6 MiB** of memory (verified with `PORT=3000`, which is what Voroa injects,
and the Docker `HEALTHCHECK` reporting `healthy`). On 0.1 vCPU the CPU-bound
part of that scales roughly 5–10×, so a deploy-side restart is tens of seconds —
paid once, not per visitor.

The open risk is not boot time, it is the 0.1 vCPU share during a turn:
resampling, per-frame JSON and the state machine are local work. Watch two
numbers after the first deploy:

- `/healthz` `ready: true` and the probe's p50 (see §6)
- peak memory and CPU on the service's **Compute** page

If p50 climbs materially above the Fly baseline (1676 ms EOS→wire) or memory
crosses ~400 MB, move to **Nano Plus ₹149** (0.5 CPU, same 512 MB). Paying does
not change sleep behaviour for a Docker build — it buys processor.

## 5. Endpoint window

The first Voroa deploy keeps `ENDPOINT_SILENCE_MS=800` so the host change is the
only variable being measured. The 500 ms value is the measured safe floor on the
reference clip (`bench/audio/question-16k.wav`, longest internal pause 460 ms)
and is the one to re-test here, where the provider legs are shorter and jitter
should be lower: deploy 800 and 500 with 20–30 turns each and compare p50/p90,
not 5. The 650 ms point was inside the noise band on Fly and is not worth
repeating.

## 6. Verify

```bash
curl -s https://<service>.onvoroa.com/healthz | jq
curl -s https://<service>.onvoroa.com/api/config | jq '{profile, asr, llm, tts, retrieval}'
```

Expected: `ok: true`, `ready: true`, `audio_format: "pcm16-v2"`,
`endpoint_silence_ms: 800`, `vad: "silero/800ms"`, and the `bom-india` provider
set (`sarvam` / `sarvam` / `smallest`).

Then, in order:

```bash
node scripts/smoke.mjs --url https://<service>.onvoroa.com/     # HTTP surface
```

Open the URL in Chrome, allow the microphone, and run one spoken turn. The
first turn also warms the filler bank; if it is noticeably slower than the
second, that is the TTS render, not the host.

Latency, same probe and fixture as the Fly baseline:

```bash
python bench/probe/e2e_turn.py --url https://<service>.onvoroa.com --runs 5 --label voroa-800ms
```

Read `eos_to_transport_first_frame_ms` (server, last speech frame → first frame
on the wire) and `client` (last speech → first audible). Compare against the Fly
`sin` baseline in `docs/voice-pipeline-fixes.md` §7.7 rather than against a
number quoted from memory.

## 7. Rollback

Voroa keeps every release: open the service's deploy history and roll back to
the previous release. The old version starts before traffic moves, so a failed
rollback does not take the site down.

If the gateway itself is suspect, the independent fallback is the Fly app:
`flyctl deploy --image <previous image>` (see the Fly section in the runbook).
The two hosts do not share state, so switching is safe; live sessions drop and
clients reconnect.

## 8. Notes

- WebSockets are supported on the same port as HTTP. Voroa closes a connection
  that carries no data for ~100 s; the gateway already sends an application-level
  `{"type":"ping"}` every 15 s while a session is idle, so no extra keepalive is
  needed.
- WebSocket traffic *from* the service counts as outbound bandwidth; inbound
  audio does not. A voice turn is roughly 96 kB/s down at 48 kHz PCM, so 5 GB is
  in the tens of hours of speech — fine for a portfolio, not for a launch.
- There is no persistent filesystem. `SARANG_CACHE_DIR` is inside the image and
  filler clips are re-rendered after a deploy; that cost is one turn, not a
  failure.
