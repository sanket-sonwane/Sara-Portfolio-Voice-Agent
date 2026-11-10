# ADR 0004 — Smallest Lightning is the TTS default; Cartesia is the comparison arm

**Date:** 2026-11-10
**Status:** accepted
**Amends:** ADR 0002 §3 and the TTS row of its §6.
**Context:** Cartesia's free allowance was exhausted mid-demo, and its free tier
caps the account at two concurrent contexts. The turn engine submits one TTS request
per answer segment, so a three-segment answer lost its last segment. The audience is
Indian and the pinned Cartesia voice (`Ella`) is en-AU.

---

## 1. Decision

`bom-india` uses **Smallest AI Lightning** for TTS:

| Setting | Value |
|---|---|
| Provider | Smallest AI |
| Model | `lightning_v3.1_pro` |
| Voice | `rhea` (en-IN) |
| Language | `hi` — Indian-accented English with Hindi code-switching |
| Transport | WebSocket, one socket per in-flight segment |
| Sample rate | 24 kHz PCM16, resampled to 48 kHz egress |

`TTS_PROVIDER` still selects the arm, so `cartesia`, `sarvam` and `synthetic`
remain available. `iad-global` keeps Cartesia, where its lower latency and RTF are
worth more than accent.

## 2. Measurements

Same passage, same segmentation, same metric names, five runs of five segments, one
laptop on a consumer link in India (`smallest_ttfa.py`, `cartesia_ttfa.py`):

| Metric | Cartesia p50 | Smallest p50 |
|---|---:|---:|
| TTFA, first segment | 156 ms | 246 ms |
| Subsequent segment TTFA | 4.9 ms | 259 ms |
| RTF | 0.10 | 0.58 |
| Cancel-to-silence | 137 ms | 89 ms (socket close) |

From the production region (`sin`), single sentence, 5 samples each: Cartesia
**186 ms**, Smallest **216 ms** to first audio chunk. In a Fly machine, TCP RTT to
either provider is 1–2 ms because Fly proxies egress locally, so only the full
TLS+WebSocket measurement means anything.

End to end, through the deployed gateway (`e2e_turn.py --url`), Smallest lands
end-of-speech → first answer audio at **926 ms** p50, with 234 ms of that being TTS
TTFA.

## 3. Why slower is the right trade

1. **Concurrency.** Cartesia's free tier allows two contexts; Lightning allows
   10–15. The orchestrator fans out one task per segment. Two contexts meant the
   third segment was dropped, and `TTS_MAX_CONCURRENCY=2` was the workaround. That
   is a hard ceiling on a product that speaks multi-segment answers.
2. **Cost.** $0.195 vs $0.50 per 10K characters (2.6×), and $10 of free credit is
   roughly 51K characters against Cartesia's 20K credits/month.
3. **Accent.** `language: "hi"` on `rhea` produces Indian-accented English. Verified
   by round-tripping six rendered samples through Sarvam ASR as `en-IN`: all six came
   back as the exact input sentence, proper nouns included.
4. **The gap is not the budget.** 30–60 ms against a ~930 ms server turn, and
   smaller than the turn's own run-to-run spread, is not worth an accent change.

RTF 0.58 is the one number that got worse. It is still below 1.0, so synthesis keeps
ahead of playback for a single segment, and the orchestrator pipelines segments
anyway.

## 4. Protocol findings that shape the adapter

These were measured, not read off the docs, and each one is easy to reintroduce:

| Finding | Consequence |
|---|---|
| `complete_backoff_ms` defaults to **4 s** | Every request sends `0`. With the default, RTF measures 2.38 and every turn looks 4 s slow. |
| `context_close: true` does **not** stop audio | It kept sending for 9.3 s p50 (143–226 late chunks). Barge-in closes the socket instead (89 ms). |
| `context_id` is not echoed; `request_id` comes back as `external_request_id` | Attribution is by socket, not by frame. One in-flight request per socket. |
| Two contexts on one socket did not both receive `complete` | One socket per in-flight segment, drawn from a pool and reused sequentially. |
| `continue: true` buys prosody across segments | Rejected: +90–160 ms to first audio, measured. |
| No 48 kHz output | 24 kHz requested, upsampled with a stateful `StreamResampler`. |
| Max 250 characters per request | The segmenter already emits ~140. |

## 5. Consequences

- `apps/api/sarang/tts/smallest.py` is the adapter; `CartesiaTTS` is untouched.
- `SMALLEST_API_KEY` is a backend-only secret (`fly secrets`), never a browser
  variable. `/healthz` reports `tts: smallest` and `tts_ready: true`.
- `fly.toml` pins `TTS_PROVIDER = "smallest"` in `[env]` so the deployed arm is
  reviewable in the repository rather than hidden in a secret.
- The filler bank is rendered through whichever arm is active, so cached fillers
  change voice when the provider changes. The cache directory is keyed by provider
  and voice, so stale Cartesia audio cannot leak into a Smallest session.

## 6. Revisit when

- Cartesia raises its free-tier concurrency above the number of segments a
  35-word answer produces, **and** an en-IN voice measures within ~20 ms of the
  current one.
- Smallest ships a 48 kHz output (the resampler disappears).
- The measured RTF rises toward 1.0 under multi-segment load, which would mean
  synthesis can no longer keep ahead of playback.
