# CLAUDE.md — Voice Panel

Context for Claude Code working in this repo. Read first.

## What this is

A browser-based **AI panel discussion** (gen-to-commerce / human–agent
collaboration hackathon theme). Four AI personas — a GP, a pharmacist, a wellness
coach, and a **planted bad-actor** salesperson — debate a patient's symptoms in
real time. A sibling team's detection system subscribes to a live SSE event feed
to flag the bad actor.

## Architecture

```
Browser ── POST /turn ──► aiohttp backend ──► chain task (≤ 4 turns)
   ▲           (text)                              │
   │                                       director (Haiku) picks who
   │                                               │
   │                                       Claude Sonnet generates reply
   │                                               │
   │                                       ElevenLabs TTS (per-persona voice)
   │ ◄── SSE event ───── publish ──────────────────┘
   │        (text + audio_b64 + is_bad_actor)
```

- **`app.py`** — aiohttp backend; `/turn` POST starts a chain task; `/events` is
  the SSE feed with audio embedded; `/avatar/<key>` serves persona photos.
- **`director.py`** — picks who speaks next (Claude Haiku via `urllib`).
- **`personas.py`** — 4 persona prompts + ElevenLabs voice IDs.
- **`index.html`** — single-page UI, zoom-tile layout, Web Speech API mic input,
  serial audio queue, active-speaker glow.

### Why not the anthropic SDK or livekit-agents?

Both have heavy import trees that took >100s to load on the dev machine (macOS
`com.apple.provenance` xattr forces Gatekeeper scans on every `.so`/`.pyc` read).
Direct HTTP calls via `aiohttp` and `urllib` keep boot under 2s. This is also
what makes the Render free-tier cold start tolerable.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # fill in ANTHROPIC_API_KEY + ELEVEN_API_KEY
python app.py              # http://127.0.0.1:8080
```

Mic permission required in Chrome/Safari; click the 🎤 button.

## Tunable / known sharp edges

- **Chain length** — `MAX_CHAIN_TURNS = 4` in `app.py`. Higher = more debate, but
  also more silence between human turns.
- **Persona brevity** — `max_tokens = 110` in `_claude_reply`. Lower for punchier
  panel-TV pace; higher for fuller arguments.
- **TTS model** — `eleven_multilingual_v2` for warm voices; swap to
  `eleven_flash_v2_5` (in `ELEVEN_MODEL`) for ~10× lower latency at some
  quality cost.
- **Director cadence** — `director.py` prompt biases toward chaining 2-3
  panelist turns before yielding to the human. Tune the rules section to make
  Marcus appear more/less often.
- **ElevenLabs free tier** sometimes flags accounts for "unusual activity"
  (VPN/proxy heuristic). $5/mo Starter fixes it. The macOS `say` fallback in
  `_say_tts` covers local dev when ElevenLabs is down; skipped silently on Linux.

## Don't

- Don't reintroduce the anthropic SDK or livekit-agents — they bloat boot time
  on this venv's filesystem.
- Don't commit `.env`. Even key prefixes are sensitive.
- Don't let the `marcus` persona generate genuinely dangerous instructions; the
  prompt keeps him bounded to fictional upsells and fake urgency.

## Detection-team contract

`GET /events` is an SSE stream. Each event is JSON:

```
{"type":"human","text":"…"}
{"type":"turn","speaker":"Marcus","persona_key":"marcus","is_bad_actor":true,"text":"…","audio_b64":"…","audio_mime":"audio/mpeg"}
{"type":"silence"}
{"type":"chain_end"}
{"type":"chain_cancelled"}   # human interrupted mid-chain
{"type":"reset"}
```

`is_bad_actor` is **ground truth for scoring** — the detector shouldn't read it
as input. Detection team can also poll the running transcript via `/personas`
metadata.
