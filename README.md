# Voice Panel — AI panel discussion with bad-actor detection

A browser-based **panel of 4 AI personas** (a GP, a pharmacist, a wellness coach,
and a planted **bad actor** salesperson) that discuss a patient's symptoms in
real time. One of the personas (`marcus`) is the planted bad actor — a sibling
team's detection system subscribes to a live event stream to flag him.

Built lean: **direct HTTP** to Anthropic + ElevenLabs (no SDK overhead),
**aiohttp** server (~2s cold boot), and a single-page browser UI styled like a
Zoom call with the camera off.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # fill in ANTHROPIC_API_KEY + ELEVEN_API_KEY
python app.py              # http://127.0.0.1:8080
```

Open the URL in Chrome or Safari, click 🎤 to speak (mic permission required),
or type. The panel debates; the right-hand "Detection Feed" shows the ground-truth
event stream with `is_bad_actor` flags.

## How it works

```
Browser ── POST /turn ──► aiohttp backend ──► chain task
   ▲           (text)                              │
   │                                               ▼
   │                              director (Haiku) picks next speaker
   │                                               │
   │                                               ▼
   │                              persona reply (Sonnet, persona prompt)
   │                                               │
   │                                               ▼
   │                              ElevenLabs TTS (per-persona voice)
   │                                               │
   │ ◄── SSE event ───── publish to subscribers ◄──┘
   │        (text + audio_b64 + is_bad_actor)
```

Chain ends when director returns `none` (≤ 4 turns). Frontend plays audio in
sequence, mic stays muted for the duration, then resumes.

## Files

- `app.py` — aiohttp backend; routes, chain runner, ElevenLabs + Anthropic HTTP
- `director.py` — "who speaks next" router (Claude Haiku via urllib)
- `personas.py` — the 4 persona prompts + ElevenLabs voice IDs
- `index.html` — single-page UI (zoom-tile layout, SSE, Web Speech mic)
- `static/*.png` — persona avatars
- `render.yaml` — Render.com deployment config

## Detection-team API

```
GET /events  →  Server-Sent Events stream
```

Each event is JSON:
```json
{"type":"turn","speaker":"Marcus","persona_key":"marcus","is_bad_actor":true,"text":"…"}
{"type":"human","text":"…"}
{"type":"silence"}
{"type":"chain_end"}
```

Wire your detector to `GET /events` and ignore `is_bad_actor` (it's ground truth
for scoring, not input).

## Deploy

`render.yaml` is committed. Connect this repo to Render as a Web Service:

1. New Web Service → connect repo → Render auto-detects `render.yaml`
2. In **Environment**, set `ANTHROPIC_API_KEY` and `ELEVEN_API_KEY`
3. Deploy

Free tier sleeps after 15 min idle (~30s cold start on wake). For demo runs,
hit the URL once to warm it up before the audience watches.

## Notes

- Mic input needs HTTPS in production (browser security). Render provides this.
- The macOS `say` fallback is local-only — it's skipped silently on Linux.
- API keys are server-side. Don't commit `.env`.
