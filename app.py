"""
Voice Panel — browser-based backend (no livekit-agents, no anthropic SDK).

The panel is now a fluid multi-turn conversation:
- Human speaks (POST /turn).
- Backend spawns a background CHAIN: director picks a panelist, that persona
  replies, director picks the next (often a challenger), and so on, capped at
  MAX_CHAIN_TURNS. Every turn — text + audio — is published live over SSE.
- The director ends the chain by returning 'none', handing back to the human.
- If a new /turn comes in mid-chain (human interrupts), the prior chain is
  cancelled.

Endpoints:
  POST /turn       human utterance → starts a panel chain. Returns immediately.
  GET  /events     SSE stream of every event (human, turn, silence, reset,
                   chain_end). Turn events include base64 audio.
  GET  /personas   metadata for the UI.
  POST /reset      clear transcript + event log.
  GET  /           single-page UI (index.html).
"""

from __future__ import annotations

import asyncio
import base64
import datetime as _dt
import json
import logging
import os
import re
import subprocess
import tempfile

from aiohttp import ClientSession, web
from dotenv import load_dotenv

load_dotenv(override=True)  # shell may pre-set blank ANTHROPIC_API_KEY

from director import (  # noqa: E402
    detect_direct_address,
    orchestrate,
    reset_history,
    set_enabled as set_persona_enabled,
    is_enabled as persona_is_enabled,
)
from personas import PERSONAS  # noqa: E402
from health_record import get_record, set_record, format_for_prompt as record_for_prompt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("voice-panel")

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
ELEVEN_API_KEY = os.environ["ELEVEN_API_KEY"]

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
ANTHROPIC_HEADERS = {
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

# How many panelist turns can chain before we hand back to the human, no
# matter what the director wants. Keeps the panel from rambling.
MAX_CHAIN_TURNS = 4
INTER_TURN_DELAY = 0.35  # seconds between panelist turns — feels conversational


def _resolve_version() -> str:
    """Return the running build's git SHA. Render injects RENDER_GIT_COMMIT
    automatically on every deploy; locally we fall back to `git rev-parse`."""
    sha = os.environ.get("RENDER_GIT_COMMIT") or os.environ.get("GIT_COMMIT")
    if sha:
        return sha.strip()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2,
            cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "dev"


VERSION_FULL = _resolve_version()
VERSION_SHORT = VERSION_FULL[:7] if VERSION_FULL != "dev" else "dev"
STARTED_AT = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
log.info("voice-panel version=%s started_at=%s", VERSION_SHORT, STARTED_AT)

# Panel state.
_transcript: list[str] = []
_events_log: list[dict] = []
_subscribers: set[asyncio.Queue] = set()
_http: ClientSession | None = None
_chain_task: asyncio.Task | None = None


# ---------------------------------------------------------------------------
# Model calls (HTTP, no SDKs)
# ---------------------------------------------------------------------------

async def _claude_reply(persona, transcript_context: str) -> str:
    # Prepend the patient's health record so each persona can personalise.
    system = persona.system_prompt + "\n\n" + record_for_prompt()
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 110,  # ~2 sentences. Brevity is felt as energy in a demo.
        "system": system,
        "messages": [{"role": "user", "content": transcript_context}],
    }
    async with _http.post(ANTHROPIC_URL, headers=ANTHROPIC_HEADERS, json=body) as r:
        r.raise_for_status()
        data = await r.json()
    for part in data.get("content", []):
        if part.get("type") == "text":
            return part["text"].strip()
    return ""


# Per-persona voice tuning. Lower stability = more expressive prosody;
# higher style = more characterful delivery (helps salesy Marcus). All
# numbers in [0, 1]. See https://elevenlabs.io/docs/api-reference/text-to-speech
_VOICE_SETTINGS = {
    "vale":   {"stability": 0.50, "similarity_boost": 0.85, "style": 0.30, "use_speaker_boost": True},
    "pri":    {"stability": 0.45, "similarity_boost": 0.85, "style": 0.40, "use_speaker_boost": True},
    "sam":    {"stability": 0.40, "similarity_boost": 0.85, "style": 0.45, "use_speaker_boost": True},
    "marcus": {"stability": 0.35, "similarity_boost": 0.85, "style": 0.60, "use_speaker_boost": True},
}
# Multilingual v2 sounds noticeably warmer & less robotic than turbo, at the
# cost of ~400ms extra latency per call. Worth it for a panel demo.
ELEVEN_MODEL = "eleven_multilingual_v2"


async def _eleven_tts(persona, text: str) -> bytes:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{persona.voice_id}"
    payload = {
        "text": text,
        "model_id": ELEVEN_MODEL,
        "voice_settings": _VOICE_SETTINGS.get(persona.key, _VOICE_SETTINGS["vale"]),
    }
    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    async with _http.post(url, json=payload, headers=headers) as r:
        r.raise_for_status()
        return await r.read()


# macOS `say` fallback when ElevenLabs is unavailable.
_SAY_VOICE_BY_PERSONA = {
    "vale": "Samantha",
    "pri": "Karen",
    "sam": "Daniel",
    "marcus": "Alex",
}


async def _say_tts(persona_key: str, text: str) -> bytes:
    # `say` is a macOS-only binary. In production (Linux) skip it.
    import platform
    if platform.system() != "Darwin":
        raise RuntimeError("`say` is macOS-only")
    voice = _SAY_VOICE_BY_PERSONA.get(persona_key, "Samantha")
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
        path = f.name
    proc = await asyncio.create_subprocess_exec(
        "say", "-v", voice, "-o", path, text,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    with open(path, "rb") as f:
        data = f.read()
    os.unlink(path)
    return data


async def _tts(persona, text: str) -> tuple[str | None, str]:
    """Try ElevenLabs, fall back to `say`. Returns (base64, mime)."""
    try:
        audio = await _eleven_tts(persona, text)
        return base64.b64encode(audio).decode(), "audio/mpeg"
    except Exception as e:
        log.warning("ElevenLabs TTS failed (%s); falling back to macOS `say`", e)
    try:
        audio = await _say_tts(persona.key, text)
        return base64.b64encode(audio).decode(), "audio/aiff"
    except Exception as e2:
        log.warning("`say` fallback also failed (%s); text only", e2)
        return None, "audio/mpeg"


# ---------------------------------------------------------------------------
# Event pub/sub
# ---------------------------------------------------------------------------

async def _publish(event: dict) -> None:
    # Don't store the audio_b64 in the historical event log — too heavy on
    # SSE reconnects. Strip it before logging.
    light = {k: v for k, v in event.items() if k != "audio_b64"}
    _events_log.append(light)
    dead = []
    for q in _subscribers:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _subscribers.discard(q)


# ---------------------------------------------------------------------------
# Panel chain
# ---------------------------------------------------------------------------

def _strip_name_prefix(text: str, persona) -> str:
    """Belt-and-braces removal of '**Name:**' or 'Name:' leaks."""
    pat = re.compile(
        rf"^\s*(?:\*\*)?(?:{re.escape(persona.name)}|{re.escape(persona.key)})(?:\*\*)?\s*[:\-—]\s*",
        re.IGNORECASE,
    )
    return pat.sub("", text).strip()


async def _run_chain(forced_first_speaker: str | None = None) -> None:
    """Loop: orchestrate → reply → speak → publish. Stops on silence or cap.

    `forced_first_speaker`: when set, the first turn skips the multi-agent
    vote and goes directly to that persona. Used for direct-address routing
    ("Vale, what do you think?"). The chain then continues normally.
    """
    last_who: str | None = None
    spoke_count = 0
    try:
        for i in range(MAX_CHAIN_TURNS):
            transcript_ctx = "\n".join(_transcript[-24:])
            forced = forced_first_speaker if i == 0 else None
            who, considerations = await orchestrate(
                transcript_ctx,
                forced_first=forced,
                last_speaker=last_who,
            )
            # Always log the parallel vote for visibility — show raw urgency,
            # diversity bonus, and the final adjusted score that decided it.
            log.info(
                "chain turn %d: orchestrate → %s | %s",
                i, who,
                ", ".join(
                    f"{c['persona']}={c['urgency']}{c.get('bonus',0):+d}={c.get('adjusted', c['urgency'])}"
                    for c in considerations
                ),
            )

            if who is None or who not in PERSONAS:
                if spoke_count == 0:
                    await _publish({"type": "silence"})
                break

            persona = PERSONAS[who]
            try:
                reply_text = await _claude_reply(persona, transcript_ctx)
            except Exception as e:
                log.exception("claude failed in chain turn %d", i)
                await _publish({"type": "error", "text": f"claude: {e}"})
                break
            reply_text = _strip_name_prefix(reply_text, persona)
            if not reply_text:
                break

            _transcript.append(f"{persona.name}: {reply_text}")
            audio_b64, audio_mime = await _tts(persona, reply_text)

            await _publish({
                "type": "turn",
                "speaker": persona.name,
                "persona_key": persona.key,
                "is_bad_actor": persona.is_bad_actor,
                "text": reply_text,
                "audio_b64": audio_b64,
                "audio_mime": audio_mime,
            })
            last_who = who
            spoke_count += 1
            await asyncio.sleep(INTER_TURN_DELAY)
    except asyncio.CancelledError:
        log.info("chain cancelled (human interrupted)")
        await _publish({"type": "chain_cancelled"})
        raise
    finally:
        await _publish({"type": "chain_end"})


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------

async def handle_turn(request: web.Request) -> web.Response:
    body = await request.json()
    user_text = (body.get("text") or "").strip()
    if not user_text:
        return web.json_response({"error": "empty"}, status=400)

    # Cancel any in-flight chain — human is interrupting.
    global _chain_task
    if _chain_task and not _chain_task.done():
        _chain_task.cancel()
        try:
            await _chain_task
        except asyncio.CancelledError:
            pass

    _transcript.append(f"Human: {user_text}")
    forced = detect_direct_address(user_text)
    if forced:
        log.info("direct-address detected: forcing %s as first speaker", forced)
    await _publish({
        "type": "human",
        "text": user_text,
        **({"directly_addressed": forced} if forced else {}),
    })

    _chain_task = asyncio.create_task(_run_chain(forced_first_speaker=forced))
    return web.json_response({"ok": True, "directly_addressed": forced})


async def handle_events(request: web.Request) -> web.StreamResponse:
    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )
    await resp.prepare(request)

    # Replay history (without audio — too heavy).
    for ev in list(_events_log):
        await resp.write(f"data: {json.dumps(ev)}\n\n".encode())

    q: asyncio.Queue = asyncio.Queue(maxsize=128)
    _subscribers.add(q)
    try:
        while True:
            ev = await q.get()
            await resp.write(f"data: {json.dumps(ev)}\n\n".encode())
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        _subscribers.discard(q)
    return resp


async def handle_personas(request: web.Request) -> web.Response:
    return web.json_response(
        {
            k: {
                "key": p.key,
                "name": p.name,
                "is_bad_actor": p.is_bad_actor,
                "voice_id": p.voice_id,
                "enabled": persona_is_enabled(k),
            }
            for k, p in PERSONAS.items()
        }
    )


async def handle_persona_enabled(request: web.Request) -> web.Response:
    key = request.match_info["key"]
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    enabled = bool(body.get("enabled"))
    if not set_persona_enabled(key, enabled):
        return web.json_response({"error": f"unknown persona '{key}'"}, status=404)
    log.info("persona %s -> %s", key, "enabled" if enabled else "disabled")
    return web.json_response({"key": key, "enabled": enabled})


async def handle_version(request: web.Request) -> web.Response:
    return web.json_response({
        "commit": VERSION_FULL,
        "short": VERSION_SHORT,
        "started_at": STARTED_AT,
    })


async def handle_get_record(request: web.Request) -> web.Response:
    return web.json_response(get_record())


async def handle_set_record(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    merged = set_record(body)
    log.info("health record updated: name=%s, conditions=%s",
             merged.get("name"), merged.get("conditions"))
    return web.json_response(merged)


async def handle_reset(request: web.Request) -> web.Response:
    global _chain_task
    if _chain_task and not _chain_task.done():
        _chain_task.cancel()
        try:
            await _chain_task
        except asyncio.CancelledError:
            pass
    _transcript.clear()
    _events_log.clear()
    reset_history()   # also clear diversity-weighting memory
    await _publish({"type": "reset"})
    return web.json_response({"ok": True})


async def handle_index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


# Avatar lookup: serves the first matching static/<key>.{jpg,jpeg,png,webp,gif}
# Returns 404 cleanly so the frontend can fall back to an emoji avatar.
_AVATAR_EXTS = ("jpg", "jpeg", "png", "webp", "gif")

async def handle_avatar(request: web.Request) -> web.FileResponse:
    key = request.match_info["key"]
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    for ext in _AVATAR_EXTS:
        path = os.path.join(static_dir, f"{key}.{ext}")
        if os.path.isfile(path):
            return web.FileResponse(path, headers={"Cache-Control": "public, max-age=300"})
    raise web.HTTPNotFound()


async def on_startup(app: web.Application) -> None:
    global _http
    _http = ClientSession()


async def on_cleanup(app: web.Application) -> None:
    if _http is not None:
        await _http.close()


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/personas", handle_personas)
    app.router.add_post("/turn", handle_turn)
    app.router.add_post("/reset", handle_reset)
    app.router.add_get("/events", handle_events)
    app.router.add_get("/avatar/{key}", handle_avatar)
    app.router.add_get("/version", handle_version)
    app.router.add_get("/health_record", handle_get_record)
    app.router.add_post("/health_record", handle_set_record)
    app.router.add_post("/personas/{key}/enabled", handle_persona_enabled)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


if __name__ == "__main__":
    # In production (e.g. Render), PORT is injected and we must bind 0.0.0.0.
    # Locally, default to 127.0.0.1:8080 so we don't surprise-expose the dev server.
    port = int(os.environ.get("PORT", 8080))
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    print(f"voice-panel up on http://{host}:{port}", flush=True)
    web.run_app(make_app(), host=host, port=port, print=None)
