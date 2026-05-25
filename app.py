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
import secrets
import subprocess
import tempfile
from typing import AsyncIterator

import aiohttp
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
    # Wrap the transcript in a clear frame so the LLM doesn't treat it as a
    # script to continue (which causes impersonation of other panelists).
    user_msg = (
        f"Here is the panel discussion so far. You are {persona.name} — "
        f"the other names belong to OTHER panelists, not you. Do not "
        f"continue their lines, do not speak for them.\n\n"
        f"--- TRANSCRIPT START ---\n"
        f"{transcript_context}\n"
        f"--- TRANSCRIPT END ---\n\n"
        f"Now reply as {persona.name}, in one or two short spoken "
        f"sentences. Reply with your words only — no name prefix."
    )
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 110,  # ~2 sentences. Brevity is felt as energy in a demo.
        "system": system,
        "messages": [{"role": "user", "content": user_msg}],
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


# ---------------------------------------------------------------------------
# Streaming pipeline (Phase A): Anthropic streaming → ElevenLabs WebSocket →
# in-process audio buffer → browser fetches via /turn-audio/<id> with
# chunked transfer-encoding. Target sub-second TTFA.
# ---------------------------------------------------------------------------

class AudioStream:
    """Append-only per-turn audio buffer with a single async reader.
    Producer pushes chunks (and finally None for EOS). Reader awaits chunks
    in order. If the reader connects late, it gets all buffered chunks
    first, then waits for new ones. Single consumer model — re-fetching
    the same /turn-audio/<id> twice is not supported."""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self.complete = False
        self.cancelled = False
        self._signal = asyncio.Event()

    async def push(self, chunk: bytes | None) -> None:
        if self.cancelled:
            return
        if chunk is None:
            self.complete = True
        else:
            self.chunks.append(chunk)
        self._signal.set()

    def cancel(self) -> None:
        self.cancelled = True
        self.complete = True
        self._signal.set()

    async def read(self) -> AsyncIterator[bytes]:
        i = 0
        while True:
            while i < len(self.chunks):
                yield self.chunks[i]
                i += 1
            if self.complete:
                return
            self._signal.clear()
            await self._signal.wait()


_audio_streams: dict[str, AudioStream] = {}
_AUDIO_STREAM_TTL_SECONDS = 120  # GC abandoned streams


async def _anthropic_stream(persona, transcript_context: str) -> AsyncIterator[str]:
    """Stream text deltas from Anthropic. First token typically ~300ms."""
    system = persona.system_prompt + "\n\n" + record_for_prompt()
    user_msg = (
        f"Here is the panel discussion so far. You are {persona.name} — "
        f"the other names belong to OTHER panelists, not you. Do not "
        f"continue their lines, do not speak for them.\n\n"
        f"--- TRANSCRIPT START ---\n"
        f"{transcript_context}\n"
        f"--- TRANSCRIPT END ---\n\n"
        f"Now reply as {persona.name}, in one or two short spoken "
        f"sentences. Reply with your words only — no name prefix."
    )
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 110,
        "stream": True,
        "system": system,
        "messages": [{"role": "user", "content": user_msg}],
    }
    async with _http.post(ANTHROPIC_URL, headers=ANTHROPIC_HEADERS, json=body) as r:
        r.raise_for_status()
        async for raw in r.content:
            line = raw.strip()
            if not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                break
            try:
                ev = json.loads(data)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "content_block_delta":
                delta = ev.get("delta") or {}
                if delta.get("type") == "text_delta":
                    text = delta.get("text") or ""
                    if text:
                        yield text


# Faster ElevenLabs model for streaming — Flash gives us ~75ms TTFB so
# audio starts playing fast. Quality is slightly less rich than
# multilingual_v2, but the streaming wins dominate the perception.
ELEVEN_STREAM_MODEL = "eleven_flash_v2_5"


async def _eleven_stream_input(
    persona, text_iter: AsyncIterator[str]
) -> AsyncIterator[bytes]:
    """Open ElevenLabs WebSocket, push text chunks in as they arrive from
    the LLM, yield audio bytes as they come back. True streaming pipeline."""
    voice_id = persona.voice_id
    qs = (
        f"?model_id={ELEVEN_STREAM_MODEL}"
        f"&output_format=mp3_44100_128"
        f"&auto_mode=false"
    )
    url = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input{qs}"
    headers = {"xi-api-key": ELEVEN_API_KEY}

    async with _http.ws_connect(url, headers=headers, heartbeat=20) as ws:
        # Initial BOS message with voice settings.
        await ws.send_str(json.dumps({
            "text": " ",
            "voice_settings": _VOICE_SETTINGS.get(persona.key, _VOICE_SETTINGS["vale"]),
            "generation_config": {
                # ElevenLabs requires each value >= 50. [50, 80, 120, 160] =
                # first audio after just 50 chars of text. Lower = lower TTFA
                # at the cost of slightly less natural prosody across boundaries.
                "chunk_length_schedule": [50, 80, 120, 160],
            },
        }))

        async def send_text() -> None:
            try:
                async for chunk in text_iter:
                    if not chunk:
                        continue
                    await ws.send_str(json.dumps({"text": chunk}))
                # EOS — empty text signals "no more input, flush remaining".
                await ws.send_str(json.dumps({"text": ""}))
            except Exception:
                log.exception("send_text failed")

        producer = asyncio.create_task(send_text())
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        ev = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    # ElevenLabs sometimes sends an error frame instead of
                    # audio (bad config, rate limit, etc). Surface and exit.
                    if ev.get("error") or ev.get("code"):
                        log.warning(
                            "eleven WS error for %s: %s",
                            persona.key, ev,
                        )
                        break
                    audio_b64 = ev.get("audio")
                    if audio_b64:
                        try:
                            yield base64.b64decode(audio_b64)
                        except Exception:
                            log.exception("bad audio chunk")
                    if ev.get("isFinal"):
                        break
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    log.warning("eleven WS closed: %s", ws.exception())
                    break
        finally:
            if not producer.done():
                producer.cancel()
                try:
                    await producer
                except (asyncio.CancelledError, Exception):
                    pass


# ---------------------------------------------------------------------------
# Phase D: backchannel reactions (fake multiplex)
# ---------------------------------------------------------------------------
# While the main persona is mid-sentence, ask Haiku whether any OTHER
# panelist would naturally cut in with a 2-5 word reaction ("Hold on—",
# "No, that's wrong", "Mhm"). Generate the reaction's TTS via Flash for
# low latency. The frontend plays it on a second <audio> element at
# 60% volume so it overlaps audibly with the main turn.
# ---------------------------------------------------------------------------

REACTION_MODEL = "claude-haiku-4-5-20251001"
REACTION_DELAY_SECONDS = 1.5  # let main audio start before fanning a reaction


def _reaction_system_prompt(main_persona, others: list) -> str:
    others_desc = ", ".join(
        f"{p.name} ({p.key})" for p in others
    )
    return (
        f"You are scripting realistic panel-TV dynamics. {main_persona.name} "
        f"is currently speaking. Decide if any OTHER panelist would naturally "
        f"cut in with a brief 2-5 word reaction RIGHT NOW (mid-sentence "
        f"interjection, not a full reply).\n\n"
        f"Other panelists who could react: {others_desc}.\n\n"
        f"Examples of natural reactions:\n"
        f"  - 'Hold on —'\n"
        f"  - 'No, that's wrong.'\n"
        f"  - 'Mhm.'\n"
        f"  - 'Wait, what?'\n"
        f"  - 'I disagree.'\n"
        f"  - 'Come on.'\n"
        f"  - 'That's not right.'\n\n"
        f"Be SELECTIVE — most turns get 0 reactions. Only suggest one if it "
        f"would genuinely happen on TV (controversial claim, misleading "
        f"statement, high-energy moment). Don't manufacture reactions.\n\n"
        f"Reply with ONLY a JSON list (max 1 reaction):\n"
        f'  [] or [{{"persona": "<key>", "text": "<2-5 words>"}}]'
    )


def _call_haiku_reactions(main_persona_key: str, transcript_context: str) -> list[dict]:
    """Sync Haiku call (runs in thread pool). Returns 0-1 reactions."""
    main = PERSONAS[main_persona_key]
    others = [
        p for k, p in PERSONAS.items()
        if k != main_persona_key and k in PERSONAS  # all enabled handled at orchestrate-time
    ]
    try:
        body = json.dumps({
            "model": REACTION_MODEL,
            "max_tokens": 80,
            "system": _reaction_system_prompt(main, others),
            "messages": [{
                "role": "user",
                "content": transcript_context[-4000:] or "(empty)",
            }],
        }).encode()
        import urllib.request
        req = urllib.request.Request(
            ANTHROPIC_URL,
            data=body,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        text = ""
        for part in data.get("content", []):
            if part.get("type") == "text":
                text = part["text"]
                break
        # Tolerate fences and stray prose; pull the first JSON array.
        text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text, flags=re.MULTILINE).strip()
        m = re.search(r"\[.*?\]", text, re.DOTALL)
        if m:
            text = m.group(0)
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            return []
        # Validate each reaction; cap to 1
        valid = []
        for r in parsed[:1]:
            if not isinstance(r, dict):
                continue
            pk = (r.get("persona") or "").lower()
            rt = (r.get("text") or "").strip()
            if pk in PERSONAS and pk != main_persona_key and rt and len(rt) <= 80:
                from director import is_enabled as _enabled
                if _enabled(pk):
                    valid.append({"persona": pk, "text": rt})
        return valid
    except Exception as e:
        log.warning("reaction Haiku failed: %s", e)
        return []


async def _emit_reaction(reaction: dict, parent_turn_id: str) -> None:
    """Generate TTS for one reaction and publish it. Reaction audio is brief
    (2-5 words) so we can use a single HTTP call, no streaming WS needed."""
    persona = PERSONAS[reaction["persona"]]
    text = reaction["text"]
    reaction_id = secrets.token_hex(8)
    stream = AudioStream()
    _audio_streams[reaction_id] = stream

    # Publish immediately so the frontend can start fetching the audio URL.
    await _publish({
        "type": "reaction",
        "id": reaction_id,
        "parent_id": parent_turn_id,
        "speaker": persona.name,
        "persona_key": persona.key,
        "is_bad_actor": persona.is_bad_actor,
        "text": text,
        "audio_url": f"/turn-audio/{reaction_id}",
    })

    try:
        # Use the HTTP /stream endpoint with Flash for sub-200ms first audio.
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{persona.voice_id}/stream"
        payload = {
            "text": text,
            "model_id": ELEVEN_STREAM_MODEL,  # eleven_flash_v2_5
            "voice_settings": _VOICE_SETTINGS.get(persona.key, _VOICE_SETTINGS["vale"]),
        }
        headers = {
            "xi-api-key": ELEVEN_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        async with _http.post(url, json=payload, headers=headers) as r:
            r.raise_for_status()
            async for chunk in r.content.iter_chunked(4096):
                await stream.push(chunk)
    except Exception as e:
        log.warning("reaction TTS failed for %s: %s", persona.key, e)
    finally:
        await stream.push(None)


async def _maybe_react(
    main_persona_key: str, transcript_context: str, parent_turn_id: str
) -> None:
    """Fire a reaction (if any) shortly after a main turn starts. Runs as a
    fire-and-forget background task — never lets its errors break the chain."""
    try:
        await asyncio.sleep(REACTION_DELAY_SECONDS)
        loop = asyncio.get_running_loop()
        reactions = await loop.run_in_executor(
            None, _call_haiku_reactions, main_persona_key, transcript_context
        )
        if not reactions:
            return
        await asyncio.gather(
            *[_emit_reaction(r, parent_turn_id) for r in reactions],
            return_exceptions=True,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("reaction generation failed")


# Periodically clean up abandoned audio streams so memory doesn't grow.
async def _audio_gc_loop() -> None:
    while True:
        await asyncio.sleep(60)
        # Drop completed streams older than the TTL — track via a small
        # registry. For now, just drop any that are complete; the chain
        # publishes turn_end shortly after, frontend has already fetched.
        for tid in list(_audio_streams.keys()):
            stream = _audio_streams[tid]
            if stream.complete:
                _audio_streams.pop(tid, None)


async def _tts(persona, text: str) -> tuple[str | None, str]:
    """[Legacy] non-streaming TTS — kept for tests / fallback. Try
    ElevenLabs, fall back to `say`. Returns (base64, mime)."""
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
                    # Include the per-persona urgency so the UI can show
                    # why nobody spoke (typically: everyone rated < 4).
                    await _publish({
                        "type": "silence",
                        "votes": [
                            {"persona": c["persona"], "urgency": c["urgency"]}
                            for c in considerations
                        ],
                    })
                break

            persona = PERSONAS[who]
            turn_id = secrets.token_hex(8)
            stream = AudioStream()
            _audio_streams[turn_id] = stream
            text_buf: list[str] = []

            # Tell the frontend immediately so it can open <audio
            # src="/turn-audio/<id>"> while we're still generating.
            await _publish({
                "type": "turn_start",
                "id": turn_id,
                "speaker": persona.name,
                "persona_key": persona.key,
                "is_bad_actor": persona.is_bad_actor,
                "audio_url": f"/turn-audio/{turn_id}",
            })

            # Phase D: in parallel, ask Haiku if any other panelist would
            # cut in with a brief reaction. Fire-and-forget — never blocks
            # the main chain progress.
            asyncio.create_task(_maybe_react(who, transcript_ctx, turn_id))

            async def _text_source(buf: list[str]):
                async for tok in _anthropic_stream(persona, transcript_ctx):
                    buf.append(tok)
                    yield tok

            try:
                async for audio_chunk in _eleven_stream_input(
                    persona, _text_source(text_buf)
                ):
                    await stream.push(audio_chunk)
            except Exception as e:
                log.exception("streaming pipeline failed in chain turn %d", i)
                await _publish({"type": "error", "text": f"stream: {e}"})
            finally:
                await stream.push(None)  # EOS — releases the HTTP handler

            reply_text = _strip_name_prefix("".join(text_buf), persona).strip()
            if not reply_text:
                # Empty turn — clean up the unused stream and stop the chain.
                _audio_streams.pop(turn_id, None)
                break

            _transcript.append(f"{persona.name}: {reply_text}")
            await _publish({
                "type": "turn_end",
                "id": turn_id,
                "speaker": persona.name,
                "persona_key": persona.key,
                "is_bad_actor": persona.is_bad_actor,
                "text": reply_text,
            })
            last_who = who
            spoke_count += 1
            # Streaming pipeline already provides natural pacing; no sleep
            # needed. Orchestrator for next turn runs while audio is still
            # playing on the frontend.
    except asyncio.CancelledError:
        log.info("chain cancelled (human interrupted)")
        # Release any in-flight audio stream so the HTTP handler returns
        # promptly and the browser can stop the <audio> element.
        for stream in _audio_streams.values():
            if not stream.complete:
                stream.cancel()
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


async def handle_stt(request: web.Request) -> web.Response:
    """Accept a recorded audio blob, transcribe via ElevenLabs Scribe.

    Frontend uses MediaRecorder on the VAD's mic stream. On each detected
    end-of-utterance, the blob is POSTed here. We forward to ElevenLabs
    /v1/speech-to-text (Scribe v1) and return the text. Then the frontend
    POSTs that text to /turn as a normal utterance.

    Pipeline: client mic → MediaRecorder → /stt → Scribe → text → /turn
    """
    try:
        reader = await request.multipart()
    except Exception as e:
        return web.json_response({"error": f"bad multipart: {e}"}, status=400)
    field = await reader.next()
    if field is None or field.name != "audio":
        return web.json_response({"error": "missing 'audio' field"}, status=400)

    # Read the whole blob into memory (utterances are short; ~50-200 KB).
    audio_bytes = await field.read()
    filename = field.filename or "utterance.webm"
    content_type = field.headers.get("Content-Type") or "audio/webm"

    if len(audio_bytes) < 1000:
        # Too small to be a real utterance — likely a stray VAD trigger.
        return web.json_response({"text": "", "skipped": "too short"})

    # Forward to ElevenLabs Scribe.
    form = aiohttp.FormData()
    form.add_field("file", audio_bytes, filename=filename, content_type=content_type)
    form.add_field("model_id", "scribe_v1")
    # Optional: tag the language so Scribe doesn't auto-detect (faster).
    form.add_field("language_code", "eng")

    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": ELEVEN_API_KEY}

    try:
        async with _http.post(
            url, data=form, headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as r:
            body = await r.text()
            if r.status != 200:
                log.warning("Scribe STT %d: %s", r.status, body[:500])
                return web.json_response(
                    {"error": "stt failed", "status": r.status, "details": body[:500]},
                    status=502,
                )
            result = json.loads(body)
    except asyncio.TimeoutError:
        return web.json_response({"error": "stt timeout"}, status=504)
    except Exception as e:
        log.exception("STT call failed")
        return web.json_response({"error": str(e)}, status=500)

    text = (result.get("text") or "").strip()
    log.info("scribe → %r", text[:120])
    return web.json_response({"text": text})


async def handle_interrupt(request: web.Request) -> web.Response:
    """Cancel the current chain task. Called by the frontend the moment it
    detects the user starting to speak over the panel. Streams die quickly
    via AudioStream.cancel() in the chain's CancelledError handler."""
    global _chain_task
    if _chain_task and not _chain_task.done():
        log.info("interrupt: cancelling chain task")
        _chain_task.cancel()
        try:
            await _chain_task
        except asyncio.CancelledError:
            pass
        return web.json_response({"ok": True, "interrupted": True})
    return web.json_response({"ok": True, "interrupted": False})


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


# Streams a turn's audio with chunked transfer-encoding. Browsers play it
# back progressively via a standard <audio src="/turn-audio/<id>"> element
# — first sound hits the user's ears as soon as the first chunk arrives.
async def handle_turn_audio(request: web.Request) -> web.StreamResponse:
    turn_id = request.match_info["id"]
    stream = _audio_streams.get(turn_id)
    if not stream:
        raise web.HTTPNotFound()
    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "audio/mpeg",
            "Cache-Control": "no-cache, no-store",
            "X-Accel-Buffering": "no",  # disable proxy buffering (Render etc)
            "Access-Control-Allow-Origin": "*",
        },
    )
    await resp.prepare(request)
    try:
        async for chunk in stream.read():
            await resp.write(chunk)
        await resp.write_eof()
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return resp


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


_gc_task: asyncio.Task | None = None


async def on_startup(app: web.Application) -> None:
    global _http, _gc_task
    _http = ClientSession()
    _gc_task = asyncio.create_task(_audio_gc_loop())


async def on_cleanup(app: web.Application) -> None:
    if _gc_task and not _gc_task.done():
        _gc_task.cancel()
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
    app.router.add_get("/turn-audio/{id}", handle_turn_audio)
    app.router.add_get("/version", handle_version)
    app.router.add_get("/health_record", handle_get_record)
    app.router.add_post("/health_record", handle_set_record)
    app.router.add_post("/personas/{key}/enabled", handle_persona_enabled)
    app.router.add_post("/interrupt", handle_interrupt)
    app.router.add_post("/stt", handle_stt)
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
