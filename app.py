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
import time
from collections import deque
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
from personas import PERSONAS, PERSONA_SETS  # noqa: E402
from health_record import get_record, set_record, format_for_prompt as record_for_prompt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("voice-panel")

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
ELEVEN_API_KEY = os.environ["ELEVEN_API_KEY"]
# Cartesia key: env var wins; fallback is committed for demo expedience.
# ROTATE THIS in the Cartesia dashboard right after the demo — it's public.
CARTESIA_API_KEY = os.environ.get("CARTESIA_API_KEY", "sk_car_uAPGrHGAx7CbGAKEmBPiXZ")

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

# Runtime TTS provider toggle. Default elevenlabs; switchable via
# POST /tts/provider {"provider": "cartesia"|"elevenlabs"}. Captured at
# turn-start so a mid-turn switch never strands a half-streamed reply.
_tts_provider: str = "elevenlabs"


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
    the same /turn-audio/<id> twice is not supported.

    `mime` is the Content-Type for the eventual /turn-audio HTTP response.
    ElevenLabs streams produce audio/mpeg; Cartesia raw-PCM streams produce
    audio/wav (with a streaming WAV header pushed as the first chunk)."""

    def __init__(self, mime: str = "audio/mpeg") -> None:
        self.chunks: list[bytes] = []
        self.complete = False
        self.cancelled = False
        self.mime = mime
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


# ---------------------------------------------------------------------------
# Diagnostics — server-side ring buffer + per-stream metadata
# ---------------------------------------------------------------------------
# Append-only structures inspected via GET /diag/state. Cheap to maintain
# (no I/O), bounded so they can't grow unbounded under demo load.

_diag_audio: "deque[dict]" = deque(maxlen=40)   # one entry per /turn-audio request
_diag_chain: "deque[dict]" = deque(maxlen=40)   # turn_start / turn_end / chain_end markers
_diag_eleven: "deque[dict]" = deque(maxlen=40)  # ElevenLabs streaming errors / completions

def _diag_log(bucket: "deque[dict]", **kw) -> None:
    """Push a timestamped event into a diagnostic ring buffer. Cheap; never raises."""
    try:
        kw.setdefault("ts", time.time())
        bucket.append(kw)
    except Exception:
        pass

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
                        _diag_log(_diag_eleven, persona=persona.key, error=str(ev)[:200])
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
# Cartesia TTS — WebSocket streaming (raw PCM → streaming WAV)
# ---------------------------------------------------------------------------
# Symmetric to the ElevenLabs WS path. LLM tokens flow in as Cartesia
# `continue: true` continuation messages on a shared context_id, audio chunks
# (base64-encoded raw 16-bit mono PCM @ 24kHz) flow back, and we yield them
# to the caller prepended by a streaming WAV header so a plain <audio>
# element can decode them progressively over chunked transfer-encoding.

CARTESIA_WS_URL = "wss://api.cartesia.ai/tts/websocket?cartesia_version=2026-03-01"
CARTESIA_WS_MODEL = "sonic-3"
CARTESIA_SAMPLE_RATE = 24000  # 24 kHz mono 16-bit PCM


def _streaming_wav_header(sample_rate: int = CARTESIA_SAMPLE_RATE) -> bytes:
    """44-byte RIFF/WAVE header with size fields set to 0xFFFFFFFF — tells the
    browser the stream length is unknown so it keeps playing until the
    connection closes. Mono 16-bit PCM little-endian."""
    import struct
    byte_rate = sample_rate * 1 * 16 // 8   # mono, 16 bits per sample
    block_align = 1 * 16 // 8
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH",
                                 16,            # PCM fmt chunk size
                                 1,             # AudioFormat = PCM
                                 1,             # NumChannels = 1 (mono)
                                 sample_rate,
                                 byte_rate,
                                 block_align,
                                 16)            # BitsPerSample
        + b"data" + struct.pack("<I", 0xFFFFFFFE)
    )


async def _cartesia_stream_input(
    persona, text_iter: AsyncIterator[str]
) -> AsyncIterator[bytes]:
    """Open a Cartesia WebSocket and stream LLM tokens in via continuation
    messages (same context_id, `continue: true`). Yield raw PCM bytes back to
    the caller, prepended with a streaming WAV header on the first chunk.

    Symmetric to _eleven_stream_input — both providers expose a true
    bidirectional streaming pipeline so the chain runner does not branch on
    "REST collects full text first" anymore.
    """
    if not CARTESIA_API_KEY:
        _diag_log(_diag_eleven, persona=persona.key, error="cartesia: CARTESIA_API_KEY not set")
        log.warning("Cartesia selected but CARTESIA_API_KEY is missing")
        return

    context_id = secrets.token_hex(8)
    voice = {"mode": "id", "id": persona.cartesia_voice_id}
    output_format = {
        "container": "raw",
        "encoding": "pcm_s16le",
        "sample_rate": CARTESIA_SAMPLE_RATE,
    }

    def _gen(transcript: str, cont: bool) -> str:
        return json.dumps({
            "model_id": CARTESIA_WS_MODEL,
            "transcript": transcript,
            "voice": voice,
            "language": "en",
            "context_id": context_id,
            "output_format": output_format,
            "continue": cont,
            "max_buffer_delay_ms": 0,   # disable Cartesia's 3s default buffer
        })

    headers = {"X-API-Key": CARTESIA_API_KEY}
    header_emitted = False

    try:
        async with _http.ws_connect(
            CARTESIA_WS_URL, headers=headers, heartbeat=20,
        ) as ws:
            # Cartesia rejects empty transcripts when continue:true, so we
            # don't pre-send a priming message — just start streaming tokens
            # the moment the LLM yields its first delta.

            async def send_tokens() -> None:
                try:
                    async for tok in text_iter:
                        if not tok:
                            continue
                        await ws.send_str(_gen(tok, True))
                    # EOS: empty transcript + continue:false flushes & closes ctx.
                    await ws.send_str(_gen("", False))
                except Exception:
                    log.exception("cartesia send_tokens failed")

            producer = asyncio.create_task(send_tokens())
            try:
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        if msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            log.warning("cartesia WS closed: %s", ws.exception())
                            break
                        continue
                    try:
                        ev = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    ev_type = ev.get("type")
                    if ev_type == "error":
                        log.warning(
                            "cartesia WS error for %s: code=%s msg=%s",
                            persona.key, ev.get("error_code"), ev.get("message", "")[:200],
                        )
                        _diag_log(
                            _diag_eleven, persona=persona.key,
                            error=f"cartesia {ev.get('error_code')}: {str(ev.get('message',''))[:200]}",
                        )
                        break
                    if ev_type == "chunk":
                        b64 = ev.get("data")
                        if not b64:
                            continue
                        try:
                            pcm = base64.b64decode(b64)
                        except Exception:
                            log.exception("cartesia bad audio chunk")
                            continue
                        if not header_emitted:
                            header_emitted = True
                            yield _streaming_wav_header()
                        yield pcm
                    elif ev_type == "done" and ev.get("done"):
                        break
                    # ignore flush_done, timestamps, etc.
            finally:
                if not producer.done():
                    producer.cancel()
                    try:
                        await producer
                    except (asyncio.CancelledError, Exception):
                        pass
    except Exception as e:
        log.exception("cartesia WS connect failed for %s", persona.key)
        _diag_log(_diag_eleven, persona=persona.key, error=f"cartesia exception: {e}")




# ---------------------------------------------------------------------------
# Cartesia Line voice agents - WebSocket loopback with parallel STT
# ---------------------------------------------------------------------------
# When provider == "cartesia-agents", each panel turn is generated by one of
# Dave's configured Cartesia Line agents (doctor / holistic / lawyer). The
# agent runtime does its own STT-LLM-TTS pipeline inside Cartesia. We pass the
# panel transcript via the agent.system_prompt override so each agent hears
# what the previous panelists said. The actual "user audio" we feed is a
# brief seed - the system_prompt carries the real context.

CARTESIA_AGENT_WS = "wss://api.cartesia.ai/agents/stream/{agent_id}"
CARTESIA_AGENT_VERSION = "2025-04-16"
# A short PCM blob of near-silence to trigger the agent's endpointing. We
# build this once at module load. 16-bit mono PCM at 24kHz, 600ms.
# Cartesia's STT VAD endpoints on silence-in-the-audio-stream, not on
# absence of new bytes. We need to *send* silence after the speech seed.
# 0.6 s at 24 kHz mono 16-bit s16le = 28800 bytes of zeros.
# (Reduced from 1.5 s — Cartesia's default endpointing fires within ~0.4 s.)
_AGENT_SILENCE_PAD = b"\x00\x00" * (24000 * 6 // 10)

# Cached seed PCM. Generated ONCE on first agent call (or at startup if
# pre_warm_agent_seed is called). The seed audio just needs to trigger
# VAD endpointing; the agent's LLM context comes entirely from the
# system_prompt override, so the seed text content is irrelevant.
_CACHED_SEED_PCM: bytes | None = None
_SEED_PHRASE = "Go ahead."
_SEED_VOICE_ID = "6d14ac2a-4dda-46f8-bd6f-0722db08ec00"  # Mae - neutral


async def _get_cached_seed_pcm() -> bytes:
    """Return the cached 1-second PCM seed, generating it lazily on first call.
    Generating once at startup saves a Cartesia REST round-trip per agent turn.
    """
    global _CACHED_SEED_PCM
    if _CACHED_SEED_PCM is not None:
        return _CACHED_SEED_PCM
    if not CARTESIA_API_KEY:
        # Fall back to a brief 440Hz tone if no key configured.
        import math as _m, struct as _s
        n = 24000  # 1 second
        buf = bytearray(n * 2)
        for i in range(n):
            v = int(0.22 * 32767 * _m.sin(2 * _m.pi * 440 * i / 24000))
            _s.pack_into("<h", buf, i * 2, v)
        _CACHED_SEED_PCM = bytes(buf)
        return _CACHED_SEED_PCM
    headers = {
        "X-API-Key": CARTESIA_API_KEY,
        "Cartesia-Version": "2026-03-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model_id": "sonic-3",
        "transcript": _SEED_PHRASE,
        "voice": {"mode": "id", "id": _SEED_VOICE_ID},
        "language": "en",
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": 24000,
        },
    }
    try:
        async with _http.post(
            "https://api.cartesia.ai/tts/bytes",
            json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as r:
            if r.status != 200:
                log.warning("cached seed TTS failed: %d", r.status)
                _CACHED_SEED_PCM = b"\x00\x00" * 24000  # 1s silence fallback
                return _CACHED_SEED_PCM
            _CACHED_SEED_PCM = await r.read()
            log.info("cached agent seed PCM: %d bytes (%.1fs of audio)",
                     len(_CACHED_SEED_PCM), len(_CACHED_SEED_PCM) / 2 / 24000)
            return _CACHED_SEED_PCM
    except Exception:
        log.exception("cached seed TTS exception")
        _CACHED_SEED_PCM = b"\x00\x00" * 24000
        return _CACHED_SEED_PCM


def _build_agent_system_prompt(persona, panel_context: str) -> str:
    """Compose the per-turn system_prompt for a Cartesia Line agent."""
    return (
        persona.system_prompt
        + "\n\n--- PANEL CONTEXT (the conversation so far) ---\n"
        + (panel_context or "(this is the opening turn)")
        + "\n--- END PANEL CONTEXT ---\n\n"
        + "You are speaking as part of a live panel. The audio you are about "
        + "to hear is a placeholder seed - DO NOT respond to it literally. "
        + "Instead, continue the panel discussion as your character, "
        + "responding to what the OTHER panelists have just said. Push back "
        + "where it's appropriate to your persona. Reply in 1-2 short spoken "
        + "sentences (under 30 words). No name prefix, no stage directions."
    )


async def _cartesia_agent_stream(
    persona, panel_context: str,
) -> AsyncIterator[bytes]:
    """Open WS to Cartesia Line agent, run one panel turn, yield response audio.

    The agent uses its OWN STT-LLM-TTS pipeline inside Cartesia. We override
    its system_prompt with the panel context so it responds in character to
    what the other panelists have said. The audio we feed is a brief seed
    blob - the real context lives in the system_prompt override.

    Yields raw PCM 24kHz mono 16-bit LE bytes, prepended with a streaming WAV
    header on the first chunk so the frontend's <audio> element can play it
    progressively over the existing /turn-audio chunked-transfer endpoint.
    """
    agent_id = (persona.cartesia_agent_id or "").strip()
    if not agent_id:
        _diag_log(_diag_eleven, persona=persona.key,
                  error="agent: persona has no cartesia_agent_id")
        return
    if not CARTESIA_API_KEY:
        _diag_log(_diag_eleven, persona=persona.key,
                  error="agent: CARTESIA_API_KEY not set")
        return

    url = CARTESIA_AGENT_WS.format(agent_id=agent_id)
    headers = {
        "Authorization": f"Bearer {CARTESIA_API_KEY}",
        "Cartesia-Version": CARTESIA_AGENT_VERSION,
    }
    stream_id = secrets.token_hex(8)
    header_emitted = False
    system_prompt = _build_agent_system_prompt(persona, panel_context)

    try:
        async with _http.ws_connect(url, headers=headers, heartbeat=15) as ws:
            # Send start event with per-turn system_prompt override.
            # Empty introduction so the agent doesn't open with a greeting.
            await ws.send_str(json.dumps({
                "event": "start",
                "stream_id": stream_id,
                "config": {
                    "input_format": "pcm_24000",
                    "voice_id": persona.cartesia_voice_id,
                },
                "agent": {
                    "system_prompt": system_prompt,
                    "introduction": "",
                },
                "metadata": {
                    "panel_turn": persona.key,
                    "from": "voice-panel-orchestrator",
                },
            }))

            # The seed audio is just a VAD trigger. The agent's LLM context
            # is in system_prompt, so the seed phrase doesn't matter. We use
            # a 1s cached "Go ahead." PCM generated once at startup, followed
            # by 0.6s of silence to trip Cartesia's endpointing.
            speech_pcm = await _get_cached_seed_pcm()
            audio_in = speech_pcm + _AGENT_SILENCE_PAD

            # Cartesia's streaming STT buffers — we don't need realtime pacing.
            # Blast all chunks at ~5x realtime; the agent processes at its
            # own rate and endpoints when it sees the silence segment.
            CHUNK_SIZE = 24000 * 40 // 1000 * 2  # 1920 bytes per 40ms
            async def feed_seed():
                try:
                    for i in range(0, len(audio_in), CHUNK_SIZE):
                        chunk = audio_in[i:i + CHUNK_SIZE]
                        await ws.send_str(json.dumps({
                            "event": "media_input",
                            "stream_id": stream_id,
                            "media": {"payload": base64.b64encode(chunk).decode()},
                        }))
                        await asyncio.sleep(0.008)  # ~5x realtime, still gentle
                except Exception:
                    log.exception("agent feed_seed failed")

            seeder = asyncio.create_task(feed_seed())

            # Now consume agent's media_output events. Cartesia Line emits
            # raw PCM in base64 via media_output events. We yield those bytes
            # after a one-time WAV header. We close on `clear` (agent done) or
            # when no media for ~3s after first byte.
            had_first_byte = False
            AGENT_TIMEOUT_S = 18.0   # hard ceiling: cached seed (~1s) + endpointing (~0.6s) + LLM + TTS
            INACTIVITY_S = 8.0       # if no event in 8s and we have audio, end
            turn_start_at = time.monotonic()
            last_event_at = turn_start_at
            try:
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                    except asyncio.TimeoutError:
                        now = time.monotonic()
                        if now - turn_start_at > AGENT_TIMEOUT_S:
                            log.warning("agent %s: hard timeout (%ds, no media_output)",
                                        persona.key, int(AGENT_TIMEOUT_S))
                            _diag_log(_diag_eleven, persona=persona.key,
                                      error=f"agent timeout after {int(AGENT_TIMEOUT_S)}s")
                            break
                        if had_first_byte and now - last_event_at > INACTIVITY_S:
                            log.info("agent %s: inactivity end after audio captured", persona.key)
                            break
                        continue
                    last_event_at = time.monotonic()
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        if msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            log.warning("agent WS closed for %s: %s",
                                        persona.key, ws.exception())
                            break
                        continue
                    try:
                        ev = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    et = ev.get("event")
                    if et == "ack":
                        continue
                    if et == "media_output":
                        b64 = ev.get("media", {}).get("payload")
                        if not b64:
                            continue
                        try:
                            pcm = base64.b64decode(b64)
                        except Exception:
                            log.exception("agent bad audio chunk")
                            continue
                        if not header_emitted:
                            header_emitted = True
                            had_first_byte = True
                            yield _streaming_wav_header(24000)
                        yield pcm
                    elif et == "clear":
                        # Agent finished or wants to interrupt; close cleanly.
                        if had_first_byte:
                            break
                    # ignore other events
            finally:
                if not seeder.done():
                    seeder.cancel()
                    try:
                        await seeder
                    except (asyncio.CancelledError, Exception):
                        pass
                # Best-effort done frame so Cartesia tears down the agent run.
                try:
                    await ws.send_str("done")
                except Exception:
                    pass
    except Exception as e:
        log.exception("agent WS connect failed for %s", persona.key)
        _diag_log(_diag_eleven, persona=persona.key, error=f"agent exception: {e}")


async def _cartesia_stt_pcm(pcm_bytes: bytes) -> str | None:
    """Run Cartesia /stt on a raw PCM 24kHz mono blob. Returns text or None.

    Used to recover the agent's response text from its TTS audio so the next
    panel turn's system_prompt can include what was just said.
    """
    if not pcm_bytes or not CARTESIA_API_KEY:
        return None
    headers = {
        "Authorization": f"Bearer {CARTESIA_API_KEY}",
        "Cartesia-Version": "2026-03-01",
    }
    form = aiohttp.FormData()
    form.add_field("file", pcm_bytes, filename="agent.pcm", content_type="audio/pcm")
    form.add_field("model", CARTESIA_STT_MODEL)
    form.add_field("language", "en")
    form.add_field("encoding", "pcm_s16le")
    form.add_field("sample_rate", "24000")
    try:
        async with _http.post(
            CARTESIA_STT_URL, data=form, headers=headers,
            timeout=aiohttp.ClientTimeout(total=20),
        ) as r:
            if r.status != 200:
                body = await r.text()
                log.warning("agent-output STT %d: %s", r.status, body[:300])
                return None
            return ((await r.json()).get("text") or "").strip()
    except Exception:
        log.exception("agent-output STT failed")
        return None


# ---------------------------------------------------------------------------
# TTS dispatcher — single entry point used by the chain runner
# ---------------------------------------------------------------------------

async def _stream_tts(
    persona, anthropic_iter: AsyncIterator[str],
    text_buf: list[str], provider: str,
) -> AsyncIterator[bytes]:
    """Pipe LLM tokens through the chosen TTS provider, yield audio bytes,
    fill text_buf with each token so the caller can reconstruct reply_text.

    Both providers now expose true bidirectional streaming: tokens flow
    LLM → TTS as they arrive; audio bytes flow back the moment the provider
    produces them. No "collect-then-call" branch.
    """
    if provider == "cartesia-agents":
        # Drain the anthropic_iter (or cancel it) - the agent generates its
        # own response, we don't need Claude tokens. Skip the stream entirely.
        # text_buf stays empty; transcript text is recovered post-hoc via STT
        # on the agent's output audio.
        panel_context = "\n".join(_transcript[-24:])
        async for chunk in _cartesia_agent_stream(persona, panel_context):
            yield chunk
        return

    async def _text_source():
        async for tok in anthropic_iter:
            text_buf.append(tok)
            yield tok

    if provider == "cartesia":
        async for chunk in _cartesia_stream_input(persona, _text_source()):
            yield chunk
    else:
        async for chunk in _eleven_stream_input(persona, _text_source()):
            yield chunk


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
            # Snapshot provider here so the AudioStream's mime matches the
            # actual TTS path we'll take for this turn.
            provider_for_turn = _tts_provider
            stream = AudioStream(
                mime="audio/wav" if provider_for_turn in ("cartesia", "cartesia-agents") else "audio/mpeg"
            )
            _audio_streams[turn_id] = stream
            text_buf: list[str] = []

            # Tell the frontend immediately so it can open <audio
            # src="/turn-audio/<id>"> while we're still generating.
            _diag_log(_diag_chain, kind="turn_start", id=turn_id, who=persona.key)
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

            _diag_log(_diag_chain, kind="tts_provider", id=turn_id, who=persona.key, provider=provider_for_turn)

            try:
                async for audio_chunk in _stream_tts(
                    persona,
                    _anthropic_stream(persona, transcript_ctx),
                    text_buf,
                    provider_for_turn,
                ):
                    await stream.push(audio_chunk)
            except Exception as e:
                log.exception("streaming pipeline failed in chain turn %d (%s)", i, provider_for_turn)
                await _publish({"type": "error", "text": f"stream: {e}"})
            finally:
                await stream.push(None)  # EOS — releases the HTTP handler

            if provider_for_turn == "cartesia-agents":
                # Agent generated audio directly; recover its text via STT
                # on the captured PCM bytes so the next turn's system_prompt
                # includes what was just said. Strip the leading WAV header
                # (44 bytes) before sending raw PCM to STT.
                pcm_bytes = b"".join(stream.chunks)
                if len(pcm_bytes) > 44:
                    pcm_bytes = pcm_bytes[44:]
                reply_text = await _cartesia_stt_pcm(pcm_bytes) or ""
                reply_text = reply_text.strip()
            else:
                reply_text = _strip_name_prefix("".join(text_buf), persona).strip()
            if not reply_text:
                # Empty turn — clean up the unused stream and stop the chain.
                _audio_streams.pop(turn_id, None)
                break

            _transcript.append(f"{persona.name}: {reply_text}")
            _diag_log(_diag_chain, kind="turn_end", id=turn_id, who=persona.key, reply_chars=len(reply_text))
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
        _diag_log(_diag_chain, kind="chain_end")
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



CARTESIA_STT_URL = "https://api.cartesia.ai/stt"
CARTESIA_STT_MODEL = "ink-whisper"


async def _cartesia_stt(audio_bytes: bytes, filename: str, content_type: str) -> str | None:
    """POST the recorded audio blob to Cartesia /stt, return transcript text.

    Returns None on any error (HTTP non-200, missing key, exception) so the
    caller can fall through to the ElevenLabs Scribe path if it wants.
    """
    if not CARTESIA_API_KEY:
        return None
    headers = {
        "Authorization": f"Bearer {CARTESIA_API_KEY}",
        "Cartesia-Version": "2026-03-01",
    }
    form = aiohttp.FormData()
    form.add_field("file", audio_bytes, filename=filename, content_type=content_type)
    form.add_field("model", CARTESIA_STT_MODEL)
    form.add_field("language", "en")
    try:
        async with _http.post(
            CARTESIA_STT_URL, data=form, headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as r:
            body = await r.text()
            if r.status != 200:
                log.warning("Cartesia STT %d: %s", r.status, body[:500])
                _diag_log(_diag_eleven, persona="stt", error=f"cartesia stt {r.status}: {body[:200]}")
                return None
            result = json.loads(body)
            return (result.get("text") or "").strip()
    except asyncio.TimeoutError:
        log.warning("Cartesia STT timeout")
        _diag_log(_diag_eleven, persona="stt", error="cartesia stt timeout")
        return None
    except Exception as e:
        log.exception("Cartesia STT call failed")
        _diag_log(_diag_eleven, persona="stt", error=f"cartesia stt exception: {e}")
        return None


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

    # Prefer Cartesia Ink-Whisper when configured (works regardless of
    # ElevenLabs quota). Fall back to ElevenLabs Scribe only if Cartesia
    # returns None (no key set, or error).
    if CARTESIA_API_KEY:
        text = await _cartesia_stt(audio_bytes, filename, content_type)
        if text is not None:
            log.info("cartesia stt → %r", text[:120])
            return web.json_response({"text": text, "provider": "cartesia"})
        # else: fall through to ElevenLabs

    # ElevenLabs Scribe fallback.
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
    return web.json_response({"text": text, "provider": "elevenlabs"})


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
    return web.FileResponse(
        os.path.join(os.path.dirname(__file__), "index.html"),
        headers={"Cache-Control": "no-cache, no-store"},
    )


# Streams a turn's audio with chunked transfer-encoding. Browsers play it
# back progressively via a standard <audio src="/turn-audio/<id>"> element
# — first sound hits the user's ears as soon as the first chunk arrives.
async def handle_turn_audio(request: web.Request) -> web.StreamResponse:
    turn_id = request.match_info["id"]
    stream = _audio_streams.get(turn_id)
    if not stream:
        log.warning("turn-audio %s: stream missing (404) — client requested expired/unknown id", turn_id)
        _diag_log(_diag_audio, turn_id=turn_id, status="404_missing")
        raise web.HTTPNotFound()
    t0 = time.monotonic()
    first_byte_at: float | None = None
    total_bytes = 0
    chunks = 0
    client_ua = request.headers.get("User-Agent", "?")[:80]
    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": stream.mime,   # audio/mpeg for elevenlabs, audio/wav for cartesia
            "Cache-Control": "no-cache, no-store",
            "X-Accel-Buffering": "no",  # disable proxy buffering (Render etc)
            "Access-Control-Allow-Origin": "*",
            "Connection": "close",       # avoid keep-alive races on chunked
        },
    )
    await resp.prepare(request)
    log.info("turn-audio %s: stream OPENED ua=%s", turn_id, client_ua)
    _diag_log(_diag_audio, turn_id=turn_id, status="opened", ua=client_ua)
    outcome = "completed"
    try:
        async for chunk in stream.read():
            if first_byte_at is None:
                first_byte_at = time.monotonic()
                ttfb_ms = int((first_byte_at - t0) * 1000)
                log.info("turn-audio %s: FIRST BYTE after %dms", turn_id, ttfb_ms)
                _diag_log(_diag_audio, turn_id=turn_id, status="first_byte", ttfb_ms=ttfb_ms)
            total_bytes += len(chunk)
            chunks += 1
            await resp.write(chunk)
        await resp.write_eof()
    except (ConnectionResetError, asyncio.CancelledError):
        outcome = "client_disconnected"
    except Exception as e:
        outcome = f"error:{type(e).__name__}"
        log.exception("turn-audio %s: unexpected error", turn_id)
    finally:
        total_ms = int((time.monotonic() - t0) * 1000)
        ttfb_ms = int((first_byte_at - t0) * 1000) if first_byte_at else None
        log.info(
            "turn-audio %s: CLOSED outcome=%s bytes=%d chunks=%d ttfb_ms=%s total_ms=%d",
            turn_id, outcome, total_bytes, chunks, ttfb_ms, total_ms,
        )
        _diag_log(_diag_audio, turn_id=turn_id, status=outcome,
                  bytes=total_bytes, chunks=chunks,
                  ttfb_ms=ttfb_ms, total_ms=total_ms)
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




# ---------------------------------------------------------------------------
# Diagnostic endpoints — inspect runtime state without restarting / redeploying
# ---------------------------------------------------------------------------

async def handle_diag_state(request: web.Request) -> web.Response:
    """Return a JSON snapshot of server-side audio + chain diagnostics."""
    chain_state = "idle"
    if _chain_task and not _chain_task.done():
        chain_state = "running"
    elif _chain_task and _chain_task.done():
        chain_state = "finished"
    return web.json_response({
        "version": {"commit": VERSION_FULL, "short": VERSION_SHORT, "started_at": STARTED_AT},
        "now": time.time(),
        "chain_state": chain_state,
        "active_audio_streams": [
            {"id": tid, "complete": s.complete, "cancelled": s.cancelled, "chunks": len(s.chunks)}
            for tid, s in _audio_streams.items()
        ],
        "transcript_len": len(_transcript),
        "subscribers": len(_subscribers),
        "events_log_len": len(_events_log),
        "recent_audio": list(_diag_audio)[-30:],
        "recent_chain": list(_diag_chain)[-30:],
        "recent_eleven": list(_diag_eleven)[-30:],
    }, headers={"Cache-Control": "no-store", "Access-Control-Allow-Origin": "*"})


async def handle_diag_audio_test(request: web.Request) -> web.Response:
    """Return a 0.5s 440Hz sine wave as a known-good audio playback positive control.

    If the browser can play /diag/audio-test but not /turn-audio/<id>, the bug is
    isolated to the chunked-transfer path or the page's persistent <audio> wiring.
    """
    import math, struct
    sr = 22050
    duration = 0.6
    freq = 440
    n_samples = int(sr * duration)
    body = bytearray()
    # 50ms fade in/out to avoid click
    fade = int(sr * 0.05)
    for i in range(n_samples):
        env = 1.0
        if i < fade: env = i / fade
        elif i > n_samples - fade: env = (n_samples - i) / fade
        sample = int(0.3 * env * 32767 * math.sin(2 * math.pi * freq * i / sr))
        body.extend(struct.pack("<h", sample))
    header = (
        b"RIFF" + struct.pack("<I", 36 + len(body)) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
        + b"data" + struct.pack("<I", len(body))
    )
    wav = bytes(header) + bytes(body)
    return web.Response(
        body=wav,
        headers={
            "Content-Type": "audio/wav",
            "Cache-Control": "no-store",
            "Access-Control-Allow-Origin": "*",
        },
    )


def _sync_personas_to_mode(provider: str) -> None:
    """Toggle which personas are enabled to match the active provider mode.

    Each mode has its own persona set (declared in personas.PERSONA_SETS).
    The orchestrator only considers enabled personas, so flipping these
    flags is what makes mode switching feel instantaneous in the UI.
    """
    target = set(PERSONA_SETS.get(provider, PERSONA_SETS["elevenlabs"]))
    for key in PERSONAS:
        set_persona_enabled(key, key in target)


async def handle_tts_provider_get(request: web.Request) -> web.Response:
    return web.json_response({
        "provider": _tts_provider,
        "available": ["elevenlabs", "cartesia", "cartesia-agents"],
        "cartesia_configured": bool(CARTESIA_API_KEY),
    })


async def handle_tts_provider_set(request: web.Request) -> web.Response:
    global _tts_provider
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    p = (body.get("provider") or "").strip().lower()
    if p not in ("elevenlabs", "cartesia", "cartesia-agents"):
        return web.json_response(
            {"error": "provider must be 'elevenlabs', 'cartesia', or 'cartesia-agents'"},
            status=400,
        )
    if p in ("cartesia", "cartesia-agents") and not CARTESIA_API_KEY:
        return web.json_response(
            {"error": "CARTESIA_API_KEY env var not set on the server."},
            status=400,
        )
    _tts_provider = p
    _sync_personas_to_mode(p)
    log.info("tts_provider switched -> %s", p)
    return web.json_response({"provider": _tts_provider})

_gc_task: asyncio.Task | None = None


async def on_startup(app: web.Application) -> None:
    global _http, _gc_task
    _http = ClientSession()
    _gc_task = asyncio.create_task(_audio_gc_loop())
    _sync_personas_to_mode(_tts_provider)
    # Pre-warm the cached agent seed so the first cartesia-agents turn
    # doesn't pay a ~1s Cartesia REST round-trip.
    try:
        await _get_cached_seed_pcm()
    except Exception:
        log.exception("pre-warm agent seed failed (non-fatal)")


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
    app.router.add_get("/diag/state", handle_diag_state)
    app.router.add_get("/diag/audio-test", handle_diag_audio_test)
    app.router.add_get("/tts/provider", handle_tts_provider_get)
    app.router.add_post("/tts/provider", handle_tts_provider_set)
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
