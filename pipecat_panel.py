"""
`cartesia-agents` panel mode — Anthropic → Cartesia, true streaming end-to-end.

Originally this module wrapped the Cartesia TTS call in a per-turn Pipecat
`Pipeline([CartesiaTTSService, _AudioSink])`. That path produced choppy audio
in the browser, while the existing `cartesia` mode (which calls Cartesia's WS
directly with the same raw PCM + streaming WAV header) stayed smooth.

The root cause was at the boundary of the panel-mode path: it asked Claude
for the FULL reply (REST, non-streaming), then asked Cartesia to synthesize
the whole utterance in a single `continue: false` WS message. Cartesia then
flushed audio chunks back as fast as the socket could carry them — far
faster than realtime. By the time the browser drained its initial
chunked-transfer buffer, the producer had already paused (synthesis done)
and the chunked TCP stream had idled; some intermediaries reset framing,
others delivered chunks in bursts that the <audio> element couldn't keep
ahead of. Net effect: choppy playback. The cartesia mode avoided this
because it streamed tokens INTO Cartesia (`continue: true`) so audio came
back paced by LLM token rate — naturally close to realtime.

The fix here mirrors that streaming contract. We:
  1. Stream Anthropic deltas (Anthropic SSE), accumulate the reply text in
     parallel for the transcript.
  2. Forward each delta into a Cartesia WS as a `continue: true` message
     (same context_id), exactly like `_cartesia_stream_input` in app.py.
  3. Yield each `chunk` event's PCM bytes through, prefixed once by the
     streaming WAV header.

Audio pacing is now LLM-bound, identical to the smooth path. Pipecat is no
longer in the per-turn hot path.

We keep `pipecat_available()` / `pipecat_import_error()` so /diag/state can
still report whether the heavy import succeeded — informational only.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import secrets
import time
from typing import AsyncIterator

import aiohttp

log = logging.getLogger("voice-panel.pipecat")

CARTESIA_SAMPLE_RATE = 24000
CARTESIA_WS_URL = "wss://api.cartesia.ai/tts/websocket?cartesia_version=2026-03-01"
CARTESIA_WS_MODEL = "sonic-3.5"  # latest stable (May 2026) — more natural pacing/emotion than sonic-3

# Pipecat import is now purely informational — we don't use it on the hot
# path. Kept so /diag/state can still expose whether the dep is installed.
_PIPECAT_AVAILABLE = False
_PIPECAT_IMPORT_ERROR: str | None = None
try:
    import pipecat  # noqa: F401, E402

    _PIPECAT_AVAILABLE = True
except Exception as e:  # pragma: no cover
    _PIPECAT_IMPORT_ERROR = f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Anthropic streaming for panel-mode prompts
# ---------------------------------------------------------------------------

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"


def _build_system_prompt(persona, panel_context: str) -> str:
    """Per-turn system prompt: persona character + brief panel transcript +
    instruction to respond in 1-2 short sentences and push back where natural."""
    return (
        persona.system_prompt
        + "\n\n--- PANEL TRANSCRIPT (the conversation so far) ---\n"
        + (panel_context or "(this is the opening turn — kick things off)")
        + "\n--- END PANEL TRANSCRIPT ---\n\n"
        + "Respond as your character in 1-2 short spoken sentences "
        + "(under 30 words). Push back where it's natural. Reply with ONLY "
        + "the words you would say — no name prefix, no stage directions."
    )


async def _claude_stream_text(
    persona,
    panel_context: str,
    anthropic_api_key: str,
    http_session: aiohttp.ClientSession,
) -> AsyncIterator[str]:
    """Stream Anthropic text deltas for one panel turn. First delta typically
    ~300-500 ms."""
    system = _build_system_prompt(persona, panel_context)
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 110,
        "stream": True,
        "system": system,
        "messages": [{
            "role": "user",
            "content": (
                f"You are {persona.name}. Continue the panel as your character."
            ),
        }],
    }
    headers = {
        "x-api-key": anthropic_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    async with http_session.post(ANTHROPIC_URL, headers=headers, json=body) as r:
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


# ---------------------------------------------------------------------------
# Cartesia streaming WS — token-paced, same protocol as app.py's
# _cartesia_stream_input (the smooth `cartesia` mode reference).
# ---------------------------------------------------------------------------

async def _cartesia_stream_from_tokens(
    persona,
    text_iter: AsyncIterator[str],
    cartesia_api_key: str,
    http_session: aiohttp.ClientSession,
    *,
    text_out: list[str],
) -> AsyncIterator[bytes]:
    """Open a Cartesia WS, pipe text deltas in via `continue: true` messages
    sharing one context_id, and yield raw PCM as Cartesia emits it (prefixed
    once by the streaming WAV header). Mirrors `_cartesia_stream_input` in
    app.py, with the extra duty of mirroring each delta into text_out so the
    caller can reconstruct the spoken text for the transcript."""
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
            "max_buffer_delay_ms": 3000,  # Cartesia's recommended default — they explicitly tune for 3000ms
        })

    headers = {"X-API-Key": cartesia_api_key}

    try:
        async with http_session.ws_connect(
            CARTESIA_WS_URL, headers=headers, heartbeat=20,
        ) as ws:
            async def send_tokens() -> None:
                try:
                    async for tok in text_iter:
                        if not tok:
                            continue
                        text_out.append(tok)
                        await ws.send_str(_gen(tok, True))
                    await ws.send_str(_gen("", False))
                except Exception:
                    log.exception("cartesia-agents send_tokens failed")

            producer = asyncio.create_task(send_tokens())
            try:
                async for ws_msg in ws:
                    if ws_msg.type != aiohttp.WSMsgType.TEXT:
                        if ws_msg.type in (
                            aiohttp.WSMsgType.ERROR,
                            aiohttp.WSMsgType.CLOSED,
                        ):
                            log.warning(
                                "cartesia-agents WS closed: %s", ws.exception(),
                            )
                            break
                        continue
                    try:
                        ev = json.loads(ws_msg.data)
                    except json.JSONDecodeError:
                        continue
                    ev_type = ev.get("type")
                    if ev_type == "error":
                        log.warning(
                            "cartesia-agents WS error for %s: code=%s msg=%s",
                            persona.key, ev.get("error_code"),
                            ev.get("message", "")[:200],
                        )
                        break
                    if ev_type == "chunk":
                        b64 = ev.get("data")
                        if not b64:
                            continue
                        try:
                            pcm = base64.b64decode(b64)
                        except Exception:
                            log.exception("cartesia-agents bad audio chunk")
                            continue
                        yield pcm  # raw s16le PCM — browser decodes via Web Audio API
                    elif ev_type == "done" and ev.get("done"):
                        break
            finally:
                if not producer.done():
                    producer.cancel()
                    try:
                        await producer
                    except (asyncio.CancelledError, Exception):
                        pass
    except Exception:
        log.exception("cartesia-agents WS connect failed for %s", persona.key)


# ---------------------------------------------------------------------------
# Public entry point: produce a panel-mode audio stream for one persona turn.
# ---------------------------------------------------------------------------

async def agents_mode_stream(
    persona,
    panel_context: str,
    *,
    cartesia_api_key: str,
    anthropic_api_key: str,
    http_session: aiohttp.ClientSession,
    text_out: list[str],
) -> AsyncIterator[bytes]:
    """One panel turn: Claude (streaming) → Cartesia WS (token-paced) → PCM.

    Audio bytes are yielded with a streaming WAV header prepended on the
    first chunk. After the iterator drains, ``text_out`` contains exactly
    one entry — the full spoken text — so the caller can do
    ``text_out[0]`` (the contract app.py's _stream_tts expects).
    """
    text_out.clear()

    if not cartesia_api_key:
        log.warning("cartesia-agents: CARTESIA_API_KEY missing for %s", persona.key)
        return
    if not anthropic_api_key:
        log.warning("cartesia-agents: ANTHROPIC_API_KEY missing for %s", persona.key)
        return

    t0 = time.monotonic()
    first_audio_at: float | None = None

    token_iter = _claude_stream_text(
        persona, panel_context, anthropic_api_key, http_session,
    )

    # Accumulate deltas in a local buffer; finalise text_out as a single
    # joined string once streaming completes so the existing chain runner
    # (which reads text_out[0]) gets the full reply.
    delta_buf: list[str] = []

    async for chunk in _cartesia_stream_from_tokens(
        persona,
        token_iter,
        cartesia_api_key,
        http_session,
        text_out=delta_buf,
    ):
        if first_audio_at is None:
            first_audio_at = time.monotonic()
            log.info(
                "cartesia-agents: first audio for %s in %dms",
                persona.key, int((first_audio_at - t0) * 1000),
            )
        yield chunk

    if delta_buf:
        text_out.append("".join(delta_buf))


def pipecat_available() -> bool:
    return _PIPECAT_AVAILABLE


def pipecat_import_error() -> str | None:
    return _PIPECAT_IMPORT_ERROR
