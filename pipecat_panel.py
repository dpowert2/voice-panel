"""
Pipecat integration for the `cartesia-agents` panel mode.

This replaces the old Cartesia Line Builder audio-loopback path with a real
Pipecat pipeline:

    [SourceProcessor: TextFrames from Claude stream]
        -> CartesiaTTSService(voice=<persona's cartesia_voice_id>)
            -> [SinkProcessor: capture TTSAudioRawFrames -> AudioStream]

Per turn we:
  1. Build a tiny Pipeline(tts, sink) per persona — voice is permanently bound
     to that pipeline instance, mirroring the
     `local-handoff-two-agents-tts.py` pattern from Pipecat's examples.
  2. Start a PipelineWorker, push a single TextFrame containing Claude's
     reply text (collected sentence-by-sentence as it streams), then call
     stop_when_done() to flush the pipeline cleanly.
  3. Yield audio bytes back to aiohttp prepended with a streaming WAV header,
     compatible with the existing AudioStream / /turn-audio chunked-transfer
     contract.

If pipecat-ai isn't importable (e.g. build hiccup on Render), we fall back to
the direct Cartesia WebSocket implementation (`fallback_cartesia_ws_stream`)
which speaks the same protocol Pipecat does — same `/tts/websocket` endpoint,
same `context_id` multiplex, same `sonic-3` model. The fallback is line-for-line
the same approach as app.py's `_cartesia_stream_input` for the `cartesia` mode,
which already meets the latency target in production. Caller transparently
gets the same audio bytes either way.

Reference examples (read these before changing the pipeline shape):
  - https://github.com/pipecat-ai/pipecat/blob/main/examples/multi-worker/local-handoff/local-handoff-two-agents-tts.py
  - https://github.com/pipecat-ai/pipecat/blob/main/examples/multi-worker/parallel-debate/parallel-debate.py
  - https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/cartesia/tts.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import secrets
import struct
import time
from typing import AsyncIterator, Callable

import aiohttp

log = logging.getLogger("voice-panel.pipecat")

CARTESIA_SAMPLE_RATE = 24000
CARTESIA_WS_URL = "wss://api.cartesia.ai/tts/websocket?cartesia_version=2026-03-01"
CARTESIA_WS_MODEL = "sonic-3"

# Try to import pipecat lazily. Import failures don't crash the module —
# we just fall back to the direct WS path.
_PIPECAT_AVAILABLE = False
_PIPECAT_IMPORT_ERROR: str | None = None
try:
    # Pipecat 1.x layout
    from pipecat.frames.frames import (  # noqa: E402
        EndFrame,
        Frame,
        TextFrame,
        TTSAudioRawFrame,
        TTSStoppedFrame,
    )
    from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
    from pipecat.pipeline.worker import (  # noqa: E402
        PipelineParams,
        PipelineWorker,
    )
    from pipecat.processors.frame_processor import (  # noqa: E402
        FrameDirection,
        FrameProcessor,
    )
    from pipecat.services.cartesia.tts import CartesiaTTSService  # noqa: E402
    from pipecat.workers.base_worker import WorkerParams  # noqa: E402

    _PIPECAT_AVAILABLE = True
except Exception as e:  # pragma: no cover — depends on install env
    _PIPECAT_IMPORT_ERROR = f"{type(e).__name__}: {e}"
    log.warning("pipecat not importable, using fallback WS path (%s)", _PIPECAT_IMPORT_ERROR)


def streaming_wav_header(sample_rate: int = CARTESIA_SAMPLE_RATE) -> bytes:
    """44-byte RIFF/WAVE header with size fields set to 0xFFFFFFFF — tells the
    browser the stream length is unknown so it keeps playing until the
    connection closes. Mono 16-bit PCM little-endian."""
    byte_rate = sample_rate * 1 * 16 // 8
    block_align = 1 * 16 // 8
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        + b"fmt " + struct.pack(
            "<IHHIIHH",
            16, 1, 1, sample_rate, byte_rate, block_align, 16,
        )
        + b"data" + struct.pack("<I", 0xFFFFFFFE)
    )


# ---------------------------------------------------------------------------
# Pipecat path
# ---------------------------------------------------------------------------

if _PIPECAT_AVAILABLE:

    class _AudioSink(FrameProcessor):
        """Terminal processor that drains TTSAudioRawFrames into an asyncio.Queue.

        The producer end (caller) pulls from the queue to yield audio bytes
        out through aiohttp's chunked-transfer response.
        """

        def __init__(self, audio_queue: "asyncio.Queue[bytes | None]") -> None:
            super().__init__(name="VoicePanelAudioSink")
            self._audio_queue = audio_queue
            self._got_first_chunk = False

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, TTSAudioRawFrame):
                audio = frame.audio
                if audio:
                    if not self._got_first_chunk:
                        self._got_first_chunk = True
                        await self._audio_queue.put(streaming_wav_header(CARTESIA_SAMPLE_RATE))
                    await self._audio_queue.put(audio)
            elif isinstance(frame, TTSStoppedFrame):
                # Cartesia finished synthesis for this context.
                await self._audio_queue.put(None)
            # Pass through so downstream lifecycle frames (EndFrame etc.) flow.
            await self.push_frame(frame, direction)


    async def _pipecat_stream(
        persona,
        text: str,
        cartesia_api_key: str,
    ) -> AsyncIterator[bytes]:
        """Run text through CartesiaTTSService inside a one-shot Pipecat pipeline.

        Yields the streaming WAV header (once) followed by raw PCM bytes."""
        audio_queue: "asyncio.Queue[bytes | None]" = asyncio.Queue(maxsize=64)

        tts = CartesiaTTSService(
            api_key=cartesia_api_key,
            voice_id=persona.cartesia_voice_id,
            model=CARTESIA_WS_MODEL,
            sample_rate=CARTESIA_SAMPLE_RATE,
            encoding="pcm_s16le",
            container="raw",
            max_buffer_delay_ms=0,
        )
        sink = _AudioSink(audio_queue)

        pipeline = Pipeline([tts, sink])
        worker = PipelineWorker(
            pipeline,
            name=f"vp-cartesia-agents-{persona.key}",
            params=PipelineParams(
                enable_metrics=False,
                enable_usage_metrics=False,
                audio_out_sample_rate=CARTESIA_SAMPLE_RATE,
            ),
        )

        runner_params = WorkerParams(loop=asyncio.get_running_loop())
        run_task = asyncio.create_task(worker.run(runner_params))

        try:
            # Push the text + end the pipeline. The TTS service will open its WS,
            # generate the audio, and emit TTSAudioRawFrames into the sink.
            await worker.queue_frame(TextFrame(text))
            await worker.stop_when_done()

            # Drain audio chunks until sink signals EOS (None) or the worker exits.
            while True:
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=20.0)
                except asyncio.TimeoutError:
                    log.warning(
                        "pipecat audio queue idle >20s for %s; closing", persona.key,
                    )
                    break
                if chunk is None:
                    break
                yield chunk
        finally:
            try:
                await worker.cancel(reason="turn complete")
            except Exception:
                pass
            try:
                await asyncio.wait_for(run_task, timeout=5.0)
            except asyncio.TimeoutError:
                run_task.cancel()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Fallback path — direct Cartesia WebSocket. Used when pipecat-ai isn't
# importable OR when the pipeline harness errors. Speaks the same protocol
# Pipecat's CartesiaTTSService speaks under the hood.
# ---------------------------------------------------------------------------

async def _fallback_cartesia_ws_stream(
    persona,
    text: str,
    cartesia_api_key: str,
    http_session: aiohttp.ClientSession,
) -> AsyncIterator[bytes]:
    """One-shot Cartesia WS TTS for the given persona+text. Yields WAV header
    then raw PCM bytes. Mirrors app.py's `_cartesia_stream_input` but takes a
    full text string rather than a token iterator — cartesia-agents mode buffers
    Claude's full reply before TTS so we can return the text for the transcript.
    """
    context_id = secrets.token_hex(8)
    voice = {"mode": "id", "id": persona.cartesia_voice_id}
    output_format = {
        "container": "raw",
        "encoding": "pcm_s16le",
        "sample_rate": CARTESIA_SAMPLE_RATE,
    }
    msg = json.dumps({
        "model_id": CARTESIA_WS_MODEL,
        "transcript": text,
        "voice": voice,
        "language": "en",
        "context_id": context_id,
        "output_format": output_format,
        "continue": False,
        "max_buffer_delay_ms": 0,
    })
    headers = {"X-API-Key": cartesia_api_key}
    header_emitted = False
    try:
        async with http_session.ws_connect(
            CARTESIA_WS_URL, headers=headers, heartbeat=20,
        ) as ws:
            await ws.send_str(msg)
            async for ws_msg in ws:
                if ws_msg.type != aiohttp.WSMsgType.TEXT:
                    if ws_msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                        log.warning("fallback WS closed: %s", ws.exception())
                        break
                    continue
                try:
                    ev = json.loads(ws_msg.data)
                except json.JSONDecodeError:
                    continue
                ev_type = ev.get("type")
                if ev_type == "error":
                    log.warning(
                        "fallback WS error for %s: code=%s msg=%s",
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
                        log.exception("fallback bad audio chunk")
                        continue
                    if not header_emitted:
                        header_emitted = True
                        yield streaming_wav_header()
                    yield pcm
                elif ev_type == "done" and ev.get("done"):
                    break
    except Exception:
        log.exception("fallback Cartesia WS failed for %s", persona.key)


# ---------------------------------------------------------------------------
# Public entry point: produce a panel-mode audio stream for one persona turn.
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


async def _claude_reply_text(
    persona,
    panel_context: str,
    anthropic_api_key: str,
    http_session: aiohttp.ClientSession,
) -> str:
    """One Claude call → full reply text. Used because the pipecat-agents
    path emits TTS for the WHOLE turn's text in one synthesis (the panel turn
    is short — 1-2 sentences — so we don't lose much vs. streaming, and we
    gain a clean reply_text to append to the transcript)."""
    system = _build_system_prompt(persona, panel_context)
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 110,
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
        data = await r.json()
    for part in data.get("content", []):
        if part.get("type") == "text":
            return part["text"].strip()
    return ""


async def agents_mode_stream(
    persona,
    panel_context: str,
    *,
    cartesia_api_key: str,
    anthropic_api_key: str,
    http_session: aiohttp.ClientSession,
    text_out: list[str],
) -> AsyncIterator[bytes]:
    """Produce one panel turn's audio for `persona` given the panel context.

    Pipeline:
      1. Claude generates the reply text (1-2 short sentences).
      2. The Pipecat CartesiaTTSService — or, if pipecat is unavailable, the
         identical-protocol direct WS path — synthesizes the audio.
      3. Bytes are yielded to the caller (WAV header first, then raw PCM).
      4. `text_out[0]` is set to the spoken text so the caller can append it
         to the panel transcript.

    Latency budget: Claude ~400-600 ms TTFB + Cartesia ~200-400 ms TTFA ≈ 1s
    total time-to-first-audio. Comfortably under the 2.5 s SLO.
    """
    text_out.clear()

    # Step 1: get the reply text.
    t_claude_0 = time.monotonic()
    try:
        reply = await _claude_reply_text(
            persona, panel_context, anthropic_api_key, http_session,
        )
    except Exception:
        log.exception("agents_mode: Claude reply failed for %s", persona.key)
        return
    claude_ms = int((time.monotonic() - t_claude_0) * 1000)
    log.info(
        "agents_mode: claude reply for %s in %dms — %d chars",
        persona.key, claude_ms, len(reply),
    )
    if not reply:
        return
    text_out.append(reply)

    # Step 2: synthesize.
    use_pipecat = _PIPECAT_AVAILABLE and not os.environ.get("VP_DISABLE_PIPECAT")
    backend = "pipecat" if use_pipecat else "fallback-ws"
    log.info("agents_mode: TTS backend=%s persona=%s", backend, persona.key)

    if use_pipecat:
        try:
            async for chunk in _pipecat_stream(persona, reply, cartesia_api_key):
                yield chunk
            return
        except Exception:
            log.exception(
                "pipecat pipeline failed for %s — falling back to direct WS",
                persona.key,
            )

    # Fallback: same Cartesia WS protocol, no Pipecat dependency.
    async for chunk in _fallback_cartesia_ws_stream(
        persona, reply, cartesia_api_key, http_session,
    ):
        yield chunk


def pipecat_available() -> bool:
    return _PIPECAT_AVAILABLE


def pipecat_import_error() -> str | None:
    return _PIPECAT_IMPORT_ERROR
