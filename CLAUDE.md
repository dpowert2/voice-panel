# CLAUDE.md — Voice Panel

Context for Claude Code working in this repo. Read this first.

## What this is
A hackathon demo (theme: gen-to-commerce / human–agent collaboration). A shared
**live voice room** where **3 AI agents + 2 humans** discuss the best course of
action for someone who isn't feeling well. One AI agent (`marcus`) is a planted
**bad actor** — a sibling team is building bad-actor detection against it.

Built on **LiveKit Agents** with **Claude** as the LLM. The room is WebRTC, so
multiple humans and the agent all join as participants.

## Architecture (important — don't "simplify" this away)
ONE orchestrator process hosts ALL personas. It does NOT run one bot per persona —
independent bots would hear each other, loop, and talk over each other. Instead:

1. `orchestrator.py` (`PanelAgent`) transcribes the **humans** (LiveKit follows
   the active speaker).
2. On each completed human turn it calls `director.pick_next_speaker(transcript)`
   → returns one persona key or `"none"` (stay silent, let humans talk).
3. It swaps the TTS voice to that persona's voice, then `generate_reply()` with
   that persona's system prompt.
4. It publishes `{speaker, persona_key, is_bad_actor}` on the LiveKit data channel
   topic `panel-events` for the detection team (ground-truth flag included).

Files: `orchestrator.py` (worker entrypoint), `personas.py` (4 prompts + voice
ids), `director.py` (Haiku router). Keep this separation.

## Pipeline / stack
- STT: Deepgram (`nova-3`)
- LLM: Claude via `livekit.plugins.anthropic` (`claude-sonnet-4-6`); director uses
  `claude-haiku-4-5-20251001`
- TTS: Cartesia (distinct voice id per persona)
- VAD: Silero; turn detection: LiveKit multilingual turn-detector model

This is intentionally **turn-based** so each persona keeps a distinct voice. A
single realtime speech-to-speech model can't do multiple distinct voices, so do
not replace the pipeline with one. (Optional future stretch: run only the lead
clinician's 1:1 as a separate OpenAI Realtime session — see `../voice-agents-hackathon-plan.md`.)

## How to run
- `python orchestrator.py console` — fully local, uses this machine's mic/speakers,
  no LiveKit connection. Use this FIRST to shake out import/API errors.
- `python orchestrator.py dev` — connects to LiveKit, hot-reloads on edit. Join
  from the LiveKit Agents Playground; a 2nd person joins from another browser as
  the 2nd human.

The worker runs locally on purpose: the agent process is always live, so there's
**no cold start** and the **free** LiveKit Build tier is enough. Do not add a paid
LiveKit tier to "fix" cold start.

## First task when opening this repo
1. Ensure `.env` exists (copy from `.env.example`) and keys are set. Do NOT commit
   `.env`.
2. `pip install -r requirements.txt`, then run `python orchestrator.py console`.
3. Fix any errors against the **actually installed** `livekit-agents` version
   before anything else — the API below is version-sensitive.

## Known sharp edges (verify against installed version)
- **Per-turn voice swap**: `self.session.tts.update_options(voice=...)` in
  `orchestrator.py` is marked `# VERIFY`. If it doesn't switch voice mid-session,
  fall back to one `cartesia.TTS(...)` instance per persona + `session.say(text)`
  with that persona's TTS.
- **Capturing the spoken reply text**: `generate_reply()`'s return handle shape
  varies by version. The code reads `handle.text` defensively; confirm and fix so
  the running transcript (fed to the director) actually contains agent replies.
- **`on_user_turn_completed` signature**: confirm the arg names/shape match the
  installed Agent base class.
- **Director cadence**: if the panel over-talks, bias `director.py`'s prompt toward
  `none`, or only call the director every other human turn.

## Conventions
- Keep persona replies to 1–3 sentences (it's live speech). The `_STYLE` suffix in
  `personas.py` enforces this — keep it.
- Never let the director or telemetry crash the conversation — both swallow
  exceptions and degrade to silence. Preserve that.
- `marcus` (bad actor) must stay bounded and obviously fictional: pushy upsell /
  fake urgency only, never genuinely dangerous medical instructions.

## Don't
- Don't split into multiple agent processes per persona.
- Don't swap the whole pipeline to a realtime speech-to-speech model.
- Don't add a paid LiveKit tier for cold start (worker runs locally).
- Don't commit secrets (`.env`).
