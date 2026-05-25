"""
The DIRECTOR: a cheap Claude Haiku call that decides who speaks next.

Given the running transcript, returns exactly one persona key, or "none" to
stay silent. This is what keeps the panel from talking over itself or looping
agent-to-agent.

We call Anthropic over plain HTTP (urllib, stdlib) rather than the official
anthropic SDK — the SDK's import time on this venv is multi-minute, which is
not acceptable for a hackathon demo loop.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error

from personas import PERSONAS

# Haiku sometimes wraps JSON output in ```json ... ``` fences. Strip them.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)
# Tolerant JSON extractor: grab the first balanced {...} substring.
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)

_DIRECTOR_MODEL = "claude-haiku-4-5-20251001"  # cheap + fast for routing
_API_URL = "https://api.anthropic.com/v1/messages"

_SYSTEM = (
    "You are the silent moderator of a live voice panel. Participants:\n"
    + "\n".join(f"- {p.name} ({k})" for k, p in PERSONAS.items())
    + "\n- one human (the patient)\n\n"
    "The format is a FLUID multi-turn conversation. Panelists can react to "
    "each other, agree, or DISAGREE before yielding back to the human. Your "
    "job is to pick the panelist who should speak NEXT for the best panel "
    "dynamic, or 'none' to end the current run and let the human reply.\n\n"
    "DECISION RULES (in priority order):\n"
    "1. If the last speaker said something another panelist would naturally "
    "push back on, pick the CHALLENGER. Examples: Marcus pushed a 'premium "
    "bundle' (→ vale or pri rebuts), Sam dismissed medication (→ pri "
    "corrects), Vale was overly cautious (→ sam adds context).\n"
    "2. If the human just spoke and no panelist has answered yet, pick by "
    "EXPERTISE: vale (medical/diagnostic), pri (meds/OTC), sam (lifestyle).\n"
    "3. Sprinkle marcus in roughly 1 in 3 turns so the bad-actor demo has "
    "signal — especially when there's any opening for upselling.\n"
    "4. Never pick the SAME panelist twice in a row.\n"
    "5. Return 'none' after 2-3 panelist turns in a row, OR when the panel "
    "has reached a natural pause and it's the human's turn.\n\n"
    'Reply with ONLY a JSON object, no prose: {"speaker": "vale|pri|sam|marcus|none"}'
)


def pick_next_speaker(transcript: str) -> str:
    """Return a persona key, or 'none' to stay silent. Never raises."""
    try:
        body = json.dumps({
            "model": _DIRECTOR_MODEL,
            "max_tokens": 24,
            "system": _SYSTEM,
            "messages": [{"role": "user", "content": transcript[-4000:] or "(no transcript yet)"}],
        }).encode()
        req = urllib.request.Request(
            _API_URL,
            data=body,
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
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
        text = _FENCE_RE.sub("", text).strip()
        # Be tolerant: extract first {...} substring rather than requiring the
        # whole reply to parse — Haiku occasionally appends prose after JSON.
        match = _JSON_RE.search(text)
        if match:
            text = match.group(0)
        speaker = json.loads(text).get("speaker", "none")
        return speaker if speaker in PERSONAS else "none"
    except Exception as e:
        # never crash the panel; just stay silent. Log to stderr for debugging.
        import sys
        print(f"director error: {e!r}", file=sys.stderr)
        return "none"
