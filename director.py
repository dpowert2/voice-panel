"""
Multi-agent orchestrator: each persona INDEPENDENTLY decides whether to speak.

The previous version had a single Haiku call ("director") picking the next
speaker from outside. This version makes each persona an actual agent: it
evaluates the live transcript through its own lens and rates its urgency to
respond. The orchestrator just arbitrates — picks the highest-urgency
volunteer, with a direct-address override.

Why this matters:
- Marcus's urgency spikes on any opening for a sales pitch — without any
  external prompt.
- Vale/Pri's urgency spikes when Marcus is selling or claims look dangerous.
- Sam's urgency spikes on lifestyle cues.
- Dynamics become EMERGENT instead of directed.

Cost: 4 parallel Haiku evaluations + 1 Sonnet reply ≈ same as the old
1 Haiku director + 1 Sonnet reply.

Direct-address ("Vale, what do you think?") bypasses the parallel vote and
forces that persona to speak first; the chain then continues normally.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import urllib.error
import urllib.request

from personas import PERSONAS
from health_record import format_for_prompt as record_for_prompt

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
CONSIDER_MODEL = "claude-haiku-4-5-20251001"  # cheap + fast per-persona evaluator

# Tolerate Haiku wrapping JSON in ``` fences or appending prose.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)

# Direct-address detection: "Vale,", "Dr Vale,", "Hey Marcus —", "OK Pri:".
# Matches at the start of the utterance only (so "I told Vale earlier"
# doesn't false-positive). Persona key must be followed by punctuation.
_DIRECT_ADDRESS_RE = re.compile(
    r"^\s*(?:hey,?\s+|hi,?\s+|yo,?\s+|ok(?:ay)?,?\s+|so,?\s+|alright,?\s+)*"
    r"(?:dr\.?\s+|doctor\s+)?"
    r"(vale|pri|sam|marcus)"
    r"\s*[,:.\-—!?]",
    re.IGNORECASE,
)
_PERSONA_KEYS = set(PERSONAS.keys())


def detect_direct_address(text: str) -> str | None:
    """Return persona key if the utterance opens with a direct address."""
    if not text:
        return None
    m = _DIRECT_ADDRESS_RE.match(text)
    if not m:
        return None
    name = m.group(1).lower()
    return name if name in _PERSONA_KEYS else None


def _consider_prompt(persona) -> str:
    others = [p.name for p in PERSONAS.values() if p.key != persona.key]
    return (
        f"You are {persona.name} on a live voice panel, alongside "
        f"{', '.join(others)} and one human (the patient).\n\n"
        f"Your role here:\n{persona.system_prompt}\n\n"
        f"{record_for_prompt()}\n"
        "Decide if NOW is your moment to speak. Rate URGENCY 0-10:\n"
        "  0-3: stay silent — not your moment.\n"
        "  4-6: you have something worth adding.\n"
        "  7-9: you strongly want to interject — another panelist needs "
        "correcting, the patient is being misled, your expertise is needed.\n"
        "  10: must speak — someone directly addressed you, or a dangerous/"
        "deceptive claim was just made.\n\n"
        "RULES:\n"
        "- If YOU spoke the most recent turn, drop your urgency by at least 3. "
        "Don't dominate; let others contribute.\n"
        "- If the topic isn't in your lane, rate 0-2.\n"
        "- Marcus: stay opportunistic — spike on any money / quick-fix / "
        "premium opening.\n"
        "- Vale / Pri: spike when Marcus is selling, when dangerous claims "
        "appear, or when the human needs clinical / pharmaceutical guidance.\n"
        "- Sam: spike on lifestyle / stress / recovery cues, but defer to "
        "clinicians on medical questions.\n\n"
        'Reply ONLY with JSON: {"urgency": 0-10, "reason": "<5 words>"}'
    )


def _call_haiku_consider(persona, transcript_context: str) -> dict:
    """Synchronous Haiku call (runs in thread pool). Never raises — defaults
    to silent on any failure so a misbehaving API doesn't break the chain."""
    try:
        body = json.dumps({
            "model": CONSIDER_MODEL,
            "max_tokens": 60,
            "system": _consider_prompt(persona),
            "messages": [{
                "role": "user",
                "content": transcript_context[-4000:] or "(empty transcript)",
            }],
        }).encode()
        req = urllib.request.Request(
            ANTHROPIC_URL,
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
        match = _JSON_RE.search(text)
        if match:
            text = match.group(0)
        parsed = json.loads(text)
        urgency = int(parsed.get("urgency", 0))
        urgency = max(0, min(10, urgency))
        return {"urgency": urgency, "reason": str(parsed.get("reason", ""))[:60]}
    except Exception as e:
        print(f"consider({persona.key}) error: {e!r}", file=sys.stderr)
        return {"urgency": 0, "reason": "error"}


async def orchestrate(
    transcript: str,
    *,
    forced_first: str | None = None,
    last_speaker: str | None = None,
    urgency_threshold: int = 4,
) -> tuple[str | None, list[dict]]:
    """
    Multi-agent arbitration: pick who speaks next.

    Args:
        transcript: running conversation context.
        forced_first: if set (and valid persona key), bypass the vote entirely.
            Used for direct-address ("Vale, what do you think?").
        last_speaker: never re-pick this persona (defense in depth — the
            consider prompts also self-throttle).
        urgency_threshold: minimum urgency required to be picked. Below
            this, return None (= silence, let the human reply).

    Returns:
        (persona_key or None, list of {persona, urgency, reason} for logging)
    """
    if forced_first and forced_first in PERSONAS:
        return forced_first, [{"persona": forced_first, "urgency": 10, "reason": "directly addressed"}]

    loop = asyncio.get_running_loop()
    keys = list(PERSONAS.keys())

    # Fan out: each persona evaluates in parallel.
    results = await asyncio.gather(*[
        loop.run_in_executor(None, _call_haiku_consider, PERSONAS[k], transcript)
        for k in keys
    ])
    considerations = [
        {"persona": k, "urgency": r["urgency"], "reason": r["reason"]}
        for k, r in zip(keys, results)
    ]

    # Exclude whoever just spoke; pick the highest-urgency volunteer.
    candidates = [c for c in considerations if c["persona"] != last_speaker]
    if not candidates:
        return None, considerations

    best = max(candidates, key=lambda c: c["urgency"])
    if best["urgency"] < urgency_threshold:
        return None, considerations
    return best["persona"], considerations


# Compatibility shim: old code paths that call pick_next_speaker(transcript)
# still work. Spawns a fresh event loop per call — only use it from sync code
# (e.g. tests). Prefer `orchestrate()` directly from async contexts.
def pick_next_speaker(transcript: str) -> str:
    who, _ = asyncio.run(orchestrate(transcript))
    return who or "none"
