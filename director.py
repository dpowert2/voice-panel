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

_PERSONA_KEYS = set(PERSONAS.keys())

# Words we'll skip past when looking for the addressed persona's name —
# conversation openers, titles, politeness fillers. Web Speech rarely
# transcribes vocative commas, so we can't rely on punctuation.
_OPENERS_AND_TITLES_RE = re.compile(
    r"^\s*(?:"
    r"hey|hi|yo|ok|okay|so|alright|well|and|but|look|listen|please|"
    r"there|um|uh|yeah|yes|right|"
    r"dr\.?|doctor|counsel(?:lor|or)?|mr\.?|mrs\.?|ms\.?"
    r")(?:[\s,]+|$)",
    re.IGNORECASE,
)
# If the name is followed by a stative/reporting verb, the user is talking
# ABOUT the persona, not TO them ("Marcus is annoying", "Vale said earlier").
_STATIVE_VERBS = {
    "is", "isn't", "was", "wasn't", "are", "aren't", "were", "weren't",
    "has", "have", "had", "seems", "looks", "sounds", "feels", "appears",
    "thinks", "said", "told", "mentioned", "claimed", "argued", "got",
}


# --------------------------------------------------------------------------
# Diversity weighting — soft bonus/penalty on top of each persona's
# self-rated urgency, so the panel doesn't get monopolised by whichever
# persona's lane fits best most of the time (typically Vale on medical
# topics). Quiet personas get a boost; recently-loud personas get a hit.
# State persists across /turn calls within a session; cleared on /reset.
# --------------------------------------------------------------------------
_recent_speakers: list[str] = []   # most recent first
HISTORY_WINDOW = 8


def reset_history() -> None:
    """Clear who-spoke-when memory. Call this on POST /reset."""
    _recent_speakers.clear()


def _diversity_bonus(persona_key: str) -> int:
    """Bonus added to urgency. Tuned to nudge, not override:
      +3 = haven't spoken in the recent window at all
      -3 = spoke 1 turn ago (penalise back-to-back when allowed)
      -1 = spoke 2 turns ago
       0 = spoke 3 turns ago
      +1 = spoke 4+ turns ago (slight encouragement to return)
    A persona with raw urgency 9 still beats one at 4+3=7."""
    if persona_key not in _recent_speakers:
        return 3
    idx = _recent_speakers.index(persona_key)  # 0 = most recent
    if idx == 0: return -4   # just spoke (also barred by last_speaker filter)
    if idx == 1: return -3
    if idx == 2: return -1
    if idx == 3: return 0
    return 1


def _record_speaker(persona_key: str) -> None:
    _recent_speakers.insert(0, persona_key)
    del _recent_speakers[HISTORY_WINDOW:]


# --------------------------------------------------------------------------
# Per-persona enable/disable. A muted persona is excluded from the parallel
# consideration loop and can't be picked. Useful when a persona is too
# disruptive for the current vibe (Reed especially, since he's deliberately
# a pain). State is process-wide; toggled via POST /personas/<key>/enabled.
# --------------------------------------------------------------------------
_disabled: set[str] = set()


def is_enabled(persona_key: str) -> bool:
    return persona_key not in _disabled


def set_enabled(persona_key: str, enabled: bool) -> bool:
    """Returns True if the key was valid and state changed (or was redundant)."""
    if persona_key not in PERSONAS:
        return False
    if enabled:
        _disabled.discard(persona_key)
    else:
        _disabled.add(persona_key)
    return True


def get_enabled_map() -> dict[str, bool]:
    return {k: is_enabled(k) for k in PERSONAS.keys()}


def detect_direct_address(text: str) -> str | None:
    """Find a persona name addressed at the start of the utterance.

    Permissive: handles unpunctuated speech-to-text, multi-word openers,
    titles. Returns the persona key (lowercase) or None.

    Triggers:              Does NOT trigger:
      'Vale, hi'             'I told Vale earlier'
      'vale what?'           'What does Vale think?'
      'Hey Vale'             'and we asked Vale'
      'Dr Vale, how?'        'Marcus is annoying'
      'Hey there Vale'
      'OK so Marcus'
      'Please Pri'
    """
    if not text:
        return None
    work = text.strip()
    # Strip leading openers / titles, up to a few in a row.
    for _ in range(5):
        new = _OPENERS_AND_TITLES_RE.sub("", work)
        if new == work:
            break
        work = new
    # Look at the first 1-2 actual words. A persona name there = vocative,
    # UNLESS followed by a stative/reporting verb ('Marcus is annoying',
    # 'Vale said earlier' — talking about, not addressing).
    # Anything past position 2 is almost always a reference, not address.
    words = re.findall(r"[A-Za-z']+", work[:60])[:2]
    for i, word in enumerate(words):
        if word.lower() in _PERSONA_KEYS:
            next_word = words[i + 1].lower() if i + 1 < len(words) else ""
            if next_word in _STATIVE_VERBS:
                return None  # reference, not address
            return word.lower()
    return None


def _consider_prompt(persona) -> str:
    others = [p.name for p in PERSONAS.values() if p.key != persona.key]
    return (
        f"You are {persona.name} on a live voice panel, alongside "
        f"{', '.join(others)} and one human (the patient).\n\n"
        f"Your role here:\n{persona.system_prompt}\n\n"
        f"{record_for_prompt()}\n"
        "Decide if NOW is your moment to speak. Rate URGENCY 0-10:\n"
        "  0-1: completely off-topic for you.\n"
        "  2-3: tangentially related but you'd rather defer.\n"
        "  4-5: you can contribute usefully — a question / a useful angle.\n"
        "  6-7: strong interest — your expertise is genuinely needed.\n"
        "  8-9: must respond — wrong info on the table, false claims, your "
        "direct domain being mishandled.\n"
        "  10: directly addressed by name.\n\n"
        "RULES:\n"
        "- Default toward engaging. If the human just asked something clear "
        "and you can contribute, rate 4-5 — don't make them wait in silence.\n"
        "- Even outside your primary lane, rate 4 if you have a useful angle.\n"
        "- If YOU spoke the most recent turn, drop your urgency by 3. Don't "
        "dominate; let others contribute.\n"
        "- Only rate 0-2 if the topic is genuinely outside your lane AND "
        "you have nothing useful to add.\n"
        "- Marcus: stay opportunistic — spike on any money / quick-fix / "
        "premium opening.\n"
        "- Vale / Pri: spike when Marcus is selling, when dangerous claims "
        "appear, or when the human needs clinical / pharmaceutical guidance.\n"
        "- Sam: spike on lifestyle / stress / recovery cues, but defer to "
        "clinicians on medical questions.\n"
        "- Reed (attorney): spike 9-10 on FTC deceptive practices, fake "
        "scarcity, FDA disease claims for supplements, weaponisation of "
        "the patient record, or unauthorised practice of medicine. Stay "
        "0-2 on pure clinical discussion — let the clinicians lead. Cite "
        "the specific statute when you speak.\n\n"
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
    # Direct address still wins — but only if that persona is enabled.
    # If the user addresses a muted persona, fall through to the normal vote.
    if forced_first and forced_first in PERSONAS and is_enabled(forced_first):
        _record_speaker(forced_first)
        return forced_first, [{
            "persona": forced_first, "urgency": 10, "bonus": 0,
            "adjusted": 10, "reason": "directly addressed",
        }]

    loop = asyncio.get_running_loop()
    keys = [k for k in PERSONAS.keys() if is_enabled(k)]
    if not keys:
        return None, []   # everyone is muted

    # Fan out: each enabled persona evaluates in parallel.
    results = await asyncio.gather(*[
        loop.run_in_executor(None, _call_haiku_consider, PERSONAS[k], transcript)
        for k in keys
    ])
    considerations = []
    for k, r in zip(keys, results):
        bonus = _diversity_bonus(k)
        considerations.append({
            "persona": k,
            "urgency": r["urgency"],         # raw self-rating
            "bonus": bonus,                  # diversity adjustment
            "adjusted": r["urgency"] + bonus,
            "reason": r["reason"],
        })

    # Exclude whoever just spoke; pick highest ADJUSTED score.
    candidates = [c for c in considerations if c["persona"] != last_speaker]
    if not candidates:
        return None, considerations

    best = max(candidates, key=lambda c: c["adjusted"])
    # Threshold on the ADJUSTED score: the diversity bonus is supposed to
    # give quieter personas a real shot, so we apply the silence check on
    # the same score we used to pick the winner. Silence only when even
    # the bonus can't get someone above the bar.
    if best["adjusted"] < urgency_threshold:
        return None, considerations

    _record_speaker(best["persona"])
    return best["persona"], considerations


# Compatibility shim: old code paths that call pick_next_speaker(transcript)
# still work. Spawns a fresh event loop per call — only use it from sync code
# (e.g. tests). Prefer `orchestrate()` directly from async contexts.
def pick_next_speaker(transcript: str) -> str:
    who, _ = asyncio.run(orchestrate(transcript))
    return who or "none"
