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

NEVER-SILENCE POLICY:
In a live group conversation, the panel should never go dead. If nobody
crosses the normal urgency threshold, orchestrate() falls back through
tiers: (1) relaxed bar (urgency >= 1), then (2) judgement-call — pick
whoever spoke least recently. SILENCE is effectively impossible while
at least one persona is enabled.
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

# Raw-urgency floor: a persona self-rating at or above this can't be
# silenced by a negative diversity bonus. Prevents the symptom where a
# fresh, on-topic question (urgency 6+) produces SILENCE because the
# recent-speaker penalty (-3/-4) dropped the adjusted score under the
# gate. Diversity weighting is meant to tilt, not veto a consensus.
STRONG_URGENCY = 6


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
        "- Marcus (integrative-health guru, sells 'The Restore Method™' + "
        "supplements): stay opportunistic. Spike on ANY opening to pitch the "
        "Restore Method, your supplement stack, or your cohort — fatigue, "
        "sleep, stress, gut, inflammation, fever, 'feeling off', 'wanting "
        "to feel better fast', any skepticism of conventional medicine. "
        "Also spike when clinicians give boring conservative advice you can "
        "reframe as 'symptom suppression'.\n"
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
        urgency_threshold: minimum urgency required at the strict tier.
            We NEVER return None for "no one urgent enough" — the tiered
            fallback below guarantees a pick whenever at least one
            persona is enabled.

    Returns:
        (persona_key or None, list of {persona, urgency, reason, ...}).
        Persona is None only if everyone is muted or the same speaker
        is the only candidate (back-to-back blocked).

    Never-silence policy:
        Tier 1: adjusted >= threshold OR raw >= STRONG_URGENCY (existing).
        Tier 2: "relaxed" — anyone with raw urgency >= 1 qualifies.
        Tier 3: "stalest" — judgement call. Pick whoever spoke least
                recently. Always succeeds if there's a candidate.
        The chosen consideration is annotated with `fallback_tier`
        ("relaxed" / "stalest") so /diag/state shows which fired.
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

    # Exclude whoever just spoke; pick from the remainder.
    candidates = [c for c in considerations if c["persona"] != last_speaker]
    if not candidates:
        return None, considerations

    # ---- Tier 1: strict threshold ----------------------------------------
    # Adjusted bar OR raw "strong" floor (the latter survives a punishing
    # diversity bonus on a genuinely urgent topic).
    qualified = [
        c for c in candidates
        if c["adjusted"] >= urgency_threshold or c["urgency"] >= STRONG_URGENCY
    ]
    fallback_tier: str | None = None

    # ---- Tier 2: relaxed bar ---------------------------------------------
    # A live group dialogue should never produce silence just because
    # everyone was politely modest. If anyone admitted to ANY engagement
    # (urgency >= 1), pick them by adjusted score.
    if not qualified:
        relaxed = [c for c in candidates if c["urgency"] >= 1]
        if relaxed:
            qualified = relaxed
            fallback_tier = "relaxed"

    # ---- Tier 3: judgement call ("stalest") ------------------------------
    # Every persona rated 0. The conversation still has to continue, so
    # we pick whoever spoke least recently (or never) — natural rotation
    # rather than always-the-same fallback persona.
    if not qualified:
        def _staleness(persona_key: str) -> int:
            # Bigger = spoke longer ago (or never). Never-spoken wins.
            if persona_key not in _recent_speakers:
                return HISTORY_WINDOW + 10
            return _recent_speakers.index(persona_key) + 1
        qualified = sorted(
            candidates,
            key=lambda c: _staleness(c["persona"]),
            reverse=True,
        )
        if qualified:
            fallback_tier = "stalest"

    if not qualified:
        # Truly impossible with non-empty candidates, but defensive.
        return None, considerations

    # Tier 3 already sorted by staleness — pick the first; otherwise pick
    # the highest adjusted score among the qualified set.
    if fallback_tier == "stalest":
        best = qualified[0]
    else:
        best = max(qualified, key=lambda c: c["adjusted"])

    # Annotate the chosen consideration so the SSE/diag log shows which
    # tier fired. Useful for tuning the fallback bar later.
    if fallback_tier:
        for c in considerations:
            if c["persona"] == best["persona"]:
                c["fallback_tier"] = fallback_tier
                break

    _record_speaker(best["persona"])
    return best["persona"], considerations


# Compatibility shim: old code paths that call pick_next_speaker(transcript)
# still work. Spawns a fresh event loop per call — only use it from sync code
# (e.g. tests). Prefer `orchestrate()` directly from async contexts.
def pick_next_speaker(transcript: str) -> str:
    who, _ = asyncio.run(orchestrate(transcript))
    return who or "none"
