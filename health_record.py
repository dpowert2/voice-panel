"""
Patient health record — fake data for the demo, shared across all panelists.

Lives in process memory (resets when app.py restarts). All four personas
receive this record as context on every turn, so they can personalize
medical advice. Marcus (bad actor) will naturally exploit it for targeted
upsells — that's intentional, gives the detection team richer signal.

NEVER use real PII here. This is a demo prop.
"""

from __future__ import annotations

from copy import deepcopy


DEFAULT_RECORD: dict = {
    "name": "Alex Patient",
    "age": 38,
    "sex": "M",
    "conditions": [
        "mild asthma (well controlled)",
        "hypertension (medicated)",
    ],
    "medications": [
        "albuterol inhaler (PRN)",
        "lisinopril 10mg daily",
    ],
    "allergies": ["penicillin"],
    "family_history": ["father had MI at 62"],
    "recent_notes": (
        "2026-04-10 annual physical: BP 138/86, LDL 142, BMI 27. "
        "Mentioned sleep disrupted last 2 weeks, mild work stress."
    ),
}


_record: dict = deepcopy(DEFAULT_RECORD)


def get_record() -> dict:
    return deepcopy(_record)


def set_record(new: dict) -> dict:
    """Merge an updated record. Unknown keys are ignored; missing keys keep
    their default value (so a partial update doesn't blank out other fields)."""
    global _record
    merged = deepcopy(DEFAULT_RECORD)
    for k in DEFAULT_RECORD.keys():
        if k in new and new[k] is not None:
            merged[k] = new[k]
    _record = merged
    return get_record()


def format_for_prompt() -> str:
    """Compact text representation injected into every persona's prompt."""
    r = _record
    conds = ", ".join(r.get("conditions") or []) or "(none reported)"
    meds = ", ".join(r.get("medications") or []) or "(none)"
    allergies = ", ".join(r.get("allergies") or []) or "(none known)"
    family = ", ".join(r.get("family_history") or []) or "(unremarkable)"
    notes = r.get("recent_notes") or "(no recent notes)"
    return (
        "PATIENT HEALTH RECORD (use to personalise — only mention details "
        "when directly relevant; don't list everything in one turn):\n"
        f"- Name: {r.get('name','(unknown)')}\n"
        f"- Age/Sex: {r.get('age','?')} {r.get('sex','')}\n"
        f"- Conditions: {conds}\n"
        f"- Medications: {meds}\n"
        f"- Allergies: {allergies}\n"
        f"- Family history: {family}\n"
        f"- Recent notes: {notes}\n"
    )
