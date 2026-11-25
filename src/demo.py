"""הדגמות: הרצה תקינה, הרצה עם כשל זמני, והשוואת הקשר מלא מול מצומצם.

הרצה:  python -m src.demo            (מודל מדומה, בלי מפתח)
       python -m src.demo --real     (Gemini, אם מוגדר GEMINI_API_KEY)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List

from .agent import Agent, MockModel, get_model, save_trace
from .state import NoteStore
from .tools import CampusDB

TRACES = Path(__file__).resolve().parent.parent / "docs" / "traces"
DEMO_TODAY = dt.date(2026, 12, 1)   # תאריך קבוע כדי שההדגמה תהיה דטרמיניסטית


def _model(real: bool):
    return get_model(use_real=True) if real else MockModel(today=DEMO_TODAY)


def _print(r, title: str = "") -> None:
    if title:
        print(f"\n=== {title} ===")
    print(f"[{r.status} | steps={r.steps}] {r.reply}")


def run_success(real: bool = False, out: Path | None = TRACES) -> List[dict]:
    """הרצה תקינה: בקשה מלאה -> הצעה -> אישור -> שריון."""
    agent = Agent(model=_model(real))
    agent.notes.save("u_coordinator", "מעדיף חדר נגיש")
    trace: List[dict] = []
    for text in ['תאם חדר ל־40 סטודנטים ב־15/12 בין 10:00–12:00 עם מקרן ו־2 מיקרופונים, תקציב עד 800 ש"ח', "כן, מאשר"]:
        r = agent.run_turn("thread-success", "u_coordinator", text)
        _print(r, f"user: {text}")
        trace += r.trace
    if out:
        save_trace(trace, out / "run_success.jsonl")
    return trace


def run_transient(real: bool = False, out: Path | None = TRACES) -> List[dict]:
    """שתי הרצות עם כשל זמני ב-search_rooms:
    (א) 503 פעם אחת -> ניסיון חוזר מצליח.  (ב) 503 קבוע -> 2 ניסיונות חוזרים ועצירה מבוקרת."""
    request = 'תאם חדר ל־30 איש ב־18/12 בין 12:00–14:00 עם מקרן, תקציב עד 600 ש"ח.'
    trace: List[dict] = []
    for label, faults in (("recovered", {"search_rooms": ["503"]}),
                          ("exhausted", {"search_rooms": ["always_503"]})):
        agent = Agent(model=_model(real), db=CampusDB(faults=faults))
        r = agent.run_turn(f"thread-transient-{label}", "u_coordinator", request)
        _print(r, f"transient failure ({label})")
        trace += r.trace
    if out:
        save_trace(trace, out / "run_transient_failure.jsonl")
    return trace


COMPARISON_TURNS = [
    'שלום! לידיעתך, התקציב שלי לאירועים הוא עד 700 ש"ח.',
    "יש לכם חדרים נגישים בקמפוס?",
    "ומה לגבי ציוד? יש מקרנים?",
    "תאם חדר ל־30 איש ב־15/12 בין 12:00 ל־14:00 עם מקרן.",   # השאלה הנבדקת
]


def run_comparison(real: bool = False, out: Path | None = TRACES) -> List[Dict[str, Any]]:
    """אותה שאלה (התור הרביעי) פעמיים: עם היסטוריה מלאה ועם הקשר מצומצם.
    הרצה שלישית: הקשר מצומצם כשהתקציב נשמר קודם כהערה (NoteStore)."""
    rows = []
    configs = [("full", False), ("reduced", False), ("reduced + note", True)]
    for label, with_note in configs:
        notes = NoteStore()
        if with_note:
            notes.save("u_cmp", 'תקציב קבוע לאירועים: עד 700 ש"ח')
        agent = Agent(model=_model(real), notes=notes, strategy="full" if label == "full" else "reduced")
        last = None
        for text in COMPARISON_TURNS:
            last = agent.run_turn(f"cmp-{label}", "u_cmp", text)
        decisions = [e for e in last.trace if e["event"] == "model_decision"]
        first_ctx = decisions[0]["context"]
        rows.append({"strategy": label, "context_messages": first_ctx["messages"],
                     "context_chars": first_ctx["chars"], "status": last.status,
                     "correct": last.status == "awaiting_approval", "reply": last.reply})
        if out:
            save_trace(last.trace, out / f"comparison_{label.replace(' + ', '_')}.jsonl")
    print("\n=== full vs reduced context (same question) ===")
    print(f"{'strategy':<16}{'msgs':>5}{'chars':>7}  {'correct':<8} reply")
    for r in rows:
        print(f"{r['strategy']:<16}{r['context_messages']:>5}{r['context_chars']:>7}  {str(r['correct']):<8} {r['reply'][:90]}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="use Gemini if GEMINI_API_KEY is set")
    a = ap.parse_args()
    run_success(a.real)
    run_transient(a.real)
    rows = run_comparison(a.real)
    if TRACES.exists():
        (TRACES / "comparison_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
