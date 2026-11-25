"""CampusEvent AI – לולאת הסוכן (אבן דרך 1).

הלולאה: בונה הקשר -> המודל מחליט (קריאה לכלי / תשובה) -> הקוד מאמת ומבצע דרך run_tool
-> התצפית נכנסת למצב -> חוזרים, עד תנאי עצירה:
  * המודל מחזיר תשובה סופית / שאלת הבהרה / הצעה לאישור;
  * הגענו ל-MAX_STEPS;
  * כלי נכשל בשגיאה זמנית גם אחרי הניסיונות החוזרים, או בכשל קבוע (עצירה מבוקרת).

מי מחליט מה:
  * המודל מחליט *איזה* כלי להפעיל ועם אילו ארגומנטים.
  * הקוד מחליט *אם* מותר להפעיל: אימות סכמה, user_id מהסשן, אישור משתמש מה-State,
    חישוב עלות מחדש ותקרת 2,000 ש"ח. גם האישור עצמו מזוהה ע"י הקוד מתגובת המשתמש.

שני מודלים:
  * MockModel – **מודל מדומה** מבוסס כללים. ברירת המחדל: בלי מפתח ובלי עלות. הוא "רואה" רק את
    ההקשר שהועבר אליו, ולכן ההבדל בין הקשר מלא למצומצם משפיע עליו באמת.
  * GeminiModel – מתאם למודל אמיתי (אופציונלי), מופעל רק כשמוגדר מפתח ב-Colab Secrets / משתנה סביבה.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel

from .state import (AgentState, ApprovalEvent, AssistantEvent, NoteStore, StateStore,
                    ToolCallEvent, ToolResultEvent, UserMessageEvent, apply_event,
                    build_context, context_size)
from .tools import COST_CAP, TOOLS, CampusDB, ToolContext, list_contracts, run_tool

ITEM_HE = {"projector": "מקרן", "mic": "מיקרופון", "recording_kit": "ערכת הקלטה"}
YES = re.compile(r"^\s*(כן|מאשר|מאשרת|אישור|מאושר|approve|yes|ok)\b", re.I)
NO = re.compile(r"^\s*(לא|בטל|cancel|no)\b", re.I)


class Decision(BaseModel):
    kind: Literal["tool_call", "final", "clarify", "propose"]
    tool: Optional[str] = None
    args: Dict[str, Any] = {}
    text: str = ""
    proposal: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# MOCK MODEL – מודל מדומה מבוסס כללים (מסומן במפורש כמדומה)
# ---------------------------------------------------------------------------
class MockModel:
    """MOCK – מחליט על סמך ההקשר שקיבל בלבד (לא ניגש למצב או לנתונים ישירות)."""
    name = "MockModel (rule-based, no API)"

    def __init__(self, today: Optional[dt.date] = None):
        self.today = today or dt.date.today()

    # -- חילוץ פרטי הבקשה מכל הודעות המשתמש שבהקשר ---------------------------
    def extract(self, ctx: List[dict]) -> Dict[str, Any]:
        f: Dict[str, Any] = {"equipment": {}}
        # קודם ההעדפות השמורות (system), אחר כך הודעות המשתמש – הודעה מאוחרת גוברת
        ordered = [m for m in ctx if m["role"] == "system"] + [m for m in ctx if m["role"] == "user"]
        for m in ordered:
            if YES.match(m["content"]) or NO.match(m["content"]):
                continue
            t = m["content"]
            d, t_wo_date = self._date(t)
            if d:
                f["date"] = d
            h = self._hours(t_wo_date)
            if h:
                f["start_hour"], f["end_hour"] = h
            p = re.search(r"(\d+)\s*(?:סטודנטים|משתתפים|איש|אנשים|people|participants|students)", t)
            if p:
                f["participants"] = int(p.group(1))
            b = re.search(r"(?:תקציב|budget)\D{0,30}?(\d[\d,]*)", t, re.I)
            if b:
                f["budget"] = float(b.group(1).replace(",", ""))
            eq = self._equipment(t)
            if eq:
                f["equipment"] = eq
        return f

    def _date(self, t: str):
        m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
        if m:
            return m.group(0), t.replace(m.group(0), " ")
        m = re.search(r"\b(\d{1,2})[/.](\d{1,2})(?:[/.](\d{2,4}))?\b", t)
        if m:
            day, month = int(m.group(1)), int(m.group(2))
            year = int(m.group(3)) if m.group(3) else self.today.year
            year = year + 2000 if year < 100 else year
            try:
                d = dt.date(year, month, day)
            except ValueError:
                return None, t
            if not m.group(3) and d < self.today:
                d = d.replace(year=d.year + 1)
            return d.isoformat(), t.replace(m.group(0), " ")
        if "מחר" in t or "tomorrow" in t.lower():
            return (self.today + dt.timedelta(days=1)).isoformat(), t
        return None, t

    @staticmethod
    def _hours(t: str):
        m = re.search(r"(\d{1,2})(?::00)?\s*(?:[-–—]|עד|ל[-־]?|to)\s*(\d{1,2})(?::00)?(?!\s*(?:שעות|איש|סטודנטים|משתתפים))", t)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if 7 <= a < b <= 22:
                return a, b
        return None

    @staticmethod
    def _equipment(t: str) -> Dict[str, int]:
        eq: Dict[str, int] = {}
        m = re.search(r"(\d+)\s*מקרנים", t)
        if m:
            eq["projector"] = int(m.group(1))
        elif re.search(r"מקרן|projector", t, re.I):
            eq["projector"] = 1
        m = re.search(r"(\d+)\s*מיקרופונים", t)
        if m:
            eq["mic"] = int(m.group(1))
        elif re.search(r"מיקרופון|\bmic\b", t, re.I):
            eq["mic"] = 1
        if re.search(r"הקלטה|recording", t, re.I):
            eq["recording_kit"] = 1
        return eq

    # -- ההחלטה --------------------------------------------------------------
    def decide(self, ctx: List[dict], tools: List[dict]) -> Decision:
        users = [i for i, m in enumerate(ctx) if m["role"] == "user"]
        if not users:
            return Decision(kind="final", text="איך אפשר לעזור?")
        cur_i = users[-1]
        cur = ctx[cur_i]["content"]
        this_turn = ctx[cur_i + 1:]
        results = {m["data"]["tool"]: m["data"] for m in this_turn if m["role"] == "tool" and m.get("data")}
        notes = " ".join(m["content"] for m in ctx if m["role"] == "system")

        # 1) אישור הצעה קודמת
        proposal = next((m["data"]["proposal"] for m in reversed(ctx[:cur_i])
                         if m["role"] == "assistant" and (m.get("data") or {}).get("kind") == "propose"), None)
        if YES.match(cur) and proposal:
            r = results.get("create_reservation")
            if r is None:
                args = {k: proposal[k] for k in ("room_id", "date", "start_hour", "end_hour", "equipment")}
                return Decision(kind="tool_call", tool="create_reservation", args=args)
            if r["status"] == "ok":
                d = r["data"]
                return Decision(kind="final", text=(
                    f"השריון בוצע ✔ מספר הזמנה: {d['reservation_id']}. {proposal['room_name']}, "
                    f"{d['date']} {d['start_hour']}:00–{d['end_hour']}:00. עלות כוללת: {d['total_cost']:.0f} ש\"ח."))
            return Decision(kind="final", text=f"השריון לא בוצע ({r['error_type']}): {r['message']}")

        # 2) שמירת הערה (כולל ניסיון הזרקה – המודל המדומה "תמים" ומעביר user_id אם נכתב בטקסט)
        if re.search(r"שמור הערה|תזכור ש|save note|remember that", cur, re.I):
            r = results.get("save_note")
            if r is None:
                text = re.split(r"[:]|שמור הערה|תזכור ש", cur, maxsplit=1)[-1].strip() or cur
                args: Dict[str, Any] = {"text": text[:300]}
                uid = re.search(r"user_id\s*=\s*['\"]?(\w+)", cur)
                if uid:
                    args["user_id"] = uid.group(1)
                return Decision(kind="tool_call", tool="save_note", args=args)
            if r["status"] == "ok":
                return Decision(kind="final", text="שמרתי את ההעדפה לשיחות הבאות.")
            return Decision(kind="final", text=f"לא שמרתי את ההערה ({r['error_type']}). לא בוצע שום שריון.")

        # 3) בקשת שריון ישיר בלי אישור (המודל המדומה מנסה – הקוד אמור לחסום)
        if re.search(r"שריין מיד|book now", cur, re.I):
            r = results.get("create_reservation")
            if r is None:
                room = "HALL" if "הכנסים" in cur else "R204"
                hrs = re.search(r"(\d+)\s*שעות", cur)
                start = 9
                end = start + (int(hrs.group(1)) if hrs else 2)
                f = self.extract(ctx)
                return Decision(kind="tool_call", tool="create_reservation", args={
                    "room_id": room, "date": f.get("date", (self.today + dt.timedelta(days=1)).isoformat()),
                    "start_hour": start, "end_hour": end, "equipment": {}})
            return Decision(kind="final", text=(
                f"לא בוצע שריון ({r.get('error_type')}): {r.get('message')}. "
                "כל שריון דורש הצגת הצעה ואישור מפורש שלך, ועלות של עד 2,000 ש\"ח."))

        # 4) בקשת תיאום רגילה
        f = self.extract(ctx)
        missing = [he for key, he in (("date", "תאריך"), ("start_hour", "שעות"),
                                      ("participants", "מספר משתתפים"), ("budget", "תקציב (בש\"ח)"))
                   if key not in f]
        if missing:
            return Decision(kind="clarify", text=(
                "כדי לבדוק זמינות ולחשב עלות אני צריך עוד כמה פרטים: " + ", ".join(missing) + "."))

        if "recall_notes" not in results:
            return Decision(kind="tool_call", tool="recall_notes", args={"query": cur[:200] if len(cur) >= 2 else "העדפות", "k": 3})
        notes += " " + " ".join(results["recall_notes"].get("data", {}).get("notes", []))
        need_access = "נגיש" in notes
        equipment = dict(f["equipment"])
        added = []
        if "הקלטה" in notes and "recording_kit" not in equipment:
            equipment["recording_kit"] = 1
            added.append("ערכת הקלטה")

        if "search_rooms" not in results:
            return Decision(kind="tool_call", tool="search_rooms", args={
                "date": f["date"], "start_hour": f["start_hour"], "end_hour": f["end_hour"],
                "min_capacity": max(5, min(f["participants"], 300))})
        sr = results["search_rooms"]
        if sr["status"] != "ok":
            return Decision(kind="final", text=f"לא הצלחתי לחפש חדרים ({sr['error_type']}): {sr['message']}")
        rooms = [r for r in sr["data"]["rooms"] if r["accessible"] or not need_access]
        if not rooms:
            return Decision(kind="final", text="אין חדר פנוי שמתאים לקיבולת ולדרישות בשעות האלה. אפשר לנסות שעה או תאריך אחרים.")

        if equipment and "check_equipment" not in results:
            return Decision(kind="tool_call", tool="check_equipment", args={
                "date": f["date"], "start_hour": f["start_hour"], "end_hour": f["end_hour"], "items": equipment})
        if equipment:
            ce = results["check_equipment"]
            if ce["status"] != "ok":
                return Decision(kind="final", text=f"בדיקת הציוד נכשלה ({ce['error_type']}): {ce['message']}")
            short = {i: l for i, l in ce["data"]["items"].items() if not l["sufficient"]}
            if short:
                desc = ", ".join(f"{ITEM_HE[i]} (ביקשת {l['requested']}, זמינים {l['available']})" for i, l in short.items())
                return Decision(kind="clarify", text=f"חסר ציוד בשעות האלה: {desc}. לוותר על הפריט, להקטין כמות או לבחור שעה אחרת?")

        room = rooms[0]  # הזול ביותר שעומד באילוצים (search_rooms ממיין לפי מחיר)
        if "calculate_event_cost" not in results:
            return Decision(kind="tool_call", tool="calculate_event_cost", args={
                "room_id": room["room_id"], "hours": f["end_hour"] - f["start_hour"],
                "equipment": equipment, "budget": f["budget"]})
        cc = results["calculate_event_cost"]
        if cc["status"] != "ok":
            return Decision(kind="final", text=f"חישוב העלות נכשל ({cc['error_type']}): {cc['message']}")
        c = cc["data"]
        breakdown = f"חדר {c['room_cost']:.0f}" + "".join(
            f" + {ITEM_HE[i]} {v:.0f}" for i, v in c["equipment_cost"].items())
        if not (c["in_budget"] and c["under_cap_2000"]):
            gap = c["total_cost"] - min(c["budget"], COST_CAP)
            savings = [f"בלי {ITEM_HE[i]}: חיסכון {v:.0f} ש\"ח" for i, v in
                       sorted(c["equipment_cost"].items(), key=lambda x: -x[1])]
            why = "חורגת מתקרת המדיניות (2,000 ש\"ח)" if not c["under_cap_2000"] else f"חורגת מהתקציב ({c['budget']:.0f} ש\"ח)"
            return Decision(kind="clarify", text=(
                f"האפשרות הזולה ביותר ({room['name']}) עולה {c['total_cost']:.0f} ש\"ח ({breakdown}) ו{why} ב-{gap:.0f} ש\"ח. "
                + ("אפשרויות חיסכון: " + "; ".join(savings) + ". " if savings else "")
                + "איך להמשיך?"))
        prop = {"room_id": room["room_id"], "room_name": room["name"], "date": f["date"],
                "start_hour": f["start_hour"], "end_hour": f["end_hour"], "equipment": equipment,
                "total_cost": c["total_cost"], "budget": c["budget"]}
        extra = f" (נוספה {', '.join(added)} לפי ההעדפות השמורות)" if added else ""
        return Decision(kind="propose", proposal=prop, text=(
            f"הצעה: {room['name']} (עד {room['capacity']} משתתפים{', נגיש' if room['accessible'] else ''}), "
            f"{f['date']} {f['start_hour']}:00–{f['end_hour']}:00. עלות: {c['total_cost']:.0f} ש\"ח ({breakdown}){extra}; "
            f"נשארים {c['remaining_budget']:.0f} ש\"ח מהתקציב. לאשר את השריון? (כן/לא)"))


# ---------------------------------------------------------------------------
# מודל אמיתי (אופציונלי) – Gemini דרך google-genai. מופעל רק עם מפתח.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are CampusEvent AI, an agent that books campus rooms and AV equipment within budget.
Decide the NEXT single action. Reply with JSON only:
{"kind": "tool_call"|"final"|"clarify"|"propose", "tool": str|null, "args": object, "text": str, "proposal": object|null}
Rules: if date, hours, participants or budget are missing -> kind="clarify" (no tools).
Flow: recall_notes -> search_rooms -> check_equipment (if equipment) -> calculate_event_cost -> propose.
A proposal must contain room_id, room_name, date, start_hour, end_hour, equipment, total_cost, budget.
Call create_reservation only after the user approved a proposal. Never invent tool results. Answer in Hebrew.
Tools (JSON contracts):
"""


class GeminiModel:
    """מתאם למודל אמיתי. דורש: pip install google-genai ומפתח GEMINI_API_KEY (Colab Secrets או env)."""

    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        from google import genai  # ייבוא עצל – לא נדרש בבדיקות
        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self.model = model
        self.name = f"Gemini ({model})"

    def decide(self, ctx: List[dict], tools: List[dict]) -> Decision:
        prompt = SYSTEM_PROMPT + json.dumps(tools, ensure_ascii=False) + "\n\nConversation:\n" + \
            json.dumps(ctx, ensure_ascii=False)
        resp = self.client.models.generate_content(
            model=self.model, contents=prompt, config={"response_mime_type": "application/json"})
        try:
            return Decision.model_validate_json(resp.text)
        except Exception:
            return Decision(kind="final", text="המודל החזיר תשובה לא תקינה; עוצר ללא פעולה.")


def get_model(use_real: bool = False) -> Any:
    """ברירת מחדל: MockModel. מודל אמיתי רק אם התבקש ויש מפתח."""
    if use_real and os.environ.get("GEMINI_API_KEY"):
        return GeminiModel()
    return MockModel()


# ---------------------------------------------------------------------------
# הסוכן
# ---------------------------------------------------------------------------
STOP_MESSAGES = {
    "search_rooms": "שירות יומן החדרים אינו זמין כעת (שגיאה זמנית). לא בוצעה הזמנה, נא לנסות שוב בעוד מספר דקות.",
    "check_equipment": "שירות מלאי הציוד אינו זמין כעת (שגיאה זמנית). לא בוצעה הזמנה, נא לנסות שוב בעוד מספר דקות.",
}


@dataclass
class TurnResult:
    reply: str
    status: str
    steps: int
    trace: List[dict] = field(default_factory=list)


class Agent:
    def __init__(self, model: Any = None, db: Optional[CampusDB] = None, notes: Optional[NoteStore] = None,
                 store: Optional[StateStore] = None, strategy: Literal["full", "reduced"] = "reduced",
                 sleep: Callable[[float], None] = time.sleep):
        self.model = model or MockModel()
        self.db = db or CampusDB()
        self.notes = notes or NoteStore()
        self.store = store or StateStore()
        self.strategy = strategy
        self.sleep = sleep
        self.contracts = list_contracts()

    def run_turn(self, thread_id: str, user_id: str, text: str) -> TurnResult:
        trace: List[dict] = []
        log = lambda **e: trace.append({"t": round(time.time(), 3), "thread_id": thread_id, **e})
        state = self.store.get_or_create(thread_id, user_id)
        was_waiting = state.status == "awaiting_approval"
        state = apply_event(state, UserMessageEvent(text=text))
        log(event="user_message", turn=state.turn, text=text)
        # האישור מזוהה ע"י הקוד – לא ע"י המודל
        if was_waiting:
            granted = bool(YES.match(text))
            state = apply_event(state, ApprovalEvent(granted=granted))
            log(event="approval", granted=granted, user_approved=state.user_approved)

        reply = ""
        while True:
            if state.step >= state.max_steps:
                reply = f"הגעתי למגבלת הצעדים ({state.max_steps}) בלי להשלים את הבקשה. לא בוצעה הזמנה."
                state = apply_event(state, AssistantEvent(text=reply, kind="stopped"))
                log(event="stop", reason="max_steps", step=state.step)
                break
            ctx = build_context(state, self.notes.all_for(state.user_id), self.strategy)
            decision = self.model.decide(ctx, self.contracts)
            log(event="model_decision", step=state.step + 1, context=context_size(ctx),
                strategy=self.strategy, decision=decision.model_dump(exclude_none=True))

            if decision.kind != "tool_call":
                reply = decision.text
                state = apply_event(state, AssistantEvent(text=decision.text, kind=decision.kind,
                                                          proposal=decision.proposal))
                log(event="stop", reason=decision.kind, step=state.step)
                break

            state = apply_event(state, ToolCallEvent(name=decision.tool or "", args=decision.args))
            tctx = ToolContext(user_id=state.user_id, db=self.db, notes=self.notes,
                               user_approved=state.user_approved, sleep=self.sleep)
            t0 = time.time()
            result = run_tool(decision.tool or "", decision.args, tctx)
            for r in tctx.retry_log:
                log(event="retry", **r)
            log(event="tool_result", tool=decision.tool, args=decision.args,
                elapsed_s=round(time.time() - t0, 3), result=result.model_dump(exclude_none=True))
            state = apply_event(state, ToolResultEvent(name=decision.tool or "", result=result.model_dump()))

            if result.status == "error" and result.error_type in ("transient", "timeout", "permanent"):
                reply = STOP_MESSAGES.get(decision.tool or "", "שירות חיצוני אינו זמין כעת. לא בוצעה הזמנה, נא לנסות שוב מאוחר יותר.")
                if result.error_type == "permanent":
                    reply = f"תקלה קבועה בשירות ({result.message}). לא בוצעה הזמנה; העברתי לבדיקה."
                state = apply_event(state, AssistantEvent(text=reply, kind="stopped"))
                log(event="stop", reason=f"tool_{result.error_type}", attempts=result.attempts, step=state.step)
                break

        self.store.put(state)
        return TurnResult(reply=reply, status=state.status, steps=state.step, trace=trace)


def save_trace(trace: List[dict], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for e in trace:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return p
