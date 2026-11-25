"""CampusEvent AI – סכמת המצב, זיכרון מתמשך וניהול ההקשר (אבן דרך 1).

* AgentState  – מצב ריצה לכל שיחה (thread_id). מתעדכן רק דרך apply_event (פונקציה טהורה).
* StateStore  – מחזיק מצב נפרד לכל thread_id; שיחה שייכת למשתמש אחד בלבד.
* NoteStore   – זיכרון מתמשך בין שיחות, מבודד לפי user_id שמוזרק מהסשן.
* build_context – בונה את ההקשר למודל: "full" (כל ההיסטוריה) או "reduced"
                  (notes + הבקשה הנוכחית + עבודת התור הנוכחי + last_n הודעות קודמות).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

MAX_STEPS = 8
LAST_N = 4

Role = Literal["user", "assistant", "tool", "system"]
Status = Literal["idle", "running", "awaiting_approval", "clarify", "done", "stopped"]


class Message(BaseModel):
    role: Role
    content: str
    turn: int                                  # מספר תור המשתמש שאליו ההודעה שייכת
    tool_call: Optional[Dict[str, Any]] = None  # {"name":..., "args":...} בהודעות assistant
    data: Optional[Dict[str, Any]] = None       # תוצאה מובנית (tool) או הצעה (assistant)


class AgentState(BaseModel):
    thread_id: str
    user_id: str
    turn: int = 0
    step: int = 0                               # צעדים בתור הנוכחי
    max_steps: int = MAX_STEPS
    messages: List[Message] = Field(default_factory=list)
    user_approved: bool = False                 # נקבע רק ע"י הקוד מתגובת המשתמש
    pending_proposal: Optional[Dict[str, Any]] = None
    status: Status = "idle"


# ---------------------------------------------------------------------------
# אירועים – הדרך היחידה לשנות מצב
# ---------------------------------------------------------------------------
class UserMessageEvent(BaseModel):
    type: Literal["user_message"] = "user_message"
    text: str


class ToolCallEvent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    name: str
    args: Dict[str, Any]


class ToolResultEvent(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    name: str
    result: Dict[str, Any]


class AssistantEvent(BaseModel):
    type: Literal["assistant"] = "assistant"
    text: str
    kind: Literal["final", "clarify", "propose", "stopped"] = "final"
    proposal: Optional[Dict[str, Any]] = None


class ApprovalEvent(BaseModel):
    type: Literal["approval"] = "approval"
    granted: bool


Event = Union[UserMessageEvent, ToolCallEvent, ToolResultEvent, AssistantEvent, ApprovalEvent]


def apply_event(state: AgentState, event: Event) -> AgentState:
    """פונקציה טהורה: מחזירה מצב חדש ואינה משנה את הקיים."""
    s = state.model_copy(deep=True)
    if isinstance(event, UserMessageEvent):
        s.turn += 1
        s.step = 0
        s.status = "running"
        s.messages.append(Message(role="user", content=event.text, turn=s.turn))
    elif isinstance(event, ApprovalEvent):
        s.user_approved = event.granted and s.pending_proposal is not None
        if not event.granted:
            s.pending_proposal = None
    elif isinstance(event, ToolCallEvent):
        s.step += 1
        s.messages.append(Message(role="assistant", turn=s.turn,
                                  content=f"[tool_call] {event.name}",
                                  tool_call={"name": event.name, "args": event.args}))
    elif isinstance(event, ToolResultEvent):
        s.messages.append(Message(role="tool", turn=s.turn,
                                  content=json.dumps(event.result, ensure_ascii=False),
                                  data=event.result))
        if event.name == "create_reservation":
            # אישור חד-פעמי: נצרך בניסיון השריון, הצליח או נכשל
            s.user_approved = False
            if event.result.get("status") == "ok":
                s.pending_proposal = None
    elif isinstance(event, AssistantEvent):
        if event.kind != "stopped":      # עצירה שנוצרה ע"י הקוד אינה צעד של המודל
            s.step += 1
        s.messages.append(Message(role="assistant", content=event.text, turn=s.turn,
                                  data={"kind": event.kind, "proposal": event.proposal}))
        if event.kind == "propose":
            s.pending_proposal = event.proposal
            s.user_approved = False
            s.status = "awaiting_approval"
        elif event.kind == "clarify":
            s.status = "clarify"
        elif event.kind == "stopped":
            s.status = "stopped"
        else:
            s.status = "done"
    else:  # pragma: no cover
        raise TypeError(f"unknown event {event!r}")
    return s


# ---------------------------------------------------------------------------
# מאגר מצבים – שיחה נפרדת לכל thread_id
# ---------------------------------------------------------------------------
class StateStore:
    def __init__(self) -> None:
        self._states: Dict[str, AgentState] = {}

    def get_or_create(self, thread_id: str, user_id: str) -> AgentState:
        st = self._states.get(thread_id)
        if st is None:
            st = AgentState(thread_id=thread_id, user_id=user_id)
            self._states[thread_id] = st
        elif st.user_id != user_id:
            raise PermissionError("thread belongs to a different user")
        return st.model_copy(deep=True)

    def put(self, state: AgentState) -> None:
        self._states[state.thread_id] = state.model_copy(deep=True)

    def threads(self) -> List[str]:
        return list(self._states)


# ---------------------------------------------------------------------------
# זיכרון מתמשך בין שיחות – מבודד לפי user_id
# ---------------------------------------------------------------------------
_SENSITIVE = [re.compile(r"\b\d{9}\b"),                  # ת"ז
              re.compile(r"\b(?:\d[ -]?){13,19}\b"),     # כרטיס אשראי
              re.compile(r"(?:password|סיסמ)", re.I),
              re.compile(r"\b05\d[- ]?\d{7}\b")]         # טלפון נייד


class NoteStore:
    def __init__(self) -> None:
        self._notes: Dict[str, List[str]] = {}

    def save(self, user_id: str, text: str) -> int:
        if any(p.search(text) for p in _SENSITIVE):
            raise PermissionError("sensitive personal data is never stored")
        self._notes.setdefault(user_id, []).append(text.strip())
        return len(self._notes[user_id])

    def recall(self, user_id: str, query: str, k: int = 3) -> List[str]:
        notes = self._notes.get(user_id, [])
        q = set(_tokens(query))
        scored = [(len(q & set(_tokens(n))), i, n) for i, n in enumerate(notes)]
        scored.sort(key=lambda x: (-x[0], -x[1]))
        hits = [n for score, _, n in scored if score > 0][:k]
        return hits or notes[-k:]          # אין התאמה => ההעדפות האחרונות

    def all_for(self, user_id: str) -> List[str]:
        return list(self._notes.get(user_id, []))


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[\w֐-׿]+", text.lower()) if len(t) > 1]


# ---------------------------------------------------------------------------
# בניית ההקשר
# ---------------------------------------------------------------------------
def build_context(state: AgentState, notes: List[str], strategy: Literal["full", "reduced"] = "reduced",
                  last_n: int = LAST_N) -> List[Dict[str, Any]]:
    """full    – כל ההודעות בשיחה.
    reduced – הודעת מערכת עם ההעדפות השמורות (notes)
              + last_n הודעות אחרונות מהתורות הקודמים
              + כל ההודעות של התור הנוכחי (הבקשה הנוכחית ועבודת הכלים שלה – תמיד נשמרות;
                התור מוגבל ממילא ע"י MAX_STEPS).
    """
    msgs = state.messages
    if strategy == "full":
        chosen = msgs
    else:
        current = [m for m in msgs if m.turn == state.turn]
        earlier = [m for m in msgs if m.turn < state.turn]
        chosen = (earlier[-last_n:] if last_n > 0 else []) + current
    ctx: List[Dict[str, Any]] = []
    if notes:
        ctx.append({"role": "system", "content": "העדפות שמורות של המשתמש: " + " | ".join(notes)})
    ctx.extend(m.model_dump(exclude_none=True) for m in chosen)
    return ctx


def context_size(ctx: List[Dict[str, Any]]) -> Dict[str, int]:
    return {"messages": len(ctx), "chars": sum(len(m.get("content", "")) for m in ctx)}
