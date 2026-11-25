"""CampusEvent AI – הכלים והחוזים שלהם (אבן דרך 1).

כל כלי מוגדר בחוזה מפורש: שם, תיאור, סכמת קלט (Pydantic, extra="forbid") וסכמת פלט.
כל הפעלה עוברת דרך run_tool, שאחראי על:
  1. אימות קלט (Pydantic) לפני ביצוע.
  2. בדיקת הרשאות בקוד (user_id מהסשן, אישור משתמש מה-State, תקרת 2,000 ש"ח).
  3. מגבלת זמן לכל קריאה.
  4. הבחנה בין שגיאה זמנית (transient) לכשל קבוע (permanent).
  5. ניסיונות חוזרים עם Exponential Backoff – רק לכלי קריאה/חישוב, לעולם לא לכלי כתיבה.
  6. החזרת תוצאה מובנית (JSON) בכל מקרה – גם בכשל.

שירות היומן כאן הוא **MOCK** (CampusDB): הוא קורא נתונים סינתטיים מקובצי JSON מקומיים
ומאפשר להזריק כשלים (503, timeout, כשל קבוע) לצורך בדיקות. אין שום שירות חיצוני.
"""
from __future__ import annotations

import concurrent.futures
import copy
import datetime as dt
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

DATA_DIR = Path(__file__).parent / "data"
COST_CAP = 2000.0
MAX_RETRIES = 2                 # עד 2 ניסיונות חוזרים => לכל היותר 3 קריאות
BACKOFF_DELAYS = (0.5, 1.0)     # Exponential Backoff בשניות
EquipmentItem = Literal["projector", "mic", "recording_kit"]


# ---------------------------------------------------------------------------
# שגיאות שהשירות המדומה יכול לזרוק
# ---------------------------------------------------------------------------
class TransientToolError(Exception):
    """שגיאה זמנית (למשל 503) – מותר לנסות שוב בכלי קריאה."""


class ToolTimeoutError(TransientToolError):
    """חריגה ממגבלת הזמן – נחשבת זמנית."""


class PermanentToolError(Exception):
    """כשל קבוע (למשל נתונים פגומים / 500 קבוע) – אין טעם לנסות שוב."""


# ---------------------------------------------------------------------------
# MOCK: שירות יומן הקמפוס (נתונים סינתטיים + הזרקת כשלים)
# ---------------------------------------------------------------------------
class CampusDB:
    """MOCK – יומן קמפוס מדומה בזיכרון. כל מופע מקבל עותק נפרד של הנתונים.

    faults: מילון  tool_name -> רשימת קודי כשל שנצרכים לפי הסדר, למשל
            {"search_rooms": ["503", "503"]}  => שתי הקריאות הראשונות נכשלות.
            הקוד "always_503" גורם לכשל זמני בכל קריאה.
            קודים: "503", "timeout", "hang", "permanent", "always_503".
    """

    def __init__(self, data_dir: Path = DATA_DIR, faults: Optional[Dict[str, List[str]]] = None,
                 hang_seconds: float = 5.0):
        self.hang_seconds = hang_seconds
        self.rooms = {r["room_id"]: r for r in _load(data_dir / "rooms.json")["rooms"]}
        self.equipment = {e["item"]: e for e in _load(data_dir / "equipment.json")["equipment"]}
        self.reservations: List[dict] = _load(data_dir / "reservations.json")["reservations"]
        self.faults: Dict[str, List[str]] = copy.deepcopy(faults or {})
        self.calls: Dict[str, int] = {}
        self._next_res = 104

    # -- הזרקת כשלים --------------------------------------------------------
    def maybe_fail(self, tool: str) -> None:
        self.calls[tool] = self.calls.get(tool, 0) + 1
        queue = self.faults.get(tool)
        if not queue:
            return
        code = queue[0] if queue[0] == "always_503" else queue.pop(0)
        if code in ("503", "always_503"):
            raise TransientToolError("503 Service Unavailable")
        if code == "timeout":
            raise ToolTimeoutError("upstream timeout")
        if code == "hang":
            time.sleep(self.hang_seconds)          # נתפס ע"י מגבלת הזמן של run_tool
            return
        if code == "permanent":
            raise PermanentToolError("500 calendar data corrupted")

    # -- שאילתות -----------------------------------------------------------
    def overlapping(self, date: str, start: int, end: int) -> List[dict]:
        return [r for r in self.reservations
                if r["date"] == date and r["start_hour"] < end and start < r["end_hour"]]

    def equipment_free(self, date: str, start: int, end: int) -> Dict[str, int]:
        used: Dict[str, int] = {k: 0 for k in self.equipment}
        for r in self.overlapping(date, start, end):
            for item, qty in r.get("equipment", {}).items():
                used[item] = used.get(item, 0) + qty
        return {k: self.equipment[k]["stock"] - used[k] for k in self.equipment}

    def compute_cost(self, room_id: str, hours: int, equipment: Dict[str, int]) -> Dict[str, Any]:
        room = self.rooms[room_id]
        room_cost = room["hourly_rate"] * hours
        lines = {item: self.equipment[item]["unit_price"] * qty for item, qty in equipment.items()}
        return {"room_cost": room_cost, "equipment_cost": lines,
                "total_cost": float(room_cost + sum(lines.values()))}

    def new_reservation_id(self) -> str:
        rid = f"RES-{self._next_res}"
        self._next_res += 1
        return rid


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# סכמות קלט (חוזה) – extra="forbid": שדה זר (למשל user_id) נדחה
# ---------------------------------------------------------------------------
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_date(v: str) -> str:
    try:
        dt.date.fromisoformat(v)
    except ValueError as e:
        raise ValueError("date must be YYYY-MM-DD") from e
    return v


class SearchRoomsInput(StrictModel):
    date: str = Field(description="תאריך בפורמט YYYY-MM-DD")
    start_hour: int = Field(ge=8, le=20)
    end_hour: int = Field(ge=9, le=21)
    min_capacity: int = Field(ge=5, le=300)
    v_date = field_validator("date")(_check_date)

    @model_validator(mode="after")
    def _hours(self):
        if self.end_hour <= self.start_hour:
            raise ValueError("end_hour must be after start_hour")
        return self


class CheckEquipmentInput(StrictModel):
    date: str
    start_hour: int = Field(ge=8, le=20)
    end_hour: int = Field(ge=9, le=21)
    items: Dict[EquipmentItem, int] = Field(min_length=1, description="פריט -> כמות (1-10)")
    v_date = field_validator("date")(_check_date)

    @field_validator("items")
    @classmethod
    def _qty(cls, v):
        for item, qty in v.items():
            if not 1 <= qty <= 10:
                raise ValueError(f"quantity for {item} must be 1-10")
        return v

    @model_validator(mode="after")
    def _hours(self):
        if self.end_hour <= self.start_hour:
            raise ValueError("end_hour must be after start_hour")
        return self


class CalculateEventCostInput(StrictModel):
    room_id: str = Field(min_length=2, max_length=10)
    hours: int = Field(ge=1, le=8)
    equipment: Dict[EquipmentItem, int] = Field(default_factory=dict)
    budget: float = Field(gt=0, le=100_000)

    @field_validator("equipment")
    @classmethod
    def _qty(cls, v):
        for item, qty in v.items():
            if not 0 <= qty <= 10:
                raise ValueError(f"quantity for {item} must be 0-10")
        return {k: q for k, q in v.items() if q > 0}


class CreateReservationInput(StrictModel):
    """אין total_cost ואין user_approved: העלות מחושבת מחדש בקוד והאישור נקרא מה-State."""
    room_id: str = Field(min_length=2, max_length=10)
    date: str
    start_hour: int = Field(ge=8, le=20)
    end_hour: int = Field(ge=9, le=21)
    equipment: Dict[EquipmentItem, int] = Field(default_factory=dict)
    v_date = field_validator("date")(_check_date)

    @model_validator(mode="after")
    def _hours(self):
        if self.end_hour <= self.start_hour:
            raise ValueError("end_hour must be after start_hour")
        if self.end_hour - self.start_hour > 8:
            raise ValueError("reservation longer than 8 hours")
        return self


class SaveNoteInput(StrictModel):
    text: str = Field(min_length=3, max_length=300)


class RecallNotesInput(StrictModel):
    query: str = Field(min_length=2, max_length=200)
    k: int = Field(default=3, ge=1, le=5)


# ---------------------------------------------------------------------------
# סכמות פלט
# ---------------------------------------------------------------------------
ErrorType = Literal["invalid_input", "permission_denied", "not_found", "conflict",
                    "transient", "timeout", "permanent"]


class ToolResult(BaseModel):
    """המעטפת האחידה שכל כלי מחזיר (תמיד JSON)."""
    tool: str
    status: Literal["ok", "error"]
    data: Optional[Dict[str, Any]] = None
    error_type: Optional[ErrorType] = None
    retryable: bool = False
    message: str = ""
    attempts: int = 1

    def to_json(self) -> str:
        return self.model_dump_json()


class RoomOut(BaseModel):
    room_id: str
    name: str
    capacity: int
    hourly_rate: float
    accessible: bool


class SearchRoomsOutput(BaseModel):
    rooms: List[RoomOut]


class EquipmentLine(BaseModel):
    requested: int
    available: int
    unit_price: float
    sufficient: bool


class CheckEquipmentOutput(BaseModel):
    items: Dict[str, EquipmentLine]
    all_available: bool


class CalculateEventCostOutput(BaseModel):
    room_id: str
    room_cost: float
    equipment_cost: Dict[str, float]
    total_cost: float
    budget: float
    in_budget: bool
    under_cap_2000: bool
    remaining_budget: float


class CreateReservationOutput(BaseModel):
    reservation_id: str
    room_id: str
    date: str
    start_hour: int
    end_hour: int
    equipment: Dict[str, int]
    total_cost: float


class SaveNoteOutput(BaseModel):
    saved: bool
    note_count: int


class RecallNotesOutput(BaseModel):
    notes: List[str]


# ---------------------------------------------------------------------------
# הקשר ההרצה – מגיע מהקוד ומה-State, לעולם לא מהמודל
# ---------------------------------------------------------------------------
@dataclass
class ToolContext:
    user_id: str                                  # מוזרק מהסשן
    db: CampusDB
    notes: "Any"                                  # NoteStore (state.py)
    user_approved: bool = False                   # נקרא מה-AgentState
    sleep: Callable[[float], None] = time.sleep   # ניתן להחלפה בבדיקות (בלי המתנה אמיתית)
    retry_log: List[dict] = field(default_factory=list)
    timeout_override: Optional[float] = None      # לבדיקות: מגבלת זמן קצרה יותר


# ---------------------------------------------------------------------------
# מימוש הכלים (כל אחד מקבל קלט שכבר אומת)
# ---------------------------------------------------------------------------
def _search_rooms(inp: SearchRoomsInput, ctx: ToolContext) -> dict:
    ctx.db.maybe_fail("search_rooms")
    busy = {r["room_id"] for r in ctx.db.overlapping(inp.date, inp.start_hour, inp.end_hour)}
    rooms = [r for r in ctx.db.rooms.values()
             if r["capacity"] >= inp.min_capacity and r["room_id"] not in busy]
    rooms.sort(key=lambda r: (r["hourly_rate"], r["capacity"]))
    return SearchRoomsOutput(rooms=[RoomOut(**r) for r in rooms]).model_dump()


def _check_equipment(inp: CheckEquipmentInput, ctx: ToolContext) -> dict:
    ctx.db.maybe_fail("check_equipment")
    free = ctx.db.equipment_free(inp.date, inp.start_hour, inp.end_hour)
    lines = {}
    for item, qty in inp.items.items():
        lines[item] = EquipmentLine(requested=qty, available=max(free[item], 0),
                                    unit_price=ctx.db.equipment[item]["unit_price"],
                                    sufficient=free[item] >= qty)
    return CheckEquipmentOutput(items=lines,
                                all_available=all(l.sufficient for l in lines.values())).model_dump()


def _calculate_event_cost(inp: CalculateEventCostInput, ctx: ToolContext) -> dict:
    ctx.db.maybe_fail("calculate_event_cost")
    if inp.room_id not in ctx.db.rooms:
        raise LookupError(f"room {inp.room_id} not found")
    c = ctx.db.compute_cost(inp.room_id, inp.hours, inp.equipment)
    total = c["total_cost"]
    return CalculateEventCostOutput(
        room_id=inp.room_id, room_cost=c["room_cost"], equipment_cost=c["equipment_cost"],
        total_cost=total, budget=inp.budget, in_budget=total <= inp.budget,
        under_cap_2000=total <= COST_CAP, remaining_budget=inp.budget - total).model_dump()


def _create_reservation(inp: CreateReservationInput, ctx: ToolContext) -> dict:
    # הרשאה 1: אישור אנושי מפורש – נקרא מה-State בלבד
    if not ctx.user_approved:
        raise PermissionError("explicit user approval is required before reserving")
    if inp.room_id not in ctx.db.rooms:
        raise LookupError(f"room {inp.room_id} not found")
    # הרשאה 2: העלות מחושבת מחדש בקוד – לא נלקחת מהמודל
    cost = ctx.db.compute_cost(inp.room_id, inp.end_hour - inp.start_hour, inp.equipment)
    if cost["total_cost"] > COST_CAP:
        raise PermissionError(f"total cost {cost['total_cost']:.0f} exceeds policy cap {COST_CAP:.0f}")
    ctx.db.maybe_fail("create_reservation")
    # התנגשויות: חדר תפוס או ציוד לא מספיק
    overlap = ctx.db.overlapping(inp.date, inp.start_hour, inp.end_hour)
    if any(r["room_id"] == inp.room_id for r in overlap):
        raise FileExistsError("room already reserved for this time")
    free = ctx.db.equipment_free(inp.date, inp.start_hour, inp.end_hour)
    short = [i for i, q in inp.equipment.items() if free[i] < q]
    if short:
        raise FileExistsError(f"equipment not available: {', '.join(short)}")
    res = {"reservation_id": ctx.db.new_reservation_id(), "user_id": ctx.user_id,
           "room_id": inp.room_id, "date": inp.date, "start_hour": inp.start_hour,
           "end_hour": inp.end_hour, "equipment": dict(inp.equipment),
           "total_cost": cost["total_cost"]}
    ctx.db.reservations.append(res)
    out = {k: v for k, v in res.items() if k != "user_id"}
    return CreateReservationOutput(**out).model_dump()


def _save_note(inp: SaveNoteInput, ctx: ToolContext) -> dict:
    ctx.db.maybe_fail("save_note")       # מאגר ההערות הוא חלק מהשירות המדומה
    count = ctx.notes.save(ctx.user_id, inp.text)
    return SaveNoteOutput(saved=True, note_count=count).model_dump()


def _recall_notes(inp: RecallNotesInput, ctx: ToolContext) -> dict:
    ctx.db.maybe_fail("recall_notes")
    return RecallNotesOutput(notes=ctx.notes.recall(ctx.user_id, inp.query, inp.k)).model_dump()


# ---------------------------------------------------------------------------
# החוזים
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolContract:
    name: str
    description: str
    kind: Literal["read", "compute", "write", "memory"]
    input_model: type
    output_model: type
    fn: Callable[[Any, ToolContext], dict]
    requires_approval: bool = False
    timeout_s: float = 2.0
    retry: bool = True        # ניסיון חוזר רק לפעולות ללא תופעות לוואי (לעולם לא לכתיבה)

    @property
    def retryable_kind(self) -> bool:
        return self.retry

    def describe(self) -> dict:
        return {"name": self.name, "description": self.description, "kind": self.kind,
                "requires_approval": self.requires_approval, "timeout_s": self.timeout_s,
                "retry": f"up to {MAX_RETRIES} retries on transient errors" if self.retryable_kind else "never",
                "input_schema": self.input_model.model_json_schema(),
                "output_schema": self.output_model.model_json_schema()}


TOOLS: Dict[str, ToolContract] = {c.name: c for c in [
    ToolContract("search_rooms", "איתור חדרים פנויים לפי תאריך, שעות וקיבולת מינימלית. ממוין מהזול ליקר.",
                 "read", SearchRoomsInput, SearchRoomsOutput, _search_rooms),
    ToolContract("check_equipment", "בדיקת זמינות ומחיר ליחידה לכל פריט ציוד לתאריך ולטווח השעות.",
                 "read", CheckEquipmentInput, CheckEquipmentOutput, _check_equipment),
    ToolContract("calculate_event_cost", "חישוב דטרמיניסטי של עלות חדר + ציוד מול תקציב ותקרת 2,000 ש\"ח.",
                 "compute", CalculateEventCostInput, CalculateEventCostOutput, _calculate_event_cost),
    ToolContract("create_reservation", "שריון סופי של חדר וציוד. דורש אישור משתמש מפורש; העלות מחושבת מחדש בקוד.",
                 "write", CreateReservationInput, CreateReservationOutput, _create_reservation,
                 requires_approval=True, retry=False),
    ToolContract("save_note", "שמירת העדפה קבועה של המשתמש (user_id מוזרק מהסשן).",
                 "memory", SaveNoteInput, SaveNoteOutput, _save_note, retry=False),
    ToolContract("recall_notes", "שליפת העדפות קבועות של המשתמש לפי שאילתה (user_id מוזרק מהסשן).",
                 "memory", RecallNotesInput, RecallNotesOutput, _recall_notes),
]}


def list_contracts() -> List[dict]:
    return [c.describe() for c in TOOLS.values()]


# ---------------------------------------------------------------------------
# המבצע: אימות -> הרשאות -> מגבלת זמן -> ניסיונות חוזרים -> תוצאה מובנית
# ---------------------------------------------------------------------------
def _call_with_timeout(contract: ToolContract, inp: Any, ctx: ToolContext) -> dict:
    limit = ctx.timeout_override or contract.timeout_s
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        fut = pool.submit(contract.fn, inp, ctx)
        try:
            return fut.result(timeout=limit)
        except concurrent.futures.TimeoutError as e:
            raise ToolTimeoutError(f"tool exceeded {limit}s time limit") from e
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def run_tool(name: str, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    """נקודת הכניסה היחידה להפעלת כלי. תמיד מחזירה ToolResult (JSON)."""
    contract = TOOLS.get(name)
    if contract is None:
        return ToolResult(tool=name, status="error", error_type="not_found",
                          message=f"unknown tool '{name}'")
    # 1) אימות קלט
    try:
        inp = contract.input_model.model_validate(args or {})
    except ValidationError as e:
        errs = "; ".join(f"{'.'.join(map(str, x['loc'])) or 'input'}: {x['msg']}" for x in e.errors())
        return ToolResult(tool=name, status="error", error_type="invalid_input", message=errs)

    max_attempts = 1 + (MAX_RETRIES if contract.retryable_kind else 0)
    attempt = 0
    while True:
        attempt += 1
        try:
            data = _call_with_timeout(contract, inp, ctx)
            return ToolResult(tool=name, status="ok", data=data, attempts=attempt)
        except PermissionError as e:
            return ToolResult(tool=name, status="error", error_type="permission_denied",
                              message=str(e), attempts=attempt)
        except LookupError as e:
            return ToolResult(tool=name, status="error", error_type="not_found",
                              message=str(e), attempts=attempt)
        except FileExistsError as e:
            return ToolResult(tool=name, status="error", error_type="conflict",
                              message=str(e), attempts=attempt)
        except PermanentToolError as e:
            return ToolResult(tool=name, status="error", error_type="permanent",
                              retryable=False, message=str(e), attempts=attempt)
        except TransientToolError as e:
            etype = "timeout" if isinstance(e, ToolTimeoutError) else "transient"
            if attempt < max_attempts:
                delay = BACKOFF_DELAYS[min(attempt - 1, len(BACKOFF_DELAYS) - 1)]
                ctx.retry_log.append({"tool": name, "attempt": attempt, "error": str(e), "backoff_s": delay})
                ctx.sleep(delay)
                continue
            msg = str(e)
            if not contract.retryable_kind:
                msg += " (write tool – not retried to avoid double booking; check reservations before retrying)"
            return ToolResult(tool=name, status="error", error_type=etype, retryable=True,
                              message=msg, attempts=attempt)
