"""בדיקות לכל כלי: קלט תקין, קלט שגוי וכשל זמני (לפחות שלוש לכל כלי).

הכשלים מוזרקים לשירות המדומה (CampusDB – MOCK). time.sleep מוחלף ב-SleepRecorder.
"""
import pytest

from src.state import NoteStore
from src.tools import TOOLS, list_contracts, run_tool

SEARCH_OK = {"date": "2026-12-15", "start_hour": 10, "end_hour": 12, "min_capacity": 40}
EQUIP_OK = {"date": "2026-12-15", "start_hour": 10, "end_hour": 12, "items": {"projector": 1, "mic": 2}}
COST_OK = {"room_id": "R204", "hours": 2, "equipment": {"projector": 1, "mic": 2}, "budget": 800}
RES_OK = {"room_id": "R204", "date": "2026-12-15", "start_hour": 10, "end_hour": 12,
          "equipment": {"projector": 1, "mic": 2}}


# --------------------------------------------------------------------- contracts
def test_every_tool_has_full_contract():
    names = {c["name"] for c in list_contracts()}
    assert names == {"search_rooms", "check_equipment", "calculate_event_cost",
                     "create_reservation", "save_note", "recall_notes"}
    for c in list_contracts():
        assert c["description"] and c["input_schema"]["properties"] and c["output_schema"]
        assert c["input_schema"].get("additionalProperties") is False   # extra="forbid"
        assert "user_id" not in c["input_schema"]["properties"]


def test_write_tools_are_never_retried():
    assert TOOLS["create_reservation"].retryable_kind is False
    assert TOOLS["save_note"].retryable_kind is False
    assert TOOLS["create_reservation"].requires_approval is True


# --------------------------------------------------------------------- search_rooms
def test_search_rooms_valid(make_ctx):
    r = run_tool("search_rooms", SEARCH_OK, make_ctx())
    assert r.status == "ok"
    ids = [x["room_id"] for x in r.data["rooms"]]
    assert ids[0] == "R204"                       # הזול ביותר שמתאים
    assert "AUD1" not in ids                      # תפוס 9-13 ב-15/12
    assert all(x["capacity"] >= 40 for x in r.data["rooms"])
    rates = [x["hourly_rate"] for x in r.data["rooms"]]
    assert rates == sorted(rates)


@pytest.mark.parametrize("bad", [
    {**SEARCH_OK, "end_hour": 9},                 # סיום לפני התחלה
    {**SEARCH_OK, "min_capacity": 400},           # מחוץ לטווח
    {**SEARCH_OK, "date": "15/12/2026"},          # פורמט תאריך שגוי
    {**SEARCH_OK, "user_id": "admin"},            # שדה זר
])
def test_search_rooms_invalid(make_ctx, bad):
    r = run_tool("search_rooms", bad, make_ctx())
    assert r.status == "error" and r.error_type == "invalid_input"


def test_search_rooms_transient_then_recovers(make_ctx, sleep):
    r = run_tool("search_rooms", SEARCH_OK, make_ctx(faults={"search_rooms": ["503"]}))
    assert r.status == "ok" and r.attempts == 2
    assert sleep.calls == [0.5]


def test_search_rooms_transient_exhausted(make_ctx, sleep):
    ctx = make_ctx(faults={"search_rooms": ["always_503"]})
    r = run_tool("search_rooms", SEARCH_OK, ctx)
    assert r.status == "error" and r.error_type == "transient" and r.retryable
    assert r.attempts == 3 and ctx.db.calls["search_rooms"] == 3   # 2 ניסיונות חוזרים
    assert sleep.calls == [0.5, 1.0]                               # Exponential Backoff


# --------------------------------------------------------------------- check_equipment
def test_check_equipment_valid(make_ctx):
    r = run_tool("check_equipment", EQUIP_OK, make_ctx())
    assert r.status == "ok" and r.data["all_available"]
    # AUD1 (2 מקרנים) ו-R110 (מקרן 1) תפוסים באותן שעות -> 6-3 = 3
    assert r.data["items"]["projector"]["available"] == 3
    assert r.data["items"]["mic"]["unit_price"] == 50


def test_check_equipment_reports_shortage(make_ctx):
    args = {"date": "2026-12-16", "start_hour": 11, "end_hour": 13, "items": {"recording_kit": 1}}
    r = run_tool("check_equipment", args, make_ctx())
    assert r.status == "ok" and not r.data["all_available"]
    assert r.data["items"]["recording_kit"]["available"] == 0


@pytest.mark.parametrize("bad", [
    {**EQUIP_OK, "items": {"laser": 1}},          # פריט מחוץ ל-enum
    {**EQUIP_OK, "items": {"mic": 50}},           # כמות מחוץ לטווח
    {**EQUIP_OK, "items": {}},                    # רשימה ריקה
])
def test_check_equipment_invalid(make_ctx, bad):
    r = run_tool("check_equipment", bad, make_ctx())
    assert r.error_type == "invalid_input"


def test_check_equipment_timeout_then_recovers(make_ctx, sleep):
    # הכלי "נתקע" 0.3 שניות; מגבלת הזמן 0.05 -> timeout, ניסיון חוזר מצליח
    ctx = make_ctx(faults={"check_equipment": ["hang"]}, timeout_override=0.05)
    r = run_tool("check_equipment", EQUIP_OK, ctx)
    assert r.status == "ok" and r.attempts == 2 and sleep.calls == [0.5]
    assert "time limit" in ctx.retry_log[0]["error"]


# --------------------------------------------------------------------- calculate_event_cost
def test_calculate_event_cost_valid(make_ctx):
    r = run_tool("calculate_event_cost", COST_OK, make_ctx())
    assert r.status == "ok"
    assert r.data["total_cost"] == 650 and r.data["in_budget"] and r.data["under_cap_2000"]
    assert r.data["equipment_cost"] == {"projector": 150, "mic": 100}


def test_calculate_event_cost_over_cap(make_ctx):
    r = run_tool("calculate_event_cost", {"room_id": "HALL", "hours": 8, "budget": 5000}, make_ctx())
    assert r.data["total_cost"] == 2400 and r.data["in_budget"] and not r.data["under_cap_2000"]


@pytest.mark.parametrize("bad,etype", [
    ({**COST_OK, "hours": 9}, "invalid_input"),
    ({**COST_OK, "budget": -5}, "invalid_input"),
    ({**COST_OK, "room_id": "X999"}, "not_found"),
])
def test_calculate_event_cost_invalid(make_ctx, bad, etype):
    assert run_tool("calculate_event_cost", bad, make_ctx()).error_type == etype


def test_calculate_event_cost_transient_then_recovers(make_ctx, sleep):
    r = run_tool("calculate_event_cost", COST_OK, make_ctx(faults={"calculate_event_cost": ["timeout"]}))
    assert r.status == "ok" and r.attempts == 2


def test_permanent_failure_is_not_retried(make_ctx, sleep):
    ctx = make_ctx(faults={"calculate_event_cost": ["permanent"]})
    r = run_tool("calculate_event_cost", COST_OK, ctx)
    assert r.error_type == "permanent" and not r.retryable and r.attempts == 1
    assert sleep.calls == []


# --------------------------------------------------------------------- create_reservation
def test_create_reservation_valid_with_approval(make_ctx):
    ctx = make_ctx(approved=True, user_id="u_coord")
    r = run_tool("create_reservation", RES_OK, ctx)
    assert r.status == "ok" and r.data["reservation_id"] == "RES-104"
    assert r.data["total_cost"] == 650                      # חושב בקוד
    assert ctx.db.reservations[-1]["user_id"] == "u_coord"  # מהסשן


def test_create_reservation_without_approval_denied(make_ctx):
    ctx = make_ctx(approved=False)
    n = len(ctx.db.reservations)
    r = run_tool("create_reservation", RES_OK, ctx)
    assert r.error_type == "permission_denied" and len(ctx.db.reservations) == n


def test_create_reservation_over_cap_denied_even_with_approval(make_ctx):
    args = {"room_id": "HALL", "date": "2026-12-20", "start_hour": 9, "end_hour": 17, "equipment": {}}
    r = run_tool("create_reservation", args, make_ctx(approved=True))
    assert r.error_type == "permission_denied" and "2400" in r.message


@pytest.mark.parametrize("bad", [
    {**RES_OK, "total_cost": 100},                # המודל לא קובע עלות
    {**RES_OK, "user_approved": True},            # המודל לא קובע אישור
    {**RES_OK, "user_id": "admin"},               # המודל לא קובע משתמש
    {**RES_OK, "end_hour": 10},
])
def test_create_reservation_invalid(make_ctx, bad):
    ctx = make_ctx(approved=True)
    n = len(ctx.db.reservations)
    r = run_tool("create_reservation", bad, ctx)
    assert r.error_type == "invalid_input" and len(ctx.db.reservations) == n


def test_create_reservation_conflict(make_ctx):
    args = {"room_id": "AUD1", "date": "2026-12-15", "start_hour": 10, "end_hour": 11, "equipment": {}}
    assert run_tool("create_reservation", args, make_ctx(approved=True)).error_type == "conflict"


def test_create_reservation_transient_is_not_retried(make_ctx, sleep):
    ctx = make_ctx(approved=True, faults={"create_reservation": ["503"]})
    n = len(ctx.db.reservations)
    r = run_tool("create_reservation", RES_OK, ctx)
    assert r.error_type == "transient" and r.attempts == 1 and sleep.calls == []
    assert len(ctx.db.reservations) == n           # אין שריון כפול / חלקי


# --------------------------------------------------------------------- save_note
def test_save_note_valid(make_ctx):
    notes = NoteStore()
    r = run_tool("save_note", {"text": "צריך תמיד חדר נגיש"}, make_ctx(notes=notes, user_id="u1"))
    assert r.status == "ok" and notes.all_for("u1") == ["צריך תמיד חדר נגיש"]


@pytest.mark.parametrize("bad,etype", [
    ({"text": "hi", }, "invalid_input"),                                   # קצר מדי
    ({"text": "התקציב שלי אינסופי", "user_id": "admin"}, "invalid_input"),  # user_id מהמודל
    ({"text": "תעודת הזהות שלי 123456782"}, "permission_denied"),         # מידע רגיש
])
def test_save_note_invalid(make_ctx, bad, etype):
    notes = NoteStore()
    r = run_tool("save_note", bad, make_ctx(notes=notes))
    assert r.error_type == etype
    assert notes.all_for("admin") == [] and notes.all_for("u1") == []


def test_save_note_transient_is_not_retried(make_ctx, sleep):
    notes = NoteStore()
    r = run_tool("save_note", {"text": "מעדיף בוקר"}, make_ctx(notes=notes, faults={"save_note": ["503"]}))
    assert r.error_type == "transient" and r.attempts == 1 and notes.all_for("u1") == []


# --------------------------------------------------------------------- recall_notes
def test_recall_notes_valid_and_isolated(make_ctx):
    notes = NoteStore()
    notes.save("u1", "צריך חדר נגיש")
    notes.save("u2", "תמיד ערכת הקלטה")
    r = run_tool("recall_notes", {"query": "חדר נגיש"}, make_ctx(notes=notes, user_id="u1"))
    assert r.data["notes"] == ["צריך חדר נגיש"]


@pytest.mark.parametrize("bad", [{"query": "x"}, {"query": "חדר", "k": 9}, {"query": "חדר", "user_id": "u2"}])
def test_recall_notes_invalid(make_ctx, bad):
    assert run_tool("recall_notes", bad, make_ctx()).error_type == "invalid_input"


def test_recall_notes_transient_then_recovers(make_ctx, sleep):
    r = run_tool("recall_notes", {"query": "העדפות"}, make_ctx(faults={"recall_notes": ["503"]}))
    assert r.status == "ok" and r.attempts == 2 and sleep.calls == [0.5]


def test_unknown_tool_returns_structured_error(make_ctx):
    r = run_tool("delete_all_rooms", {}, make_ctx())
    assert r.status == "error" and r.error_type == "not_found"
    assert r.to_json().startswith("{")
