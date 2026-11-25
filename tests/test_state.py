"""בדיקות מצב וזיכרון: סכמה, apply_event טהורה, ושתי שיחות שאינן מתערבבות."""
import datetime as dt

import pytest

from src.agent import Agent, MockModel
from src.state import (AgentState, ApprovalEvent, AssistantEvent, StateStore, ToolResultEvent,
                       UserMessageEvent, apply_event, build_context)

TODAY = dt.date(2026, 12, 1)
REQ_A = 'תאם חדר ל־40 סטודנטים ב־15/12 בין 10:00–12:00 עם מקרן ו־2 מיקרופונים, תקציב עד 800 ש"ח'
REQ_B = 'תאם חדר ל־20 איש ב־17/12 בין 14:00–16:00, תקציב עד 300 ש"ח'


def test_apply_event_is_pure():
    s0 = AgentState(thread_id="t", user_id="u")
    s1 = apply_event(s0, UserMessageEvent(text="שלום"))
    assert s0.messages == [] and s0.turn == 0
    assert len(s1.messages) == 1 and s1.turn == 1 and s1.status == "running"


def test_approval_only_with_pending_proposal_and_is_consumed():
    s = apply_event(AgentState(thread_id="t", user_id="u"), UserMessageEvent(text="x"))
    s = apply_event(s, ApprovalEvent(granted=True))
    assert s.user_approved is False                         # אין הצעה => אין אישור
    s = apply_event(s, AssistantEvent(text="הצעה", kind="propose", proposal={"room_id": "R204"}))
    s = apply_event(s, ApprovalEvent(granted=True))
    assert s.user_approved is True
    s = apply_event(s, ToolResultEvent(name="create_reservation", result={"status": "ok"}))
    assert s.user_approved is False and s.pending_proposal is None   # חד-פעמי


def test_two_threads_do_not_mix():
    agent = Agent(model=MockModel(today=TODAY), sleep=lambda s: None)
    agent.notes.save("alice", "צריך חדר נגיש")
    ra = agent.run_turn("thread-A", "alice", REQ_A)
    rb = agent.run_turn("thread-B", "bob", REQ_B)
    assert ra.status == "awaiting_approval" and rb.status == "awaiting_approval"

    a = agent.store.get_or_create("thread-A", "alice")
    b = agent.store.get_or_create("thread-B", "bob")
    a_text = " ".join(m.content for m in a.messages)
    b_text = " ".join(m.content for m in b.messages)
    assert "40 סטודנטים" in a_text and "40 סטודנטים" not in b_text
    assert "20 איש" in b_text and "20 איש" not in a_text
    assert a.pending_proposal["date"] == "2026-12-15" and b.pending_proposal["date"] == "2026-12-17"

    # אישור בשיחה B לא משפיע על שיחה A
    rb2 = agent.run_turn("thread-B", "bob", "כן")
    assert rb2.status == "done"
    a_after = agent.store.get_or_create("thread-A", "alice")
    assert a_after.status == "awaiting_approval" and a_after.user_approved is False
    assert a_after.messages == a.messages

    # זיכרון מתמשך מבודד לפי משתמש
    assert agent.notes.all_for("bob") == []
    owners = {r["user_id"] for r in agent.db.reservations if r["reservation_id"] == "RES-104"}
    assert owners == {"bob"}


def test_thread_cannot_be_hijacked_by_other_user():
    store = StateStore()
    store.get_or_create("thread-A", "alice")
    with pytest.raises(PermissionError):
        store.get_or_create("thread-A", "mallory")


def test_reduced_context_keeps_current_request_and_last_n():
    s = AgentState(thread_id="t", user_id="u")
    for i in range(5):
        s = apply_event(s, UserMessageEvent(text=f"שאלה {i}"))
        s = apply_event(s, AssistantEvent(text=f"תשובה {i}", kind="clarify"))
    s = apply_event(s, UserMessageEvent(text="הבקשה הנוכחית"))
    full = build_context(s, [], "full")
    red = build_context(s, ["העדפה"], "reduced", last_n=4)
    assert len(full) == 11
    assert red[0]["role"] == "system" and "העדפה" in red[0]["content"]
    assert [m["content"] for m in red[1:]] == ["שאלה 3", "תשובה 3", "שאלה 4", "תשובה 4", "הבקשה הנוכחית"]
