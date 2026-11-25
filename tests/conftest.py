"""Fixtures משותפות. כל הבדיקות רצות מול שירות מדומה (MOCK) בלי רשת ובלי מפתח."""
import pytest

from src.state import NoteStore
from src.tools import CampusDB, ToolContext


class SleepRecorder:
    """מחליף את time.sleep: רושם את זמני ההמתנה בלי לחכות באמת."""
    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


@pytest.fixture
def sleep():
    return SleepRecorder()


@pytest.fixture
def make_ctx(sleep):
    def _make(faults=None, user_id="u1", approved=False, notes=None, **kw):
        db = kw.pop("db", None) or CampusDB(faults=faults, hang_seconds=kw.pop("hang_seconds", 0.3))
        return ToolContext(user_id=user_id, db=db, notes=notes or NoteStore(),
                           user_approved=approved, sleep=sleep, **kw)
    return _make
