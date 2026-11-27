"""Approval for sensitive actions (no LLM, no network).

A sensitive tool call is not run: it is saved here as a pending action and the user is asked to
confirm. The user's very next message decides: a clear yes runs exactly the saved action, a clear no
cancels it, anything else drops it (nothing runs). A pending action expires after a few minutes and
only its owner can confirm it. One pending action per user: a new one replaces the old one.
"""
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from recipe_agent.db.models import PendingAction, User

TTL_MINUTES = 10

_MARKS = re.compile("[\u064b-\u065f\u0670\u0640]")


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = _MARKS.sub("", t)
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ة", "ه")):
        t = t.replace(a, b)
    return " ".join(re.sub(r"[^\w\s']", " ", t).split())


YES = {"نعم", "ايوه", "ايوا", "اه", "ايوه ماشي", "تمام", "اكد", "تاكيد", "اكيد", "موافق", "موافقه", "نفذ", "نفذي",
       "كمل", "كملي", "ماشي", "اوك", "اوكي", "yes", "y", "yep", "yeah", "ok", "okay", "confirm", "sure",
       "go ahead", "do it", "yes please", "تمام نفذ", "نعم نفذ", "نعم اكد", "اه نفذ"}
NO = {"لا", "لاء", "لأ", "لا شكرا", "لا خلاص", "الغي", "الغيه", "الغاء", "لغي", "كنسل", "بلاش", "خلاص", "مش عايز", "مش عايزه",
      "no", "n", "nope", "cancel", "stop", "never mind", "dont", "don't", "no thanks"}
_YES, _NO = {_norm(x) for x in YES}, {_norm(x) for x in NO}


def is_yes(text: str) -> bool:
    return _norm(text) in _YES


def is_no(text: str) -> bool:
    return _norm(text) in _NO


@dataclass
class Pending:
    tool: str
    args: dict
    summary_ar: str
    summary_en: str


def _aware(d: datetime) -> datetime:
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)   # SQLite drops the time zone


class ApprovalService:
    def __init__(self, session: Session, ttl_minutes: int = TTL_MINUTES):
        self.session, self.ttl = session, ttl_minutes

    def stage(self, user_id: str, tool: str, args: dict, summary_ar: str, summary_en: str,
              now: Optional[datetime] = None) -> None:
        now = now or datetime.now(timezone.utc)
        if self.session.get(User, user_id) is None:
            self.session.add(User(id=user_id))
            self.session.flush()
        self.session.execute(delete(PendingAction).where(PendingAction.user_id == user_id))
        self.session.add(PendingAction(user_id=user_id, tool=tool, args_json=json.dumps(args, ensure_ascii=False),
                                       summary_ar=summary_ar, summary_en=summary_en, created_at=now,
                                       expires_at=now + timedelta(minutes=self.ttl)))
        self.session.commit()

    def take(self, user_id: str, now: Optional[datetime] = None) -> Optional[Pending]:
        """Return the user's pending action (None if none or expired) and remove it: it is answered once."""
        now = now or datetime.now(timezone.utc)
        row = self.session.scalars(select(PendingAction).where(PendingAction.user_id == user_id)
                                   .order_by(PendingAction.id.desc())).first()
        if row is None:
            return None
        live = _aware(row.expires_at) >= now
        out = Pending(row.tool, json.loads(row.args_json), row.summary_ar, row.summary_en) if live else None
        self.session.execute(delete(PendingAction).where(PendingAction.user_id == user_id))
        self.session.commit()
        return out

    def peek(self, user_id: str) -> Optional[Pending]:
        row = self.session.scalars(select(PendingAction).where(PendingAction.user_id == user_id)).first()
        return Pending(row.tool, json.loads(row.args_json), row.summary_ar, row.summary_en) if row else None
