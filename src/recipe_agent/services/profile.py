"""User profile and conversation history (Phase 5b). No LLM, no network.

Profile: household preferences and business info, in the database (one row per user).
Allergies: kept in the existing memory file (recipe_agent.memory), the same store the recipe
search reads, so there is one source of truth for allergies. They are only saved when the
caller asks (the profile agent does that on an explicit request).
History: every message and answer, per user, with read and delete.
Nothing is shared between users.
"""
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from recipe_agent.db.models import ChatMessage, User, UserProfile

Mode = Literal["household", "business"]
Period = Literal["daily", "weekly", "monthly"]
MAX_TEXT = 4000


def _clean_list(items) -> list:
    out = []
    for i in items or []:
        t = re.sub(r"\s+", " ", str(i)).strip()
        if t and t.lower() not in [o.lower() for o in out]:
            out.append(t[:80])
    return out[:30]


class ProfileUpdate(BaseModel):
    """Only the fields that are set are changed. Lists REPLACE the old list."""
    mode: Optional[Mode] = None
    family_size: Optional[int] = Field(default=None, ge=1, le=50)
    favorite_cuisines: Optional[list[str]] = None
    disliked_ingredients: Optional[list[str]] = None
    diet: Optional[list[str]] = None
    max_minutes: Optional[int] = Field(default=None, ge=5, le=1440)
    budget_egp: Optional[float] = Field(default=None, ge=0)
    budget_period: Optional[Period] = None
    business_name: Optional[str] = Field(default=None, max_length=200)
    business_type: Optional[str] = Field(default=None, max_length=200)

    @field_validator("favorite_cuisines", "disliked_ingredients", "diet")
    @classmethod
    def _lists(cls, v):
        return None if v is None else _clean_list(v)


class ProfileView(BaseModel):
    user_id: str
    exists: bool = False
    mode: str = "household"
    family_size: Optional[int] = None
    favorite_cuisines: list = Field(default_factory=list)
    disliked_ingredients: list = Field(default_factory=list)
    diet: list = Field(default_factory=list)
    max_minutes: Optional[int] = None
    budget_egp: Optional[float] = None
    budget_period: Optional[str] = None
    business_name: Optional[str] = None
    business_type: Optional[str] = None
    currency: str = "EGP"
    allergies: Optional[list] = None        # None = the memory file could not be read
    allergies_error: bool = False


class HistoryEntry(BaseModel):
    id: int
    role: str
    text: str
    agent: Optional[str] = None


class ProfileService:
    def __init__(self, session: Session, memory=None):
        self.session = session
        if memory is None:
            from recipe_agent import memory
        self.memory = memory

    def _ensure_user(self, user_id: str) -> None:
        if self.session.get(User, user_id) is None:
            self.session.add(User(id=user_id))
            self.session.flush()

    def get(self, user_id: str) -> ProfileView:
        row = self.session.get(UserProfile, user_id)
        view = ProfileView(user_id=user_id)
        if row is not None:
            view = ProfileView(user_id=user_id, exists=True, mode=row.mode, family_size=row.family_size,
                               favorite_cuisines=list(row.favorite_cuisines or []),
                               disliked_ingredients=list(row.disliked_ingredients or []),
                               diet=list(row.diet or []), max_minutes=row.max_minutes,
                               budget_egp=row.budget_egp, budget_period=row.budget_period,
                               business_name=row.business_name, business_type=row.business_type,
                               currency=row.currency)
        try:
            view.allergies = self.memory.get_exclusions(user_id)
        except self.memory.MemoryReadError:
            view.allergies, view.allergies_error = None, True
        return view

    def update(self, user_id: str, upd: ProfileUpdate) -> ProfileView:
        changes = upd.model_dump(exclude_none=True)
        if not changes:
            raise ValueError("nothing to update")
        self._ensure_user(user_id)
        row = self.session.get(UserProfile, user_id)
        if row is None:
            row = UserProfile(user_id=user_id, favorite_cuisines=[], disliked_ingredients=[], diet=[])
            self.session.add(row)
        for k, v in changes.items():
            setattr(row, k, v)
        self.session.commit()
        return self.get(user_id)

    def add_allergies(self, user_id: str, items) -> list:
        """Save allergies in the memory file (same store the recipe search uses)."""
        items = _clean_list(items)
        if not items:
            raise ValueError("no allergy given")
        return self.memory.remember_exclusions(user_id, items)

    def clear_allergies(self, user_id: str) -> None:
        self.memory.forget(user_id)

    def delete_profile(self, user_id: str) -> bool:
        row = self.session.get(UserProfile, user_id)
        if row is None:
            return False
        self.session.delete(row)
        self.session.commit()
        return True


class HistoryService:
    def __init__(self, session: Session):
        self.session = session

    def add(self, user_id: str, role: str, text: str, agent: Optional[str] = None) -> None:
        if role not in ("user", "assistant"):
            raise ValueError("role must be user or assistant")
        if self.session.get(User, user_id) is None:
            self.session.add(User(id=user_id))
            self.session.flush()
        self.session.add(ChatMessage(user_id=user_id, role=role, text=(text or "")[:MAX_TEXT], agent=agent))
        self.session.commit()

    def recent(self, user_id: str, n: int = 6) -> list:
        """Last n entries, oldest first."""
        rows = list(self.session.scalars(select(ChatMessage).where(ChatMessage.user_id == user_id)
                                         .order_by(ChatMessage.id.desc()).limit(max(0, n))))
        return [HistoryEntry(id=r.id, role=r.role, text=r.text, agent=r.agent) for r in reversed(rows)]

    def count(self, user_id: str) -> int:
        return self.session.scalar(select(func.count()).select_from(ChatMessage)
                                   .where(ChatMessage.user_id == user_id)) or 0

    def clear(self, user_id: str) -> int:
        n = self.count(user_id)
        self.session.execute(delete(ChatMessage).where(ChatMessage.user_id == user_id))
        self.session.commit()
        return n
