"""Tables for the pantry. Quantities are stored in a base unit (g, ml or a count unit)."""
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import JSON, Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class PantryItem(Base):
    """One stock row per user + ingredient identity + base unit.
    `expiry` is the earliest expiry among what was added (conservative)."""
    __tablename__ = "pantry_items"
    __table_args__ = (UniqueConstraint("user_id", "identity", "unit", name="uq_pantry_user_item_unit"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    identity: Mapped[str] = mapped_column(String(200))
    name: Mapped[str] = mapped_column(String(200))
    unit: Mapped[str] = mapped_column(String(20))
    quantity: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(20), default="verified")  # verified / estimated
    expiry: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    low_stock_threshold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
    transactions: Mapped[list["InventoryTransaction"]] = relationship(back_populates="item")


class InventoryTransaction(Base):
    """Every change to a stock row: add, use, adjust, discard."""
    __tablename__ = "inventory_transactions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("pantry_items.id"), index=True)
    kind: Mapped[str] = mapped_column(String(20))
    delta: Mapped[float] = mapped_column(Float)
    quantity_after: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(20))
    note: Mapped[str] = mapped_column(String(300), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    item: Mapped[PantryItem] = relationship(back_populates="transactions")


class UserProfile(Base):
    """Household preferences and business profile. One row per user. Allergies are NOT stored
    here: they stay in the memory file (recipe_agent.memory), which the recipe search already uses."""
    __tablename__ = "user_profiles"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), primary_key=True)
    mode: Mapped[str] = mapped_column(String(20), default="household")   # household / business
    family_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    favorite_cuisines: Mapped[list] = mapped_column(JSON, default=list)
    disliked_ingredients: Mapped[list] = mapped_column(JSON, default=list)
    diet: Mapped[list] = mapped_column(JSON, default=list)
    max_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    budget_egp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    budget_period: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # daily / weekly / monthly
    business_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    business_type: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    currency: Mapped[str] = mapped_column(String(10), default="EGP")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class Account(Base):
    """A login. One per user; the user's data (pantry, profile, history) hangs on users.id as before."""
    __tablename__ = "accounts"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), primary_key=True)
    username: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(300))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class PendingAction(Base):
    """A sensitive action waiting for the user's yes (one per user; the next message answers it)."""
    __tablename__ = "pending_actions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    tool: Mapped[str] = mapped_column(String(100))
    args_json: Mapped[str] = mapped_column(Text)
    summary_ar: Mapped[str] = mapped_column(Text, default="")
    summary_en: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ChatMessage(Base):
    """Saved conversation history (what the user wrote and what the assistant answered)."""
    __tablename__ = "chat_messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    role: Mapped[str] = mapped_column(String(10))          # user / assistant
    text: Mapped[str] = mapped_column(Text)
    agent: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)   # e.g. "cost,shopping"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
