"""Tables for the pantry. Quantities are stored in a base unit (g, ml or a count unit)."""
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
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
