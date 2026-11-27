"""Pantry / inventory service. Every query is limited to one user_id (per-user ownership)."""
from datetime import date, timedelta
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import InventoryTransaction, PantryItem, User
from .schemas import Measure
from .units import UnitError, convert, resolve_ingredient


class InsufficientStock(ValueError):
    def __init__(self, name, have, need, unit):
        super().__init__(f"not enough {name}: have {have:g} {unit}, need {need:g} {unit}")
        self.name, self.have, self.need, self.unit = name, have, need, unit


class PantryItemView(BaseModel):
    id: int
    name: str
    identity: str
    unit: str
    quantity: float
    status: str
    expiry: Optional[date] = None
    low_stock_threshold: Optional[float] = None


def _view(i: PantryItem) -> PantryItemView:
    return PantryItemView(id=i.id, name=i.name, identity=i.identity, unit=i.unit,
                          quantity=round(i.quantity, 3), status=i.status, expiry=i.expiry,
                          low_stock_threshold=i.low_stock_threshold)


class PantryService:
    def __init__(self, session: Session):
        self.session = session

    # ---- helpers ----
    def ensure_user(self, user_id: str, name: Optional[str] = None) -> User:
        if not user_id or not str(user_id).strip():
            raise ValueError("user_id is required")
        u = self.session.get(User, user_id)
        if u is None:
            u = User(id=user_id, name=name)
            self.session.add(u)
            self.session.flush()
        return u

    def _measure(self, name, quantity, unit) -> tuple:
        ing = resolve_ingredient(name)
        m = convert(quantity, unit, ing)
        if m.value is None or m.unit is None:
            raise UnitError(f"cannot convert {quantity} {unit} of {name}: {m.note}")
        return ing, m

    def _find(self, user_id, identity, unit) -> Optional[PantryItem]:
        return self.session.scalars(select(PantryItem).where(
            PantryItem.user_id == user_id, PantryItem.identity == identity,
            PantryItem.unit == unit)).first()

    def _log(self, item, kind, delta, note=""):
        self.session.add(InventoryTransaction(user_id=item.user_id, item_id=item.id, kind=kind,
                                              delta=delta, quantity_after=item.quantity,
                                              unit=item.unit, note=note))

    # ---- changes ----
    def add_item(self, user_id: str, name: str, quantity: float, unit: Optional[str] = None,
                 expiry: Optional[date] = None, low_stock_threshold: Optional[float] = None,
                 note: str = "") -> PantryItemView:
        """Add stock. The quantity is converted to a base unit (g, ml or count)."""
        if quantity is None or quantity <= 0:
            raise ValueError("quantity must be positive")
        self.ensure_user(user_id)
        ing, m = self._measure(name, quantity, unit)
        item = self._find(user_id, ing.identity, m.unit)
        if item is None:
            item = PantryItem(user_id=user_id, identity=ing.identity, name=name, unit=m.unit,
                              quantity=0.0, status=m.status, expiry=expiry,
                              low_stock_threshold=low_stock_threshold)
            self.session.add(item)
            self.session.flush()
        else:
            if m.status == "estimated":
                item.status = "estimated"
            if expiry and (item.expiry is None or expiry < item.expiry):
                item.expiry = expiry
            if low_stock_threshold is not None:
                item.low_stock_threshold = low_stock_threshold
        item.quantity = item.quantity + m.value
        self._log(item, "add", m.value, note)
        self.session.commit()
        return _view(item)

    def use_item(self, user_id: str, name: str, quantity: float, unit: Optional[str] = None,
                 allow_partial: bool = False, note: str = "") -> PantryItemView:
        """Take stock out. Raises InsufficientStock (nothing changes) unless allow_partial."""
        if quantity is None or quantity <= 0:
            raise ValueError("quantity must be positive")
        ing, m = self._measure(name, quantity, unit)
        item = self._find(user_id, ing.identity, m.unit)
        have = item.quantity if item else 0.0
        if item is None or have + 1e-9 < m.value:
            if not allow_partial or item is None:
                raise InsufficientStock(name, have, m.value, m.unit)
            take = have
        else:
            take = m.value
        item.quantity = max(item.quantity - take, 0.0)
        self._log(item, "use", -take, note)
        self.session.commit()
        return _view(item)

    def adjust_item(self, user_id: str, name: str, new_quantity: float, unit: Optional[str] = None,
                    note: str = "stock count") -> PantryItemView:
        """Set the stock to an exact amount (after counting the shelf)."""
        if new_quantity is None or new_quantity < 0:
            raise ValueError("quantity cannot be negative")
        self.ensure_user(user_id)
        ing = resolve_ingredient(name)
        m = convert(new_quantity, unit, ing) if new_quantity > 0 else None
        if m is not None and m.value is None:
            raise UnitError(f"cannot convert {new_quantity} {unit} of {name}")
        if m is None:  # zero: need the existing row's unit
            rows = self.session.scalars(select(PantryItem).where(
                PantryItem.user_id == user_id, PantryItem.identity == ing.identity)).all()
            if not rows:
                raise KeyError(name)
            for r in rows:
                self._log_change(r, 0.0, note)
            self.session.commit()
            return _view(rows[0])
        item = self._find(user_id, ing.identity, m.unit)
        if item is None:
            item = PantryItem(user_id=user_id, identity=ing.identity, name=name, unit=m.unit,
                              quantity=0.0, status=m.status)
            self.session.add(item)
            self.session.flush()
        self._log_change(item, m.value, note)
        self.session.commit()
        return _view(item)

    def _log_change(self, item, new_qty, note):
        delta = new_qty - item.quantity
        item.quantity = new_qty
        self._log(item, "adjust", delta, note)

    def discard_item(self, user_id: str, name: str, unit: Optional[str] = None) -> PantryItemView:
        """Throw away all the stock of an item (expired or spoiled)."""
        ing = resolve_ingredient(name)
        q = select(PantryItem).where(PantryItem.user_id == user_id, PantryItem.identity == ing.identity)
        if unit:
            q = q.where(PantryItem.unit == convert(1, unit, ing).unit)
        rows = self.session.scalars(q).all()
        if not rows:
            raise KeyError(name)
        for r in rows:
            delta = -r.quantity
            r.quantity = 0.0
            r.expiry = None
            self._log(r, "discard", delta)
        self.session.commit()
        return _view(rows[0])

    # ---- reads ----
    def list_items(self, user_id: str, include_empty: bool = False) -> list:
        q = select(PantryItem).where(PantryItem.user_id == user_id).order_by(PantryItem.name)
        if not include_empty:
            q = q.where(PantryItem.quantity > 0)
        return [_view(i) for i in self.session.scalars(q)]

    def get_stock(self, user_id: str, name: str) -> list:
        ing = resolve_ingredient(name)
        return [Measure(value=i.quantity, unit=i.unit, status=i.status)
                for i in self.session.scalars(select(PantryItem).where(
                    PantryItem.user_id == user_id, PantryItem.identity == ing.identity))]

    def stock_map(self, user_id: str) -> dict:
        """{(identity, unit): quantity}, the shape the shopping list and budget services take."""
        return {(i.identity, i.unit): i.quantity for i in self.session.scalars(
            select(PantryItem).where(PantryItem.user_id == user_id, PantryItem.quantity > 0))}

    def expired(self, user_id: str, today: Optional[date] = None) -> list:
        today = today or date.today()
        return [_view(i) for i in self.session.scalars(select(PantryItem).where(
            PantryItem.user_id == user_id, PantryItem.quantity > 0, PantryItem.expiry.is_not(None),
            PantryItem.expiry < today).order_by(PantryItem.expiry))]

    def expiring_soon(self, user_id: str, within_days: int = 3, today: Optional[date] = None) -> list:
        """Not yet expired, expiring within `within_days` days (inclusive)."""
        today = today or date.today()
        end = today + timedelta(days=within_days)
        return [_view(i) for i in self.session.scalars(select(PantryItem).where(
            PantryItem.user_id == user_id, PantryItem.quantity > 0, PantryItem.expiry.is_not(None),
            PantryItem.expiry >= today, PantryItem.expiry <= end).order_by(PantryItem.expiry))]

    def low_stock(self, user_id: str) -> list:
        """Items with a threshold set whose quantity is at or below it (empty ones included)."""
        return [_view(i) for i in self.session.scalars(select(PantryItem).where(
            PantryItem.user_id == user_id, PantryItem.low_stock_threshold.is_not(None),
            PantryItem.quantity <= PantryItem.low_stock_threshold).order_by(PantryItem.name))]

    def transactions(self, user_id: str, name: Optional[str] = None) -> list:
        q = select(InventoryTransaction).where(InventoryTransaction.user_id == user_id)
        if name:
            ing = resolve_ingredient(name)
            ids = [i.id for i in self.session.scalars(select(PantryItem).where(
                PantryItem.user_id == user_id, PantryItem.identity == ing.identity))]
            q = q.where(InventoryTransaction.item_id.in_(ids))
        return list(self.session.scalars(q.order_by(InventoryTransaction.id)))
