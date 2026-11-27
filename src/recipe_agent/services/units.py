"""Units and ingredient-aware conversions. Reuses the Phase 2 parser and price map.

Everything is converted to a base dimension: grams ('g'), millilitres ('ml') or a count
('piece', 'clove', 'head', 'can'). A conversion that needs an ingredient (cup -> grams) uses the
price-map row (g_per_cup, g_per_piece, density...). If it cannot be done the result is
`unknown` with value None. Nothing is guessed.
"""
import re
from dataclasses import dataclass
from typing import Optional

from ..egypt_cost import load_price_map
from ..egypt_parse import f, grams, match, norm, parse_amount
from .schemas import Measure, Status

UNIT_ALIASES = {
    "g": "g", "gram": "g", "grams": "g", "gm": "g", "جرام": "g", "جم": "g", "غرام": "g",
    "kg": "kg", "kilo": "kg", "kilos": "kg", "kilogram": "kg", "كيلو": "kg", "كجم": "kg",
    "ml": "ml", "milliliter": "ml", "millilitre": "ml", "مل": "ml",
    "l": "l", "liter": "l", "litre": "l", "liters": "l", "litres": "l", "لتر": "l",
    "tbsp": "tbsp", "tablespoon": "tbsp", "tablespoons": "tbsp", "tbs": "tbsp",
    "tsp": "tsp", "teaspoon": "tsp", "teaspoons": "tsp",
    "cup": "cup", "cups": "cup", "كوب": "cup",
    "piece": "piece", "pieces": "piece", "pc": "piece", "pcs": "piece", "unit": "piece",
    "units": "piece", "each": "piece", "حبة": "piece", "حبه": "piece",
    "clove": "clove", "cloves": "clove", "فص": "clove",
    "head": "head", "heads": "head", "راس": "head",
    "can": "can", "cans": "can", "tin": "can", "tins": "can", "علبة": "can", "علبه": "can",
    "pinch": "pinch", "pinches": "pinch", "رشة": "pinch", "رشه": "pinch",
    "whole_chicken": "whole_chicken",
}
COUNT_UNITS = ("piece", "clove", "head", "can")
VOLUME_UNITS = ("tbsp", "tsp", "cup")

# Price-map groups that are too broad to merge by key ("other spices", "any fish"...).
# Their identity is the ingredient name instead, so cumin-like mistakes cannot happen.
COARSE_KEYS = {"spices_other", "fresh_herbs", "cheese_generic", "beef_generic", "fish_generic",
               "non_food"}

_AR = re.compile(r"[\u0600-\u06FF]")


class UnitError(ValueError):
    """Unknown unit, or a quantity that cannot be converted."""


def normalize_unit(unit) -> Optional[str]:
    """Canonical unit or None. Raises UnitError for a unit we do not know."""
    if unit is None or str(unit).strip() == "":
        return None
    u = norm(str(unit)).strip().lower()
    if u not in UNIT_ALIASES:
        u2 = u.rstrip(".")
        if u2 not in UNIT_ALIASES:
            raise UnitError(f"unknown unit: {unit!r}")
        u = u2
    return UNIT_ALIASES[u]


@dataclass
class Ingredient:
    """One ingredient name resolved against the price map."""
    name: str
    key: Optional[str]
    identity: str
    row: Optional[dict]


def _plain_name(name: str) -> str:
    t = norm(str(name)).strip().lower()
    t = re.sub(r"[^\w\u0600-\u06FF]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def resolve_ingredient(name: str, price_map=None) -> Ingredient:
    pm = price_map if price_map is not None else load_price_map()
    text = str(name).strip()
    row = None
    if _AR.search(text):
        row = match(text, pm)
    else:
        from ..egypt_store import _user_key  # English name -> price-map group
        key = _user_key(text, pm)
        if key:
            row = next((r for r in pm if r["key"] == key), None)
    key = row["key"] if row else None
    identity = key if (key and key not in COARSE_KEYS) else "name:" + _plain_name(text)
    return Ingredient(name=text, key=key, identity=identity, row=row)


def _convert(qty, unit, ing: Ingredient, conf: str = "ok") -> Measure:
    """qty/unit are already canonical (unit may include 'whole_chicken' or None for a bare count)."""
    if qty is None:
        return Measure(status="unknown", note="quantity missing")
    row = ing.row
    notes = []
    if unit == "g":
        return Measure(value=qty, unit="g", status="verified")
    if unit == "kg":
        return Measure(value=qty * 1000, unit="g", status="verified")
    a = {"qty": qty, "unit": unit, "to_taste": False}
    if row is None:
        # no ingredient data: only unit-only conversions
        if unit == "ml":
            return Measure(value=qty, unit="ml", status="verified")
        if unit == "l":
            return Measure(value=qty * 1000, unit="ml", status="verified")
        if unit in VOLUME_UNITS:
            ml = qty * {"tbsp": 15, "tsp": 5, "cup": 240}[unit]
            return Measure(value=ml, unit="ml", status="estimated",
                           note="standard spoon/cup size; ingredient not in price map")
        if unit == "pinch":
            return Measure(value=0.5 * qty, unit="g", status="estimated", note="pinch ~ 0.5 g")
        if unit in COUNT_UNITS:
            return Measure(value=qty, unit=unit, status="verified",
                           note="count only; ingredient not in price map")
        if unit == "whole_chicken" or unit is None:
            return Measure(value=qty, unit="piece", status="estimated",
                           note="count only; ingredient not in price map")
        return Measure(status="unknown", note=f"cannot convert {unit}")
    g = grams(a, row, ing.name)
    if g is not None:
        if unit in ("ml", "l") and not f(row.get("density_g_per_ml")):
            notes.append("density assumed 1 g/ml")
        if conf in ("low", "count_assumed", "range_midpoint"):
            notes.append(f"parse: {conf}")
        return Measure(value=g, unit="g", status="estimated", note="; ".join(notes))
    # no gram weight available for this unit: keep the count, do not invent a weight
    if unit in COUNT_UNITS or unit is None or unit == "whole_chicken":
        return Measure(value=qty, unit="piece" if unit in (None, "whole_chicken") else unit,
                       status="estimated" if unit in (None, "whole_chicken") else "verified",
                       note="no gram weight in price map; kept as a count")
    if unit == "ml":
        return Measure(value=qty, unit="ml", status="verified")
    if unit == "l":
        return Measure(value=qty * 1000, unit="ml", status="verified")
    return Measure(status="unknown", note=f"cannot convert {unit}")


def convert(quantity, unit, ingredient) -> Measure:
    """Convert (quantity, unit) of an ingredient name or `Ingredient` to a base-dimension Measure."""
    ing = ingredient if isinstance(ingredient, Ingredient) else resolve_ingredient(ingredient)
    if quantity is None:
        return Measure(status="unknown", note="quantity missing")
    if quantity < 0:
        raise UnitError("quantity cannot be negative")
    return _convert(float(quantity), normalize_unit(unit), ing)


def convert_text(amount_text, ingredient) -> tuple:
    """Convert an Arabic/English amount string ('2 كوب', '500 جرام'). Returns (Measure, to_taste)."""
    ing = ingredient if isinstance(ingredient, Ingredient) else resolve_ingredient(ingredient)
    a = parse_amount(amount_text)
    if a["to_taste"]:
        return Measure(status="unknown", note="to taste"), True
    return _convert(a["qty"], a["unit"], ing, a["conf"]), False


# ---------- price ----------
PRICE_STATUS = {"wfp": "verified", "current_web": "verified", "estimated": "estimated"}


@dataclass
class Price:
    cost_egp: Optional[float]
    price_egp_per_kg: Optional[float]
    source: Optional[str]
    status: Status
    reliability: Optional[str] = None
    price_date: Optional[str] = None


def price_of(measure: Measure, ing: Ingredient) -> Price:
    """Cost of a Measure. Only grams of an ingredient with a price-map price can be costed;
    anything else stays unavailable (None)."""
    row = ing.row
    ppk = f(row["price_egp_per_kg"]) if row else None
    if measure.value is None or measure.unit != "g" or ppk is None:
        return Price(None, ppk, None, "unknown")
    src = row["price_source"]
    return Price(round(measure.value / 1000 * ppk, 2), ppk, src, PRICE_STATUS.get(src, "estimated"),
                 row.get("current_price_reliability") or None, row.get("current_price_date") or None)
