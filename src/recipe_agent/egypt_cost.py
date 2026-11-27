"""Cost estimate (EGP) and time filter for Egyptian (fifi.cooking) recipes. Offline.

Prices come from data/egypt/ingredient_price_map.csv. Each row has a price_source:
  wfp          WFP row dated 2026 or later
  current_web  public Egyptian price page (URL and date in the CSV row)
  estimated    rough placeholder, NOT sourced
The recipe-level source is the least trusted source among the lines that were priced.
Totals are always approximate. fifi's own `estimate.cost` is never used (unknown currency).
"""
from pathlib import Path

from .price_range import label_text, make_range, range_text
from .egypt_parse import (f, grams, load_map, match, parse_amount, parse_minutes,
                          parse_servings)

DEFAULT_MAP = Path(__file__).resolve().parents[2] / "data" / "egypt" / "ingredient_price_map.csv"
_TRUST_ORDER = ["estimated", "current_web", "wfp"]  # least trusted first
SOURCE_LABELS = {
    "ar": {"wfp": "سعر WFP", "current_web": "سعر السوق الحالي", "estimated": "سعر تقديري"},
    "en": {"wfp": "WFP price", "current_web": "current market price", "estimated": "rough estimate"},
}

_cache = {}


def load_price_map(path=None):
    path = str(path or DEFAULT_MAP)
    if path not in _cache:
        _cache[path] = load_map(path)
    return _cache[path]


def estimate_line(ingredient: dict, price_map) -> dict:
    """Price one fifi masterIngredients entry. cost_egp is None when it cannot be priced."""
    amount = ingredient.get("standardAmount")
    a = parse_amount(amount)
    row = match(ingredient.get("name", ""), price_map)
    g = None if a["to_taste"] else grams(a, row, ingredient.get("name", ""))
    ppk = f(row["price_egp_per_kg"]) if row else None
    cost = round(g / 1000 * ppk, 2) if (g is not None and ppk is not None) else None
    return {
        "name": ingredient.get("name", ""),
        "amount_text": amount,
        "to_taste": a["to_taste"],
        "parse_confidence": a["conf"],
        "grams": round(g, 1) if g is not None else None,
        "mapped_key": row["key"] if row else None,
        "price_source": row["price_source"] if (row and cost is not None) else None,
        "price_reliability": (row.get("current_price_reliability") or None) if (row and cost is not None) else None,
        "price_date": (row.get("current_price_date") or None) if (row and cost is not None) else None,
        "price_egp_per_kg": ppk,
        "cost_egp": cost,
    }


def estimate_recipe_cost(entry: dict, price_map=None) -> dict:
    """entry is a fifi recipe file ({'recipe': {...}, ...}) or the inner recipe dict."""
    price_map = price_map if price_map is not None else load_price_map()
    rec = entry.get("recipe", entry)
    lines = [estimate_line(m, price_map) for m in rec.get("masterIngredients", [])]
    counted = [l for l in lines if not l["to_taste"]]
    priced = [l for l in counted if l["cost_egp"] is not None]
    total = round(sum(l["cost_egp"] for l in priced), 2) if priced else None
    servings = parse_servings(rec.get("servings"))
    sources = {l["price_source"] for l in priced}
    source = next((s for s in _TRUST_ORDER if s in sources), None)
    rng = make_range([{"cost": l["cost_egp"], "source": l["price_source"], "reliability": l["price_reliability"],
                       "price_date": l["price_date"]} for l in priced])
    return {
        "recipe_id": rec.get("id"),
        "low_egp": rng["low"] if rng else None,
        "high_egp": rng["high"] if rng else None,
        "price_label": rng["label_key"] if rng else None,
        "stale_prices": rng["stale"] if rng else False,
        "currency": "EGP",
        "approximate": True,
        "total_egp": total,
        "per_serving_egp": round(total / servings, 2) if (total is not None and servings) else None,
        "servings": servings,
        "price_source": source,
        "coverage": round(len(priced) / len(counted), 2) if counted else None,
        "unpriced": [l["name"] for l in counted if l["cost_egp"] is None],
        "lines": lines,
    }


def format_cost(est: dict, lang: str = "ar") -> str:
    """One display line. Always says it is approximate and which kind of price it used."""
    if est["total_egp"] is None:
        return "التكلفة غير متاحة" if lang == "ar" else "Cost not available"
    pct = int(round((est["coverage"] or 0) * 100))
    if est.get("low_egp") is not None:
        rng = {"label_key": est["price_label"], "stale": est.get("stale_prices")}
        amount, label = range_text(est["low_egp"], est["high_egp"], lang), label_text(rng, lang)
    else:   # a cost made without ranges (older data): one number, labelled by its weakest source
        amount, label = (f"{est['total_egp']:.0f} " + ("جنيه" if lang == "ar" else "EGP")), SOURCE_LABELS[lang][est["price_source"]]
    if lang == "ar":
        return f"التكلفة التقريبية: {amount} ({label}، تغطي {pct}% من المكونات)"
    return f"Approx. cost: {amount} ({label}, covers {pct}% of ingredients)"


def recipe_minutes(entry: dict):
    """Prep + cook minutes. None if neither time can be parsed (unknown time)."""
    rec = entry.get("recipe", entry)
    parts = [parse_minutes(rec.get("prepTime")), parse_minutes(rec.get("cookTime"))]
    known = [p for p in parts if p is not None]
    return sum(known) if known else None


def within_minutes(entry: dict, max_minutes, unknown_ok: bool = False) -> bool:
    """Under-N-minutes filter. Unknown time is excluded unless unknown_ok."""
    if max_minutes is None:
        return True
    m = recipe_minutes(entry)
    return unknown_ok if m is None else m <= max_minutes
