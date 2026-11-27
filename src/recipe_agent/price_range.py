"""Price uncertainty (price plan, step 3): turn per-line sources into one honest range and label.

Deterministic. The spread of a line depends on how good its price is: a high-reliability or WFP price
is +-10%, a medium or low web price +-20%, an estimated price +-35%. These are rough rules, not
statistics. A total's range is the sum of its lines' ranges (no cancelling out). The label comes
from the share of the total that rests on estimated prices.
"""
import math
from datetime import date, datetime

STALE_DAYS = 30            # a price older than this is shown as "old"


def parse_date(text):
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


SPREAD_HIGH, SPREAD_MEDIUM, SPREAD_ESTIMATED = 0.10, 0.20, 0.35
MARKET_MAX_SHARE = 0.10   # up to 10% of the total on estimates: still "market prices"
APPROX_MAX_SHARE = 0.50   # more than half on estimates: "rough estimate"

LABELS = {
    "ar": {"market": "سعر السوق", "approx": "سعر تقريبي", "rough": "سعر تقديري جداً"},
    "en": {"market": "market prices", "approx": "approximate prices", "rough": "rough estimate"},
}
STALE_NOTE = {"ar": "بعض الأسعار قديمة", "en": "some prices are old"}


def line_spread(source, reliability=None) -> float:
    if source == "wfp":
        return SPREAD_HIGH
    if source == "current_web":
        return SPREAD_HIGH if reliability == "high" else SPREAD_MEDIUM
    return SPREAD_ESTIMATED


def _is_stale(price_date, today) -> bool:
    d = parse_date(price_date)
    return d is not None and (today - d).days > STALE_DAYS


def make_range(parts, today: date = None, fixed: float = 0.0):
    """parts: dicts with cost, source, reliability, price_date. fixed: an exact amount (extra expenses).

    Returns None when nothing is priced, else {low, high, point, estimated_share, label_key, stale}.
    """
    today = today or date.today()
    low = high = point = est = 0.0
    stale = False
    n = 0
    for p in parts:
        cost = p.get("cost")
        if cost is None:
            continue
        n += 1
        s = line_spread(p.get("source"), p.get("reliability"))
        low += cost * (1 - s)
        high += cost * (1 + s)
        point += cost
        if p.get("source") not in ("wfp", "current_web"):
            est += cost
        stale = stale or _is_stale(p.get("price_date"), today)
    if not n:
        return None
    share = est / point if point > 0 else 0.0
    key = "market" if share <= MARKET_MAX_SHARE else ("approx" if share <= APPROX_MAX_SHARE else "rough")
    return {"low": round(low + fixed, 2), "high": round(high + fixed, 2), "point": round(point + fixed, 2),
            "estimated_share": round(share, 2), "label_key": key, "stale": stale}


def _step(high: float) -> float:
    return 1 if high < 20 else (5 if high < 200 else 10)


def round_range(low: float, high: float):
    """Round outwards to a friendly step (1, 5 or 10 EGP) so the range never looks more exact than it is."""
    step = _step(high)
    lo = max(0, math.floor(low / step) * step)
    hi = math.ceil(high / step) * step
    return lo, hi


def range_text(low: float, high: float, lang: str = "ar") -> str:
    lo, hi = round_range(low, high)
    if lo == hi:
        return f"حوالي {hi:g} جنيه" if lang == "ar" else f"about {hi:g} EGP"
    return f"حوالي {lo:g} إلى {hi:g} جنيه" if lang == "ar" else f"about {lo:g} to {hi:g} EGP"


def label_text(rng: dict, lang: str = "ar") -> str:
    out = LABELS[lang][rng["label_key"]]
    if not rng.get("stale"):
        return out
    return f"{out}{'،' if lang == 'ar' else ','} {STALE_NOTE[lang]}"
