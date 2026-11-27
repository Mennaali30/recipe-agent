"""Update the Egyptian price map from a small CSV (price plan, step 2). Offline, no LLM, no network.

Each update row sets one ingredient group to a sourced price. Every row is checked before anything
is written; one bad row stops the whole run, so the map is never half updated.
Columns: key, price_egp_per_kg, source_url, date (YYYY-MM-DD), type (retail|wholesale|festival retail),
reliability (high|medium|low), note (optional).
"""
import csv
import io
import math
from datetime import date
from pathlib import Path

from .egypt_cost import DEFAULT_MAP
from .price_range import STALE_DAYS, parse_date  # noqa: F401  (shared with the range code)

TYPES = {"retail", "wholesale", "festival retail"}
RELIABILITY = {"high", "medium", "low"}
UPDATE_COLUMNS = ["key", "price_egp_per_kg", "source_url", "date", "type", "reliability", "note"]
MAX_PRICE = 20000.0        # EGP per kg; saffron-level prices are below this
MAX_AGE_DAYS = 120         # a source older than this is refused
JUMP_FACTOR = 5.0          # a price 5x higher or lower than the current one needs --force


def read_rows(path=None):
    path = Path(path or DEFAULT_MAP)
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rd = csv.DictReader(fh)
        return rd.fieldnames, list(rd)


def write_rows(fieldnames, rows, path=None):
    path = Path(path or DEFAULT_MAP)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8-sig", newline="")


def read_updates(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rd = csv.DictReader(fh)
        missing = [c for c in UPDATE_COLUMNS[:6] if c not in (rd.fieldnames or [])]
        if missing:
            raise ValueError("update file is missing columns: " + ", ".join(missing))
        return [{k: (v or "").strip() for k, v in r.items()} for r in rd if any((v or "").strip() for v in r.values())]


def validate_update(u: dict, by_key: dict, today: date, force: bool = False) -> list:
    """Return the problems with one update row (empty list = fine)."""
    errs = []
    row = by_key.get(u.get("key"))
    if row is None:
        return [f"unknown key '{u.get('key')}'"]
    try:
        price = float(u.get("price_egp_per_kg"))
    except (TypeError, ValueError):
        price = None
    if price is None or not math.isfinite(price) or price <= 0 or price > MAX_PRICE:
        errs.append(f"{u['key']}: price must be a number above 0 and at most {MAX_PRICE:g}")
    url = u.get("source_url", "")
    if not (url.startswith("https://") or url.startswith("http://")) or " " in url:
        errs.append(f"{u['key']}: source_url must be a web address")
    d = parse_date(u.get("date"))
    if d is None:
        errs.append(f"{u['key']}: date must be YYYY-MM-DD")
    elif d > today:
        errs.append(f"{u['key']}: date is in the future")
    elif (today - d).days > MAX_AGE_DAYS:
        errs.append(f"{u['key']}: source is older than {MAX_AGE_DAYS} days")
    if u.get("type") not in TYPES:
        errs.append(f"{u['key']}: type must be one of {sorted(TYPES)}")
    if u.get("reliability") not in RELIABILITY:
        errs.append(f"{u['key']}: reliability must be one of {sorted(RELIABILITY)}")
    if price and not force:
        try:
            old = float(row.get("price_egp_per_kg") or 0)
        except ValueError:
            old = 0
        if old > 0 and (price > old * JUMP_FACTOR or price < old / JUMP_FACTOR):
            errs.append(f"{u['key']}: {price:g} is more than {JUMP_FACTOR:g}x away from the current {old:g} (check the unit, or use force)")
    return errs


def apply_updates(rows: list, updates: list, today: date = None, force: bool = False):
    """Return (new_rows, changes). Raises ValueError with every problem if any row is bad."""
    today = today or date.today()
    by_key = {r["key"]: r for r in rows}
    errors, seen = [], set()
    for u in updates:
        if u.get("key") in seen:
            errors.append(f"{u.get('key')}: appears twice in the update file")
        seen.add(u.get("key"))
        errors += validate_update(u, by_key, today, force)
    if errors:
        raise ValueError("\n".join(errors))
    new_rows, changes = [dict(r) for r in rows], []
    index = {r["key"]: r for r in new_rows}
    for u in updates:
        r = index[u["key"]]
        old_price, old_source = r["price_egp_per_kg"], r["price_source"]
        r["price_source"] = "current_web"
        r["price_egp_per_kg"] = f"{float(u['price_egp_per_kg']):g}"
        r["current_price_source_url"] = u["source_url"]
        r["current_price_date"] = u["date"]
        r["current_price_type"] = u["type"]
        r["current_price_reliability"] = u["reliability"]
        r["notes"] = ((u.get("note") or "") + f" (was {old_price} EGP/kg, {old_source})").strip()
        changes.append({"key": u["key"], "old_price": old_price, "new_price": r["price_egp_per_kg"],
                        "old_source": old_source})
    return new_rows, changes


def price_age_days(row: dict, today: date = None):
    """Days since the price date, or None when the row has no date (estimated rows)."""
    d = parse_date((row or {}).get("current_price_date"))
    return None if d is None else ((today or date.today()) - d).days


def is_stale(row: dict, today: date = None) -> bool:
    age = price_age_days(row, today)
    return age is not None and age > STALE_DAYS


def update_file(updates_path, map_path=None, write: bool = False, today: date = None, force: bool = False):
    """Check an update CSV against the map; write the map only when write=True. Returns the changes."""
    fields, rows = read_rows(map_path)
    new_rows, changes = apply_updates(rows, read_updates(updates_path), today, force)
    if write:
        write_rows(fields, new_rows, map_path)
    return changes
