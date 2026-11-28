"""Meal plans from the curated Egyptian recipes. Pure code: no LLM, no network, same answer for the same input.

A "meal" is one serving. A plan is N meals (or days x meals per day), every recipe different, buckets
rotated for variety, saved allergies and extra exclusions respected. With a budget the plan is made to
fit the HIGH end of the price range, so "within budget" is not a hope. A price that is missing stays
missing: the plan says its cost is partial, never guesses.
"""
import random
from typing import Optional

from recipe_agent import egypt_store
from recipe_agent.egypt_parse import parse_servings

MIN_COVERAGE = 0.9      # recipes with a bigger unpriced part are left out: their cost would look too low
MAX_SERVINGS = 20       # a dataset 'servings' above this is not a number of people (e.g. grams)
MAIN_BUCKETS = ("chicken", "meat", "soups", "mahshi", "street")
SLOT_BUCKETS = {"breakfast": ("breakfast", "street"), "lunch": MAIN_BUCKETS, "dinner": MAIN_BUCKETS, "meal": MAIN_BUCKETS}
DAY_SLOTS = {1: ("lunch",), 2: ("breakfast", "dinner"), 3: ("breakfast", "lunch", "dinner")}


class PlanError(Exception):
    """code: bad_args | no_candidates"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def per_meal(entry: dict) -> Optional[tuple]:
    """(low, high) EGP for one serving, or None when the recipe has no usable price."""
    c = entry["cost"]
    if c.get("low_egp") is None or c.get("high_egp") is None or (c.get("coverage") or 0) < MIN_COVERAGE:
        return None
    n = parse_servings((entry["raw"].get("recipe") or entry["raw"]).get("servings"))   # unknown servings: cannot scale to a meal
    if not n or n > MAX_SERVINGS:
        return None
    return (round(c["low_egp"] / n, 2), round(c["high_egp"] / n, 2))


def slots_for(meals: Optional[int], days: Optional[int], meals_per_day: Optional[int]) -> list:
    """[(day, meal_type)] in order."""
    if meals is not None:
        if days is not None or meals_per_day is not None:
            raise PlanError("bad_args", "give either meals, or days with meals_per_day, not both")
        if not 1 <= meals <= 21:
            raise PlanError("bad_args", "meals must be between 1 and 21")
        return [((i // 3) + 1, "meal") for i in range(meals)]
    if days is None:
        raise PlanError("bad_args", "give meals, or days (1-7) with meals_per_day (1-3)")
    mpd = meals_per_day or 1
    if not (1 <= days <= 7 and 1 <= mpd <= 3):
        raise PlanError("bad_args", "days must be 1-7 and meals_per_day 1-3")
    return [(d + 1, t) for d in range(days) for t in DAY_SLOTS[mpd]]


_CACHE: dict = {}


def candidates(exclude, max_minutes=None, buckets=None) -> list:
    """Usable recipes (cached per filter: the allergy scan over all recipes is the slow part)."""
    key = (tuple(sorted(str(x).lower() for x in exclude or ())), max_minutes, tuple(buckets or ()))
    if key not in _CACHE:
        if len(_CACHE) > 200:
            _CACHE.clear()
        _CACHE[key] = _candidates(exclude, max_minutes, buckets)
    return list(_CACHE[key])


def _candidates(exclude, max_minutes=None, buckets=None) -> list:
    out = []
    for e in egypt_store.load_store()["entries"].values():
        if buckets and e["category"] not in buckets:
            continue
        if per_meal(e) is None:
            continue
        if max_minutes and (e["total_minutes"] is None or e["total_minutes"] > max_minutes):
            continue
        if egypt_store.excluded(e, exclude):
            continue
        out.append(e)
    return sorted(out, key=lambda e: e["recipe_id"])


def _totals(picks):
    lo = round(sum(per_meal(e)[0] for e in picks), 2)
    hi = round(sum(per_meal(e)[1] for e in picks), 2)
    return lo, hi


def verdict(picks, budget_egp: Optional[float]) -> Optional[str]:
    if budget_egp is None:
        return None
    lo, hi = _totals(picks)
    if hi <= budget_egp:
        return "within_budget"
    return "over_budget" if lo > budget_egp else "might_exceed"


def with_fallback(exclude, max_minutes, mt) -> list:
    """The meal type's own buckets first, then every other bucket (used only when the own ones run out)."""
    own = candidates(exclude, max_minutes, SLOT_BUCKETS[mt])
    ids = {e["recipe_id"] for e in own}
    return own + [e for e in candidates(exclude, max_minutes, None) if e["recipe_id"] not in ids]


def _pick_variety(pool_by_slot, slots, rng, taken):
    """One recipe per slot: all different, neighbouring meals from different buckets when possible."""
    picks, used, last = [], set(taken), None
    for _, mt in slots:
        own = set(SLOT_BUCKETS[mt])
        pool = [e for e in pool_by_slot[mt] if e["recipe_id"] not in used and e["category"] in own]
        pool = pool or [e for e in pool_by_slot[mt] if e["recipe_id"] not in used]
        if not pool:                                   # short of recipes: repeat the least used one (flagged in the result)
            allp = pool_by_slot[mt]
            if not allp:
                raise PlanError("no_candidates", "no recipe matches the allergies and limits")
            use = {}
            for p in picks:
                use[p["recipe_id"]] = use.get(p["recipe_id"], 0) + 1
            low = min(use.get(e["recipe_id"], 0) for e in allp)
            pool = [e for e in allp if use.get(e["recipe_id"], 0) == low and (not picks or e["recipe_id"] != picks[-1]["recipe_id"])] \
                or [e for e in allp if use.get(e["recipe_id"], 0) == low]
        fresh = [e for e in pool if e["category"] != last] or pool
        # rotate through buckets: prefer the bucket used least so far
        counts = {}
        for p in picks:
            counts[p["category"]] = counts.get(p["category"], 0) + 1
        least = min(counts.get(e["category"], 0) for e in fresh)
        fresh = [e for e in fresh if counts.get(e["category"], 0) == least]
        e = rng.choice(fresh)
        picks.append(e)
        used.add(e["recipe_id"])
        last = e["category"]
    return picks


def _fit_budget(picks, slots, pool_by_slot, budget_egp):
    """Swap the most expensive meals for cheaper unused ones until the high end fits (or nothing cheaper is left)."""
    picks = list(picks)
    while _totals(picks)[1] > budget_egp:
        used = {p["recipe_id"] for p in picks}
        best = None
        for i, (_, mt) in enumerate(slots):
            cur = per_meal(picks[i])[1]
            cheaper = [e for e in pool_by_slot[mt] if e["recipe_id"] not in used and per_meal(e)[1] < cur]
            if cheaper:
                c = max(cheaper, key=lambda e: (per_meal(e)[1], e["recipe_id"]))     # smallest step down: keep variety
                saving = cur - per_meal(c)[1]
                if best is None or cur > best[0]:
                    best = (cur, i, c, saving)
        if best is None:
            break
        picks[best[1]] = best[2]
    return picks


def _view(entry, slot_no, day, mt, lang="ar"):
    lo, hi = per_meal(entry)
    return {"slot": slot_no, "day": day, "meal_type": mt, "recipe_id": entry["recipe_id"],
            "name": entry["name"] if lang == "ar" else (entry["name_en"] or entry["name"]),
            "category": entry["category"], "total_minutes": entry["total_minutes"],
            "cost_low_egp": lo, "cost_high_egp": hi,
            "price_complete": (entry["cost"].get("coverage") or 0) >= 0.999,
            "unpriced": entry["cost"].get("unpriced") or []}


def _result(picks, slots, budget_egp, lang, suggestions=None):
    lo, hi = _totals(picks)
    partial = [p["name"] for p in picks if (p["cost"].get("coverage") or 0) < 0.999]
    views, seen = [], set()
    for i, (e, (d, mt)) in enumerate(zip(picks, slots)):
        v = _view(e, i + 1, d, mt, lang)
        v["repeat"] = e["recipe_id"] in seen
        seen.add(e["recipe_id"])
        views.append(v)
    return {"meals": views, "repeats": sum(v["repeat"] for v in views), "suggestions": list(suggestions or []),
            "total_low_egp": lo, "total_high_egp": hi, "budget_egp": budget_egp,
            "budget_status": verdict(picks, budget_egp), "partial_cost": bool(partial),
            "partial_cost_recipes": partial, "currency": "EGP"}


def _tip(code, ar, en, **extra):
    return {"code": code, "message_ar": ar, "message_en": en, **extra}


def _suggestions(picks, slots, budget_egp, exclude, relaxable, max_minutes, mt_pool):
    """Plain advice when the plan is not what was asked: too few recipes, or too small a budget."""
    tips = []
    unique = len({p["recipe_id"] for p in picks})
    if unique < len(picks):
        avail = len(candidates(exclude, max_minutes, None))
        tips.append(_tip("recipes_short",
                         f"المتاح {avail} وصفة بس بتناسب الشروط، فالخطة فيها {len(picks) - unique} وجبة متكررة. ممكن تقللي عدد الوجبات لـ {unique} أو تقبلي التكرار.",
                         f"Only {avail} different recipes fit your allergies and limits, so {len(picks) - unique} meals are repeats. You can plan {unique} meals instead, or accept repeats.",
                         available_recipes=avail, planned_meals=len(picks), unique_meals=unique))
        best = None
        for item in relaxable:
            rest = [x for x in exclude if x != item]
            gain = len(candidates(rest, max_minutes, None)) - avail
            if gain > 0 and (best is None or gain > best[1]):
                best = (item, gain)
        if best:
            tips.append(_tip("relax_exclusion",
                             f"لو شلتي «{best[0]}» من الاستثناءات هيبقى فيه {best[1]} وصفة زيادة.",
                             f"Removing \"{best[0]}\" from the exclusions would add {best[1]} recipes.",
                             exclusion=best[0], extra_recipes=best[1]))
        if max_minutes:
            gain = len(candidates(exclude, None, None)) - avail
            if gain > 0:
                tips.append(_tip("relax_time", f"لو شلتي حد الوقت هيبقى فيه {gain} وصفة زيادة.",
                                 f"Without the time limit {gain} more recipes would fit.", extra_recipes=gain))
    if budget_egp is not None and verdict(picks, budget_egp) == "over_budget":
        lo, hi = _totals(picks)
        tips.append(_tip("budget_too_small",
                         f"أرخص خطة ممكنة تقريبًا {lo:g} إلى {hi:g} جنيه، أكتر من الميزانية {budget_egp:g}. ارفعي الميزانية أو قللي الوجبات.",
                         f"The cheapest plan I can make costs about {lo:g} to {hi:g} EGP, above your {budget_egp:g} budget. Raise the budget or plan fewer meals.",
                         cheapest_low_egp=lo, cheapest_high_egp=hi))
    return tips


def make_plan(meals=None, days=None, meals_per_day=None, budget_egp=None, exclude=(), max_minutes=None,
              seed: int = 0, lang: str = "ar", relaxable=()) -> dict:
    slots = slots_for(meals, days, meals_per_day)
    pool = {mt: with_fallback(exclude, max_minutes, mt) for mt in {m for _, m in slots}}
    picks = _pick_variety(pool, slots, random.Random(seed), ())
    if budget_egp is not None:
        picks = _fit_budget(picks, slots, pool, budget_egp)
    return _result(picks, slots, budget_egp, lang,
                   _suggestions(picks, slots, budget_egp, list(exclude), list(relaxable), max_minutes, pool))


def replan(current_ids, replace, budget_egp=None, exclude=(), max_minutes=None, seed: int = 0, cheaper: bool = False,
           meal_types=None, lang: str = "ar") -> dict:
    """Swap the meals in `replace` (0-based positions or recipe ids) for different ones; the rest stay."""
    store = egypt_store.load_store()["entries"]
    if not current_ids:
        raise PlanError("bad_args", "the current plan is empty")
    for rid in current_ids:
        if rid not in store or per_meal(store[rid]) is None:
            raise PlanError("bad_args", f"unknown recipe in the plan: {rid}")
    meal_types = list(meal_types or ["meal"] * len(current_ids))
    if len(meal_types) != len(current_ids) or any(m not in SLOT_BUCKETS for m in meal_types):
        raise PlanError("bad_args", "meal_types must match the plan, each breakfast, lunch, dinner or meal")
    idx = set()
    for r in replace:
        if isinstance(r, int) and 0 <= r < len(current_ids):
            idx.add(r)
        elif r in current_ids:
            idx.add(current_ids.index(r))
        else:
            raise PlanError("bad_args", f"not in the plan: {r}")
    if not idx:
        raise PlanError("bad_args", "nothing to replace")
    picks = [store[r] for r in current_ids]
    stuck = []
    rng = random.Random(seed)
    taken = set(current_ids)
    for i in sorted(idx):
        mt = meal_types[i]
        cur_hi = per_meal(picks[i])[1]
        allp = [e for e in with_fallback(exclude, max_minutes, mt) if e["recipe_id"] not in taken]
        pool = [e for e in allp if e["category"] in SLOT_BUCKETS[mt]] or allp
        if cheaper:
            pool = [e for e in pool if per_meal(e)[1] < cur_hi]
        if not pool:
            stuck.append(i)                     # nothing else fits: this meal stays, the answer says so
            continue
        others = {p["category"] for j, p in enumerate(picks) if j != i}
        fresh = [e for e in pool if e["category"] not in others] or pool
        new = rng.choice(fresh)
        taken.add(new["recipe_id"])
        picks[i] = new
    if budget_egp is not None:
        slots_ = [(j // 3 + 1, meal_types[j]) for j in range(len(picks))]
        keep = {j for j in range(len(picks)) if j not in idx}
        pool = {mt: with_fallback(exclude, max_minutes, mt) for mt in set(meal_types)}
        # only replaced meals may change while fitting the budget
        while _totals(picks)[1] > budget_egp:
            used = {p["recipe_id"] for p in picks}
            best = None
            for j in sorted(idx):
                cur = per_meal(picks[j])[1]
                cheaper_ = [e for e in pool[meal_types[j]] if e["recipe_id"] not in used and per_meal(e)[1] < cur]
                if cheaper_ and (best is None or cur > best[0]):
                    best = (cur, j, max(cheaper_, key=lambda e: (per_meal(e)[1], e["recipe_id"])))
            if best is None:
                break
            picks[best[1]] = best[2]
    slots_ = [(j // 3 + 1, meal_types[j]) for j in range(len(picks))]
    tips = [_tip("no_other_recipe",
                 f"مفيش وصفة تانية تناسب الشروط للوجبة رقم {i + 1}، فاتساب زي ما هي.",
                 f"No other recipe fits the limits for meal {i + 1}, so it stays as it was.", position=i) for i in stuck]
    tips += [t for t in _suggestions(picks, slots_, budget_egp, list(exclude), [], max_minutes, None)
             if t["code"] == "budget_too_small"]
    return _result(picks, slots_, budget_egp, lang, tips)
