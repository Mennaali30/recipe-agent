"""Recipe cost and cost per serving, with extra expenses. Same numbers as egypt_cost.

Prices that are not in the price map stay unavailable (None); the total then only covers the
priced lines and `complete` is False.
"""
from typing import Optional

from ..egypt_parse import parse_servings
from ..price_range import make_range
from .schemas import (CostLine, ExtraExpense, IngredientInput, Measure, RecipeCost, RecipeInput,
                      worst_status)
from .units import (Ingredient, convert, convert_text, price_of, resolve_ingredient)


def recipe_from_fifi(entry: dict, extra_expenses=None) -> RecipeInput:
    """Build a RecipeInput from a fifi recipe file or an egypt_store entry."""
    raw = entry.get("raw", entry)
    rec = raw.get("recipe", raw)
    ings = [IngredientInput(name=m.get("name") or "", amount_text=m.get("standardAmount"))
            for m in rec.get("masterIngredients", [])]
    return RecipeInput(name=rec.get("title") or rec.get("id") or "recipe",
                       servings=parse_servings(rec.get("servings")),
                       ingredients=ings, extra_expenses=list(extra_expenses or []))


def cost_line(ing_in: IngredientInput, factor: float = 1.0) -> tuple:
    """Return (CostLine, Ingredient). `factor` scales the quantity (recipe scaling)."""
    ing = resolve_ingredient(ing_in.name)
    to_taste = ing_in.to_taste
    if ing_in.quantity is not None and not to_taste:
        measure = convert(ing_in.quantity, ing_in.unit, ing)
    elif ing_in.amount_text and not to_taste:
        measure, to_taste = convert_text(ing_in.amount_text, ing)
    else:
        measure = Measure(status="unknown", note="to taste" if to_taste else "quantity missing")
    measure = measure.scaled(factor)
    line = CostLine(name=ing_in.name, key=ing.key, identity=ing.identity,
                    amount_text=ing_in.amount_text, to_taste=to_taste, measure=measure,
                    note=measure.note)
    if not to_taste:
        p = price_of(measure, ing)
        line.cost_egp, line.price_egp_per_kg = p.cost_egp, p.price_egp_per_kg
        line.price_source, line.price_status = p.source, p.status
        line.price_reliability, line.price_date = p.reliability, p.price_date
        line.status = (worst_status([measure.status, p.status])
                       if p.cost_egp is not None else "unknown")
    return line, ing


def extras_total(extras, meals: Optional[float]) -> tuple:
    """(total, notes). A per_meal expense needs the number of meals; without it it is skipped."""
    total, notes = 0.0, []
    for e in extras:
        if e.basis == "fixed":
            total += e.amount_egp
        elif meals:
            total += e.amount_egp * meals
        else:
            notes.append(f"extra expense '{e.name}' is per meal but the number of meals is unknown; not counted")
    return round(total, 2), notes


def summarize_lines(lines):
    counted = [l for l in lines if not l.to_taste]
    priced = [l for l in counted if l.cost_egp is not None]
    cost = round(sum(l.cost_egp for l in priced), 2) if priced else None
    unpriced = [l.name for l in counted if l.cost_egp is None]
    coverage = round(len(priced) / len(counted), 2) if counted else None
    status = worst_status([l.status for l in priced]) if priced else "unknown"
    return cost, unpriced, coverage, status, bool(counted) and not unpriced


def recipe_cost(recipe: RecipeInput, factor: float = 1.0,
                meals: Optional[float] = None) -> RecipeCost:
    """Cost of one batch of the recipe (as written, or scaled by `factor`).

    `meals` = how many meals that batch makes (default: the recipe's servings * factor)."""
    lines = [cost_line(i, factor)[0] for i in recipe.ingredients]
    cost, unpriced, coverage, status, complete = summarize_lines(lines)
    if meals is None and recipe.servings:
        meals = recipe.servings * factor
    extras, notes = extras_total(recipe.extra_expenses, meals)
    total = None
    if cost is not None:
        total = round(cost + extras, 2)
    elif extras > 0 and not [l for l in lines if not l.to_taste]:
        total = extras
    per = round(total / meals, 2) if (total is not None and meals) else None
    if not meals:
        notes.append("servings unknown; cost per serving not available")
    if unpriced:
        notes.append("some ingredients have no price; total covers the priced ones only")
    rng = make_range([{"cost": l.cost_egp, "source": l.price_source, "reliability": l.price_reliability,
                       "price_date": l.price_date} for l in lines if not l.to_taste], fixed=extras) if total is not None else None
    return RecipeCost(recipe=recipe.name, servings=meals, lines=lines, ingredient_cost_egp=cost,
                      low_egp=rng["low"] if rng else None, high_egp=rng["high"] if rng else None,
                      per_serving_low_egp=round(rng["low"] / meals, 2) if (rng and meals) else None,
                      per_serving_high_egp=round(rng["high"] / meals, 2) if (rng and meals) else None,
                      price_label=rng["label_key"] if rng else None, stale_prices=rng["stale"] if rng else False,
                      extras_egp=extras, total_egp=total, per_serving_egp=per,
                      complete=complete and (meals is not None or not any(
                          e.basis == "per_meal" for e in recipe.extra_expenses)),
                      coverage=coverage, unpriced=unpriced, status=status, notes=notes)
