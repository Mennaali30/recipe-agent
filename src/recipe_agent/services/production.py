"""Production planning for N meals of one recipe: scaling, cost per meal, revenue, margin."""
from typing import Optional

from .costing import cost_line, extras_total, summarize_lines
from .schemas import ExtraExpense, ProductionPlan, RecipeInput
from .shopping import build_shopping_list


def min_price_for_margin(cost_per_meal: float, target_margin_pct: float) -> float:
    """Lowest selling price that gives the target gross margin (percent of the price)."""
    if not 0 <= target_margin_pct < 100:
        raise ValueError("target margin must be in [0, 100)")
    return round(cost_per_meal / (1 - target_margin_pct / 100), 2)


def plan_production(recipe: RecipeInput, meals: int, selling_price_per_meal: Optional[float] = None,
                    extra_expenses=None, stock=None) -> ProductionPlan:
    """Plan `meals` meals. The recipe is scaled by meals / recipe.servings. If the recipe has no
    servings the scale is unknown and no quantity or cost is produced."""
    if meals <= 0:
        raise ValueError("meals must be positive")
    if selling_price_per_meal is not None and selling_price_per_meal < 0:
        raise ValueError("selling price cannot be negative")
    extras = list(recipe.extra_expenses) + list(extra_expenses or [])
    plan = ProductionPlan(recipe=recipe.name, meals=meals, recipe_servings=recipe.servings,
                          selling_price_per_meal_egp=selling_price_per_meal)
    if not recipe.servings:
        plan.notes.append("recipe servings unknown; cannot scale or cost it. Enter the servings.")
        return plan
    factor = meals / recipe.servings
    plan.scale_factor = round(factor, 4)
    lines = [cost_line(i, factor)[0] for i in recipe.ingredients]
    plan.scaled_lines = lines
    plan.missing_quantities = [l.name for l in lines if l.to_taste or l.measure.value is None]
    cost, unpriced, _cov, status, complete = summarize_lines(lines)
    extra_total, notes = extras_total(extras, meals)
    plan.notes += notes
    plan.extras_egp = extra_total
    plan.ingredient_cost_egp = cost
    plan.unpriced = unpriced
    plan.status = status
    plan.cost_complete = complete and not notes
    if cost is not None:
        plan.total_cost_egp = round(cost + extra_total, 2)
        plan.cost_per_meal_egp = round(plan.total_cost_egp / meals, 2)
    if unpriced:
        plan.notes.append("some ingredients have no price; cost covers the priced ones only")
    if plan.missing_quantities:
        plan.notes.append("quantity not available for: " + ", ".join(plan.missing_quantities))
    if stock is not None:
        plan.shopping = build_shopping_list(lines, stock)
    if selling_price_per_meal is not None:
        plan.revenue_egp = round(selling_price_per_meal * meals, 2)
        if plan.cost_complete:
            plan.gross_profit_egp = round(plan.revenue_egp - plan.total_cost_egp, 2)
            plan.margin_pct = (round(plan.gross_profit_egp / plan.revenue_egp * 100, 1)
                               if plan.revenue_egp > 0 else None)
        else:
            plan.notes.append("profit and margin not given: cost is incomplete, so they would be too high")
    return plan
