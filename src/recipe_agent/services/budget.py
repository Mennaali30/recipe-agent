"""Simple budget check for a list of recipes."""
from typing import Optional

from .costing import recipe_cost
from .schemas import BudgetCheck, BudgetEntry, RecipeInput, worst_status
from .shopping import build_shopping_list


def check_budget(items, budget_egp: float, stock=None) -> BudgetCheck:
    """items: list of (RecipeInput, meals). Each recipe is scaled to `meals` meals (needs its
    servings). With `stock` (pantry) only what must be bought is counted (basis 'to_buy').

    Verdict: within_budget only if every cost is known and the total fits. over_budget when the
    known part already exceeds the budget (more unpriced items can only add to it). Otherwise
    cannot_tell. Never guesses a missing price."""
    if budget_egp < 0:
        raise ValueError("budget cannot be negative")
    entries, unpriced, statuses, all_lines = [], [], [], []
    complete = True
    total = 0.0
    any_known = False
    for recipe, meals in items:
        if meals <= 0:
            raise ValueError("meals must be positive")
        if not recipe.servings:
            entries.append(BudgetEntry(recipe=recipe.name, meals=meals))
            unpriced.append(f"{recipe.name} (servings unknown)")
            complete = False
            continue
        rc = recipe_cost(recipe, factor=meals / recipe.servings, meals=meals)
        all_lines += rc.lines
        if stock is None:
            entries.append(BudgetEntry(recipe=recipe.name, meals=meals, cost_egp=rc.total_egp,
                                       complete=rc.complete))
            if rc.total_egp is not None:
                total += rc.total_egp
                any_known = True
                statuses.append(rc.status)
            complete = complete and rc.complete
            unpriced += [f"{recipe.name}: {n}" for n in rc.unpriced]
        else:
            entries.append(BudgetEntry(recipe=recipe.name, meals=meals, cost_egp=rc.total_egp,
                                       complete=rc.complete))
            total += rc.extras_egp  # extras are never covered by the pantry
            any_known = any_known or rc.extras_egp > 0
    basis = "full_cost"
    if stock is not None:
        basis = "to_buy"
        sl = build_shopping_list(all_lines, stock)
        if sl.estimated_total_egp is not None:
            total += sl.estimated_total_egp
            any_known = True
            statuses.append(sl.price_status)
        complete = complete and sl.total_complete
        unpriced += sl.unpriced + sl.unknown_quantities
    known = round(total, 2) if any_known else None
    chk = BudgetCheck(budget_egp=budget_egp, basis=basis, known_total_egp=known, complete=complete,
                      entries=entries, unpriced=unpriced,
                      status=worst_status(statuses) if statuses else "unknown")
    if known is not None and known > budget_egp:
        chk.verdict, chk.over_by_egp = "over_budget", round(known - budget_egp, 2)
        chk.note = "already over budget on the known costs" if not complete else ""
    elif complete and known is not None:
        chk.verdict, chk.remaining_egp = "within_budget", round(budget_egp - known, 2)
    else:
        chk.verdict = "cannot_tell"
        if known is not None:
            chk.remaining_egp = round(budget_egp - known, 2)
        chk.note = "some prices are missing, so the real total may be higher than the known total"
    return chk
