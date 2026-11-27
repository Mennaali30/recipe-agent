"""Shopping list: merge duplicate ingredients, subtract the pantry, group by category, estimate cost.

Only lines with the same ingredient identity AND the same base unit are added together.
A pantry stock in another unit (grams vs pieces) is not subtracted; the item says so.
"""
from typing import Optional

from .schemas import CostLine, Measure, ShoppingItem, ShoppingList, worst_status
from .units import Ingredient, price_of, resolve_ingredient

CATEGORY_KEYS = {
    "produce": "lemon apple pineapple banana lettuce celery artichoke tomato bell_pepper green_pepper garlic onion potato carrot peas eggplant cucumber arugula fresh_herbs spinach zucchini okra molokhia green_beans mallow_khobeiza",
    "meat_poultry": "minced_beef gizzards_offal chicken buffalo_meat beef_generic",
    "seafood": "shrimp tilapia fish_generic tuna",
    "dairy_eggs": "roumi_cheese cream_cheese kiri_cheese koreish_cheese cheese_generic yogurt milk cream eggs butter",
    "grains_legumes": "chickpeas oats_bran_freekeh lentils rice pasta peanuts fava_beans",
    "bakery": "phyllo_pastry breadcrumbs bread flour yeast_baking",
    "spices": "bay_leaf cardamom cumin turmeric coriander_dry paprika_curry saffron_special garlic_powder onion_powder spices_black_pepper salt spices_other",
    "oils_fats": "industrial_ghee ghee oil",
    "condiments_sweets": "bbq_sauce ketchup tomato_paste_sauce chocolate sugar honey_molasses coconut",
    "non_food": "non_food",
}
KEY_CATEGORY = {k: cat for cat, keys in CATEGORY_KEYS.items() for k in keys.split()}
CATEGORY_ORDER = list(CATEGORY_KEYS) + ["other"]


def category_of(key: Optional[str]) -> str:
    return KEY_CATEGORY.get(key, "other")


def build_shopping_list(lines, stock=None) -> ShoppingList:
    """lines: list of CostLine (from costing.cost_line / recipe_cost). stock: dict
    {(identity, unit): quantity} (see PantryService.stock_map) or None."""
    stock = stock or {}
    acc = {}      # (identity, unit) -> dict
    unknown = {}  # identity -> display name (quantity unknown)
    taste = {}
    for l in lines:
        ing_row_key = l.key
        if l.to_taste:
            taste.setdefault(l.identity, (l.name, ing_row_key))
            continue
        m = l.measure
        if m.value is None or m.unit is None:
            unknown.setdefault(l.identity, (l.name, ing_row_key))
            continue
        a = acc.setdefault((l.identity, m.unit),
                           dict(name=l.name, key=ing_row_key, qty=0.0, statuses=[], row_name=l.name))
        a["qty"] += m.value
        a["statuses"].append(m.status)

    items = []
    for (identity, unit), a in acc.items():
        have = float(stock.get((identity, unit), 0.0))
        note = ""
        other = [u for (i, u) in stock if i == identity and u != unit and stock[(i, u)] > 0]
        if other and have == 0:
            note = f"pantry has this in {other[0]}; not subtracted (different unit)"
        need = a["qty"]
        to_buy = max(need - have, 0.0)
        ing = resolve_ingredient(a["name"])
        item = ShoppingItem(name=a["name"], identity=identity, category=category_of(a["key"]),
                            unit=unit, needed=round(need, 2), in_pantry=round(min(have, need), 2),
                            to_buy=round(to_buy, 2),
                            quantity_status=worst_status(a["statuses"]), note=note)
        if to_buy > 0:
            p = price_of(Measure(value=to_buy, unit=unit, status="verified"), ing)
            item.est_cost_egp, item.price_source, item.price_status = p.cost_egp, p.source, p.status
        else:
            item.est_cost_egp, item.price_status = 0.0, "verified"
            item.note = (note + "; " if note else "") + "covered by pantry"
        items.append(item)
    for identity, (name, key) in unknown.items():
        items.append(ShoppingItem(name=name, identity=identity, category=category_of(key),
                                  note="quantity unknown; add manually"))
    for identity, (name, key) in taste.items():
        items.append(ShoppingItem(name=name, identity=identity, category=category_of(key),
                                  to_taste=True, note="to taste; no quantity"))
    items.sort(key=lambda i: (CATEGORY_ORDER.index(i.category), i.name))

    buy = [i for i in items if i.to_buy and i.to_buy > 0]
    priced = [i for i in buy if i.est_cost_egp is not None]
    total = round(sum(i.est_cost_egp for i in priced), 2) if priced else (0.0 if not buy and not unknown else None)
    unpriced = [i.name for i in buy if i.est_cost_egp is None]
    unk = [i.name for i in items if not i.to_taste and i.needed is None]
    complete = total is not None and not unpriced and not unk
    status = worst_status([i.price_status for i in priced]) if priced else ("verified" if complete else "unknown")
    return ShoppingList(items=items, estimated_total_egp=total, total_complete=complete,
                        unpriced=unpriced, unknown_quantities=unk, price_status=status)
