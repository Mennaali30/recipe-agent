# Phase 4: deterministic services

Scope: the calculation layer only. No LLM, no network, not wired into the LangGraph app yet (Phase 5 does that).

## Modules

| Module | What it does |
| --- | --- |
| `services/schemas.py` | Pydantic models: inputs (`RecipeInput`, `IngredientInput`, `ExtraExpense`) and results (`RecipeCost`, `ShoppingList`, `ProductionPlan`, `BudgetCheck`). |
| `services/units.py` | `convert`, `convert_text`, `resolve_ingredient`, `price_of`. Base dimensions: g, ml, count. |
| `services/costing.py` | `recipe_cost`, `recipe_from_fifi`, `cost_line`. |
| `services/pantry.py` | `PantryService(session)`: add/use/adjust/discard, expired, expiring_soon, low_stock, stock_map, transactions. |
| `services/shopping.py` | `build_shopping_list(lines, stock)`. |
| `services/production.py` | `plan_production`, `min_price_for_margin`. |
| `services/budget.py` | `check_budget`. |
| `db/models.py`, `db/session.py` | `User`, `PantryItem`, `InventoryTransaction`; `make_engine`, `init_db`. |

## Status words

`verified`, `estimated`, `unknown` on measures, prices, lines and totals. A total takes the weakest status of the lines it is built from. `price_source` keeps the original `wfp` / `current_web` / `estimated` label from the price map.

## Identity of an ingredient

An ingredient is matched to a price-map group (Arabic keywords, or the English phrases already used by `egypt_store`). Two lines are the same ingredient when the group key is the same. For broad groups (`spices_other`, `fresh_herbs`, `cheese_generic`, `beef_generic`, `fish_generic`, `non_food`) the cleaned name is used instead, so different spices are not merged. Ingredients not in the price map are matched by cleaned name.

## Known limits

- The price map has about 89 groups and mostly rough or national prices. Prices are approximate. `price_source` says how far to trust each one.
- Most densities in the price map are blank, so millilitres of a mapped ingredient are converted at 1 g/ml (noted on the line, status `estimated`).
- The pantry keeps one stock row per ingredient and unit. It keeps the earliest expiry of what was added, not one expiry per batch.
- No unit conversion between count and weight when the price map has no weight for the item (kept as a count, not costed).
- Home-food business costs besides ingredients (gas, labour, packaging) are entered by the user as `ExtraExpense`. Nothing is assumed.
- Allergy checks are not part of this layer. They stay in `allergy.py` / `verifier.py`, and are never a guarantee.
- Sizes of fifi recipes are the servings written in the recipe; if missing, the recipe cannot be scaled.
