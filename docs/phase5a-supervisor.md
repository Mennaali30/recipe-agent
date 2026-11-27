# Phase 5a: supervisor and specialist agents

New package `src/recipe_agent/multi/`. Nothing existing was changed: the Gradio app and the old
pipeline still work as before. Phase 5a adds a second entry point:

```python
from recipe_agent.multi import run_supervisor
out = run_supervisor("تكلفة فراخ معصجة", user_id="waad")
print(out["answer"])
```

## How it works

```
message -> route (LLM, keyword fallback) -> agent 1 [-> agent 2] -> finalize
```

| Agent | Does | Tools it may call |
|---|---|---|
| recipes | Delegates to the existing pipeline (allergies, memory, Egyptian recipes, verifier) | none (calls `graph.answer`) |
| pantry | add / use / list stock, expiring, low stock | pantry_add, pantry_use, pantry_list, pantry_expiring, pantry_low_stock |
| cost | recipe cost, cost per serving, extras, unit conversion | recipe_cost, convert_units |
| shopping | merged shopping list minus pantry, grouped, priced | shopping_list |
| business | scale to N meals, cost per meal, revenue, profit, margin, minimum price | production_plan |
| budget | does it fit a budget in EGP | budget_check |

Rules the code enforces:

- The LLM only turns the message into typed arguments (and picks the agent). Every number in an
  answer comes from the Phase 4 services. Answers are written by code, in Arabic or English.
- Each agent can only call its own tools. Tool arguments are validated (Pydantic); a tool never
  raises, it returns `ok=False` with an error code.
- Missing information is asked for, not guessed (which recipe? how many meals? what budget?).
  An ambiguous recipe name lists the choices.
- Incomplete cost stays visible: unpriced items are listed, the total says "incomplete", and profit,
  margin and suggested price are not shown when the cost is incomplete.
- `pantry_use` is all or nothing: if any item is short, nothing is taken.
- If the router LLM fails, a keyword router is used. If an agent's LLM call fails, the user gets a
  plain "try again" message. If an agent needs input or fails, the chain stops.
- Pantry rows are per `user_id`.

## LLM calls

Router: 1 call. Each specialist (not recipes): 1 call. The recipes agent adds whatever the old
pipeline already used. A cost question is 2 calls in total. Tests use a fake LLM (no network).

## Not in 5a

Users/profiles and chat history (5b), meal calendar agent, Business Mode data (menu, products),
Gradio/FastAPI wiring of `run_supervisor`. The structured-output schemas have only been checked
offline: the first live Gemini run is the real test of them.
