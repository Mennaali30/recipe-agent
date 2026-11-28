# Matbakhak AI: HTTP API (version 1)

Run: `uvicorn recipe_agent.api:get_app --factory --port 8000`. Interactive docs: `/docs` (Swagger) and `/openapi.json`.
JSON in, JSON out. Money is EGP. Every number comes from code and data, never from the language model; only `POST /chat` uses Gemini.

## Setup flags (environment)
| Variable | Meaning |
|---|---|
| `RECIPE_REQUIRE_LOGIN=1` | `/chat`, `/pantry...`, `/recipes...`, `/users/...` and the other feature endpoints need a login. Use this for the UI. |
| `RECIPE_AUTH_SECRET` | Secret that signs login tokens. Set it, or every token stops working when the server restarts. |
| `RECIPE_API_KEY` | Optional. If set, every call except `/health` also needs the header `X-API-Key`. |
| `RECIPE_APPROVAL_TOOLS` | Which chat actions ask "yes/no" first. Default `pantry_discard,pantry_adjust,profile_clear`; `none` turns it off. |
| `RECIPE_AGENT_DB_URL` | Database URL (SQLite by default). |

## Login
`POST /auth/register` and `POST /auth/login` take `{"username": "sara", "password": "..."}` and return `{"token": "...", "user_id": "sara"}`.
Username: 3-32 characters `a-z 0-9 _ . -` (lower case). Password: 8-128 characters. 5 wrong passwords lock that username for 5 minutes (429).
Send the token on every other call: `Authorization: Bearer <token>`. It lasts 7 days; the app logs out by forgetting it.
`GET /auth/me` returns `{"user_id"}`. `POST /auth/password` takes `{"old_password","new_password"}`.
With login required, the user is always the token's user. A request for another user's id gets 403.

## Errors (always this shape)
```json
{"error": {"code": "invalid_request", "message": "the request is not valid", "fields": [{"field": "items", "message": "..."}]}}
```
| HTTP | code | When |
|---|---|---|
| 401 | unauthorized | no token, bad or expired token, wrong password, missing API key |
| 403 | forbidden | another user's data |
| 404 | not_found | unknown recipe or nothing pending |
| 409 | conflict | not enough stock, name matches several recipes (`error.candidates` lists them), recipe blocked by saved allergies, username taken |
| 422 | invalid_request | the body or a parameter is wrong (`error.fields` says which); unknown unit |
| 429 | rate_limited | too many wrong passwords, or the Gemini quota is used up (`/chat`) |
| 503 | unavailable | the model is busy (`/chat`), or saved allergies could not be read |
| 500 | server_error | unexpected |

## Endpoints
### Chat
`POST /chat` `{"message": "عايزة اعمل كشري"}` -> `{"answer": "...", "user_id": "sara"}`. The assistant may answer with a question
("confirm?") for sensitive actions; the next message answers it, or use the pending endpoints below. Without login required you may also send `"user_id"`.

### Pending confirmations (buttons for the chat's "yes / no")
`GET /pending` -> `{"pending": null}` or `{"pending": {"tool": "pantry_discard", "summary_ar": "...", "summary_en": "..."}}` (one per user, expires after 10 minutes).
`POST /pending/confirm` -> `{"done": true, "tool": "..."}` (404 if nothing is waiting). `POST /pending/cancel` -> `{"cancelled": true|false}`.

### Profile, allergies, history
`GET|PUT|DELETE /users/{user_id}/profile` (PUT changes only the fields sent). `POST /users/{user_id}/allergies` `{"items": ["milk"]}`, `DELETE` clears.
`GET /users/{user_id}/history?limit=20`, `DELETE` clears. `user_id` must be your own.
Saved allergies are applied to every recipe endpoint below. They are never relaxed.

### Pantry
Quantities are stored in a base unit (g, ml, or a count unit); `"unit": "kg"` is converted.
| Call | Body / query | Result |
|---|---|---|
| `GET /pantry` | | list of `{name, quantity, unit, expiry, status, ...}` |
| `POST /pantry/items` | `{"items": [{"name": "rice", "quantity": 2, "unit": "kg", "expiry": "2026-12-30", "low_stock_threshold": 500}]}` | updated rows |
| `POST /pantry/use` | `{"items": [{"name": "rice", "quantity": 500, "unit": "g"}]}` | updated rows; 409 `conflict` if not enough (nothing is changed) |
| `PUT /pantry/items` | `{"items": [{"name": "rice", "quantity": 1, "unit": "kg"}]}` | sets the exact amount |
| `POST /pantry/discard` | `{"items": [{"name": "rice"}]}` | throws away all of the item |
| `GET /pantry/expiring?within_days=3` | | `{"expired": [...], "expiring_soon": [...], "within_days": 3}` |
| `GET /pantry/low-stock` | | items at or below their threshold |

### Recipes
| Call | Notes |
|---|---|
| `GET /recipes/search?ingredients=رز,فراخ&max_minutes=45&exclude=milk&limit=5&lang=ar` | `{"recipes": [...], "applied_exclusions": [...], "international_available": true}`. Egyptian recipes have `cost_text` and `egyptian: true`; international ones have no EGP cost. |
| `GET /recipes/by-name?q=كشري&lang=ar` | Dish name. `{"recipes", "egyptian", "blocked_by_allergy", "note"}`. If there is no Egyptian recipe, `egyptian` is false, `note` is `no_egyptian_recipe` and `recipes` holds close international ones (the UI should say so first). |
| `GET /recipes/{recipe_id}?lang=ar` | Egyptian recipe (`egy:fah-096`): ingredients with amounts and price, steps, nutrition, `cost_text`, `cost` (`low_egp`, `high_egp`, `price_label`, `coverage`, `unpriced`). 409 if it contains one of your saved allergens. |
A cost is a range with a label (market / approx / rough). When `coverage` is below 1 the cost is partial: `unpriced` names what is missing. Show it as "at least".

### Cost, units, shopping, budget, business
A recipe is `{"recipe_id": "egy:fah-096"}` or `{"name": "كشري"}` (409 with `candidates` if the name matches several) or your own `{"name": "x", "ingredients": [{"name": "rice", "quantity": 200, "unit": "g"}]}`.
| Call | Body |
|---|---|
| `POST /cost` | `{"recipe": R, "extra_expenses": [{"name": "gas", "amount_egp": 5}]}` |
| `POST /convert` | `{"quantity": 2, "unit": "cup", "ingredient": "rice"}` |
| `POST /shopping` | `{"recipes": [{"recipe": R, "meals": 2}], "use_pantry": true}` |
| `POST /budget/check` | `{"items": [{"recipe": R, "meals": 2}], "budget_egp": 300, "use_pantry": false}` -> verdict `within_budget` / `over_budget` / `unknown` (unknown = some prices missing) |
| `POST /business/production-plan` | `{"recipe": R, "meals": 20, "selling_price_per_meal": 60, "extra_expenses": [], "target_margin_pct": 30}` -> cost, revenue, profit, margin, suggested price |
Prices that are not known stay unknown: they are listed, never guessed.

## Not in version 1 yet
Meal plans with budget and replanning, Business Mode beyond the production plan, saved shopping lists, recipe filters. They will be added as new endpoints; existing ones will not change shape.
