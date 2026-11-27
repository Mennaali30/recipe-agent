# Phase 6a: HTTP API

A FastAPI layer over the supervisor. Same answers as the Gradio app; no new logic.

Needs fastapi, uvicorn and httpx. They already come with gradio; if missing: `pip install fastapi uvicorn httpx`.

Run: `python -m uvicorn recipe_agent.api:get_app --factory --app-dir src --port 8000`
Docs page: http://localhost:8000/docs

| Method | Path | Body | Result |
|---|---|---|---|
| GET | /health | | `{"status":"ok"}` |
| POST | /chat | `{"message","user_id"?}` | `{"answer","user_id"}` |
| GET/PUT/DELETE | /users/{id}/profile | PUT: any profile fields | profile (PUT changes only the fields sent) |
| POST/DELETE | /users/{id}/allergies | POST: `{"items":[...]}` | allergies (kept in the memory file) |
| GET/DELETE | /users/{id}/history | GET: `?limit=20` | messages / number deleted |

Errors: 422 bad input, 401 wrong or missing key, 429 Gemini quota used up, 503 model busy, 500 anything else (no internals shown).
Set `RECIPE_API_KEY` to require the header `X-API-Key` (everything except /health). user_id is 1-64 characters: letters, digits, `_ . @ -`.
The key is one shared secret: whoever holds it can act as any user_id. Real per-user login comes later.
