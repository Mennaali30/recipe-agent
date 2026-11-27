"""Phase 6a: FastAPI layer over the supervisor. No new logic: every answer, number and allergy rule
comes from the supervisor and the Phase 4/5b services. Everything external is injectable for tests.

Endpoints
  GET    /health
  POST   /chat                         {"message": "...", "user_id": "waad"}  -> {"answer", "user_id"}
  GET    /users/{id}/profile
  PUT    /users/{id}/profile           only the fields sent are changed
  DELETE /users/{id}/profile
  POST   /users/{id}/allergies         {"items": ["milk"]}   (saved in the memory file)
  DELETE /users/{id}/allergies
  GET    /users/{id}/history?limit=20
  DELETE /users/{id}/history

Set RECIPE_API_KEY to require the header X-API-Key on everything except /health.
"""
import hmac
import logging
import os
import re
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from recipe_agent.services.profile import HistoryService, ProfileService, ProfileUpdate

log = logging.getLogger(__name__)
USER_ID = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")
QUOTA_WORDS = ("429", "resource_exhausted", "quota")


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    user_id: str = "default"


class ChatOut(BaseModel):
    answer: str
    user_id: str


class AllergiesIn(BaseModel):
    items: list[str] = Field(min_length=1, max_length=30)


def _uid(user_id: str) -> str:
    if not USER_ID.match(user_id or ""):
        raise HTTPException(422, "user_id must be 1-64 characters: letters, digits, _ . @ -")
    return user_id


def create_app(answer_fn=None, session_factory=None, memory=None, api_key: Optional[str] = None) -> FastAPI:
    """answer_fn(text, user_id) -> str. Default: the supervisor (built on first use)."""
    api_key = api_key if api_key is not None else os.getenv("RECIPE_API_KEY")
    app = FastAPI(title="Matbakhak AI API", version="6a")
    state = {"sf": session_factory}

    def sf():
        if state["sf"] is None:
            from recipe_agent.db import init_db, make_engine, make_session_factory
            engine = make_engine()
            init_db(engine)
            state["sf"] = make_session_factory(engine)
        return state["sf"]

    def check_key(x_api_key: Optional[str] = Header(default=None)):
        if api_key and not (x_api_key and hmac.compare_digest(x_api_key, api_key)):
            raise HTTPException(401, "missing or wrong X-API-Key")

    def db():
        s = sf()()
        try:
            yield s
        finally:
            s.close()

    auth = [Depends(check_key)]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatOut, dependencies=auth)
    def chat(body: ChatIn):
        uid = _uid(body.user_id)
        fn = answer_fn
        if fn is None:
            from recipe_agent.multi import app as supervisor_app
            fn = supervisor_app.answer
        try:
            return ChatOut(answer=fn(body.message, uid), user_id=uid)
        except Exception as exc:
            log.exception("chat failed")
            from recipe_agent.llm_call import is_transient
            if is_transient(exc):
                low = str(exc).lower()
                if any(w in low for w in QUOTA_WORDS):
                    raise HTTPException(429, "Gemini free quota is used up for now. Try again later.")
                raise HTTPException(503, "The model is busy. Try again in a moment.")
            raise HTTPException(500, "Could not answer right now.")

    @app.get("/users/{user_id}/profile", dependencies=auth)
    def get_profile(user_id: str, s=Depends(db)):
        return ProfileService(s, memory).get(_uid(user_id))

    @app.put("/users/{user_id}/profile", dependencies=auth)
    def put_profile(user_id: str, upd: ProfileUpdate, s=Depends(db)):
        try:
            return ProfileService(s, memory).update(_uid(user_id), upd)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.delete("/users/{user_id}/profile", dependencies=auth)
    def delete_profile(user_id: str, s=Depends(db)):
        return {"deleted": ProfileService(s, memory).delete_profile(_uid(user_id))}

    @app.post("/users/{user_id}/allergies", dependencies=auth)
    def add_allergies(user_id: str, body: AllergiesIn, s=Depends(db)):
        try:
            return {"allergies": ProfileService(s, memory).add_allergies(_uid(user_id), body.items)}
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.delete("/users/{user_id}/allergies", dependencies=auth)
    def clear_allergies(user_id: str, s=Depends(db)):
        ProfileService(s, memory).clear_allergies(_uid(user_id))
        return {"allergies": []}

    @app.get("/users/{user_id}/history", dependencies=auth)
    def get_history(user_id: str, limit: int = Query(20, ge=1, le=200), s=Depends(db)):
        return {"messages": [m.model_dump() for m in HistoryService(s).recent(_uid(user_id), limit)]}

    @app.delete("/users/{user_id}/history", dependencies=auth)
    def clear_history(user_id: str, s=Depends(db)):
        return {"deleted": HistoryService(s).clear(_uid(user_id))}

    return app


def get_app() -> FastAPI:
    """For `uvicorn recipe_agent.api:get_app --factory`."""
    return create_app()
