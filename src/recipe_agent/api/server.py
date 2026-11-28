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

Accounts (login)
  POST   /auth/register     {"username", "password"}      -> {"token", "user_id"}
  POST   /auth/login        {"username", "password"}      -> {"token", "user_id"}
  GET    /auth/me                                         -> {"user_id"}
  POST   /auth/password     {"old_password", "new_password"}
Send the token as "Authorization: Bearer <token>". Set RECIPE_REQUIRE_LOGIN=1 (or create_app(require_login=True))
to make /chat and /users/... need it: the user is then the token's user and any other user id gets 403.
Without it, the old behaviour stays (user_id in the request). Set RECIPE_AUTH_SECRET so tokens survive a restart.
"""
import hmac
import logging
import os
import re
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from recipe_agent.services.accounts import (AccountError, AccountService, auth_secret, issue_token,
                                             verify_token)
from recipe_agent.services.profile import HistoryService, ProfileService, ProfileUpdate

log = logging.getLogger(__name__)
USER_ID = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")
API_VERSION = "1"
_CODES = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found", 409: "conflict",
          422: "invalid_request", 429: "rate_limited", 500: "server_error", 503: "unavailable"}
QUOTA_WORDS = ("429", "resource_exhausted", "quota")


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    user_id: str = "default"


class ChatOut(BaseModel):
    answer: str
    user_id: str


class CodedError(HTTPException):
    """An HTTP error with its own machine-readable code in the error body."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(status, message)
        self.code = code


class RegisterIn(BaseModel):
    email: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=1, max_length=256)


class LoginIn(BaseModel):
    email: str = Field(min_length=1, max_length=254, description="the email (an old username also works)")
    password: str = Field(min_length=1, max_length=256)


class EmailIn(BaseModel):
    email: str = Field(min_length=1, max_length=254)


class TokenIn(BaseModel):
    token: str = Field(min_length=1, max_length=200)


class ResetIn(BaseModel):
    token: str = Field(min_length=1, max_length=200)
    new_password: str = Field(min_length=1, max_length=256)


class PasswordIn(BaseModel):
    old_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=1, max_length=256)


class TokenOut(BaseModel):
    token: Optional[str] = None
    user_id: str
    email: Optional[str] = None
    verified: bool = False
    verification_required: bool = False
    dev_link: Optional[str] = None


_AUTH_STATUS = {"bad_username": 422, "bad_email": 422, "bad_password": 422, "taken": 409, "invalid_credentials": 401,
                "locked": 429, "invalid_token": 400}


class AllergiesIn(BaseModel):
    items: list[str] = Field(min_length=1, max_length=30)


def _uid(user_id: str) -> str:
    if not USER_ID.match(user_id or ""):
        raise HTTPException(422, "user_id must be 1-64 characters: letters, digits, _ . @ -")
    return user_id


def create_app(answer_fn=None, session_factory=None, memory=None, api_key: Optional[str] = None,
               require_login: Optional[bool] = None, secret: Optional[bytes] = None, clock=None,
               mailer=None, require_verified: Optional[bool] = None, dev_mail: Optional[bool] = None) -> FastAPI:
    """answer_fn(text, user_id) -> str. Default: the supervisor (built on first use)."""
    api_key = api_key if api_key is not None else os.getenv("RECIPE_API_KEY")
    if require_login is None:
        require_login = os.getenv("RECIPE_REQUIRE_LOGIN", "").strip().lower() in ("1", "true", "yes")
    secret = secret or auth_secret()
    flag = lambda name: os.getenv(name, "").strip().lower() in ("1", "true", "yes")
    require_verified = flag("RECIPE_REQUIRE_VERIFIED") if require_verified is None else require_verified
    dev_mail = flag("RECIPE_DEV_MAIL") if dev_mail is None else dev_mail
    from recipe_agent.services import mailer as mail_mod
    mailer = mailer or mail_mod.get_mailer()
    import time as _time
    clock = clock or _time.time
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

    def token_user(authorization: Optional[str]) -> Optional[str]:
        """The user id of a valid Bearer token whose account still exists; else None."""
        if not authorization or not authorization.lower().startswith("bearer "):
            return None
        uid = verify_token(authorization[7:].strip(), secret, clock())
        if uid is None:
            return None
        with sf()() as s:
            return uid if AccountService(s, clock).exists(uid) else None

    def who(requested: Optional[str], authorization: Optional[str]) -> str:
        """The user this request is for. With login required it is always the token's user."""
        if not require_login:
            return _uid(requested or "default")
        uid = token_user(authorization)
        if uid is None:
            raise HTTPException(401, "login required: send Authorization: Bearer <token>")
        if requested not in (None, "", "default", uid):
            raise HTTPException(403, "this is not your account")
        return uid

    def auth_call(fn):
        try:
            return fn()
        except AccountError as exc:
            raise CodedError(_AUTH_STATUS.get(exc.code, 400), exc.code, str(exc))

    def send_mail(pair, builder):
        """Send a reset/verify mail for (token, email). Never raises. Returns the link (only used in dev mode)."""
        if not pair:
            return None
        subject, text, link = builder(pair[0])
        try:
            mailer.send(pair[1], subject, text)
        except Exception:
            log.exception("mailer failed")
        return link

    def session_out(svc, uid, token_ok=True):
        verified = svc.is_verified(uid)
        return TokenOut(token=issue_token(uid, secret, clock()) if token_ok else None, user_id=uid,
                        email=svc.email_of(uid), verified=verified,
                        verification_required=bool(require_verified and not verified))

    @app.post("/auth/register", response_model=TokenOut, dependencies=auth)
    def register(body: RegisterIn):
        with sf()() as s:
            svc = AccountService(s, clock)
            uid = auth_call(lambda: svc.register_email(body.email, body.password))
            link = send_mail(svc.request_verification(uid), mail_mod.verify_mail)
            out = session_out(svc, uid, token_ok=not require_verified)
        if dev_mail:
            out.dev_link = link
        return out

    @app.post("/auth/login", response_model=TokenOut, dependencies=auth)
    def login(body: LoginIn):
        with sf()() as s:
            svc = AccountService(s, clock)
            uid = auth_call(lambda: svc.authenticate(body.email, body.password))
            if require_verified and svc.email_of(uid) and not svc.is_verified(uid):
                raise CodedError(403, "email_not_verified", "confirm your email first: open the link we sent you")
            return session_out(svc, uid)

    @app.post("/auth/forgot", dependencies=auth)
    def forgot(body: EmailIn):
        """Always answers the same, so it never shows which emails have an account."""
        with sf()() as s:
            link = send_mail(AccountService(s, clock).request_reset(body.email), mail_mod.reset_mail)
        out = {"sent": True}
        if dev_mail and link:
            out["dev_link"] = link
        return out

    @app.post("/auth/reset", dependencies=auth)
    def reset(body: ResetIn):
        with sf()() as s:
            auth_call(lambda: AccountService(s, clock).reset_password(body.token, body.new_password))
        return {"reset": True}

    @app.post("/auth/verify", dependencies=auth)
    def verify(body: TokenIn):
        with sf()() as s:
            auth_call(lambda: AccountService(s, clock).verify_email(body.token))
        return {"verified": True}

    @app.post("/auth/resend", dependencies=auth)
    def resend(body: EmailIn):
        """Sends the confirmation mail again (same answer whatever the email is)."""
        with sf()() as s:
            svc = AccountService(s, clock)
            uid = svc.user_for_email(body.email)
            link = send_mail(svc.request_verification(uid), mail_mod.verify_mail) if uid else None
        out = {"sent": True}
        if dev_mail and link:
            out["dev_link"] = link
        return out

    @app.get("/auth/me", dependencies=auth)
    def me(authorization: Optional[str] = Header(default=None)):
        uid = token_user(authorization)
        if uid is None:
            raise HTTPException(401, "login required")
        with sf()() as s:
            svc = AccountService(s, clock)
            return {"user_id": uid, "email": svc.email_of(uid), "verified": svc.is_verified(uid)}

    @app.post("/auth/password", dependencies=auth)
    def change_password(body: PasswordIn, authorization: Optional[str] = Header(default=None)):
        uid = token_user(authorization)
        if uid is None:
            raise HTTPException(401, "login required")
        with sf()() as s:
            auth_call(lambda: AccountService(s, clock).change_password(uid, body.old_password, body.new_password))
        return {"changed": True}

    @app.get("/health")
    def health():
        return {"status": "ok", "api_version": API_VERSION}

    @app.post("/chat", response_model=ChatOut, dependencies=auth)
    def chat(body: ChatIn, authorization: Optional[str] = Header(default=None)):
        uid = who(body.user_id, authorization)
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
    def get_profile(user_id: str, s=Depends(db), authorization: Optional[str] = Header(default=None)):
        return ProfileService(s, memory).get(who(user_id, authorization))

    @app.put("/users/{user_id}/profile", dependencies=auth)
    def put_profile(user_id: str, upd: ProfileUpdate, s=Depends(db), authorization: Optional[str] = Header(default=None)):
        try:
            return ProfileService(s, memory).update(who(user_id, authorization), upd)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.delete("/users/{user_id}/profile", dependencies=auth)
    def delete_profile(user_id: str, s=Depends(db), authorization: Optional[str] = Header(default=None)):
        return {"deleted": ProfileService(s, memory).delete_profile(who(user_id, authorization))}

    @app.post("/users/{user_id}/allergies", dependencies=auth)
    def add_allergies(user_id: str, body: AllergiesIn, s=Depends(db), authorization: Optional[str] = Header(default=None)):
        try:
            return {"allergies": ProfileService(s, memory).add_allergies(who(user_id, authorization), body.items)}
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.delete("/users/{user_id}/allergies", dependencies=auth)
    def clear_allergies(user_id: str, s=Depends(db), authorization: Optional[str] = Header(default=None)):
        ProfileService(s, memory).clear_allergies(who(user_id, authorization))
        return {"allergies": []}

    @app.get("/users/{user_id}/history", dependencies=auth)
    def get_history(user_id: str, limit: int = Query(20, ge=1, le=200), s=Depends(db), authorization: Optional[str] = Header(default=None)):
        return {"messages": [m.model_dump() for m in HistoryService(s).recent(who(user_id, authorization), limit)]}

    @app.delete("/users/{user_id}/history", dependencies=auth)
    def clear_history(user_id: str, s=Depends(db), authorization: Optional[str] = Header(default=None)):
        return {"deleted": HistoryService(s).clear(who(user_id, authorization))}

    from .features import build_router
    app.include_router(build_router(who, sf, memory=memory, auth=auth))

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        body = {"error": {"code": _CODES.get(exc.status_code, "error"), "message": str(exc.detail)}}
        if getattr(exc, "code", None):
            body["error"]["code"] = exc.code
        if getattr(exc, "candidates", None):
            body["error"]["candidates"] = exc.candidates
        return JSONResponse(body, status_code=exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        fields = [{"field": ".".join(str(p) for p in e["loc"][1:]) or str(e["loc"][0]), "message": e["msg"]}
                  for e in exc.errors()]
        return JSONResponse({"error": {"code": "invalid_request", "message": "the request is not valid",
                                       "fields": fields}}, status_code=422)

    return app


def get_app() -> FastAPI:
    """For `uvicorn recipe_agent.api:get_app --factory`."""
    return create_app()
