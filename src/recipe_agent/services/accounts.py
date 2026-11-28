"""Accounts: register, log in, signed tokens (standard library only, no network).

- Username: 3-32 characters, a-z 0-9 _ . - (stored lower case). It is also the user id, so every
  existing service (pantry, profile, history, allergies) keeps working per account.
- A username that already has data (an old user id such as "waad" or "default") cannot be registered:
  nobody can take over someone else's old data.
- Passwords: 8-128 characters, stored as a salted scrypt hash. Never logged, never returned.
- Too many wrong passwords for one username locks it for a few minutes.
- Token: "payload.signature" (HMAC-SHA256), carries the user id and an expiry (7 days). It cannot be
  changed without the secret. There is no server-side logout list: a token lives until it expires.
"""
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from recipe_agent.db.models import Account, User

USERNAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,31}$")
RESERVED = {"default", "admin", "root", "system", "me", "null", "none"}
MIN_PASSWORD, MAX_PASSWORD = 8, 128
MAX_FAILS, LOCK_SECONDS = 5, 300
TOKEN_TTL_SECONDS = 7 * 24 * 3600
_N, _R, _P = 2 ** 14, 8, 1


class AccountError(Exception):
    """code: bad_username | bad_password | taken | invalid_credentials | locked"""

    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code


def normalize_username(name: str) -> str:
    return (name or "").strip().lower()


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return "scrypt$%d$%d$%d$%s$%s" % (_N, _R, _P, base64.b64encode(salt).decode(), base64.b64encode(h).decode())


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, h = stored.split("$")
        want = base64.b64decode(h)
        got = hashlib.scrypt(password.encode("utf-8"), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
                             dklen=len(want))
        return hmac.compare_digest(got, want)
    except Exception:
        return False


_DUMMY = hash_password("not-a-real-password")   # so an unknown username costs the same time as a wrong password


class AccountService:
    def __init__(self, session: Session, clock=time.time, fails=None):
        self.session, self.clock = session, clock
        self.fails = fails if fails is not None else _FAILS      # {username: [count, locked_until]}

    # ---- register / login ----
    def register(self, username: str, password: str) -> str:
        name = normalize_username(username)
        if not USERNAME.match(name) or name in RESERVED:
            raise AccountError("bad_username", "username must be 3-32 characters: a-z, 0-9, _ . -")
        if not (MIN_PASSWORD <= len(password or "") <= MAX_PASSWORD):
            raise AccountError("bad_password", f"password must be {MIN_PASSWORD}-{MAX_PASSWORD} characters")
        if password.strip().lower() == name:
            raise AccountError("bad_password", "password cannot be the username")
        if self.session.get(User, name) is not None or self.session.scalars(
                select(Account).where(Account.username == name)).first() is not None:
            raise AccountError("taken", "this username is not available")
        try:
            self.session.add(User(id=name, name=name))
            self.session.flush()
            self.session.add(Account(user_id=name, username=name, password_hash=hash_password(password)))
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
            raise AccountError("taken", "this username is not available")
        return name

    def authenticate(self, username: str, password: str) -> str:
        name = normalize_username(username)
        now = self.clock()
        count, until = self.fails.get(name, [0, 0.0])
        if until > now:
            raise AccountError("locked", "too many wrong attempts, try again in a few minutes")
        acc = self.session.scalars(select(Account).where(Account.username == name)).first()
        ok = verify_password(password or "", acc.password_hash if acc else _DUMMY) and acc is not None
        if not ok:
            count += 1
            self.fails[name] = [0, now + LOCK_SECONDS] if count >= MAX_FAILS else [count, 0.0]
            raise AccountError("invalid_credentials", "wrong username or password")
        self.fails.pop(name, None)
        acc.last_login = datetime.fromtimestamp(now, timezone.utc)
        self.session.commit()
        return acc.user_id

    def change_password(self, user_id: str, old: str, new: str) -> None:
        acc = self.session.get(Account, user_id)
        if acc is None or not verify_password(old or "", acc.password_hash):
            raise AccountError("invalid_credentials", "wrong password")
        if not (MIN_PASSWORD <= len(new or "") <= MAX_PASSWORD):
            raise AccountError("bad_password", f"password must be {MIN_PASSWORD}-{MAX_PASSWORD} characters")
        acc.password_hash = hash_password(new)
        self.session.commit()

    def exists(self, user_id: str) -> bool:
        return self.session.get(Account, user_id) is not None


_FAILS: dict = {}


# ---- tokens ----
def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def auth_secret() -> bytes:
    """RECIPE_AUTH_SECRET (set it for real use, so tokens survive a restart); else a random one per run."""
    global _RUN_SECRET
    env = os.getenv("RECIPE_AUTH_SECRET")
    if env:
        return env.encode("utf-8")
    if _RUN_SECRET is None:
        _RUN_SECRET = secrets.token_bytes(32)
    return _RUN_SECRET


_RUN_SECRET = None


def issue_token(user_id: str, secret: bytes, now: Optional[float] = None, ttl: int = TOKEN_TTL_SECONDS) -> str:
    now = time.time() if now is None else now
    body = _b64(json.dumps({"u": user_id, "exp": int(now + ttl)}, separators=(",", ":")).encode())
    return body + "." + _b64(hmac.new(secret, body.encode(), hashlib.sha256).digest())


def verify_token(token: str, secret: bytes, now: Optional[float] = None) -> Optional[str]:
    """The user id in a valid, unexpired token; None otherwise."""
    try:
        body, sig = (token or "").split(".")
        if not hmac.compare_digest(_b64(hmac.new(secret, body.encode(), hashlib.sha256).digest()), sig):
            return None
        data = json.loads(_unb64(body))
        now = time.time() if now is None else now
        return data["u"] if int(data["exp"]) > now and isinstance(data["u"], str) else None
    except Exception:
        return None
