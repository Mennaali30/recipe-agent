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
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from recipe_agent.db.models import Account, AccountEmail, AuthToken, User

USERNAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,31}$")
EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$")
RESET_TTL, VERIFY_TTL, RESEND_GAP = 30 * 60, 24 * 3600, 60
RESERVED = {"default", "admin", "root", "system", "me", "null", "none"}
MIN_PASSWORD, MAX_PASSWORD = 8, 128
MAX_FAILS, LOCK_SECONDS = 5, 300
TOKEN_TTL_SECONDS = 7 * 24 * 3600
_N, _R, _P = 2 ** 14, 8, 1


class AccountError(Exception):
    """code: bad_username | bad_email | bad_password | taken | invalid_credentials | locked | invalid_token"""

    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code


def normalize_username(name: str) -> str:
    return (name or "").strip().lower()


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def _aware(d: datetime) -> datetime:
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _hash_token(raw: str) -> str:
    return hashlib.sha256((raw or "").encode("utf-8")).hexdigest()


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
        name = normalize_email(username) if "@" in (username or "") else normalize_username(username)
        now = self.clock()
        count, until = self.fails.get(name, [0, 0.0])
        if until > now:
            raise AccountError("locked", "too many wrong attempts, try again in a few minutes")
        acc = self._find(name)
        ok = verify_password(password or "", acc.password_hash if acc else _DUMMY) and acc is not None
        if not ok:
            count += 1
            self.fails[name] = [0, now + LOCK_SECONDS] if count >= MAX_FAILS else [count, 0.0]
            raise AccountError("invalid_credentials", "wrong email or password")
        self.fails.pop(name, None)
        acc.last_login = datetime.fromtimestamp(now, timezone.utc)
        self.session.commit()
        return acc.user_id

    def _find(self, identifier: str):
        """The account for an email (contains @) or an old username."""
        if "@" in identifier:
            row = self.session.scalars(select(AccountEmail).where(AccountEmail.email == identifier)).first()
            return self.session.get(Account, row.user_id) if row else None
        return self.session.scalars(select(Account).where(Account.username == identifier)).first()

    # ---- email accounts ----
    def register_email(self, email: str, password: str) -> str:
        mail = normalize_email(email)
        if len(mail) > 254 or not EMAIL.match(mail):
            raise AccountError("bad_email", "enter a valid email address")
        if not (MIN_PASSWORD <= len(password or "") <= MAX_PASSWORD):
            raise AccountError("bad_password", f"password must be {MIN_PASSWORD}-{MAX_PASSWORD} characters")
        if password.strip().lower() == mail:
            raise AccountError("bad_password", "password cannot be the email")
        if self.session.scalars(select(AccountEmail).where(AccountEmail.email == mail)).first() is not None:
            raise AccountError("taken", "this email is already registered")
        uid = "u_" + secrets.token_hex(6)
        try:
            self.session.add(User(id=uid, name=mail.split("@")[0][:60]))
            self.session.flush()
            self.session.add(Account(user_id=uid, username=uid, password_hash=hash_password(password)))
            self.session.add(AccountEmail(user_id=uid, email=mail))
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
            raise AccountError("taken", "this email is already registered")
        return uid

    def email_of(self, user_id: str) -> Optional[str]:
        row = self.session.get(AccountEmail, user_id)
        return row.email if row else None

    def is_verified(self, user_id: str) -> bool:
        row = self.session.get(AccountEmail, user_id)
        return bool(row and row.verified_at)

    def user_for_email(self, email: str) -> Optional[str]:
        row = self.session.scalars(select(AccountEmail).where(AccountEmail.email == normalize_email(email))).first()
        return row.user_id if row else None

    def create_token(self, user_id: str, kind: str, ttl: int, throttle: int = 0) -> Optional[str]:
        """A new single-use token (older unused ones of the same kind stop working). None if one was just made."""
        now = datetime.fromtimestamp(self.clock(), timezone.utc)
        old = self.session.scalars(select(AuthToken).where(
            AuthToken.user_id == user_id, AuthToken.kind == kind, AuthToken.used_at.is_(None))).all()
        if throttle and any(_aware(t.created_at) > now - timedelta(seconds=throttle) for t in old):
            return None
        for t in old:
            t.used_at = now
        raw = secrets.token_urlsafe(32)
        self.session.add(AuthToken(user_id=user_id, kind=kind, token_hash=_hash_token(raw),
                                   expires_at=now + timedelta(seconds=ttl), created_at=now))
        self.session.commit()
        return raw

    def _take_token(self, raw: str, kind: str) -> AuthToken:
        now = datetime.fromtimestamp(self.clock(), timezone.utc)
        t = self.session.scalars(select(AuthToken).where(
            AuthToken.token_hash == _hash_token(raw), AuthToken.kind == kind)).first()
        if t is None or t.used_at is not None or _aware(t.expires_at) <= now:
            raise AccountError("invalid_token", "this link is not valid any more, ask for a new one")
        return t

    def request_reset(self, email: str) -> Optional[tuple]:
        """(raw token, email) if the email has an account and no reset was just sent; else None (caller says nothing)."""
        uid = self.user_for_email(email)
        if uid is None:
            return None
        raw = self.create_token(uid, "reset", RESET_TTL, throttle=RESEND_GAP)
        return (raw, normalize_email(email)) if raw else None

    def reset_password(self, raw: str, new: str) -> str:
        t = self._take_token(raw, "reset")
        if not (MIN_PASSWORD <= len(new or "") <= MAX_PASSWORD):      # checked first: a typo does not burn the link
            raise AccountError("bad_password", f"password must be {MIN_PASSWORD}-{MAX_PASSWORD} characters")
        now = datetime.fromtimestamp(self.clock(), timezone.utc)
        acc = self.session.get(Account, t.user_id)
        acc.password_hash = hash_password(new)
        t.used_at = now
        row = self.session.get(AccountEmail, t.user_id)
        if row and not row.verified_at:
            row.verified_at = now                                    # the person read the mail: the email is theirs
        self.session.commit()
        for k in [k for k in self.fails if k == acc.username or (row and k == row.email)]:
            self.fails.pop(k, None)
        return t.user_id

    def request_verification(self, user_id: str) -> Optional[tuple]:
        row = self.session.get(AccountEmail, user_id)
        if row is None or row.verified_at:
            return None
        raw = self.create_token(user_id, "verify", VERIFY_TTL, throttle=RESEND_GAP)
        return (raw, row.email) if raw else None

    def verify_email(self, raw: str) -> str:
        t = self._take_token(raw, "verify")
        t.used_at = datetime.fromtimestamp(self.clock(), timezone.utc)
        row = self.session.get(AccountEmail, t.user_id)
        if row and not row.verified_at:
            row.verified_at = t.used_at
        self.session.commit()
        return t.user_id

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
