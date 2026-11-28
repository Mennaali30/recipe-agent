"""Email accounts: register, login, password reset, email verification. Offline: fake mailer, fake clock."""
import re

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from recipe_agent.api import create_app
from recipe_agent.db import init_db, make_engine, make_session_factory
from recipe_agent.services import AccountError, AccountService
from recipe_agent.services import mailer as mail_mod

SECRET = b"test-secret"
PW = "correct-horse"


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeMailer:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def send(self, to, subject, body):
        if self.fail:
            raise RuntimeError("smtp down")
        self.sent.append((to, subject, body))
        return True

    def token(self, i=-1):
        return re.search(r"token=([\w-]+)", self.sent[i][2]).group(1)


@pytest.fixture
def sf():
    engine = make_engine("sqlite://")
    init_db(engine)
    return make_session_factory(engine)


def make(sf, **kw):
    mailer, clock = FakeMailer(kw.pop("fail", False)), kw.pop("clock", None) or Clock()
    app = create_app(answer_fn=lambda t, u: "x", session_factory=sf, api_key="", require_login=True, secret=SECRET,
                     clock=clock, mailer=mailer, **kw)
    return TestClient(app), mailer, clock


def register(c, email="sara@example.com", pw=PW):
    return c.post("/auth/register", json={"email": email, "password": pw})


@pytest.mark.parametrize("email", ["", "sara", "sara@", "@example.com", "sara@example", "a b@example.com", "x" * 250 + "@e.com"])
def test_bad_email_is_refused(sf, email):
    c, _, _ = make(sf)
    r = c.post("/auth/register", json={"email": email or " ", "password": PW})
    assert r.status_code == 422 and r.json()["error"]["code"] in ("bad_email", "invalid_request")


def test_email_is_case_insensitive_and_unique(sf):
    c, _, _ = make(sf)
    assert register(c, "Sara@Example.com").status_code == 200
    r = register(c, "sara@example.COM")
    assert r.status_code == 409 and r.json()["error"]["code"] == "taken"


def test_register_sends_a_verification_mail_and_verification_is_optional(sf):
    c, mailer, _ = make(sf)
    r = register(c)
    assert r.json()["token"] and r.json()["verified"] is False and r.json()["verification_required"] is False
    assert len(mailer.sent) == 1 and mailer.sent[0][0] == "sara@example.com" and "verify-email?token=" in mailer.sent[0][2]
    assert c.post("/auth/verify", json={"token": mailer.token()}).json() == {"verified": True}
    h = {"Authorization": "Bearer " + r.json()["token"]}
    assert c.get("/auth/me", headers=h).json()["verified"] is True
    assert c.post("/auth/verify", json={"token": mailer.token()}).status_code == 400      # single use


def test_required_verification_blocks_login_until_confirmed(sf):
    c, mailer, _ = make(sf, require_verified=True)
    r = register(c)
    assert r.status_code == 200 and r.json()["token"] is None and r.json()["verification_required"] is True
    bad = c.post("/auth/login", json={"email": "sara@example.com", "password": PW})
    assert bad.status_code == 403 and bad.json()["error"]["code"] == "email_not_verified"
    c.post("/auth/verify", json={"token": mailer.token()})
    assert c.post("/auth/login", json={"email": "sara@example.com", "password": PW}).json()["token"]


def test_resend_gives_a_new_link_after_a_minute_and_the_old_one_stops(sf):
    c, mailer, clock = make(sf)
    register(c)
    old = mailer.token()
    assert c.post("/auth/resend", json={"email": "sara@example.com"}).json() == {"sent": True}
    assert len(mailer.sent) == 1                                   # throttled: one per minute
    clock.t += 61
    c.post("/auth/resend", json={"email": "sara@example.com"})
    assert len(mailer.sent) == 2 and mailer.token() != old
    assert c.post("/auth/verify", json={"token": old}).status_code == 400
    assert c.post("/auth/verify", json={"token": mailer.token()}).status_code == 200


def test_forgot_answers_the_same_for_unknown_emails(sf):
    c, mailer, _ = make(sf)
    register(c)
    n = len(mailer.sent)
    a = c.post("/auth/forgot", json={"email": "nobody@example.com"})
    b = c.post("/auth/forgot", json={"email": "sara@example.com"})
    assert a.status_code == b.status_code == 200 and a.json() == b.json() == {"sent": True}
    assert len(mailer.sent) == n + 1 and mailer.sent[-1][0] == "sara@example.com"


def test_reset_password_flow(sf):
    c, mailer, _ = make(sf)
    register(c)
    c.post("/auth/forgot", json={"email": "sara@example.com"})
    tok = mailer.token()
    assert "reset-password?token=" in mailer.sent[-1][2]
    short = c.post("/auth/reset", json={"token": tok, "new_password": "short"})
    assert short.status_code == 422 and short.json()["error"]["code"] == "bad_password"
    assert c.post("/auth/reset", json={"token": tok, "new_password": "brand-new-pass"}).json() == {"reset": True}   # the typo did not burn the link
    assert c.post("/auth/login", json={"email": "sara@example.com", "password": PW}).status_code == 401
    ok = c.post("/auth/login", json={"email": "sara@example.com", "password": "brand-new-pass"})
    assert ok.status_code == 200 and ok.json()["verified"] is True           # reading the mail proves the email
    again = c.post("/auth/reset", json={"token": tok, "new_password": "another-pass1"})
    assert again.status_code == 400 and again.json()["error"]["code"] == "invalid_token"


def test_reset_link_expires_after_30_minutes(sf):
    c, mailer, clock = make(sf)
    register(c)
    c.post("/auth/forgot", json={"email": "sara@example.com"})
    clock.t += 31 * 60
    assert c.post("/auth/reset", json={"token": mailer.token(), "new_password": "brand-new-pass"}).status_code == 400


def test_a_verify_token_cannot_reset_a_password_and_junk_is_refused(sf):
    c, mailer, _ = make(sf)
    register(c)
    assert c.post("/auth/reset", json={"token": mailer.token(), "new_password": "brand-new-pass"}).status_code == 400
    assert c.post("/auth/reset", json={"token": "junk", "new_password": "brand-new-pass"}).status_code == 400
    assert c.post("/auth/verify", json={"token": "junk"}).status_code == 400


def test_reset_clears_a_lockout(sf):
    c, mailer, _ = make(sf)
    register(c)
    for _ in range(5):
        c.post("/auth/login", json={"email": "sara@example.com", "password": "wrong-wrong1"})
    assert c.post("/auth/login", json={"email": "sara@example.com", "password": PW}).status_code == 429
    c.post("/auth/forgot", json={"email": "sara@example.com"})
    c.post("/auth/reset", json={"token": mailer.token(), "new_password": "brand-new-pass"})
    assert c.post("/auth/login", json={"email": "sara@example.com", "password": "brand-new-pass"}).status_code == 200


def test_tokens_are_stored_hashed(sf):
    from sqlalchemy import select
    from recipe_agent.db import AuthToken
    c, mailer, _ = make(sf)
    register(c)
    raw = mailer.token()
    with sf() as s:
        rows = s.scalars(select(AuthToken)).all()
    assert rows and all(raw not in r.token_hash for r in rows)


def test_a_failing_mailer_does_not_break_the_api(sf):
    c, _, _ = make(sf, fail=True)
    assert register(c).status_code == 200
    assert c.post("/auth/forgot", json={"email": "sara@example.com"}).json() == {"sent": True}


def test_dev_mode_returns_the_link_and_normal_mode_does_not(sf):
    c, _, _ = make(sf, dev_mail=True)
    r = register(c)
    assert r.json()["dev_link"].startswith("http") and "verify-email?token=" in r.json()["dev_link"]
    f = c.post("/auth/forgot", json={"email": "sara@example.com"})
    assert "reset-password?token=" in f.json()["dev_link"]
    c2, _, _ = make(sf, dev_mail=False)
    assert register(c2, "mona@example.com").json()["dev_link"] is None


def test_app_url_is_used_in_links(sf, monkeypatch):
    monkeypatch.setenv("RECIPE_APP_URL", "https://app.example.com/")
    c, mailer, _ = make(sf)
    register(c)
    assert "https://app.example.com/verify-email?token=" in mailer.sent[0][2]


def test_default_mailer_is_console_and_smtp_needs_a_host(monkeypatch):
    monkeypatch.delenv("RECIPE_SMTP_HOST", raising=False)
    assert isinstance(mail_mod.get_mailer(), mail_mod.ConsoleMailer)
    monkeypatch.setenv("RECIPE_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("RECIPE_SMTP_PORT", "2525")
    m = mail_mod.get_mailer()
    assert isinstance(m, mail_mod.SMTPMailer) and m.host == "smtp.example.com" and m.port == 2525


def test_smtp_mailer_failure_returns_false_without_raising():
    m = mail_mod.SMTPMailer("127.0.0.1", 1, "u", "p", "a@b.c")
    assert m.send("x@y.z", "s", "b") is False


def test_service_level_email_errors(sf):
    svc = AccountService(sf(), clock=Clock(), fails={})
    svc.register_email("a@example.com", PW)
    with pytest.raises(AccountError) as e:
        svc.register_email("A@example.com", PW)
    assert e.value.code == "taken"
    with pytest.raises(AccountError) as e:
        svc.register_email("b@example.com", "b@example.com")
    assert e.value.code == "bad_password"
