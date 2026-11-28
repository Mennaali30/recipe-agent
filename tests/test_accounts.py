"""Login and register: accounts, tokens, per-user isolation. Offline: in-memory DB, fake clock, fake answer."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from recipe_agent.api import create_app
from recipe_agent.db import init_db, make_engine, make_session_factory
from recipe_agent.services import AccountError, AccountService, issue_token, verify_token
from recipe_agent.services.accounts import hash_password, verify_password
from recipe_agent.services.profile import HistoryService

SECRET = b"test-secret"


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeMem:
    def __init__(self, d=None):
        self.d = d or {}

    def get_exclusions(self, uid="default"):
        return list(self.d.get(uid, []))

    def remember_exclusions(self, uid, items):
        self.d[uid] = sorted(set(self.d.get(uid, [])) | {i.lower() for i in items})
        return self.d[uid]

    def forget(self, uid="default"):
        self.d.pop(uid, None)


@pytest.fixture
def sf():
    engine = make_engine("sqlite://")
    init_db(engine)
    return make_session_factory(engine)


@pytest.fixture
def svc(sf):
    return AccountService(sf(), clock=Clock(), fails={})


def make(sf, require_login=True, mem=None, clock=None):
    calls = []
    app = create_app(answer_fn=lambda t, u: calls.append((t, u)) or f"hi {u}", session_factory=sf,
                     memory=mem or FakeMem(), api_key="", require_login=require_login, secret=SECRET,
                     clock=clock or Clock())
    return TestClient(app), calls


def reg(c, name="sara", pw="correct-horse"):
    return c.post("/auth/register", json={"email": f"{name}@example.com", "password": pw})


def bearer(r):
    return {"Authorization": "Bearer " + r.json()["token"]}


# ---------------- service ----------------
def test_password_hash_is_salted_and_verifies():
    a, b = hash_password("secret-pass"), hash_password("secret-pass")
    assert a != b and "secret-pass" not in a and verify_password("secret-pass", a) and not verify_password("x", a)
    assert not verify_password("secret-pass", "garbage")


def test_register_and_login(svc):
    assert svc.register("Sara", "correct-horse") == "sara"
    assert svc.authenticate("SARA", "correct-horse") == "sara"
    with pytest.raises(AccountError) as e:
        svc.authenticate("sara", "wrong-password")
    assert e.value.code == "invalid_credentials"
    with pytest.raises(AccountError) as e:
        svc.authenticate("nobody", "whatever1")
    assert e.value.code == "invalid_credentials"          # same error as a wrong password


@pytest.mark.parametrize("name,pw,code", [("ab", "correct-horse", "bad_username"), ("has space", "correct-horse", "bad_username"),
                                          ("سارة", "correct-horse", "bad_username"), ("admin", "correct-horse", "bad_username"),
                                          ("sara", "short", "bad_password"), ("sara1234", "sara1234", "bad_password")])
def test_bad_input_is_refused(svc, name, pw, code):
    with pytest.raises(AccountError) as e:
        svc.register(name, pw)
    assert e.value.code == code


def test_username_taken_even_with_other_case(svc):
    svc.register("sara", "correct-horse")
    with pytest.raises(AccountError) as e:
        svc.register("SARA", "another-pass1")
    assert e.value.code == "taken"


def test_old_user_with_data_cannot_be_taken(sf, svc):
    HistoryService(sf()).add("waad", "user", "hello")       # an old user id that already has data
    with pytest.raises(AccountError) as e:
        svc.register("waad", "correct-horse")
    assert e.value.code == "taken"


def test_lockout_after_five_wrong_passwords_then_expires(sf):
    clock = Clock()
    svc = AccountService(sf(), clock=clock, fails={})
    svc.register("sara", "correct-horse")
    for _ in range(5):
        with pytest.raises(AccountError):
            svc.authenticate("sara", "wrong-password")
    with pytest.raises(AccountError) as e:
        svc.authenticate("sara", "correct-horse")            # even the right password is refused while locked
    assert e.value.code == "locked"
    clock.t += 301
    assert svc.authenticate("sara", "correct-horse") == "sara"


def test_change_password(svc):
    svc.register("sara", "correct-horse")
    with pytest.raises(AccountError):
        svc.change_password("sara", "wrong-one", "new-password1")
    svc.change_password("sara", "correct-horse", "new-password1")
    assert svc.authenticate("sara", "new-password1") == "sara"
    with pytest.raises(AccountError):
        svc.authenticate("sara", "correct-horse")


# ---------------- tokens ----------------
def test_token_roundtrip_expiry_and_tamper():
    t = issue_token("sara", SECRET, now=1000, ttl=60)
    assert verify_token(t, SECRET, now=1030) == "sara"
    assert verify_token(t, SECRET, now=1061) is None                 # expired
    assert verify_token(t, b"other-secret", now=1030) is None         # wrong secret
    body, sig = t.split(".")
    forged = issue_token("admin", b"x", now=1000, ttl=60).split(".")[0] + "." + sig
    assert verify_token(forged, SECRET, now=1030) is None
    assert verify_token("", SECRET) is None and verify_token("a.b.c", SECRET) is None and verify_token(None, SECRET) is None


# ---------------- API ----------------
def test_register_login_me(sf):
    c, _ = make(sf)
    r = reg(c)
    uid = r.json()["user_id"]
    assert r.status_code == 200 and uid.startswith("u_") and r.json()["token"] and r.json()["email"] == "sara@example.com"
    assert reg(c).status_code == 409
    assert c.post("/auth/login", json={"email": "sara@example.com", "password": "nope-nope1"}).status_code == 401
    ok = c.post("/auth/login", json={"email": "SARA@example.com", "password": "correct-horse"})
    assert ok.status_code == 200 and ok.json()["user_id"] == uid
    assert c.get("/auth/me", headers=bearer(ok)).json() == {"user_id": uid, "email": "sara@example.com", "verified": False}
    assert c.get("/auth/me").status_code == 401
    assert "password" not in ok.text and "scrypt" not in ok.text


def test_old_username_accounts_can_still_log_in(sf):
    AccountService(sf(), clock=Clock(), fails={}).register("oldsara", "correct-horse")
    c, _ = make(sf)
    r = c.post("/auth/login", json={"email": "oldsara", "password": "correct-horse"})
    assert r.status_code == 200 and r.json()["user_id"] == "oldsara" and r.json()["email"] is None


def test_chat_needs_login_and_uses_the_token_user(sf):
    c, calls = make(sf)
    assert c.post("/chat", json={"message": "hi"}).status_code == 401
    t = reg(c)
    uid = t.json()["user_id"]
    assert c.post("/chat", json={"message": "hi", "user_id": "someone_else"}, headers=bearer(t)).status_code == 403
    r = c.post("/chat", json={"message": "hi"}, headers=bearer(t))
    assert r.json()["answer"] == f"hi {uid}" and calls[-1] == ("hi", uid)
    assert c.post("/chat", json={"message": "hi", "user_id": uid}, headers=bearer(t)).status_code == 200


def test_users_cannot_read_each_others_data(sf):
    c, _ = make(sf)
    a, b = reg(c, "sara"), reg(c, "mona")
    ua = a.json()["user_id"]
    c.post(f"/users/{ua}/allergies", json={"items": ["milk"]}, headers=bearer(a))
    assert c.get(f"/users/{ua}/profile", headers=bearer(a)).status_code == 200
    for method, url in (("get", f"/users/{ua}/profile"), ("get", f"/users/{ua}/history"), ("delete", f"/users/{ua}/history"),
                        ("delete", f"/users/{ua}/allergies"), ("delete", f"/users/{ua}/profile")):
        assert getattr(c, method)(url, headers=bearer(b)).status_code == 403
        assert getattr(c, method)(url).status_code == 401


def test_forged_or_expired_token_is_refused(sf):
    clock = Clock()
    c, _ = make(sf, clock=clock)
    t = reg(c)
    clock.t += 8 * 24 * 3600
    assert c.post("/chat", json={"message": "hi"}, headers=bearer(t)).status_code == 401
    forged = {"Authorization": "Bearer " + issue_token(t.json()["user_id"], b"not-the-secret")}
    assert c.post("/chat", json={"message": "hi"}, headers=forged).status_code == 401


def test_change_password_api(sf):
    c, _ = make(sf)
    t = reg(c)
    bad = c.post("/auth/password", json={"old_password": "wrong-one1", "new_password": "new-password1"}, headers=bearer(t))
    assert bad.status_code == 401
    ok = c.post("/auth/password", json={"old_password": "correct-horse", "new_password": "new-password1"}, headers=bearer(t))
    assert ok.status_code == 200
    assert c.post("/auth/login", json={"email": "sara@example.com", "password": "new-password1"}).status_code == 200


def test_without_require_login_the_old_behaviour_is_unchanged(sf):
    c, calls = make(sf, require_login=False)
    assert c.post("/chat", json={"message": "hi", "user_id": "waad"}).status_code == 200
    assert calls[-1][1] == "waad"
    assert c.get("/users/waad/profile").status_code == 200


def test_the_api_key_still_applies_to_login(sf):
    app = create_app(answer_fn=lambda t, u: "x", session_factory=sf, memory=FakeMem(), api_key="k", require_login=True,
                     secret=SECRET)
    c = TestClient(app)
    assert reg(c).status_code == 401
    r = c.post("/auth/register", json={"email": "sara@example.com", "password": "correct-horse"}, headers={"X-API-Key": "k"})
    assert r.status_code == 200
