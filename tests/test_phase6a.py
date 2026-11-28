"""Phase 6a: FastAPI layer. Offline: fake answer function, in-memory DB, fake memory."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from recipe_agent.api import create_app
from recipe_agent.db import init_db, make_engine, make_session_factory
from recipe_agent.services.profile import HistoryService


class FakeMem:
    class MemoryReadError(Exception):
        pass

    def __init__(self):
        self.d = {}

    def get_exclusions(self, uid="default"):
        return list(self.d.get(uid, []))

    def remember_exclusions(self, uid, items):
        self.d[uid] = sorted(set(self.d.get(uid, [])) | {i.lower() for i in items})
        return self.d[uid]

    def forget(self, uid="default"):
        self.d.pop(uid, None)


@pytest.fixture
def env():
    engine = make_engine("sqlite://")
    init_db(engine)
    sf = make_session_factory(engine)
    mem = FakeMem()
    calls = []

    def answer(text, uid):
        calls.append((text, uid))
        return "ANS:" + text

    app = create_app(answer_fn=answer, session_factory=sf, memory=mem, api_key="")
    return TestClient(app), sf, mem, calls


def test_health(env):
    c = env[0]
    assert c.get("/health").json() == {"status": "ok", "api_version": "1"}


def test_chat_returns_supervisor_answer_for_user(env):
    c, _, _, calls = env
    r = c.post("/chat", json={"message": "عايز كشري", "user_id": "waad"})
    assert r.status_code == 200 and r.json() == {"answer": "ANS:عايز كشري", "user_id": "waad"}
    assert c.post("/chat", json={"message": "hi"}).json()["user_id"] == "default"
    assert calls == [("عايز كشري", "waad"), ("hi", "default")]


def test_chat_validation(env):
    c = env[0]
    assert c.post("/chat", json={"message": ""}).status_code == 422
    assert c.post("/chat", json={"message": "x", "user_id": "../etc"}).status_code == 422
    assert c.post("/chat", json={"user_id": "a"}).status_code == 422


def test_chat_maps_quota_busy_and_other_errors():
    def boom(msg):
        def f(t, u):
            raise RuntimeError(msg)
        return f
    for msg, code in [("429 RESOURCE_EXHAUSTED", 429), ("503 UNAVAILABLE", 503), ("kaboom", 500)]:
        c = TestClient(create_app(answer_fn=boom(msg), api_key=""))
        r = c.post("/chat", json={"message": "x"})
        assert r.status_code == code
        assert "kaboom" not in r.text      # internal errors are not leaked


def test_profile_put_get_delete(env):
    c = env[0]
    assert c.get("/users/u1/profile").json()["exists"] is False
    r = c.put("/users/u1/profile", json={"family_size": 4, "budget_egp": 3000, "budget_period": "monthly"})
    assert r.status_code == 200 and r.json()["family_size"] == 4 and r.json()["exists"]
    r = c.put("/users/u1/profile", json={"diet": ["halal", "halal"]})
    assert r.json()["diet"] == ["halal"] and r.json()["family_size"] == 4     # partial update keeps the rest
    assert c.put("/users/u1/profile", json={}).status_code == 422
    assert c.put("/users/u1/profile", json={"family_size": 0}).status_code == 422
    assert c.delete("/users/u1/profile").json() == {"deleted": True}
    assert c.delete("/users/u1/profile").json() == {"deleted": False}


def test_allergies_saved_in_memory_store_and_shown_in_profile(env):
    c, _, mem, _ = env
    r = c.post("/users/u1/allergies", json={"items": ["Milk", "nuts"]})
    assert r.json()["allergies"] == ["milk", "nuts"] and mem.d["u1"] == ["milk", "nuts"]
    assert c.get("/users/u1/profile").json()["allergies"] == ["milk", "nuts"]
    assert c.post("/users/u1/allergies", json={"items": []}).status_code == 422
    assert c.post("/users/u1/allergies", json={"items": ["  "]}).status_code == 422
    assert c.delete("/users/u1/allergies").json() == {"allergies": []} and "u1" not in mem.d


def test_users_are_isolated(env):
    c, _, mem, _ = env
    c.put("/users/a/profile", json={"family_size": 2})
    c.post("/users/a/allergies", json={"items": ["milk"]})
    assert c.get("/users/b/profile").json()["family_size"] is None
    assert c.get("/users/b/profile").json()["allergies"] == []


def test_history_read_and_delete(env):
    c, sf, _, _ = env
    with sf() as s:
        h = HistoryService(s)
        for i in range(5):
            h.add("u1", "user" if i % 2 == 0 else "assistant", f"m{i}")
        h.add("u2", "user", "other")
    msgs = c.get("/users/u1/history?limit=3").json()["messages"]
    assert [m["text"] for m in msgs] == ["m2", "m3", "m4"]
    assert c.get("/users/u1/history?limit=0").status_code == 422
    assert c.delete("/users/u1/history").json() == {"deleted": 5}
    assert c.get("/users/u1/history").json()["messages"] == []
    assert len(c.get("/users/u2/history").json()["messages"]) == 1


def test_bad_user_id_in_path(env):
    assert env[0].get("/users/bad id!/profile").status_code == 422


def test_api_key_required_when_set(env):
    _, sf, mem, _ = env
    c = TestClient(create_app(answer_fn=lambda t, u: "ok", session_factory=sf, memory=mem, api_key="s3cret"))
    assert c.get("/health").status_code == 200
    assert c.post("/chat", json={"message": "x"}).status_code == 401
    assert c.post("/chat", json={"message": "x"}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/users/u/profile").status_code == 401
    assert c.post("/chat", json={"message": "x"}, headers={"X-API-Key": "s3cret"}).status_code == 200


def test_default_answer_fn_is_the_supervisor(monkeypatch, env):
    from recipe_agent.multi import app as supervisor_app
    _, sf, mem, _ = env
    monkeypatch.setattr(supervisor_app, "answer", lambda t, user_id: f"SUP:{user_id}")
    c = TestClient(create_app(session_factory=sf, memory=mem, api_key=""))
    assert c.post("/chat", json={"message": "x", "user_id": "w"}).json()["answer"] == "SUP:w"
