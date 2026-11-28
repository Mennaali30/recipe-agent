"""API feature endpoints and the standard error body. Offline: in-memory DB, fake memory, real Egyptian recipes."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from recipe_agent import egypt_store, tools as recipe_tools
from recipe_agent.api import create_app
from recipe_agent.db import init_db, make_engine, make_session_factory
from recipe_agent.services import ApprovalService

SECRET = b"s"


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
def env():
    engine = make_engine("sqlite://")
    init_db(engine)
    sf = make_session_factory(engine)
    mem = FakeMem()
    app = create_app(answer_fn=lambda t, u: "x", session_factory=sf, memory=mem, api_key="", require_login=True, secret=SECRET)
    c = TestClient(app)
    return c, sf, mem


def login(c, name="sara"):
    r = c.post("/auth/register", json={"username": name, "password": "correct-horse"})
    return {"Authorization": "Bearer " + r.json()["token"]}


def err(r):
    assert set(r.json()) == {"error"} and {"code", "message"} <= set(r.json()["error"])
    return r.json()["error"]


# ---------------- error format ----------------
def test_errors_use_one_format(env):
    c, _, _ = env
    assert err(c.get("/pantry"))["code"] == "unauthorized"                       # no token
    h = login(c)
    assert err(c.post("/pantry/items", json={"items": []}, headers=h))["code"] == "invalid_request"
    e = c.post("/pantry/items", json={"items": []}, headers=h).json()["error"]
    assert e["fields"] and "field" in e["fields"][0]
    assert err(c.get("/recipes/egy:nope", headers=h))["code"] == "not_found"
    assert err(c.post("/auth/login", json={"username": "sara", "password": "bad-bad-bad"}))["code"] == "unauthorized"


def test_every_feature_endpoint_needs_login(env):
    c, _, _ = env
    for method, url in (("get", "/pantry"), ("get", "/pantry/expiring"), ("get", "/pantry/low-stock"),
                        ("post", "/pantry/items"), ("post", "/pantry/use"), ("put", "/pantry/items"),
                        ("post", "/pantry/discard"), ("post", "/cost"), ("post", "/convert"), ("post", "/shopping"),
                        ("post", "/budget/check"), ("post", "/business/production-plan"),
                        ("get", "/recipes/search?ingredients=rice"), ("get", "/recipes/by-name?q=koshari"),
                        ("get", "/recipes/egy:fah-096"), ("get", "/pending"), ("post", "/pending/confirm"),
                        ("post", "/pending/cancel")):
        r = getattr(c, method)(url) if method == "get" else getattr(c, method)(url, json={})
        assert r.status_code == 401, url


# ---------------- pantry ----------------
def test_pantry_add_list_use_set_discard(env):
    c, _, _ = env
    h = login(c)
    r = c.post("/pantry/items", json={"items": [{"name": "rice", "quantity": 2, "unit": "kg"}]}, headers=h)
    assert r.status_code == 200
    rows = c.get("/pantry", headers=h).json()
    assert rows[0]["quantity"] == 2000 and rows[0]["unit"] == "g"
    assert c.post("/pantry/use", json={"items": [{"name": "rice", "quantity": 500, "unit": "g"}]}, headers=h).status_code == 200
    assert c.get("/pantry", headers=h).json()[0]["quantity"] == 1500
    assert c.put("/pantry/items", json={"items": [{"name": "rice", "quantity": 1, "unit": "kg"}]}, headers=h).status_code == 200
    assert c.get("/pantry", headers=h).json()[0]["quantity"] == 1000
    assert c.post("/pantry/discard", json={"items": [{"name": "rice"}]}, headers=h).status_code == 200
    assert c.get("/pantry", headers=h).json() == []
    assert c.get("/pantry/expiring?within_days=5", headers=h).json()["within_days"] == 5
    assert c.get("/pantry/low-stock", headers=h).json() == []


def test_using_more_than_in_stock_is_a_clear_conflict(env):
    c, _, _ = env
    h = login(c)
    c.post("/pantry/items", json={"items": [{"name": "rice", "quantity": 100, "unit": "g"}]}, headers=h)
    r = c.post("/pantry/use", json={"items": [{"name": "rice", "quantity": 500, "unit": "g"}]}, headers=h)
    assert r.status_code == 409 and err(r)["code"] == "conflict"
    assert c.get("/pantry", headers=h).json()[0]["quantity"] == 100


def test_pantry_is_private_per_user(env):
    c, _, _ = env
    a, b = login(c, "sara"), login(c, "mona")
    c.post("/pantry/items", json={"items": [{"name": "rice", "quantity": 1, "unit": "kg"}]}, headers=a)
    assert c.get("/pantry", headers=b).json() == []


# ---------------- cost, units, shopping, budget, business ----------------
def test_cost_convert_shopping_budget_production(env):
    c, _, _ = env
    h = login(c)
    rid = c.get("/recipes/by-name?q=كشري", headers=h).json()["recipes"][0]["recipe_id"]
    R = {"recipe_id": rid}
    cost = c.post("/cost", json={"recipe": R}, headers=h)
    assert cost.status_code == 200 and cost.json()
    conv = c.post("/convert", json={"quantity": 2, "unit": "cup", "ingredient": "rice"}, headers=h)
    assert conv.status_code == 200
    shop = c.post("/shopping", json={"recipes": [{"recipe": R, "meals": 2}], "use_pantry": False}, headers=h)
    assert shop.status_code == 200
    bud = c.post("/budget/check", json={"items": [{"recipe": R, "meals": 2}], "budget_egp": 1000}, headers=h)
    assert bud.status_code == 200 and bud.json()["verdict"] in ("within_budget", "over_budget", "unknown")
    prod = c.post("/business/production-plan", json={"recipe": R, "meals": 20, "selling_price_per_meal": 60}, headers=h)
    assert prod.status_code == 200 and prod.json()["plan"]["meals"] == 20


def test_unknown_recipe_name_is_not_found_or_ambiguous(env):
    c, _, _ = env
    h = login(c)
    r = c.post("/cost", json={"recipe": {"name": "zzzz no such dish"}}, headers=h)
    assert r.status_code in (404, 409) and "error" in r.json()


def test_bad_unit_is_a_422(env):
    c, _, _ = env
    h = login(c)
    r = c.post("/convert", json={"quantity": 1, "unit": "parsec", "ingredient": "rice"}, headers=h)
    assert r.status_code == 422 and err(r)["code"] == "invalid_request"


# ---------------- recipes ----------------
def test_recipe_by_name_and_detail(env):
    c, _, _ = env
    h = login(c)
    r = c.get("/recipes/by-name?q=عايز كشري", headers=h).json()
    assert r["egyptian"] and r["recipes"]
    rid = r["recipes"][0]["recipe_id"]
    d = c.get(f"/recipes/{rid}", headers=h)
    assert d.status_code == 200 and d.json()["steps"] and d.json()["ingredients"] and d.json()["cost_text"]


def test_saved_allergies_are_always_applied(env):
    c, _, mem = env
    h = login(c)
    mem.d["sara"] = ["milk"]
    r = c.get("/recipes/by-name?q=كشري&limit=10", headers=h).json()
    assert r["applied_exclusions"] == ["milk"]
    for rec in r["recipes"]:
        assert not egypt_store.excluded(egypt_store.get_recipe(rec["recipe_id"]), ["milk"])
    milky = next(e for e in egypt_store.load_store()["entries"].values() if egypt_store.excluded(e, ["milk"]))
    blocked = c.get(f"/recipes/{milky['recipe_id']}", headers=h)
    assert blocked.status_code == 409 and "allerg" in blocked.json()["error"]["message"]


def test_search_falls_back_to_egyptian_when_the_index_is_missing(env, monkeypatch):
    c, _, _ = env
    h = login(c)

    def boom(*a, **k):
        raise RuntimeError("no index")
    monkeypatch.setattr(recipe_tools, "find_recipes", boom)
    r = c.get("/recipes/search?ingredients=رز,فراخ", headers=h)
    assert r.status_code == 200 and r.json()["international_available"] is False and r.json()["recipes"]


def test_by_name_with_no_egyptian_match_offers_international(env, monkeypatch):
    c, _, _ = env
    h = login(c)
    monkeypatch.setattr(recipe_tools, "find_international_by_name", lambda q, **k: [{"recipe_id": 1, "name": "Lasagna"}])
    r = c.get("/recipes/by-name?q=لازانيا", headers=h).json()
    assert r["egyptian"] is False and r["note"] == "no_egyptian_recipe" and r["recipes"][0]["name"] == "Lasagna"


# ---------------- pending confirmations ----------------
def test_pending_confirm_and_cancel(env):
    c, sf, _ = env
    h = login(c)
    c.post("/pantry/items", json={"items": [{"name": "rice", "quantity": 1, "unit": "kg"}]}, headers=h)
    assert c.get("/pending", headers=h).json() == {"pending": None}
    with sf() as s:
        ApprovalService(s).stage("sara", "pantry_discard", {"items": [{"name": "rice"}]}, "ar", "Throw away rice")
    assert c.get("/pending", headers=h).json()["pending"]["summary_en"] == "Throw away rice"
    assert c.post("/pending/cancel", headers=h).json() == {"cancelled": True}
    assert c.get("/pantry", headers=h).json()[0]["quantity"] == 1000
    with sf() as s:
        ApprovalService(s).stage("sara", "pantry_discard", {"items": [{"name": "rice"}]}, "ar", "Throw away rice")
    assert c.post("/pending/confirm", headers=h).json() == {"done": True, "tool": "pantry_discard"}
    assert c.get("/pantry", headers=h).json() == []
    assert c.post("/pending/confirm", headers=h).status_code == 404


def test_health_has_api_version(env):
    c, _, _ = env
    assert c.get("/health").json() == {"status": "ok", "api_version": "1"}


def test_openapi_lists_the_frozen_surface(env):
    c, _, _ = env
    paths = set(c.get("/openapi.json").json()["paths"])
    for p in ("/health", "/chat", "/auth/register", "/auth/login", "/auth/me", "/auth/password", "/pantry", "/pantry/items",
              "/pantry/use", "/pantry/discard", "/pantry/expiring", "/pantry/low-stock", "/cost", "/convert", "/shopping",
              "/budget/check", "/business/production-plan", "/recipes/search", "/recipes/by-name", "/recipes/{recipe_id}",
              "/pending", "/pending/confirm", "/pending/cancel", "/users/{user_id}/profile"):
        assert p in paths, p
