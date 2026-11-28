"""Meal plans: N meals or days x meals, budget, saved allergies, replan. Offline: real Egyptian recipes, no LLM."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from recipe_agent import egypt_store
from recipe_agent.api import create_app
from recipe_agent.db import init_db, make_engine, make_session_factory
from recipe_agent.services import mealplan


class FakeMem:
    def __init__(self, d=None):
        self.d = d or {}

    def get_exclusions(self, uid="default"):
        return list(self.d.get(uid, []))


@pytest.fixture
def env():
    engine = make_engine("sqlite://")
    init_db(engine)
    sf = make_session_factory(engine)
    mem = FakeMem()
    c = TestClient(create_app(answer_fn=lambda t, u: "x", session_factory=sf, memory=mem, api_key="", require_login=True,
                              secret=b"s"))
    r = c.post("/auth/register", json={"email": "sara@example.com", "password": "correct-horse"})
    return c, mem, {"Authorization": "Bearer " + r.json()["token"]}, r.json()["user_id"]


def ids(plan):
    return [m["recipe_id"] for m in plan["meals"]]


def test_n_meals_are_n_different_meals_with_prices_and_a_shopping_list(env):
    c, _, h, _ = env
    r = c.post("/meal-plans", json={"meals": 5}, headers=h)
    assert r.status_code == 200
    p = r.json()
    assert len(p["meals"]) == 5 and len(set(ids(p))) == 5
    assert 0 < p["total_low_egp"] <= p["total_high_egp"] and p["budget_status"] is None
    assert all(m["cost_low_egp"] <= m["cost_high_egp"] and m["name"] for m in p["meals"])
    assert p["shopping"] and p["currency"] == "EGP"


def test_same_seed_same_plan_other_seed_other_plan(env):
    c, _, h, _ = env
    a = c.post("/meal-plans", json={"meals": 4, "seed": 1}, headers=h).json()
    b = c.post("/meal-plans", json={"meals": 4, "seed": 1}, headers=h).json()
    d = c.post("/meal-plans", json={"meals": 4, "seed": 2}, headers=h).json()
    assert ids(a) == ids(b) and ids(a) != ids(d)


def test_week_plan_has_days_and_meal_types(env):
    c, _, h, _ = env
    p = c.post("/meal-plans", json={"days": 7, "meals_per_day": 2}, headers=h).json()
    assert len(p["meals"]) == 14 and len(set(ids(p))) == 14
    assert [m["day"] for m in p["meals"]][:4] == [1, 1, 2, 2]
    assert {m["meal_type"] for m in p["meals"]} == {"breakfast", "dinner"}


def test_budget_the_plan_fits_the_high_end_when_it_can(env):
    c, _, h, _ = env
    p = c.post("/meal-plans", json={"days": 3, "meals_per_day": 3, "budget_egp": 250}, headers=h).json()
    assert p["budget_status"] == "within_budget" and p["total_high_egp"] <= 250


def test_a_budget_too_small_is_reported_not_hidden(env):
    c, _, h, _ = env
    r = c.post("/meal-plans", json={"days": 3, "meals_per_day": 3, "budget_egp": 5}, headers=h)
    assert r.status_code == 200 and r.json()["budget_status"] == "over_budget" and r.json()["budget_egp"] == 5


def test_saved_allergies_and_extra_exclusions_are_applied(env):
    c, mem, h, uid = env
    mem.d[uid] = ["milk"]
    p = c.post("/meal-plans", json={"days": 3, "meals_per_day": 2, "exclude": ["egg"]}, headers=h).json()
    assert p["applied_exclusions"] == ["milk", "egg"]
    for rid in ids(p):
        assert not egypt_store.excluded(egypt_store.get_recipe(rid), ["milk", "egg"])


def test_bad_requests(env):
    c, _, h, _ = env
    for body in ({}, {"meals": 3, "days": 2}, {"meals": 0}, {"days": 9}, {"days": 2, "meals_per_day": 4}):
        r = c.post("/meal-plans", json=body, headers=h)
        assert r.status_code == 422 and r.json()["error"]["code"] in ("invalid_request",), body
    assert c.post("/meal-plans", json={"meals": 3}).status_code == 401
    assert c.post("/meal-plans/replan", json={"plan": [], "replace": []}).status_code == 401


def test_nothing_at_all_fits_is_the_only_hard_error(env):
    c, _, h, _ = env
    r = c.post("/meal-plans", json={"meals": 3, "max_minutes": 1}, headers=h)
    assert r.status_code == 409 and "no recipe" in r.json()["error"]["message"]


def test_replan_changes_only_the_chosen_meals(env):
    c, _, h, _ = env
    p = c.post("/meal-plans", json={"days": 3, "meals_per_day": 3}, headers=h).json()
    plan = [{"recipe_id": m["recipe_id"], "meal_type": m["meal_type"]} for m in p["meals"]]
    r = c.post("/meal-plans/replan", json={"plan": plan, "replace": [1, 4]}, headers=h)
    assert r.status_code == 200
    q = r.json()
    assert len(q["meals"]) == 9 and len(set(ids(q))) == 9
    for i in range(9):
        assert (ids(q)[i] != ids(p)[i]) == (i in (1, 4))
    assert q["shopping"]


def test_replan_by_recipe_id_and_cheaper_under_a_budget(env):
    c, _, h, _ = env
    p = c.post("/meal-plans", json={"meals": 5, "seed": 3}, headers=h).json()
    plan = [{"recipe_id": m["recipe_id"], "meal_type": "meal"} for m in p["meals"]]
    top = max(p["meals"], key=lambda m: m["cost_high_egp"])
    q = c.post("/meal-plans/replan", json={"plan": plan, "replace": [top["recipe_id"]], "cheaper": True}, headers=h).json()
    assert top["recipe_id"] not in ids(q) and q["total_high_egp"] < p["total_high_egp"]
    fit = c.post("/meal-plans/replan", json={"plan": plan, "replace": [0, 1, 2, 3, 4], "budget_egp": 120}, headers=h).json()
    assert fit["budget_status"] in ("within_budget", "might_exceed", "over_budget") and fit["budget_egp"] == 120


def test_replan_refuses_unknown_things(env):
    c, _, h, _ = env
    p = c.post("/meal-plans", json={"meals": 2}, headers=h).json()
    plan = [{"recipe_id": m["recipe_id"], "meal_type": "meal"} for m in p["meals"]]
    assert c.post("/meal-plans/replan", json={"plan": plan, "replace": [7]}, headers=h).status_code == 422
    assert c.post("/meal-plans/replan", json={"plan": [{"recipe_id": "egy:nope"}], "replace": [0]}, headers=h).status_code == 422
    stuck = c.post("/meal-plans/replan", json={"plan": plan, "replace": [0], "cheaper": True, "max_minutes": 1}, headers=h)
    assert stuck.status_code == 200 and ids(stuck.json()) == ids(p)                  # nothing else fits: unchanged + advice
    assert [t["code"] for t in stuck.json()["suggestions"]] == ["no_other_recipe"]


def test_only_well_priced_normal_recipes_are_used():
    for e in mealplan.candidates([], None, None):
        assert (e["cost"]["coverage"] or 0) >= mealplan.MIN_COVERAGE and 0 < e["servings"] <= mealplan.MAX_SERVINGS
        lo, hi = mealplan.per_meal(e)
        assert 0 < lo <= hi


def test_plan_total_is_the_sum_of_its_meals():
    p = mealplan.make_plan(meals=6, seed=5)
    assert p["total_low_egp"] == round(sum(m["cost_low_egp"] for m in p["meals"]), 2)
    assert p["total_high_egp"] == round(sum(m["cost_high_egp"] for m in p["meals"]), 2)


def test_scarce_breakfasts_fall_back_to_other_recipes_instead_of_failing():
    p = mealplan.make_plan(days=4, meals_per_day=2, exclude=["milk"])
    assert len(set(ids(p))) == 8


def test_too_few_safe_recipes_gives_a_plan_with_flagged_repeats_and_advice():
    p = mealplan.make_plan(days=7, meals_per_day=2, exclude=["milk", "egg"], relaxable=["egg"])
    assert len(p["meals"]) == 14 and p["repeats"] == 14 - len(set(ids(p))) > 0
    assert [m["repeat"] for m in p["meals"]].count(True) == p["repeats"]
    codes = [t["code"] for t in p["suggestions"]]
    assert codes[0] == "recipes_short" and "relax_exclusion" in codes
    t = p["suggestions"][0]
    assert t["unique_meals"] < t["planned_meals"] and t["message_ar"] and t["message_en"]
    assert not any(egypt_store.excluded(egypt_store.get_recipe(r), ["milk", "egg"]) for r in ids(p))      # repeats never break allergies


def test_a_normal_plan_has_no_repeats_and_no_suggestions():
    p = mealplan.make_plan(meals=5)
    assert p["repeats"] == 0 and p["suggestions"] == [] and not any(m["repeat"] for m in p["meals"])


def test_budget_too_small_comes_with_advice():
    p = mealplan.make_plan(meals=5, budget_egp=3)
    assert p["budget_status"] == "over_budget" and p["suggestions"][-1]["code"] == "budget_too_small"


def test_api_returns_200_with_suggestions_when_recipes_are_short(env):
    c, mem, h, uid = env
    mem.d[uid] = ["milk"]
    r = c.post("/meal-plans", json={"days": 7, "meals_per_day": 2}, headers=h)
    assert r.status_code == 200
    p = r.json()
    assert len(p["meals"]) == 14 and p["repeats"] > 0 and p["suggestions"][0]["code"] == "recipes_short" and p["shopping"]
