"""Price plan, step 3: ranges and labels. Offline and deterministic."""
from datetime import date

import pytest

from recipe_agent import price_range as pr
from recipe_agent.services import IngredientInput, RecipeInput, recipe_cost

TODAY = date(2026, 11, 28)


def part(cost, source, rel=None, d=None):
    return {"cost": cost, "source": source, "reliability": rel, "price_date": d}


def test_spreads_by_source_and_reliability():
    assert pr.line_spread("wfp") == 0.10
    assert pr.line_spread("current_web", "high") == 0.10
    assert pr.line_spread("current_web", "medium") == 0.20
    assert pr.line_spread("current_web", "low") == 0.20
    assert pr.line_spread("current_web", None) == 0.20
    assert pr.line_spread("estimated") == 0.35 and pr.line_spread(None) == 0.35


def test_range_is_the_sum_of_line_ranges():
    r = pr.make_range([part(100, "current_web", "high"), part(50, "estimated")], TODAY)
    assert r["low"] == pytest.approx(90 + 32.5) and r["high"] == pytest.approx(110 + 67.5)
    assert r["point"] == 150 and r["estimated_share"] == pytest.approx(0.33)


def test_label_by_share_of_estimated_cost():
    assert pr.make_range([part(95, "current_web", "high"), part(5, "estimated")], TODAY)["label_key"] == "market"
    assert pr.make_range([part(70, "current_web", "high"), part(30, "estimated")], TODAY)["label_key"] == "approx"
    assert pr.make_range([part(40, "current_web", "high"), part(60, "estimated")], TODAY)["label_key"] == "rough"
    assert pr.make_range([part(10, "estimated")], TODAY)["label_key"] == "rough"


def test_nothing_priced_gives_no_range_and_fixed_amount_is_exact():
    assert pr.make_range([part(None, "estimated")], TODAY) is None and pr.make_range([], TODAY) is None
    r = pr.make_range([part(100, "wfp")], TODAY, fixed=20)
    assert (r["low"], r["high"], r["point"]) == (110, 130, 120)      # extras are added without a spread


def test_stale_price_is_flagged():
    assert not pr.make_range([part(10, "current_web", "high", "2026-11-20")], TODAY)["stale"]
    assert pr.make_range([part(10, "current_web", "high", "2026-10-01")], TODAY)["stale"]
    assert not pr.make_range([part(10, "estimated", None, None)], TODAY)["stale"]


def test_round_outwards_and_text():
    assert pr.round_range(91.23, 136.27) == (90, 140)
    assert pr.round_range(7.2, 9.9) == (7, 10)
    assert pr.round_range(1180, 1420) == (1180, 1420)
    assert pr.range_text(91.23, 136.27, "ar") == "حوالي 90 إلى 140 جنيه"
    assert pr.range_text(91.23, 136.27, "en") == "about 90 to 140 EGP"
    assert pr.range_text(10.1, 10.2, "en") == "about 10 to 11 EGP"
    assert pr.range_text(100, 100, "en") == "about 100 EGP"
    assert pr.round_range(0.2, 0.9) == (0, 1)


def test_labels_text():
    assert pr.label_text({"label_key": "rough", "stale": False}, "ar") == "سعر تقديري جداً"
    assert pr.label_text({"label_key": "market", "stale": True}, "en") == "market prices, some prices are old"


def test_range_always_contains_the_point_and_low_is_not_negative():
    for parts in ([part(3, "estimated")], [part(0.5, "wfp"), part(0.1, "estimated")], [part(900, "current_web", "low")]):
        r = pr.make_range(parts, TODAY)
        lo, hi = pr.round_range(r["low"], r["high"])
        assert lo >= 0 and lo <= r["point"] <= hi


def test_egyptian_recipe_cost_carries_range_and_text():
    from recipe_agent import egypt_store
    found, _ = egypt_store.find_by_name("كشري", top_k=3)
    for r in found:
        c = egypt_store.get_recipe(r["recipe_id"])["cost"]
        assert c["low_egp"] <= c["total_egp"] <= c["high_egp"]
        assert c["price_label"] in {"market", "approx", "rough"}
        assert r["cost_text"].startswith("التكلفة التقريبية: حوالي ") and " إلى " in r["cost_text"]


def test_service_recipe_cost_range_and_labels():
    rc = recipe_cost(RecipeInput(name="Rice", servings=4, ingredients=[
        IngredientInput(name="rice", quantity=300, unit="g"), IngredientInput(name="salt", quantity=5, unit="g")]),)
    assert rc.low_egp <= rc.total_egp <= rc.high_egp and rc.price_label in {"market", "approx", "rough"}
    assert rc.per_serving_low_egp <= rc.per_serving_egp <= rc.per_serving_high_egp
    assert all(l.price_source for l in rc.lines if l.cost_egp is not None)


def test_unpriced_recipe_has_no_range():
    rc = recipe_cost(RecipeInput(name="X", servings=2, ingredients=[IngredientInput(name="zzz unobtainium", quantity=100, unit="g")]))
    assert rc.total_egp is None and rc.low_egp is None and rc.price_label is None


def test_extras_are_not_widened():
    from recipe_agent.services import ExtraExpense
    base = RecipeInput(name="Rice", servings=4, ingredients=[IngredientInput(name="rice", quantity=300, unit="g")])
    with_extra = RecipeInput(name="Rice", servings=4, ingredients=base.ingredients,
                             extra_expenses=[ExtraExpense(name="box", amount_egp=20, basis="fixed")])
    a, b = recipe_cost(base), recipe_cost(with_extra)
    assert b.low_egp == pytest.approx(a.low_egp + 20) and b.high_egp == pytest.approx(a.high_egp + 20)


def test_shopping_list_total_range_and_agent_text():
    from recipe_agent.services import build_shopping_list
    from recipe_agent.services.costing import cost_line
    from recipe_agent.multi.agents import CATEGORY_AR  # noqa: F401
    lines = [cost_line(IngredientInput(name=n, quantity=q, unit="g"))[0] for n, q in (("rice", 500), ("chicken", 1000))]
    sl = build_shopping_list(lines)
    assert sl.low_egp <= sl.estimated_total_egp <= sl.high_egp and sl.price_label in {"market", "approx", "rough"}
    assert all(i.price_source for i in sl.items if i.est_cost_egp)


def test_shopping_list_covered_by_pantry_has_no_range():
    from recipe_agent.services import build_shopping_list
    from recipe_agent.services.costing import cost_line
    line = cost_line(IngredientInput(name="rice", quantity=100, unit="g"))[0]
    sl = build_shopping_list([line], stock={("rice", "g"): 500})
    assert sl.low_egp is None and sl.estimated_total_egp == 0.0
