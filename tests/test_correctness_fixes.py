"""Wrong-route, placeholder recipe, broth/spleen prices and partial costs. Offline."""
import pytest

from recipe_agent import egypt_parse as p
from recipe_agent import egypt_store
from recipe_agent.egypt_cost import format_cost, load_price_map
from recipe_agent.multi.agents import (BusinessAgent, BusinessRequest, keyword_route, keyword_topics)
from recipe_agent.multi.tools import RecipeRef, ToolContext, build_registry
from recipe_agent.services import IngredientInput


def test_n_meals_with_ingredients_is_not_business():
    assert keyword_route("اعمل 3 وجبات و عندي باذنجان و شعير و ارز و بطاطس").intents == ["recipes"]
    assert "business" not in keyword_topics("عايزة 3 وجبات من الفراخ")
    assert keyword_route("عايزة اجهز 50 وجبة وابيعها").intents == ["business"]
    assert keyword_route("خطة انتاج 30 وجبة").intents == ["business"]


def _run(recipe):
    agent = BusinessAgent(build_registry(), lambda *a, **k: None)
    from recipe_agent.multi.agents import AgentResult
    res = AgentResult("business")
    agent.handle(BusinessRequest(recipe=recipe, meals=3), "ar", ToolContext(user_id="u"), res)
    return res


def test_production_plan_needs_a_real_recipe_not_a_placeholder():
    for ref in (None, RecipeRef(name="recipe"), RecipeRef(name="وصفة"),
                RecipeRef(ingredients=[IngredientInput(name="باذنجان"), IngredientInput(name="ارز")])):
        res = _run(ref)
        assert res.needs_input and "recipe" not in res.answer.split("\n")[0].replace("Which recipe", "")


def test_production_plan_still_runs_with_amounts_or_a_name():
    res = _run(RecipeRef(name="كشري"))
    assert not (res.needs_input and "هتنتجي أنهي" in res.answer)


@pytest.mark.parametrize("name", ["مرق لحم", "مرقة دجاج", "مرق لحم أو دجاج", "شوربة لحم"])
def test_broth_is_priced_as_broth_not_meat(name):
    assert p.match(name, load_price_map())["key"] == "broth"


def test_meat_and_chicken_still_priced_as_meat():
    pm = load_price_map()
    assert p.match("لحم مفروم", pm)["key"] != "broth" and p.match("صدور فراخ", pm)["key"] != "broth"


def test_no_soup_broth_line_costs_hundreds_per_litre():
    for e in egypt_store.load_store()["entries"].values():
        for l in e["cost"]["lines"]:
            if "مرق" in l["name"] and l["cost_egp"] is not None and "لتر" in (l["amount_text"] or ""):
                assert l["cost_egp"] <= 25, (e["name_en"], l["name"], l["cost_egp"])


def test_spleen_is_not_priced_as_beef():
    row = p.match("طحال ضأن أو بتلو", load_price_map())
    assert row["key"] == "gizzards_offal" and row["price_source"] == "estimated"


def test_partial_cost_says_it_is_partial():
    est = {"total_egp": 38.0, "coverage": 0.4, "low_egp": 30, "high_egp": 45, "price_label": "approx", "stale_prices": False,
           "unpriced": ["بط", "ملح"], "price_source": "current_web"}
    ar, en = format_cost(est, "ar"), format_cost(est, "en")
    assert "جزئية" in ar and "بط" in ar and "partial" in en and "not counted" in en
    full = dict(est, coverage=1.0, unpriced=[])
    assert "جزئية" not in format_cost(full, "ar")


def test_plural_cups_are_understood():
    pm = load_price_map()
    am = p.parse_amount("5 أكواب")
    assert am["unit"] == "cup" and p.grams(am, p.match("مرق لحم", pm), "مرق لحم") == 1200.0
