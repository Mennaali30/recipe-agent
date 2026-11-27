"""Phase 5a: supervisor + specialist agents over typed tools. Offline: fake LLM, in-memory SQLite,
no Chroma, no network. Checks that numbers come from the Phase 4 services and nothing is guessed."""
import pytest

from recipe_agent.db import init_db, make_engine, make_session_factory
from recipe_agent.multi import ToolContext, build_registry, build_supervisor, run_supervisor
from recipe_agent.multi.agents import (BudgetRequest, BusinessRequest, CostRequest, PantryRequest,
                                       RouteDecision, ShoppingRequest, keyword_route)
from recipe_agent.multi.tools import (BudgetItem, ConvertArgs, PantryEntry, RecipeRef, ShoppingRecipe,
                                      find_egyptian_recipe, norm_text, ToolError)
from recipe_agent.services import (ExtraExpense, IngredientInput, PantryService, RecipeInput,
                                   plan_production, recipe_cost)


def ing(name, q=None, unit=None, taste=False):
    return IngredientInput(name=name, quantity=q, unit=unit, to_taste=taste)


RICE = [ing("rice", 400, "g"), ing("chicken", 1, "kg"), ing("salt", taste=True)]


class FakeLLM:
    """extract(schema, system, text): returns what the test queued for that schema, or raises."""

    def __init__(self, **by_schema):
        self.by_schema = by_schema
        self.calls = []

    def __call__(self, schema, system, text):
        self.calls.append(schema.__name__)
        v = self.by_schema.get(schema.__name__)
        if v is None:
            raise RuntimeError("503 UNAVAILABLE")
        return v(text) if callable(v) else v


FAKE_STORE = {"entries": {
    "egy:a": {"recipe_id": "egy:a", "name": "كشري", "name_en": "Koshari", "servings": 4,
              "raw": {"recipe": {"title": "كشري", "servings": "4", "masterIngredients": [
                  {"name": "أرز", "standardAmount": "2 كوب"}]}}},
    "egy:b": {"recipe_id": "egy:b", "name": "فراخ مشوية", "name_en": "Grilled chicken", "servings": 4, "raw": {}},
    "egy:c": {"recipe_id": "egy:c", "name": "فراخ بالفريك", "name_en": "Chicken with freekeh", "servings": 4, "raw": {}},
}}


@pytest.fixture
def sf():
    engine = make_engine("sqlite://")
    init_db(engine)
    return make_session_factory(engine)


def graph(llm, sf, **kw):
    kw.setdefault("recipe_pipeline", lambda text, uid: "RECIPES:" + text)
    return build_supervisor(extract=llm, session_factory=sf, **kw)


def route(*intents, lang="en"):
    return RouteDecision(intents=list(intents), language=lang)


def ask(g, text="x", uid="u1"):
    return g.invoke({"text": text, "user_id": uid})


# ---------------- tools ----------------
def test_registry_describes_typed_tools():
    reg = build_registry()
    assert {"recipe_cost", "shopping_list", "production_plan", "budget_check", "pantry_add", "pantry_use",
            "convert_units"} <= set(reg.names())
    d = {x["name"]: x for x in reg.describe()}
    assert d["pantry_add"]["writes"] and not d["recipe_cost"]["writes"]
    assert "properties" in d["recipe_cost"]["args"]


def test_tool_rejects_bad_args_and_disallowed_agent_without_raising(sf):
    reg, ctx = build_registry(), ToolContext("u1", sf)
    bad = reg.call("convert_units", {"quantity": -1, "ingredient": "rice"}, ctx)
    assert not bad.ok and bad.error_code == "bad_args"
    nope = reg.call("pantry_add", {"items": [{"name": "rice", "quantity": 1, "unit": "kg"}]}, ctx, allowed={"recipe_cost"})
    assert not nope.ok and nope.error_code == "not_allowed"
    assert PantryService(sf()).list_items("u1") == []     # the disallowed call changed nothing
    assert reg.call("nope", {}, ctx).error_code == "not_found"


def test_tool_cost_matches_the_service(sf):
    reg, ctx = build_registry(), ToolContext("u1", sf)
    res = reg.call("recipe_cost", {"recipe": {"name": "Rice", "servings": 4, "ingredients": [i.model_dump() for i in RICE]},
                                   "extra_expenses": [{"name": "box", "amount_egp": 10, "basis": "per_meal"}]}, ctx)
    direct = recipe_cost(RecipeInput(name="Rice", servings=4, ingredients=RICE,
                                     extra_expenses=[ExtraExpense(name="box", amount_egp=10, basis="per_meal")]))
    assert res.ok and res.data.total_egp == direct.total_egp and res.data.per_serving_egp == direct.per_serving_egp


def test_recipe_lookup_exact_ambiguous_missing():
    ctx = ToolContext(store_loader=lambda: FAKE_STORE)
    assert find_egyptian_recipe(ctx, "كشري")["recipe_id"] == "egy:a"
    assert find_egyptian_recipe(ctx, "koshari")["recipe_id"] == "egy:a"
    with pytest.raises(ToolError) as e:
        find_egyptian_recipe(ctx, "فراخ")
    assert e.value.code == "ambiguous" and len(e.value.candidates) == 2
    with pytest.raises(ToolError) as e:
        find_egyptian_recipe(ctx, "بيتزا")
    assert e.value.code == "not_found"
    assert norm_text("الكُشَري") == norm_text("الكشري")


def test_pantry_use_is_all_or_nothing(sf):
    reg, ctx = build_registry(), ToolContext("u1", sf)
    reg.call("pantry_add", {"items": [{"name": "rice", "quantity": 1, "unit": "kg"}, {"name": "sugar", "quantity": 100, "unit": "g"}]}, ctx)
    res = reg.call("pantry_use", {"items": [{"name": "rice", "quantity": 500, "unit": "g"},
                                            {"name": "sugar", "quantity": 500, "unit": "g"}]}, ctx)
    assert not res.ok and res.error_code == "insufficient_stock"
    stock = {v.name: v.quantity for v in PantryService(sf()).list_items("u1")}
    assert stock == {"rice": 1000, "sugar": 100}     # rice was NOT taken


# ---------------- routing ----------------
def test_keyword_router():
    assert keyword_route("هل ٣ وجبات داخل ميزانية 200 جنيه").intents == ["budget"]
    assert keyword_route("what is my pantry stock").intents == ["pantry"]
    assert keyword_route("تكلفة الكشري").intents == ["cost"]
    assert keyword_route("عايز قايمة شراء").intents == ["shopping"]
    assert keyword_route("عندي فراخ وأرز").intents == ["recipes"]
    assert keyword_route("عندي فراخ").language == "ar"


def test_router_llm_failure_falls_back_to_keywords(sf):
    llm = FakeLLM(CostRequest=CostRequest(task="convert", convert=ConvertArgs(quantity=2, unit="kg", ingredient="rice")))
    out = ask(graph(llm, sf), "how many grams is 2 kg rice, convert")   # RouteDecision not queued -> raises
    assert out["route_source"] == "keywords"
    assert "2000 g" in out["answer"]


def test_recipes_agent_delegates_to_existing_pipeline_and_touches_no_db(sf):
    llm = FakeLLM(RouteDecision=route("recipes", lang="ar"))
    out = ask(graph(llm, sf), "عندي فراخ وأرز")
    assert out["answer"] == "RECIPES:عندي فراخ وأرز"
    assert llm.calls == ["RouteDecision"]            # one LLM call, the pipeline does the rest


def test_recipes_agent_failure_is_reported_not_raised(sf):
    def boom(text, uid):
        raise RuntimeError("chroma missing")
    out = ask(graph(FakeLLM(RouteDecision=route("recipes")), sf, recipe_pipeline=boom))
    assert "could not fetch recipes" in out["answer"]


# ---------------- specialists ----------------
def test_cost_agent_numbers_come_from_services(sf):
    req = CostRequest(task="cost", recipe=RecipeRef(name="Rice", servings=4, ingredients=RICE))
    out = ask(graph(FakeLLM(RouteDecision=route("cost"), CostRequest=req), sf))
    direct = recipe_cost(RecipeInput(name="Rice", servings=4, ingredients=RICE))
    assert f"{direct.total_egp:,.2f}" in out["answer"] and f"{direct.per_serving_egp:,.2f}" in out["answer"]
    assert [r.tool_calls for r in out["results"]] == [[("recipe_cost", True)]]


def test_cost_agent_arabic_answer_and_real_egyptian_recipe(sf):
    req = CostRequest(task="cost", recipe=RecipeRef(name="فراخ معصجة"))
    out = ask(graph(FakeLLM(RouteDecision=route("cost", lang="ar"), CostRequest=req), sf), "تكلفة فراخ معصجة")
    assert "تكلفة" in out["answer"] and "جنيه" in out["answer"] and "للفرد" in out["answer"]


def test_incomplete_cost_is_labelled_incomplete(sf):
    req = CostRequest(task="cost", recipe=RecipeRef(name="Mystery", servings=2, ingredients=[ing("rice", 200, "g"), ing("zzz unobtainium", 100, "g")]))
    out = ask(graph(FakeLLM(RouteDecision=route("cost"), CostRequest=req), sf))
    assert "incomplete" in out["answer"] and "zzz unobtainium" in out["answer"]


def test_cost_agent_unknown_recipe_asks_instead_of_guessing(sf):
    req = CostRequest(task="cost", recipe=RecipeRef(name="بيتزا نيزكية"))
    out = ask(graph(FakeLLM(RouteDecision=route("cost"), CostRequest=req), sf))
    assert out["results"][0].needs_input and "ingredients" in out["answer"]


def test_cost_agent_ambiguous_recipe_lists_choices(sf):
    req = CostRequest(task="cost", recipe=RecipeRef(name="فراخ"))
    out = ask(graph(FakeLLM(RouteDecision=route("cost", lang="ar"), CostRequest=req), sf, store_loader=lambda: FAKE_STORE))
    assert out["results"][0].needs_input and "فراخ مشوية" in out["answer"] and "فراخ بالفريك" in out["answer"]


def test_cost_agent_missing_recipe_asks(sf):
    out = ask(graph(FakeLLM(RouteDecision=route("cost"), CostRequest=CostRequest(task="cost")), sf))
    assert out["results"][0].needs_input


def test_extraction_failure_gives_friendly_message(sf):
    out = ask(graph(FakeLLM(RouteDecision=route("budget")), sf))      # BudgetRequest not queued -> raises
    assert "could not read the request" in out["answer"] and not out["results"][0].ok


def test_pantry_agent_add_list_use_and_insufficient(sf):
    g_add = graph(FakeLLM(RouteDecision=route("pantry"), PantryRequest=PantryRequest(
        action="add", items=[PantryEntry(name="rice", quantity=2, unit="kg")])), sf)
    out = ask(g_add)
    assert "Added" in out["answer"] and "rice: 2000 g" in out["answer"]
    g_use = graph(FakeLLM(RouteDecision=route("pantry"), PantryRequest=PantryRequest(
        action="use", items=[PantryEntry(name="rice", quantity=5, unit="kg")])), sf)
    out = ask(g_use)
    assert "Not enough" in out["answer"] and not out["results"][0].ok
    g_list = graph(FakeLLM(RouteDecision=route("pantry"), PantryRequest=PantryRequest(action="list")), sf)
    assert "rice: 2000 g" in ask(g_list)["answer"]
    assert "pantry is empty" in ask(g_list, uid="someone-else")["answer"]      # per-user


def test_pantry_agent_asks_when_quantity_missing(sf):
    g = graph(FakeLLM(RouteDecision=route("pantry"), PantryRequest=PantryRequest(
        action="add", items=[PantryEntry(name="rice")])), sf)
    out = ask(g)
    assert out["results"][0].needs_input and "quantity" in out["answer"]
    assert PantryService(sf()).list_items("u1") == []


def test_pantry_agent_expiring_and_low_stock(sf):
    from datetime import date, timedelta
    today = date(2026, 11, 27)
    g_add = graph(FakeLLM(RouteDecision=route("pantry"), PantryRequest=PantryRequest(
        action="add", items=[PantryEntry(name="milk", quantity=1, unit="l", expiry=today + timedelta(days=1), low_stock_threshold=2000)])), sf, today=today)
    ask(g_add)
    out = ask(graph(FakeLLM(RouteDecision=route("pantry"), PantryRequest=PantryRequest(action="expiring")), sf, today=today))
    assert "Expiring" in out["answer"] and "milk" in out["answer"]
    out = ask(graph(FakeLLM(RouteDecision=route("pantry"), PantryRequest=PantryRequest(action="low_stock")), sf))
    assert "milk" in out["answer"]


def test_shopping_agent_subtracts_pantry(sf):
    PantryService(sf()).add_item("u1", "rice", 300, "g")
    req = ShoppingRequest(recipes=[ShoppingRecipe(recipe=RecipeRef(name="Rice", servings=4, ingredients=RICE))])
    out = ask(graph(FakeLLM(RouteDecision=route("shopping"), ShoppingRequest=req), sf))
    assert "rice: 100 g" in out["answer"] and "chicken: 1000 g" in out["answer"] and "to taste" in out["answer"]
    req2 = req.model_copy(update={"use_pantry": False})
    out2 = ask(graph(FakeLLM(RouteDecision=route("shopping"), ShoppingRequest=req2), sf))
    assert "rice: 400 g" in out2["answer"]


def test_shopping_agent_needs_a_recipe(sf):
    out = ask(graph(FakeLLM(RouteDecision=route("shopping"), ShoppingRequest=ShoppingRequest()), sf))
    assert out["results"][0].needs_input


def test_business_agent_profit_matches_service(sf):
    req = BusinessRequest(recipe=RecipeRef(name="Rice", servings=4, ingredients=RICE), meals=20,
                          selling_price_per_meal=100, extra_expenses=[ExtraExpense(name="pack", amount_egp=2, basis="per_meal")],
                          target_margin_pct=40)
    out = ask(graph(FakeLLM(RouteDecision=route("business"), BusinessRequest=req), sf))
    plan = plan_production(RecipeInput(name="Rice", servings=4, ingredients=RICE), 20, 100,
                           [ExtraExpense(name="pack", amount_egp=2, basis="per_meal")])
    assert f"{plan.gross_profit_egp:,.2f}" in out["answer"] and f"{plan.cost_per_meal_egp:,.2f}" in out["answer"]
    assert "Lowest price per meal for a 40% margin" in out["answer"]


def test_business_agent_hides_profit_when_cost_incomplete(sf):
    req = BusinessRequest(recipe=RecipeRef(name="M", servings=2, ingredients=[ing("rice", 200, "g"), ing("zzz unobtainium", 100, "g")]),
                          meals=10, selling_price_per_meal=50, target_margin_pct=30)
    out = ask(graph(FakeLLM(RouteDecision=route("business"), BusinessRequest=req), sf))
    assert "Profit not shown" in out["answer"] and "No price suggestion" in out["answer"]
    assert "Profit:" not in out["answer"]


def test_business_agent_asks_for_meals(sf):
    req = BusinessRequest(recipe=RecipeRef(name="Rice", servings=4, ingredients=RICE))
    out = ask(graph(FakeLLM(RouteDecision=route("business"), BusinessRequest=req), sf))
    assert out["results"][0].needs_input and "How many meals" in out["answer"]


def test_budget_agent_verdicts(sf):
    items = [BudgetItem(recipe=RecipeRef(name="Rice", servings=4, ingredients=RICE), meals=4)]
    cost = recipe_cost(RecipeInput(name="Rice", servings=4, ingredients=RICE)).total_egp
    ok = ask(graph(FakeLLM(RouteDecision=route("budget"), BudgetRequest=BudgetRequest(items=items, budget_egp=cost + 50)), sf))
    assert "Within budget" in ok["answer"]
    over = ask(graph(FakeLLM(RouteDecision=route("budget"), BudgetRequest=BudgetRequest(items=items, budget_egp=cost - 1)), sf))
    assert "Over budget by 1.00" in over["answer"]
    miss = [BudgetItem(recipe=RecipeRef(name="M", servings=2, ingredients=[ing("rice", 200, "g"), ing("zzz unobtainium", 1, "kg")]), meals=2)]
    cant = ask(graph(FakeLLM(RouteDecision=route("budget"), BudgetRequest=BudgetRequest(items=miss, budget_egp=10000)), sf))
    assert "Cannot tell" in cant["answer"]


def test_budget_agent_asks_for_missing_budget(sf):
    out = ask(graph(FakeLLM(RouteDecision=route("budget"), BudgetRequest=BudgetRequest(items=[])), sf))
    assert out["results"][0].needs_input and "budget" in out["answer"]


def test_convert_agent_arabic(sf):
    req = CostRequest(task="convert", convert=ConvertArgs(quantity=1, unit="cup", ingredient="rice"))
    out = ask(graph(FakeLLM(RouteDecision=route("cost", lang="ar"), CostRequest=req), sf))
    assert "تقريبي" in out["answer"]       # a cup is an estimate, and it says so


# ---------------- supervisor behaviour ----------------
def test_two_agents_run_in_order_and_answers_are_joined(sf):
    llm = FakeLLM(RouteDecision=route("pantry", "cost"),
                  PantryRequest=PantryRequest(action="add", items=[PantryEntry(name="rice", quantity=1, unit="kg")]),
                  CostRequest=CostRequest(task="convert", convert=ConvertArgs(quantity=1, unit="kg", ingredient="rice")))
    out = ask(graph(llm, sf))
    assert [r.agent for r in out["results"]] == ["pantry", "cost"]
    assert llm.calls == ["RouteDecision", "PantryRequest", "CostRequest"]
    assert out["answer"].index("Added") < out["answer"].index("1000 g")


def test_chain_stops_when_an_agent_needs_input(sf):
    llm = FakeLLM(RouteDecision=route("budget", "cost"), BudgetRequest=BudgetRequest(),
                  CostRequest=CostRequest(task="convert", convert=ConvertArgs(quantity=1, unit="kg", ingredient="rice")))
    out = ask(graph(llm, sf))
    assert [r.agent for r in out["results"]] == ["budget"] and "CostRequest" not in llm.calls


def test_duplicate_intents_run_once(sf):
    llm = FakeLLM(RouteDecision=route("recipes", "recipes"))
    out = ask(graph(llm, sf))
    assert len(out["results"]) == 1


def test_run_supervisor_entry_point(sf):
    out = run_supervisor("hi", "u1", extract=FakeLLM(RouteDecision=route("recipes")), session_factory=sf,
                         recipe_pipeline=lambda t, u: f"{u}:{t}")
    assert out["answer"] == "u1:hi"


def test_default_database_is_created_lazily(tmp_path, monkeypatch):
    monkeypatch.setenv("RECIPE_AGENT_DB_URL", f"sqlite:///{tmp_path/'m.db'}")
    out = run_supervisor("x", "u1", extract=FakeLLM(
        RouteDecision=route("pantry"), PantryRequest=PantryRequest(action="add", items=[PantryEntry(name="rice", quantity=1, unit="kg")])))
    assert "Added" in out["answer"] and (tmp_path / "m.db").exists()
