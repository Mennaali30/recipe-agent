"""Phase 4: deterministic services (units, pantry, cost, shopping, production, budget). Offline."""
from datetime import date

import pytest

from recipe_agent.db import init_db, make_engine, make_session_factory
from recipe_agent.egypt_store import load_store
from recipe_agent.services import (ExtraExpense, IngredientInput, InsufficientStock, PantryService,
                                   RecipeInput, UnitError, build_shopping_list, check_budget,
                                   convert, convert_text, min_price_for_margin, plan_production,
                                   recipe_cost, recipe_from_fifi, resolve_ingredient)
from recipe_agent.services.costing import cost_line
from recipe_agent.services.units import normalize_unit


def ing(name, q=None, unit=None, text=None, taste=False):
    return IngredientInput(name=name, quantity=q, unit=unit, amount_text=text, to_taste=taste)


@pytest.fixture
def session():
    engine = make_engine("sqlite://")
    init_db(engine)
    s = make_session_factory(engine)()
    yield s
    s.close()


@pytest.fixture
def rice_recipe():
    return RecipeInput(name="Rice and chicken", servings=4, ingredients=[
        ing("rice", 400, "g"), ing("chicken", 1, "kg"), ing("salt", taste=True)])


# ---------- units ----------
def test_metric_conversions_are_verified():
    m = convert(2, "kg", "rice")
    assert (m.value, m.unit, m.status) == (2000, "g", "verified")
    assert convert(750, "ml", "unknown thing xyz").unit == "ml"
    assert convert(1.5, "l", "unknown thing xyz").value == 1500


def test_household_units_are_estimated_and_ingredient_aware():
    cup_rice = convert(1, "cup", "rice")
    cup_flour = convert(1, "cup", "flour")
    assert cup_rice.unit == cup_flour.unit == "g"
    assert cup_rice.status == "estimated"
    assert cup_rice.value != cup_flour.value      # different g_per_cup in the price map
    eggs = convert(3, "piece", "eggs")
    assert eggs.unit == "g" and eggs.status == "estimated" and eggs.value > 100


def test_unknown_ingredient_volume_stays_volume():
    m = convert(2, "cup", "zzz mystery powder")
    assert (m.unit, m.value, m.status) == ("ml", 480, "estimated")


def test_unknown_ingredient_count_stays_count_and_missing_quantity_is_unknown():
    m = convert(3, "piece", "zzz mystery fruit")
    assert (m.unit, m.value) == ("piece", 3)
    u = convert(None, "g", "rice")
    assert u.value is None and u.status == "unknown"


def test_bad_unit_and_negative_quantity():
    with pytest.raises(UnitError):
        convert(1, "bucket", "rice")
    with pytest.raises(UnitError):
        convert(-1, "g", "rice")
    assert normalize_unit("Tablespoons") == "tbsp" and normalize_unit(None) is None


def test_arabic_amount_text_and_to_taste():
    m, taste = convert_text("2 كوب", "أرز")
    assert not taste and m.unit == "g" and m.value > 300
    m, taste = convert_text("حسب الرغبة", "ملح")
    assert taste and m.value is None
    m, _ = convert_text("500 جرام", "فراخ")
    assert (m.value, m.status) == (500, "verified")


def test_resolve_ingredient_identity():
    assert resolve_ingredient("Chicken").identity == resolve_ingredient("فراخ").identity
    a, b = resolve_ingredient("zzz powder one"), resolve_ingredient("zzz powder two")
    assert a.identity != b.identity and a.row is None


def test_coarse_groups_do_not_merge_by_key():
    a = resolve_ingredient("بقدونس")      # fresh_herbs group
    b = resolve_ingredient("شبت")
    if a.key == b.key == "fresh_herbs":
        assert a.identity != b.identity


# ---------- recipe cost ----------
def test_cost_matches_phase2_for_every_curated_recipe():
    for rid, e in load_store()["entries"].items():
        rc = recipe_cost(recipe_from_fifi(e))
        assert rc.ingredient_cost_egp == e["cost"]["total_egp"], rid
        assert rc.unpriced == e["cost"]["unpriced"], rid


def test_recipe_cost_per_serving_and_status(rice_recipe):
    rc = recipe_cost(rice_recipe)
    rice = next(l for l in rc.lines if l.name == "rice")
    chicken = next(l for l in rc.lines if l.name == "chicken")
    assert rc.ingredient_cost_egp == pytest.approx(rice.cost_egp + chicken.cost_egp)
    assert rc.per_serving_egp == pytest.approx(round(rc.total_egp / 4, 2))
    assert rc.complete and rc.coverage == 1.0
    assert rc.status in ("verified", "estimated")
    salt = next(l for l in rc.lines if l.name == "salt")
    assert salt.to_taste and salt.cost_egp is None


def test_extra_expenses_fixed_and_per_meal(rice_recipe):
    base = recipe_cost(rice_recipe)
    r = rice_recipe.model_copy(update={"extra_expenses": [
        ExtraExpense(name="gas", amount_egp=10),
        ExtraExpense(name="box", amount_egp=2, basis="per_meal")]})
    rc = recipe_cost(r)
    assert rc.extras_egp == 18.0
    assert rc.total_egp == pytest.approx(base.total_egp + 18)
    assert rc.per_serving_egp == pytest.approx(round(rc.total_egp / 4, 2))


def test_unpriced_ingredient_is_not_invented():
    r = RecipeInput(name="mystery", servings=2, ingredients=[
        ing("rice", 200, "g"), ing("zzz mystery powder", 100, "g")])
    rc = recipe_cost(r)
    assert not rc.complete
    assert rc.unpriced == ["zzz mystery powder"]
    assert rc.coverage == 0.5
    line = next(l for l in rc.lines if l.name == "zzz mystery powder")
    assert line.cost_egp is None and line.status == "unknown"
    assert rc.ingredient_cost_egp == pytest.approx(next(l for l in rc.lines if l.name == "rice").cost_egp)


def test_nothing_priced_gives_none():
    rc = recipe_cost(RecipeInput(name="x", servings=2, ingredients=[ing("zzz a", 1, "kg")]))
    assert rc.total_egp is None and rc.per_serving_egp is None and rc.status == "unknown"


def test_unknown_servings_gives_no_per_serving(rice_recipe):
    rc = recipe_cost(rice_recipe.model_copy(update={"servings": None}))
    assert rc.total_egp is not None and rc.per_serving_egp is None
    assert any("servings unknown" in n for n in rc.notes)


def test_per_meal_expense_without_servings_not_counted_and_incomplete(rice_recipe):
    r = rice_recipe.model_copy(update={"servings": None, "extra_expenses": [
        ExtraExpense(name="box", amount_egp=2, basis="per_meal")]})
    rc = recipe_cost(r)
    assert rc.extras_egp == 0 and not rc.complete


# ---------- pantry ----------
def test_pantry_add_use_and_transactions(session):
    p = PantryService(session)
    p.add_item("u1", "rice", 1, "kg")
    p.add_item("u1", "rice", 500, "g")
    v = p.use_item("u1", "rice", 300, "g")
    assert v.quantity == 1200 and v.unit == "g"
    kinds = [(t.kind, t.delta) for t in p.transactions("u1", "rice")]
    assert kinds == [("add", 1000), ("add", 500), ("use", -300)]
    assert p.transactions("u1")[-1].quantity_after == 1200


def test_pantry_insufficient_stock_changes_nothing(session):
    p = PantryService(session)
    p.add_item("u1", "rice", 100, "g")
    with pytest.raises(InsufficientStock) as e:
        p.use_item("u1", "rice", 500, "g")
    assert e.value.have == 100
    assert p.get_stock("u1", "rice")[0].value == 100
    v = p.use_item("u1", "rice", 500, "g", allow_partial=True)
    assert v.quantity == 0
    with pytest.raises(InsufficientStock):
        p.use_item("u1", "sugar", 1, "g")


def test_pantry_per_user_ownership(session):
    p = PantryService(session)
    p.add_item("alice", "rice", 1, "kg")
    p.add_item("bob", "rice", 2, "kg")
    assert p.get_stock("alice", "rice")[0].value == 1000
    assert p.get_stock("bob", "rice")[0].value == 2000
    p.use_item("bob", "rice", 2, "kg")
    assert p.get_stock("alice", "rice")[0].value == 1000
    assert p.list_items("carol") == []
    with pytest.raises(InsufficientStock):
        p.use_item("carol", "rice", 1, "g")
    assert p.transactions("alice")[0].user_id == "alice" and len(p.transactions("alice")) == 1


def test_pantry_requires_user_and_positive_quantity(session):
    p = PantryService(session)
    with pytest.raises(ValueError):
        p.add_item("", "rice", 1, "kg")
    with pytest.raises(ValueError):
        p.add_item("u", "rice", 0, "kg")
    with pytest.raises(ValueError):
        p.add_item("u", "rice", -5, "g")
    with pytest.raises(UnitError):
        p.add_item("u", "rice", 1, "bucket")


def test_pantry_expiry_and_low_stock(session):
    p = PantryService(session)
    today = date(2026, 11, 25)
    p.add_item("u1", "milk", 1, "l", expiry=date(2026, 11, 24))
    p.add_item("u1", "yogurt", 500, "g", expiry=date(2026, 11, 27))
    p.add_item("u1", "rice", 5, "kg", expiry=date(2027, 6, 1))
    p.add_item("u1", "sugar", 200, "g", low_stock_threshold=500)
    p.add_item("u1", "salt", 2000, "g", low_stock_threshold=500)
    assert [i.name for i in p.expired("u1", today)] == ["milk"]
    assert [i.name for i in p.expiring_soon("u1", 3, today)] == ["yogurt"]
    assert [i.name for i in p.low_stock("u1")] == ["sugar"]
    p.use_item("u1", "salt", 1700, "g")
    assert sorted(i.name for i in p.low_stock("u1")) == ["salt", "sugar"]


def test_earliest_expiry_is_kept(session):
    p = PantryService(session)
    p.add_item("u", "yogurt", 100, "g", expiry=date(2026, 12, 10))
    v = p.add_item("u", "yogurt", 100, "g", expiry=date(2026, 12, 1))
    assert v.expiry == date(2026, 12, 1)


def test_adjust_and_discard(session):
    p = PantryService(session)
    p.add_item("u", "rice", 1, "kg")
    assert p.adjust_item("u", "rice", 700, "g").quantity == 700
    assert p.transactions("u")[-1].delta == -300
    p.discard_item("u", "rice")
    assert p.list_items("u") == []
    assert p.list_items("u", include_empty=True)[0].quantity == 0
    with pytest.raises(KeyError):
        p.discard_item("u", "sugar")


def test_pantry_count_units_and_estimated_status(session):
    p = PantryService(session)
    v = p.add_item("u", "eggs", 6, "piece")
    assert v.unit == "g" and v.status == "estimated"
    v = p.add_item("u", "zzz mystery fruit", 4, "piece")
    assert v.unit == "piece" and v.quantity == 4


def test_pantry_persists_in_a_sqlite_file(tmp_path):
    url = f"sqlite:///{tmp_path / 'p.db'}"
    e1 = make_engine(url); init_db(e1)
    s1 = make_session_factory(e1)(); PantryService(s1).add_item("u", "rice", 1, "kg"); s1.close()
    s2 = make_session_factory(make_engine(url))()
    assert PantryService(s2).get_stock("u", "rice")[0].value == 1000


def test_db_url_from_env(monkeypatch):
    from recipe_agent.db import get_database_url
    monkeypatch.setenv("RECIPE_AGENT_DB_URL", "sqlite:///custom.db")
    assert get_database_url() == "sqlite:///custom.db"
    monkeypatch.delenv("RECIPE_AGENT_DB_URL")
    assert get_database_url() == "sqlite:///matbakhak.db"
    assert get_database_url("postgresql+psycopg://a:b@h/db").startswith("postgresql")


# ---------- shopping ----------
def _lines(*recipes):
    return [cost_line(i)[0] for r in recipes for i in r.ingredients]


def test_shopping_merges_duplicates_across_recipes():
    a = RecipeInput(name="a", servings=2, ingredients=[ing("rice", 200, "g"), ing("onion", 1, "kg")])
    b = RecipeInput(name="b", servings=2, ingredients=[ing("Rice", 300, "g"), ing("chicken", 1, "kg")])
    sl = build_shopping_list(_lines(a, b))
    rice = [i for i in sl.items if i.identity == "rice"]
    assert len(rice) == 1 and rice[0].needed == 500 and rice[0].to_buy == 500
    assert sl.total_complete and sl.estimated_total_egp > 0


def test_shopping_subtracts_pantry_and_prices_only_what_to_buy(session):
    p = PantryService(session)
    p.add_item("u", "rice", 300, "g")
    r = RecipeInput(name="a", servings=2, ingredients=[ing("rice", 1000, "g"), ing("sugar", 100, "g")])
    full = build_shopping_list(_lines(r))
    sl = build_shopping_list(_lines(r), p.stock_map("u"))
    rice = next(i for i in sl.items if i.identity == "rice")
    assert (rice.needed, rice.in_pantry, rice.to_buy) == (1000, 300, 700)
    assert rice.est_cost_egp == pytest.approx(next(i for i in full.items if i.identity == "rice").est_cost_egp * 0.7, abs=0.01)


def test_shopping_pantry_covers_item():
    r = RecipeInput(name="a", servings=2, ingredients=[ing("rice", 100, "g")])
    sl = build_shopping_list(_lines(r), {("rice", "g"): 500})
    it = sl.items[0]
    assert it.to_buy == 0 and it.est_cost_egp == 0.0 and sl.estimated_total_egp == 0.0 and sl.total_complete


def test_shopping_unit_mismatch_not_subtracted():
    r = RecipeInput(name="a", servings=2, ingredients=[ing("zzz mystery fruit", 3, "piece")])
    ident = resolve_ingredient("zzz mystery fruit").identity
    sl = build_shopping_list(_lines(r), {(ident, "g"): 500})
    assert sl.items[0].to_buy == 3 and "different unit" in sl.items[0].note


def test_shopping_unpriced_unknown_and_taste_stay_visible():
    r = RecipeInput(name="a", servings=2, ingredients=[
        ing("rice", 100, "g"), ing("zzz mystery powder", 50, "g"),
        ing("sugar", None, None), ing("salt", taste=True)])
    sl = build_shopping_list(_lines(r))
    assert not sl.total_complete
    assert sl.unpriced == ["zzz mystery powder"]
    assert sl.unknown_quantities == ["sugar"]
    assert any(i.to_taste and i.name == "salt" for i in sl.items)
    mystery = next(i for i in sl.items if i.name == "zzz mystery powder")
    assert mystery.est_cost_egp is None and mystery.price_status == "unknown"
    assert sl.estimated_total_egp == pytest.approx(next(i for i in sl.items if i.name == "rice").est_cost_egp)


def test_shopping_categories_sorted():
    r = RecipeInput(name="a", servings=2, ingredients=[
        ing("chicken", 500, "g"), ing("onion", 200, "g"), ing("rice", 300, "g"), ing("oil", 50, "ml")])
    cats = [i.category for i in build_shopping_list(_lines(r)).items]
    assert cats == ["produce", "meat_poultry", "grains_legumes", "oils_fats"]


# ---------- production ----------
def test_production_scaling_and_cost_per_meal(rice_recipe):
    one = recipe_cost(rice_recipe)
    plan = plan_production(rice_recipe, 20)
    assert plan.scale_factor == 5
    rice = next(l for l in plan.scaled_lines if l.name == "rice")
    assert rice.measure.value == 2000
    assert plan.ingredient_cost_egp == pytest.approx(one.ingredient_cost_egp * 5, abs=0.05)
    assert plan.cost_per_meal_egp == pytest.approx(one.per_serving_egp, abs=0.01)
    assert plan.missing_quantities == ["salt"]


def test_production_revenue_profit_margin(rice_recipe):
    plan = plan_production(rice_recipe, 10, selling_price_per_meal=100,
                           extra_expenses=[ExtraExpense(name="box", amount_egp=3, basis="per_meal"),
                                           ExtraExpense(name="gas", amount_egp=20)])
    assert plan.cost_complete
    assert plan.extras_egp == 50
    assert plan.revenue_egp == 1000
    assert plan.gross_profit_egp == pytest.approx(1000 - plan.total_cost_egp)
    assert plan.margin_pct == pytest.approx(plan.gross_profit_egp / 10, abs=0.06)


def test_production_without_price_has_no_revenue(rice_recipe):
    plan = plan_production(rice_recipe, 10)
    assert plan.revenue_egp is None and plan.gross_profit_egp is None and plan.margin_pct is None


def test_production_incomplete_cost_gives_no_profit():
    r = RecipeInput(name="m", servings=2, ingredients=[ing("rice", 200, "g"), ing("zzz mystery powder", 50, "g")])
    plan = plan_production(r, 4, selling_price_per_meal=50)
    assert not plan.cost_complete and plan.unpriced == ["zzz mystery powder"]
    assert plan.revenue_egp == 200
    assert plan.gross_profit_egp is None and plan.margin_pct is None
    assert plan.cost_per_meal_egp is not None   # covers priced lines only, flagged by cost_complete


def test_production_unknown_servings_cannot_scale(rice_recipe):
    plan = plan_production(rice_recipe.model_copy(update={"servings": None}), 10, 50)
    assert plan.scale_factor is None and plan.total_cost_egp is None and plan.scaled_lines == []
    assert any("servings unknown" in n for n in plan.notes)


def test_production_with_pantry_builds_shopping_list(rice_recipe):
    plan = plan_production(rice_recipe, 8, stock={("rice", "g"): 500})
    rice = next(i for i in plan.shopping.items if i.identity == "rice")
    assert (rice.needed, rice.in_pantry, rice.to_buy) == (800, 500, 300)


def test_production_validation(rice_recipe):
    with pytest.raises(ValueError):
        plan_production(rice_recipe, 0)
    with pytest.raises(ValueError):
        plan_production(rice_recipe, 5, selling_price_per_meal=-1)


def test_min_price_for_margin():
    assert min_price_for_margin(30, 25) == 40.0
    assert min_price_for_margin(30, 0) == 30.0
    with pytest.raises(ValueError):
        min_price_for_margin(30, 100)


def test_zero_selling_price_does_not_divide_by_zero(rice_recipe):
    plan = plan_production(rice_recipe, 4, selling_price_per_meal=0)
    assert plan.revenue_egp == 0 and plan.margin_pct is None and plan.gross_profit_egp < 0


# ---------- budget ----------
def test_budget_within_and_over(rice_recipe):
    cost = recipe_cost(rice_recipe).total_egp
    ok = check_budget([(rice_recipe, 4)], cost + 10)
    assert ok.verdict == "within_budget" and ok.remaining_egp == pytest.approx(10, abs=0.01)
    over = check_budget([(rice_recipe, 4)], cost - 10)
    assert over.verdict == "over_budget" and over.over_by_egp == pytest.approx(10, abs=0.01)
    double = check_budget([(rice_recipe, 4), (rice_recipe, 8)], cost * 3 + 1)
    assert double.known_total_egp == pytest.approx(cost * 3, abs=0.05)


def test_budget_missing_price_is_cannot_tell_unless_already_over():
    r = RecipeInput(name="m", servings=2, ingredients=[ing("rice", 200, "g"), ing("zzz mystery powder", 50, "g")])
    known = recipe_cost(r).total_egp
    chk = check_budget([(r, 2)], known + 100)
    assert chk.verdict == "cannot_tell" and not chk.complete and chk.unpriced
    assert check_budget([(r, 2)], known - 1).verdict == "over_budget"


def test_budget_unknown_servings_cannot_tell(rice_recipe):
    chk = check_budget([(rice_recipe.model_copy(update={"servings": None}), 4)], 1000)
    assert chk.verdict == "cannot_tell" and chk.known_total_egp is None


def test_budget_with_pantry_counts_only_to_buy(session, rice_recipe):
    p = PantryService(session)
    full = check_budget([(rice_recipe, 4)], 10_000)
    p.add_item("u", "rice", 1, "kg")
    p.add_item("u", "chicken", 1, "kg")
    chk = check_budget([(rice_recipe, 4)], 10_000, stock=p.stock_map("u"))
    assert chk.basis == "to_buy" and chk.known_total_egp == 0 and chk.verdict == "within_budget"
    assert full.known_total_egp > 0


def test_budget_validation(rice_recipe):
    with pytest.raises(ValueError):
        check_budget([(rice_recipe, 4)], -1)
    with pytest.raises(ValueError):
        check_budget([(rice_recipe, 0)], 10)


# ---------- real data ----------
def test_curated_recipe_end_to_end(session):
    e = next(iter(load_store()["entries"].values()))
    r = recipe_from_fifi(e, extra_expenses=[ExtraExpense(name="box", amount_egp=1, basis="per_meal")])
    plan = plan_production(r, 12, selling_price_per_meal=60, stock={})
    assert plan.scale_factor == pytest.approx(12 / r.servings, abs=1e-4)
    assert plan.shopping.items
    assert plan.currency == "EGP"
