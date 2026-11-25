"""Phase 2 tests: Egyptian data parsers, price map, EGP cost estimate, time filter.
Offline: uses data/egypt/ingredient_price_map.csv and tests/fixtures/. Run: pytest -q"""
import csv
import json
from pathlib import Path

import pytest

from recipe_agent.egypt_cost import (DEFAULT_MAP, estimate_line, estimate_recipe_cost,
                                     format_cost, load_price_map, recipe_minutes,
                                     within_minutes)
from recipe_agent.egypt_parse import match, parse_amount, parse_minutes, parse_servings

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def pmap():
    return load_price_map()


@pytest.fixture(scope="module")
def veg04():
    return json.load(open(FIX / "fifi_veg04.json", encoding="utf-8"))


# ---- price map file ----
def test_price_map_valid_sources_and_prices():
    rows = list(csv.DictReader(open(DEFAULT_MAP, encoding="utf-8-sig")))
    assert len(rows) > 50
    for r in rows:
        assert r["price_source"] in {"wfp", "current_web", "estimated"}
        assert float(r["price_egp_per_kg"]) >= 0
        if r["price_source"] == "current_web":
            assert r["current_price_source_url"] and r["current_price_date"]


# ---- amounts ----
def test_amount_arabic_digits_and_kg():
    a = parse_amount("٢ كيلو")
    assert (a["qty"], a["unit"]) == (2, "kg")


def test_amount_fraction_and_word_quarter():
    assert parse_amount("1/4 كيلو")["qty"] == 0.25
    assert parse_amount("ربع كيلو")["qty"] == 0.25


def test_amount_range_midpoint():
    a = parse_amount("2-4 ملعقة كبيرة")
    assert a["qty"] == 3 and a["unit"] == "tbsp"


def test_amount_to_taste_has_no_quantity():
    a = parse_amount("حسب الرغبة")
    assert a["to_taste"] and a["qty"] is None


def test_bare_spoon_is_flagged_low_confidence():
    assert parse_amount("1 ملعقة")["conf"] == "low"


# ---- times ----
@pytest.mark.parametrize("text,expected", [
    ("20 دقيقة", 20), ("ساعة", 60), ("ساعتين", 120), ("ساعة ونصف", 90),
    ("30-40 دقيقة", 40), ("بدون طهو", 0), ("بضع ساعات", None), ("-", None), ("", None)])
def test_parse_minutes(text, expected):
    assert parse_minutes(text) == expected


def test_parse_servings_range_midpoint():
    assert parse_servings("4-6 أفراد") == 5


# ---- mapping ----
def test_match_first_keyword_priority(pmap):
    assert match("بصل كبير", pmap)["key"] == "onion"
    assert match("طماطم", pmap)["key"] == "tomato"
    assert match("شيء غير موجود", pmap) is None


# ---- cost ----
def test_line_cost_math(pmap):
    line = estimate_line({"name": "طماطم", "standardAmount": "1/4 كيلو"}, pmap)
    assert line["grams"] == 250
    assert line["cost_egp"] == round(0.25 * line["price_egp_per_kg"], 2)
    assert line["price_source"] in {"wfp", "current_web", "estimated"}


def test_to_taste_line_is_not_priced(pmap):
    line = estimate_line({"name": "ملح وفلفل", "standardAmount": "حسب الرغبة"}, pmap)
    assert line["cost_egp"] is None and line["to_taste"]


def test_recipe_cost_is_egp_and_approximate(veg04, pmap):
    est = estimate_recipe_cost(veg04, pmap)
    assert est["currency"] == "EGP" and est["approximate"]
    assert est["total_egp"] == pytest.approx(sum(l["cost_egp"] for l in est["lines"] if l["cost_egp"]))
    assert est["per_serving_egp"] == pytest.approx(est["total_egp"] / 5, abs=0.01)
    assert 0 < est["coverage"] < 1
    assert "خضار مشكل" in est["unpriced"]


def test_fifi_estimate_cost_is_ignored(veg04, pmap):
    veg04 = dict(veg04, estimate={"cost": 999999})
    assert estimate_recipe_cost(veg04, pmap)["total_egp"] < 999999
    assert "999999" not in json.dumps(estimate_recipe_cost(veg04, pmap), ensure_ascii=False)


def test_recipe_source_is_least_trusted_priced_line(pmap):
    web = {"name": "طماطم", "standardAmount": "1 كيلو"}
    est_line = {"name": "خل", "standardAmount": "1 كوب"}
    assert estimate_line(web, pmap)["price_source"] == "current_web"
    assert estimate_line(est_line, pmap)["price_source"] == "estimated"
    only_web = estimate_recipe_cost({"id": "a", "servings": "2", "masterIngredients": [web]}, pmap)
    assert only_web["price_source"] == "current_web"
    mixed = estimate_recipe_cost({"id": "b", "servings": "2", "masterIngredients": [web, est_line]}, pmap)
    assert mixed["price_source"] == "estimated"


def test_nothing_priced_gives_no_total(pmap):
    est = estimate_recipe_cost({"id": "c", "masterIngredients": [
        {"name": "ملح", "standardAmount": "حسب الرغبة"}]}, pmap)
    assert est["total_egp"] is None and est["price_source"] is None
    assert format_cost(est, "en") == "Cost not available"


def test_format_cost_labels_estimates(veg04, pmap):
    est = estimate_recipe_cost(veg04, pmap)
    assert "تقديري" in format_cost(est, "ar")
    assert "EGP" in format_cost(est, "en") and "rough estimate" in format_cost(est, "en")


# ---- time filter ----
def test_recipe_minutes_prep_plus_cook(veg04):
    assert recipe_minutes(veg04) == 65


def test_within_minutes_filter(veg04):
    assert within_minutes(veg04, 90) and not within_minutes(veg04, 60)
    assert within_minutes(veg04, None)


def test_unknown_time_excluded_unless_allowed():
    r = {"recipe": {"prepTime": "-", "cookTime": "بضع ساعات"}}
    assert recipe_minutes(r) is None
    assert not within_minutes(r, 30)
    assert within_minutes(r, 30, unknown_ok=True)


def test_marinating_hours_do_not_pass_short_filter():
    r = {"recipe": {"prepTime": "12 ساعة", "cookTime": "20 دقيقة"}}
    assert not within_minutes(r, 30)
