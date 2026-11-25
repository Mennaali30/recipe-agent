"""Phase 3 tests: curated Egyptian recipes wired into the app.

Offline: reads the shipped data/egypt files, no Chroma, no LLM, no network.
Run: pytest -q
"""
import json
from pathlib import Path

import pytest

from recipe_agent import egypt_store, tools, verifier
from recipe_agent.egypt_cost import estimate_recipe_cost, load_price_map
from recipe_agent.egypt_parse import grams, match, parse_amount
from recipe_agent.schemas import UserRequest

DATA = Path(__file__).resolve().parents[1] / "data" / "egypt"
BUCKETS = ("chicken", "meat", "soups", "mahshi", "breakfast", "street")


@pytest.fixture(scope="module")
def store():
    return egypt_store.load_store()


# --------------------------------------------------------------------------- #
# Curated data
# --------------------------------------------------------------------------- #
def test_curated_set_is_150_in_six_buckets():
    curated = json.loads((DATA / "curated.json").read_text(encoding="utf-8"))
    assert set(curated["buckets"]) == set(BUCKETS)
    for bucket in BUCKETS:
        assert len(curated["buckets"][bucket]) == 25, bucket
    ids = [i for ids in curated["buckets"].values() for i in ids]
    assert len(ids) == 150 and len(set(ids)) == 150
    for rid in ids:
        assert (DATA / "recipes" / f"{rid}.json").exists(), rid


def test_every_curated_recipe_is_usable(store):
    for e in store["entries"].values():
        assert e["total_minutes"] is not None, e["recipe_id"]  # the time filter can use it
        assert e["cost"]["total_egp"] is not None, e["recipe_id"]
        assert len(e["ingredients"]) >= 3 and len(e["instructions"]) >= 2
        assert 5 <= e["cost"]["total_egp"] <= 1500  # spot-check band from the curation


# --------------------------------------------------------------------------- #
# Parser fixes (found while spot-checking costs)
# --------------------------------------------------------------------------- #
def test_whole_chicken_counts_as_whole_bird():
    a = parse_amount("دجاجة")
    assert a["unit"] == "whole_chicken" and a["qty"] == 1
    row = match("دجاجة", load_price_map())
    assert grams({"qty": 2, "unit": None}, row, "دجاج بوزن كيلو") == 2000  # name hint


def test_arabic_two_and_persian_cup():
    assert parse_amount("اثنان")["qty"] == 2
    a = parse_amount("كوپان")  # پ in some recipes
    assert (a["qty"], a["unit"]) == (2, "cup")


def test_milliliters_are_not_litres():
    a = parse_amount("500 مليلتر")
    assert (a["qty"], a["unit"]) == (500, "ml")
    assert parse_amount("1 لتر")["unit"] == "l"


def test_minced_poultry_is_not_priced_as_beef():
    pm = load_price_map()
    assert match("فراخ مفرومة", pm)["key"] == "chicken"
    assert match("جمبري مفروم", pm)["key"] == "shrimp"
    assert match("لحم مفروم", pm)["key"] in ("minced_beef", "beef_generic")


# --------------------------------------------------------------------------- #
# Search, time filter, cost
# --------------------------------------------------------------------------- #
def test_find_matches_english_ingredients(store):
    found = egypt_store.find(["chicken", "rice"], top_k=5, lang="ar")
    assert found and all(r["egyptian"] for r in found)
    assert all(r["used"] >= 1 for r in found)
    assert all(r["cost_text"].startswith("التكلفة التقريبية") for r in found)


def test_find_matches_arabic_ingredients(store):
    found = egypt_store.find(["فراخ", "أرز"], top_k=5, lang="ar")
    assert found and found[0]["used"] >= 1


def test_under_n_minutes_filter(store):
    found = egypt_store.find(["بيض"], max_minutes=30, top_k=25)
    assert found
    assert all(r["total_minutes"] <= 30 for r in found)


def test_cost_line_names_the_price_kind(store):
    found = egypt_store.find(["لحم"], top_k=3, lang="en")
    assert found
    assert "Approx. cost" in found[0]["cost_text"]
    assert any(
        kind in found[0]["cost_text"]
        for kind in ("WFP price", "current market price", "rough estimate")
    )


# --------------------------------------------------------------------------- #
# Allergy filter on Arabic data + fifi flags cross-check
# --------------------------------------------------------------------------- #
def test_allergy_filter_drops_milk_recipes(store):
    found = egypt_store.find(["بيض", "جبنة"], exclude=["milk"], top_k=25)
    for r in found:
        assert "milk" not in store["entries"][r["recipe_id"]]["families"]


def test_allergy_filter_uses_arabic_words(store):
    found = egypt_store.find(["فراخ"], exclude=["سمسم"], top_k=25)
    for r in found:
        e = store["entries"][r["recipe_id"]]
        assert "sesame" not in e["families"]


def test_fifi_flag_excludes_even_when_words_miss(store):
    """A recipe flagged by fifi but with no allergen word in the text is still dropped."""
    raw = {
        "recipe": {
            "id": "fake-1", "title": "أكلة تجريبية",
            "masterIngredients": [{"name": "دقيق", "nameEn": "flour", "standardAmount": "1 كيلو"}],
            "uniqueInstructions": [{"stepNumber": 1, "text": "تخلط وتطهى."}],
            "prepTime": "10 دقائق",
        },
        "allergens": {"contains": ["milk"], "status": "contains"},
    }
    entry = egypt_store._entry("fake-1", raw, "breakfast", load_price_map())
    assert "milk" in entry["families"]
    assert egypt_store.excluded(entry, ["milk"])
    assert not egypt_store.excluded(entry, ["egg"])


def test_plain_food_exclusion_by_key_and_name(store):
    found = egypt_store.find(["فراخ", "طماطم"], exclude=["tomato"], top_k=25)
    for r in found:
        e = store["entries"][r["recipe_id"]]
        assert "tomato" not in e["ingredient_keys"]


def test_verifier_rechecks_egyptian_recipes(store):
    found = egypt_store.find(["بيض"], top_k=5)
    milk_one = next(
        (r for r in (egypt_store.find(["بيض", "جبنة"], top_k=25))
         if "milk" in store["entries"][r["recipe_id"]]["families"]),
        None,
    )
    assert milk_one is not None  # the set has a milk dish to test with
    req = UserRequest(ingredients=["eggs"], exclude_ingredients=["milk"])
    kept, issues = verifier.verify_recipes(req, [milk_one])
    assert kept == [] and issues


# --------------------------------------------------------------------------- #
# Integration: find_recipes merges without Chroma, instructions language
# --------------------------------------------------------------------------- #
def test_find_recipes_merges_egyptian_without_chroma(monkeypatch):
    monkeypatch.setattr(tools, "search", lambda *a, **k: [])  # English index unavailable
    found = tools.find_recipes(["chicken", "rice"], top_k=3, lang="ar")
    assert found and all(r["egyptian"] for r in found)
    assert all("cost_text" in r for r in found)


def test_instructions_follow_the_request_language():
    rid = next(iter(egypt_store.load_store()["entries"]))
    ar = tools.get_instructions(rid, lang="ar")
    en = tools.get_instructions(rid, lang="en")
    assert ar and en and len(ar) == len(en)


def test_ensure_cost_lines_appends_missing():
    found = egypt_store.find(["فراخ"], top_k=1, lang="ar")
    answer = "إجابة من غير تكلفة"
    out = egypt_store.ensure_cost_lines(answer, found, "ar")
    assert found[0]["cost_text"] in out
    assert egypt_store.ensure_cost_lines(out, found, "ar") == out  # idempotent


def test_fallback_answer_shows_cost_and_handles_missing_nutrition():
    found = egypt_store.find(["فراخ"], top_k=1, lang="ar")
    r = dict(found[0])
    r["instructions"] = egypt_store.instructions(r["recipe_id"], "ar")
    req = UserRequest(ingredients=["فراخ"], language="ar")
    text = verifier.fallback_answer(req, [r], [])
    assert r["cost_text"] in text
