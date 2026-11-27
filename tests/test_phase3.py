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


# --------------------------------------------------------------------------- #
# Phase 3.1: clear note when no Egyptian recipe matches the conditions
# --------------------------------------------------------------------------- #
def _intl_recipe():
    """A Food.com-style result (as find_recipes builds for the English index)."""
    return {
        "recipe_id": 12345,
        "name": "Lemon Chicken",
        "category": "chicken",
        "total_minutes": 25,
        "servings": 4,
        "nutrition": {"calories": 300, "protein_g": 20.0, "carbs_g": 10.0,
                      "fat_g": 12.0, "sugar_g": 2.0},
        "used": 1,
        "have": ["chicken"],
        "missing": ["lemon"],
        "coverage": 0.5,
        "distance": 0.4,
        "instructions": ["Cook it."],
        "substitutions": [],
    }


def test_no_match_note_none_when_results_empty():
    assert egypt_store.no_match_note([], ["فراخ"], lang="ar") is None


def test_no_match_note_none_when_egyptian_shown(store):
    found = egypt_store.find(["فراخ"], top_k=1, lang="ar")
    assert egypt_store.no_match_note(found, ["فراخ"], lang="ar") is None


def test_no_match_note_none_when_egyptian_matches_same_conditions(store):
    # Egyptian recipes DO match chicken without constraints: they only ranked out
    # of the shown list, so there is nothing to explain.
    assert egypt_store.no_match_note([_intl_recipe()], ["فراخ"], lang="ar") is None


def test_reported_gap_chicken_milk_allergy_under_30_min(store):
    # The Phase 3.1 case: nothing Egyptian is left, only international recipes are shown.
    assert egypt_store.find(["فراخ"], max_minutes=30, exclude=["milk"], top_k=5) == []


def test_no_match_note_suggests_closest_with_more_time(store):
    note = egypt_store.no_match_note(
        [_intl_recipe()], ["فراخ"], max_minutes=30, exclude=["milk"], lang="ar"
    )
    assert note and "مفيش وصفة مصرية" in note and "عالمية" in note
    assert "أقرب وصفة مصرية" in note and "دقيقة" in note
    # The suggested recipe really is milk-free: the allergy filter is never relaxed.
    relaxed = egypt_store.find(["فراخ"], exclude=["milk"], top_k=1, lang="ar")
    assert relaxed and relaxed[0]["name"] in note
    assert "milk" not in store["entries"][relaxed[0]["recipe_id"]]["families"]


def test_no_match_note_plain_when_nothing_even_without_time_limit(store):
    # Cheese dishes all contain milk: with a milk allergy there is no closest option.
    note = egypt_store.no_match_note([_intl_recipe()], ["جبنة"], exclude=["milk"], lang="ar")
    assert note and "مفيش وصفة مصرية" in note and "أقرب وصفة مصرية" not in note


def test_no_match_note_english(store):
    note = egypt_store.no_match_note(
        [_intl_recipe()], ["chicken"], max_minutes=30, exclude=["milk"], lang="en"
    )
    assert note and note.startswith("Note: no Egyptian recipe")
    assert "closest Egyptian recipe" in note


def test_no_match_note_never_blocks_the_answer_check(store):
    req = UserRequest(
        ingredients=["فراخ"], exclude_ingredients=["milk"], max_minutes=30, language="ar"
    )
    note = egypt_store.no_match_note(
        [_intl_recipe()], req.ingredients, max_minutes=30, exclude=["milk"], lang="ar"
    )
    blocking, _ = verifier.check_answer(req, [_intl_recipe()], note + "\n\n### Lemon Chicken")
    assert blocking == []


def test_respond_node_prepends_no_match_note(monkeypatch):
    from recipe_agent import graph

    req = UserRequest(
        ingredients=["فراخ"], exclude_ingredients=["milk"], max_minutes=30, language="ar"
    )
    monkeypatch.setattr(graph, "write_answer", lambda *a, **k: "ANSWER")
    out = graph.respond_node({"request": req, "recipes": [_intl_recipe()], "shopping": []})
    assert out["answer"].startswith("ملاحظة: مفيش وصفة مصرية")
    assert "ANSWER" in out["answer"] and out["egypt_note"]


def test_respond_node_no_note_when_egyptian_recipe_shown(monkeypatch, store):
    from recipe_agent import graph

    found = egypt_store.find(["فراخ"], top_k=1, lang="ar")
    recipe = dict(found[0])
    recipe["instructions"] = egypt_store.instructions(recipe["recipe_id"], "ar")
    recipe["substitutions"] = []
    req = UserRequest(ingredients=["فراخ"], language="ar")
    monkeypatch.setattr(graph, "write_answer", lambda *a, **k: "ANSWER")
    out = graph.respond_node({"request": req, "recipes": [recipe], "shopping": []})
    # No note; only the guaranteed cost line is added to the writer's answer.
    assert out["answer"].startswith("ANSWER") and not out.get("egypt_note")


def test_fallback_answers_keep_the_note(monkeypatch):
    from recipe_agent import graph

    req = UserRequest(
        ingredients=["فراخ"], exclude_ingredients=["milk"], max_minutes=30, language="ar"
    )

    def boom(*a, **k):
        raise RuntimeError("quota")

    monkeypatch.setattr(graph, "write_answer", boom)
    out = graph.respond_node({"request": req, "recipes": [_intl_recipe()], "shopping": []})
    assert out["fallback"] == "llm_error" and out["answer"].startswith("ملاحظة")

    # The check step swaps a leaking answer for the plain one and keeps the note.
    out2 = graph.check_node(
        {
            "request": req,
            "recipes": [],
            "shopping": [],
            "answer": "Just add milk and stir.",
            "verifier_report": {},
            "egypt_note": out["egypt_note"],
        }
    )
    assert out2["fallback"] == "check_failed" and out2["answer"].startswith("ملاحظة")
