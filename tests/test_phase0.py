"""Phase 0 tests: allergy safety, ingredient matching, memory, unknown cooking time.

Run with:  pip install pytest && pytest -q
No dataset, Chroma index, GPU or API key is needed.
"""
import json

import pytest

from recipe_agent import memory, tools, verifier
from recipe_agent.allergy import arabic_allergen_families, extract_allergies
from recipe_agent.retriever import _metadata, minutes_or_none
from recipe_agent.schemas import UserRequest


# --------------------------------------------------------------------------- #
# 1. Extraction of allergies from the raw message (no LLM)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text, expected",
    [
        ("عندي حساسية من الفول السوداني والبيض وعايزة أكلة بفراخ", {"peanut", "egg"}),
        ("عندي حساسيه من الجمبري", {"shellfish"}),
        ("أنا مبكلش لبن ولا جبنة وعندي أرز", {"milk"}),
        ("عندي حساسية من القمح", {"wheat", "gluten"}),
        ("من غير سمسم", {"sesame"}),
        ("عندي حساسية من المكسرات", {"nut"}),
        ("I'm allergic to peanuts and shrimp, I have rice", {"peanut", "shellfish"}),
        ("no nuts please, I have chicken", {"nut"}),
        ("I can't eat dairy or eggs", {"dairy", "egg"}),
    ],
)
def test_extract_allergies_from_raw_message(text, expected):
    assert expected <= set(extract_allergies(text))


@pytest.mark.parametrize(
    "text",
    [
        "عندي فراخ وأرز وثوم وعايزة أكلة تخلص في نص ساعة",
        "مفيش حساسية، عايزة أكلة بالجبنة",
        "معنديش حساسية من حاجة",
        "I have eggs and cheese",
        "I have chicken and no allergies",
        "عايزة أكلة بجوز الهند",
    ],
)
def test_extract_allergies_ignores_ingredients_and_negations(text):
    assert extract_allergies(text) == []


def test_unlisted_english_food_is_kept_as_plain_name():
    assert "strawberries" in extract_allergies("I am allergic to strawberries")


def test_coconut_is_not_a_tree_nut_in_arabic():
    assert arabic_allergen_families("لبن جوز الهند") == {"milk"}  # milk word only, no nut
    assert "nut" not in arabic_allergen_families("لبن جوز الهند")
    assert "nut" in arabic_allergen_families("جوز وبندق")


def test_parse_node_adds_extracted_allergy_even_if_llm_missed_it(monkeypatch):
    pytest.importorskip("langgraph")
    from recipe_agent import graph

    monkeypatch.setattr(graph, "parse_request", lambda text: UserRequest(ingredients=["chicken"]))
    out = graph.parse_node({"text": "عندي حساسية من الفول السوداني وعندي فراخ"})
    assert "peanut" in out["request"].exclude_ingredients
    assert out["request"].ingredients == ["chicken"]


# --------------------------------------------------------------------------- #
# 2. Allergen filter: larger lexicon
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "allergen, ingredient",
    [
        ("milk", "labneh"),
        ("egg", "egg yolk"),
        ("wheat", "orzo"),
        ("gluten", "panko"),
        ("fish", "swordfish steak"),
        ("shellfish", "calamari"),
        ("nut", "marzipan"),
        ("soy", "teriyaki"),
        ("peanut", "groundnut oil"),
    ],
)
def test_extended_lexicon_excludes(allergen, ingredient):
    assert tools._is_excluded("Test dish", [ingredient, "salt"], [allergen])


def test_unrelated_recipe_is_not_excluded():
    assert not tools._is_excluded("Rice and chicken", ["chicken breast", "rice"], ["peanut", "milk"])


# --------------------------------------------------------------------------- #
# 3. Arabic allergen check on the final answer
# --------------------------------------------------------------------------- #
def _request(**kw):
    return UserRequest(language="ar", **kw)


def test_arabic_answer_with_allergen_word_is_blocked():
    req = _request(exclude_ingredients=["milk"])
    blocking, _ = verifier.check_answer(req, [], "### رز بالخضار\nضيفي كوباية لبن وقلبي.")
    assert blocking


def test_arabic_answer_with_prefix_forms_is_blocked():
    req = _request(exclude_ingredients=["peanut"])
    blocking, _ = verifier.check_answer(req, [], "رشي الفول السوداني فوق الطبق")
    assert blocking


def test_arabic_answer_with_arabic_exclusion_item_is_blocked():
    req = _request(exclude_ingredients=["بيض"])
    blocking, _ = verifier.check_answer(req, [], "اخفقي البيض مع السكر")
    assert blocking


def test_clean_arabic_answer_passes_and_allergy_note_line_is_skipped():
    req = _request(exclude_ingredients=["milk"])
    answer = "### فراخ بالأرز\nسخني الزيت وضيفي الفراخ.\nراجعي ملصقات المنتجات للحساسية من اللبن."
    blocking, _ = verifier.check_answer(req, [], answer)
    assert not blocking


# --------------------------------------------------------------------------- #
# 4. Ingredient matching false positives
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "user, recipe_ing",
    [
        ("butter", "peanut butter"),
        ("peanut butter", "butter"),
        ("milk", "coconut milk"),
        ("coconut milk", "milk"),
        ("potato", "sweet potatoes"),
        ("sweet potato", "potato"),
        ("pepper", "bell pepper"),
    ],
)
def test_compound_foods_do_not_match_plain_ingredient(user, recipe_ing):
    res = tools.compare_ingredients([user], [recipe_ing], assume_pantry=False)
    assert res["have"] == [] and res["missing"] == [recipe_ing]


@pytest.mark.parametrize(
    "user, recipe_ing",
    [
        ("butter", "unsalted butter"),
        ("milk", "whole milk"),
        ("potato", "potatoes"),
        ("sweet potato", "sweet potatoes"),
        ("peanut butter", "creamy peanut butter"),
        ("chicken", "boneless skinless chicken breast"),
    ],
)
def test_real_matches_still_work(user, recipe_ing):
    res = tools.compare_ingredients([user], [recipe_ing], assume_pantry=False)
    assert res["have"] == [recipe_ing]


@pytest.mark.parametrize(
    "ing",
    ["salt and pepper", "Salt & Pepper", "salt, pepper", "salt and freshly ground black pepper",
     "salt", "pepper", "olive oil"],
)
def test_pantry_items_are_never_missing(ing):
    res = tools.compare_ingredients(["chicken"], [ing, "chicken"])
    assert res["missing"] == []


def test_pantry_list_with_a_real_ingredient_is_not_skipped():
    res = tools.compare_ingredients(["chicken"], ["salt and garlic"])
    assert res["missing"] == ["salt and garlic"]


def test_bell_pepper_is_not_pantry():
    assert tools.compare_ingredients(["chicken"], ["bell pepper"])["missing"] == ["bell pepper"]


# --------------------------------------------------------------------------- #
# 5. memory.py: no data loss on read error, atomic writes
# --------------------------------------------------------------------------- #
@pytest.fixture
def mem_file(tmp_path, monkeypatch):
    path = tmp_path / "profiles.json"
    monkeypatch.setenv("RECIPE_MEMORY_PATH", str(path))
    return path


def test_memory_roundtrip(mem_file):
    memory.remember_exclusions("u1", ["Milk", "egg"])
    assert memory.get_exclusions("u1") == ["milk", "egg"]
    memory.forget("u1")
    assert memory.get_exclusions("u1") == []


def test_missing_file_is_empty_memory(mem_file):
    assert memory.get_exclusions("nobody") == []


def test_corrupt_file_is_not_overwritten(mem_file):
    mem_file.write_text('{"u1": {"exclude": ["milk"]', encoding="utf-8")  # truncated JSON
    before = mem_file.read_text(encoding="utf-8")
    with pytest.raises(memory.MemoryReadError):
        memory.remember_exclusions("u1", ["egg"])
    with pytest.raises(memory.MemoryReadError):
        memory.forget("u1")
    with pytest.raises(memory.MemoryReadError):
        memory.get_exclusions("u1")
    assert mem_file.read_text(encoding="utf-8") == before
    assert list(mem_file.parent.glob("profiles.json.corrupt-*"))  # backup kept


def test_non_object_json_is_not_overwritten(mem_file):
    mem_file.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(memory.MemoryReadError):
        memory.remember_exclusions("u1", ["egg"])
    assert mem_file.read_text(encoding="utf-8") == "[1, 2]"


def test_failed_write_keeps_old_file_and_leaves_no_temp(mem_file, monkeypatch):
    memory.remember_exclusions("u1", ["milk"])
    good = mem_file.read_text(encoding="utf-8")

    def boom(*a, **k):
        raise ValueError("disk problem")

    monkeypatch.setattr(memory.json, "dump", boom)
    with pytest.raises(ValueError):
        memory.remember_exclusions("u1", ["egg"])
    assert mem_file.read_text(encoding="utf-8") == good
    assert [p.name for p in mem_file.parent.iterdir() if p.name.endswith(".tmp")] == []


def test_write_uses_atomic_replace(mem_file, monkeypatch):
    calls = []
    real = memory.os.replace
    monkeypatch.setattr(memory.os, "replace", lambda a, b: (calls.append((a, b)), real(a, b))[1])
    memory.remember_exclusions("u1", ["milk"])
    assert calls and calls[0][1] == mem_file
    assert json.loads(mem_file.read_text(encoding="utf-8"))["u1"]["exclude"] == ["milk"]


def test_memory_node_keeps_message_allergies_when_file_unreadable(mem_file):
    pytest.importorskip("langgraph")
    from recipe_agent import graph

    mem_file.write_text("not json", encoding="utf-8")
    req = UserRequest(ingredients=["rice"], exclude_ingredients=["egg"], remember=True)
    out = graph.memory_node({"request": req, "user_id": "u1"})
    assert out["memory_event"] == "memory_error"
    assert out["request"].exclude_ingredients == ["egg"]
    assert mem_file.read_text(encoding="utf-8") == "not json"


# --------------------------------------------------------------------------- #
# 6. Unknown cooking time is never stored or shown as 0 minutes
# --------------------------------------------------------------------------- #
def test_minutes_or_none():
    assert minutes_or_none(None) is None
    assert minutes_or_none(float("nan")) is None
    assert minutes_or_none(0) is None
    assert minutes_or_none(35.0) == 35


def test_index_metadata_omits_unknown_time():
    assert "total_minutes" not in _metadata("A", "Dessert", float("nan"), 200)
    assert "total_minutes" not in _metadata("A", "Dessert", 0, 200)
    assert _metadata("A", "Dessert", 45, 200)["total_minutes"] == 45


def test_ingest_total_minutes():
    from recipe_agent.ingest import resolve_total_minutes

    assert resolve_total_minutes(30, None, None) == 30
    assert resolve_total_minutes(None, 10, 20) == 30
    assert resolve_total_minutes(0, None, 15) == 15
    assert resolve_total_minutes(None, None, None) is None
    assert resolve_total_minutes(float("nan"), float("nan"), float("nan")) is None
    assert resolve_total_minutes(0, 0, 0) is None


def test_fallback_answer_shows_unknown_time_not_zero():
    req = UserRequest(language="en")
    recipe = {
        "recipe_id": 1, "name": "Mystery stew", "total_minutes": None, "servings": 4,
        "nutrition": {"calories": 300, "protein_g": 10.0}, "have": [], "missing": [],
        "instructions": ["Cook."],
    }
    text = verifier.fallback_answer(req, [recipe], [])
    assert "0 minutes" not in text and "unknown" in text
    ar = verifier.fallback_answer(UserRequest(language="ar"), [recipe], [])
    assert "0 دقيقة" not in ar


def test_check_answer_ignores_unknown_time():
    req = UserRequest(language="en")
    recipe = {"name": "Stew", "total_minutes": None, "nutrition": {"calories": 300}}
    _, info = verifier.check_answer(req, [recipe], "Stew, 300 calories")
    assert info == []
