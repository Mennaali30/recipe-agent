"""A named dish with no Egyptian recipe: say so, then show close international recipes (no yes/no). Offline."""
import pytest

from recipe_agent import egypt_store, graph, tools

LASAGNA = {"recipe_id": 900001, "name": "Classic Lasagna", "category": "Main", "servings": 4,
           "ingredients": ["lasagna noodles", "ground beef", "cheese"], "instructions": ["Layer.", "Bake."],
           "calories": 500, "protein_g": 30, "carbs_g": 40, "fat_g": 20, "sugar_g": 5}
HITS = [{"recipe_id": 900001, "name": "Classic Lasagna", "category": "Main", "total_minutes": 60,
         "calories": 500, "distance": 0.2},
        {"recipe_id": 900002, "name": "Far Away Salad", "category": "Side", "total_minutes": 10,
         "calories": 90, "distance": 0.9}]


@pytest.fixture
def fake_index(monkeypatch):
    monkeypatch.setattr(tools, "search", lambda q, n_results=10, max_minutes=None: [dict(h) for h in HITS])
    from recipe_agent import verifier
    fake = lambda rid: dict(LASAGNA) if rid == 900001 else {**LASAGNA, "recipe_id": rid, "name": "Far Away Salad", "ingredients": ["lettuce"]}
    monkeypatch.setattr(tools, "get_recipe", fake)
    monkeypatch.setattr(verifier, "get_recipe", fake)
    monkeypatch.setattr(tools, "get_instructions", lambda rid, lang="en": ["Layer.", "Bake."])


def run(monkeypatch, text, lang="ar", **kw):
    from recipe_agent.schemas import UserRequest
    monkeypatch.setattr(graph, "parse_request", lambda t: UserRequest(language=lang, **kw))
    monkeypatch.setattr(graph, "write_answer", lambda *a, **k: "WRITTEN")
    monkeypatch.setattr(graph.memory, "get_exclusions", lambda uid="default": [])
    graph.build_graph.cache_clear()
    return graph.run_agent(text, "t")


def test_dish_query_drops_filler():
    assert egypt_store.dish_query("عايز لازانيا") == egypt_store.dish_query("لازانيا") != ""
    assert egypt_store.dish_query("عايز") == ""


def test_no_egyptian_match_says_so_first_then_shows_close_international(monkeypatch, fake_index):
    out = run(monkeypatch, "عايز لازانيا")
    assert out["answer"].startswith("مفيش وصفة مصرية لـ«")
    assert "عالمية" in out["answer"] and "ومعهاش تكلفة بالجنيه" in out["answer"]
    assert [r["recipe_id"] for r in out["recipes"]] == [900001]      # the far hit is not shown
    assert "?" not in out["answer"].split("\n")[0]                    # no yes/no question


def test_english_wording(monkeypatch, fake_index):
    out = run(monkeypatch, "I want lasagna", lang="en")
    assert out["answer"].startswith("There is no Egyptian recipe for") and "no EGP cost" in out["answer"]


def test_allergy_filter_still_applies(monkeypatch, fake_index):
    out = run(monkeypatch, "عايز لازانيا", exclude_ingredients=["beef"])
    assert not out.get("recipes") and "مكونات" in out["answer"]     # nothing shown, nothing invented


def test_nothing_close_says_so_and_asks_ingredients(monkeypatch):
    monkeypatch.setattr(tools, "search", lambda *a, **k: [])
    out = run(monkeypatch, "عايز لازانيا")
    assert not out.get("recipes") and out["answer"].startswith("مفيش وصفة مصرية") and "مكونات" in out["answer"]


def test_index_down_does_not_break(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no index")
    monkeypatch.setattr(tools, "search", boom)
    out = run(monkeypatch, "عايز لازانيا")
    assert not out.get("recipes") and out["answer"].startswith("مفيش وصفة مصرية")


def test_egyptian_dish_unchanged(monkeypatch, fake_index):
    out = run(monkeypatch, "عايز كشري")
    assert out["recipes"] and all(r["recipe_id"] != 900001 for r in out["recipes"])
    assert not out["answer"].startswith("مفيش وصفة مصرية لـ")
