"""Phase 5b: profile, saved allergies and chat history. Offline: fake LLM, in-memory SQLite, temp memory file."""
import pytest

from recipe_agent import memory
from recipe_agent.db import init_db, make_engine, make_session_factory
from recipe_agent.multi import build_supervisor
from recipe_agent.multi.agents import (BudgetRequest, CostRequest, ProfileRequest, RouteDecision,
                                       keyword_route)
from recipe_agent.multi.tools import ConvertArgs, BudgetItem, RecipeRef
from recipe_agent.services import (HistoryService, IngredientInput, ProfileService, ProfileUpdate)


class FakeLLM:
    def __init__(self, **by_schema):
        self.by_schema, self.calls, self.texts = by_schema, [], {}

    def __call__(self, schema, system, text):
        self.calls.append(schema.__name__)
        self.texts[schema.__name__] = text
        v = self.by_schema.get(schema.__name__)
        if v is None:
            raise RuntimeError("503 UNAVAILABLE")
        return v(text) if callable(v) else v


@pytest.fixture
def sf():
    engine = make_engine("sqlite://")
    init_db(engine)
    return make_session_factory(engine)


@pytest.fixture(autouse=True)
def mem_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RECIPE_MEMORY_PATH", str(tmp_path / "profiles.json"))


def graph(llm, sf, **kw):
    kw.setdefault("recipe_pipeline", lambda t, u: "RECIPES")
    return build_supervisor(extract=llm, session_factory=sf, memory=memory, **kw)


def ask(g, text="x", uid="u1"):
    return g.invoke({"text": text, "user_id": uid})


def route(*i, lang="en"):
    return RouteDecision(intents=list(i), language=lang)


# ---------------- services ----------------
def test_profile_defaults_then_update_keeps_other_fields(sf):
    s = sf()
    svc = ProfileService(s, memory)
    assert not svc.get("u1").exists
    v = svc.update("u1", ProfileUpdate(family_size=5, favorite_cuisines=["Egyptian", "egyptian", " Italian "]))
    assert v.exists and v.family_size == 5 and v.favorite_cuisines == ["Egyptian", "Italian"]
    v = svc.update("u1", ProfileUpdate(budget_egp=1500, budget_period="weekly"))
    assert v.family_size == 5 and v.budget_egp == 1500 and v.budget_period == "weekly"
    with pytest.raises(ValueError):
        svc.update("u1", ProfileUpdate())


def test_profile_validation():
    with pytest.raises(Exception):
        ProfileUpdate(family_size=0)
    with pytest.raises(Exception):
        ProfileUpdate(budget_egp=-5)
    with pytest.raises(Exception):
        ProfileUpdate(budget_period="yearly")


def test_profiles_are_per_user(sf):
    s = sf()
    svc = ProfileService(s, memory)
    svc.update("u1", ProfileUpdate(family_size=5))
    assert svc.get("u2").family_size is None
    svc.add_allergies("u1", ["milk"])
    assert svc.get("u1").allergies == ["milk"] and svc.get("u2").allergies == []


def test_allergies_use_the_existing_memory_file_that_recipe_search_reads(sf):
    svc = ProfileService(sf(), memory)
    svc.add_allergies("u1", ["Milk", "peanut"])
    assert memory.get_exclusions("u1") == ["milk", "peanut"] or set(map(str.lower, memory.get_exclusions("u1"))) == {"milk", "peanut"}
    svc.clear_allergies("u1")
    assert memory.get_exclusions("u1") == []
    with pytest.raises(ValueError):
        svc.add_allergies("u1", [" "])


def test_unreadable_allergy_memory_is_reported_not_hidden(sf, tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    monkeypatch.setenv("RECIPE_MEMORY_PATH", str(bad))
    v = ProfileService(sf(), memory).get("u1")
    assert v.allergies is None and v.allergies_error


def test_delete_profile(sf):
    svc = ProfileService(sf(), memory)
    assert not svc.delete_profile("u1")
    svc.update("u1", ProfileUpdate(family_size=3))
    assert svc.delete_profile("u1") and not svc.get("u1").exists


def test_history_add_recent_count_clear_and_isolation(sf):
    h = HistoryService(sf())
    for i in range(5):
        h.add("u1", "user", f"q{i}")
        h.add("u1", "assistant", f"a{i}", "cost")
    h.add("u2", "user", "other")
    assert h.count("u1") == 10 and h.count("u2") == 1
    rec = h.recent("u1", 4)
    assert [e.text for e in rec] == ["q3", "a3", "q4", "a4"] and rec[1].agent == "cost"
    assert h.clear("u1") == 10 and h.count("u1") == 0 and h.count("u2") == 1
    with pytest.raises(ValueError):
        h.add("u1", "system", "x")


def test_history_text_is_capped(sf):
    h = HistoryService(sf())
    h.add("u1", "user", "x" * 10000)
    assert len(h.recent("u1", 1)[0].text) == 4000


# ---------------- profile agent ----------------
def test_profile_agent_saves_prefs_and_shows_them(sf):
    req = ProfileRequest(action="update", update=ProfileUpdate(family_size=5, budget_egp=2000, budget_period="monthly"))
    out = ask(graph(FakeLLM(RouteDecision=route("profile", lang="ar"), ProfileRequest=req), sf), "احفظي إن عيلتي 5 وميزانيتي 2000 في الشهر")
    assert "حفظت" in out["answer"] and "5" in out["answer"] and "2000" in out["answer"]
    out = ask(graph(FakeLLM(RouteDecision=route("profile"), ProfileRequest=ProfileRequest(action="show")), sf))
    assert "Family size: 5" in out["answer"] and "monthly" in out["answer"]


def test_profile_agent_saves_allergy_the_message_names_even_if_llm_missed_it(sf):
    req = ProfileRequest(action="update", allergies=["peanut"])
    g = graph(FakeLLM(RouteDecision=route("profile"), ProfileRequest=req), sf)
    ask(g, "remember that I'm allergic to milk and peanuts")
    saved = set(map(str.lower, memory.get_exclusions("u1")))
    assert "peanut" in saved and "milk" in saved


def test_profile_agent_does_not_save_allergy_unless_asked(sf):
    req = ProfileRequest(action="update", update=ProfileUpdate(family_size=4))     # no allergies extracted
    ask(graph(FakeLLM(RouteDecision=route("profile"), ProfileRequest=req), sf), "family size 4, I'm allergic to milk")
    assert memory.get_exclusions("u1") == []


def test_profile_agent_nothing_to_save_asks(sf):
    out = ask(graph(FakeLLM(RouteDecision=route("profile"), ProfileRequest=ProfileRequest(action="update")), sf))
    assert out["results"][0].needs_input


def test_profile_agent_clear_requires_explicit_target(sf):
    ProfileService(sf(), memory).add_allergies("u1", ["milk"])
    out = ask(graph(FakeLLM(RouteDecision=route("profile"), ProfileRequest=ProfileRequest(action="clear")), sf))
    assert out["results"][0].needs_input and memory.get_exclusions("u1") == ["milk"]


def test_profile_agent_clear_history_deletes_and_does_not_log_itself(sf):
    HistoryService(sf()).add("u1", "user", "old")
    req = ProfileRequest(action="clear", clear_what="history")
    out = ask(graph(FakeLLM(RouteDecision=route("profile"), ProfileRequest=req), sf), "امسحي المحادثات")
    assert HistoryService(sf()).count("u1") == 0 and "Done, deleted: chat history" in out["answer"]


def test_profile_agent_clear_allergies_and_all(sf):
    ProfileService(sf(), memory).add_allergies("u1", ["milk"])
    ProfileService(sf(), memory).update("u1", ProfileUpdate(family_size=2))
    out = ask(graph(FakeLLM(RouteDecision=route("profile"), ProfileRequest=ProfileRequest(action="clear", clear_what="all")), sf))
    assert memory.get_exclusions("u1") == [] and not ProfileService(sf(), memory).get("u1").exists
    assert HistoryService(sf()).count("u1") == 0


# ---------------- history in the supervisor ----------------
def test_each_turn_is_saved_per_user(sf):
    g = graph(FakeLLM(RouteDecision=route("recipes")), sf)
    ask(g, "عندي فراخ", "u1")
    rec = HistoryService(sf()).recent("u1", 5)
    assert [e.role for e in rec] == ["user", "assistant"] and rec[0].text == "عندي فراخ" and rec[1].agent == "recipes"
    assert HistoryService(sf()).count("u2") == 0


def test_save_history_can_be_turned_off(sf):
    ask(graph(FakeLLM(RouteDecision=route("recipes")), sf, save_history=False))
    assert HistoryService(sf()).count("u1") == 0


def test_follow_up_gets_the_earlier_turns_as_context(sf):
    llm = FakeLLM(RouteDecision=route("recipes"))
    g = graph(llm, sf)
    ask(g, "عايز كشري")
    llm2 = FakeLLM(RouteDecision=route("cost"), CostRequest=CostRequest(task="convert", convert=ConvertArgs(quantity=1, unit="kg", ingredient="rice")))
    ask(graph(llm2, sf), "and how much does it cost?")
    t = llm2.texts["CostRequest"]
    assert "عايز كشري" in t and "CURRENT message:\nand how much does it cost?" in t
    assert llm2.texts["RouteDecision"].startswith("Earlier messages")


def test_first_message_has_no_context_block(sf):
    llm = FakeLLM(RouteDecision=route("recipes"))
    ask(graph(llm, sf), "hello")
    assert llm.texts["RouteDecision"] == "hello"


def test_history_failure_never_blocks_the_answer(sf):
    def broken():
        raise RuntimeError("db down")
    g = build_supervisor(extract=FakeLLM(RouteDecision=route("recipes")), session_factory=broken, memory=memory,
                         recipe_pipeline=lambda t, u: "STILL ANSWERS")
    assert ask(g)["answer"] == "STILL ANSWERS"


# ---------------- saved budget ----------------
def test_budget_agent_uses_saved_budget_when_none_given(sf):
    ProfileService(sf(), memory).update("u1", ProfileUpdate(budget_egp=100000, budget_period="weekly"))
    items = [BudgetItem(recipe=RecipeRef(name="Rice", servings=4, ingredients=[IngredientInput(name="rice", quantity=400, unit="g")]), meals=4)]
    out = ask(graph(FakeLLM(RouteDecision=route("budget"), BudgetRequest=BudgetRequest(items=items)), sf))
    assert "saved weekly budget" in out["answer"] and "Within budget" in out["answer"]


def test_budget_agent_still_asks_without_saved_budget(sf):
    items = [BudgetItem(recipe=RecipeRef(name="Rice", servings=4, ingredients=[IngredientInput(name="rice", quantity=400, unit="g")]), meals=4)]
    out = ask(graph(FakeLLM(RouteDecision=route("budget"), BudgetRequest=BudgetRequest(items=items)), sf))
    assert out["results"][0].needs_input


def test_keyword_router_profile():
    assert keyword_route("احفظي إن عيلتي 5").intents == ["profile"]
    assert keyword_route("show my profile").intents == ["profile"]
    assert keyword_route("عايز كشري").intents == ["recipes"]
