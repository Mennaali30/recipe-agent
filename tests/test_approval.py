"""Approval step: sensitive actions are staged, then run only after the user's yes. Offline."""
from datetime import datetime, timedelta, timezone

import pytest

from recipe_agent import memory
from recipe_agent.db import init_db, make_engine, make_session_factory
from recipe_agent.multi import build_supervisor
from recipe_agent.multi.agents import PantryRequest, ProfileRequest, RouteDecision
from recipe_agent.multi.tools import (PantryChangeArgs, PantryEntry, ToolContext, build_registry,
                                      approval_tools_from_env)
from recipe_agent.services import ApprovalService, PantryService, ProfileService, is_no, is_yes


class FakeLLM:
    def __init__(self, **by_schema):
        self.by_schema = by_schema

    def __call__(self, schema, system, text):
        v = self.by_schema.get(schema.__name__)
        if v is None:
            raise RuntimeError("503 UNAVAILABLE")
        return v


@pytest.fixture
def sf():
    engine = make_engine("sqlite://")
    init_db(engine)
    return make_session_factory(engine)


@pytest.fixture(autouse=True)
def mem_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RECIPE_MEMORY_PATH", str(tmp_path / "profiles.json"))
    monkeypatch.delenv("RECIPE_APPROVAL_TOOLS", raising=False)


def pantry_graph(sf, req, **kw):
    llm = FakeLLM(RouteDecision=RouteDecision(intents=["pantry"], language="ar"), PantryRequest=req)
    return build_supervisor(extract=llm, session_factory=sf, memory=memory, recipe_pipeline=lambda t, u: "R", **kw)


def stock(sf, name, uid="u1"):
    return [(m.value, m.unit) for m in PantryService(sf(), ).get_stock(uid, name) if m.value]


def seed(sf, uid="u1"):
    PantryService(sf()).add_item(uid, "rice", 2, "kg")


DISCARD = PantryRequest(action="remove", items=[PantryEntry(name="rice")])
ADJUST = PantryRequest(action="set", items=[PantryEntry(name="rice", quantity=500, unit="g")])


def test_yes_no_words():
    assert is_yes("تأكيد") and is_yes("  أيوه! ") and is_yes("Yes") and is_yes("ok")
    assert is_no("لا") and is_no("الغي") and is_no("Cancel")
    assert not is_yes("نعم بس زودي الأرز") and not is_no("لا تنسي الأرز") and not is_yes("عايز كشري")


def test_discard_is_staged_not_run_then_yes_runs_it(sf):
    seed(sf)
    g = pantry_graph(sf, DISCARD)
    first = g.invoke({"text": "امسحي الأرز", "user_id": "u1"})
    assert stock(sf, "rice") == [(2000, "g")] and "تأكيد" in first["answer"] and "rice" in first["answer"]
    assert first["results"][0].needs_input
    out = g.invoke({"text": "تأكيد", "user_id": "u1"})
    assert stock(sf, "rice") == [] and out["answer"] == "تمام، اتنفذ."


def test_no_cancels(sf):
    seed(sf)
    g = pantry_graph(sf, DISCARD)
    g.invoke({"text": "امسحي الأرز", "user_id": "u1"})
    out = g.invoke({"text": "لا", "user_id": "u1"})
    assert stock(sf, "rice") == [(2000, "g")] and "لغيت" in out["answer"]
    again = g.invoke({"text": "تأكيد", "user_id": "u1"})     # nothing pending any more
    assert stock(sf, "rice") == [(2000, "g")]


def test_other_message_drops_pending_and_routes_normally(sf):
    seed(sf)
    g = pantry_graph(sf, DISCARD)
    g.invoke({"text": "امسحي الأرز", "user_id": "u1"})
    out = g.invoke({"text": "عايز كشري", "user_id": "u1"})
    assert out["results"][0].agent == "pantry"                  # routed as a normal message (restaged)
    with sf() as s:
        assert ApprovalService(s).peek("u1").tool == "pantry_discard"   # the new request, not a stale one
    # an unrelated message must not leave an old action alive
    g2 = build_supervisor(extract=FakeLLM(RouteDecision=RouteDecision(intents=["recipes"], language="ar")),
                          session_factory=sf, memory=memory, recipe_pipeline=lambda t, u: "R")
    g2.invoke({"text": "عندي فراخ", "user_id": "u1"})
    with sf() as s:
        assert ApprovalService(s).peek("u1") is None
    g2.invoke({"text": "تأكيد", "user_id": "u1"})
    assert stock(sf, "rice") == [(2000, "g")]


def test_pending_action_expires(sf):
    seed(sf)
    clock = {"t": datetime(2026, 11, 27, 12, 0, tzinfo=timezone.utc)}
    with sf() as s:
        ApprovalService(s).stage("u1", "pantry_discard", {"items": [{"name": "rice"}]}, "a", "b", now=clock["t"])
    g = pantry_graph(sf, DISCARD, now=lambda: clock["t"] + timedelta(minutes=11))
    g.invoke({"text": "تأكيد", "user_id": "u1"})
    assert stock(sf, "rice") == [(2000, "g")]


def test_another_users_pending_action_is_not_run(sf):
    seed(sf, "u1")
    g = pantry_graph(sf, DISCARD)
    g.invoke({"text": "امسحي الأرز", "user_id": "u1"})
    g.invoke({"text": "تأكيد", "user_id": "u2"})
    assert stock(sf, "rice", "u1") == [(2000, "g")]
    g.invoke({"text": "تأكيد", "user_id": "u1"})
    assert stock(sf, "rice", "u1") == []


def test_staging_again_replaces_the_old_action(sf):
    seed(sf)
    with sf() as s:
        a = ApprovalService(s)
        a.stage("u1", "pantry_discard", {"items": [{"name": "rice"}]}, "1", "1")
        a.stage("u1", "pantry_adjust", {"items": [{"name": "rice", "quantity": 1, "unit": "kg"}]}, "2", "2")
        assert a.peek("u1").tool == "pantry_adjust"


def test_adjust_needs_approval_and_shows_old_and_new(sf):
    seed(sf)
    g = pantry_graph(sf, ADJUST)
    first = g.invoke({"text": "الأرز بقى 500 جرام", "user_id": "u1"})
    assert stock(sf, "rice") == [(2000, "g")] and "2000" in first["answer"] and "500" in first["answer"]
    g.invoke({"text": "yes", "user_id": "u1"})
    assert stock(sf, "rice") == [(500, "g")]


def test_discard_unknown_item_fails_after_confirm_and_changes_nothing(sf):
    seed(sf)
    req = PantryRequest(action="remove", items=[PantryEntry(name="rice"), PantryEntry(name="saffron")])
    g = pantry_graph(sf, req)
    g.invoke({"text": "امسح", "user_id": "u1"})
    out = g.invoke({"text": "تأكيد", "user_id": "u1"})
    assert stock(sf, "rice") == [(2000, "g")] and out["results"][0].ok is False


def test_english_reply_language(sf):
    seed(sf)
    g = pantry_graph(sf, DISCARD)
    g.invoke({"text": "delete my rice", "user_id": "u1"})
    out = g.invoke({"text": "yes", "user_id": "u1"})
    assert out["answer"] == "Done."


def test_pantry_add_not_gated_by_default_but_can_be(sf, monkeypatch):
    ctx = ToolContext(user_id="u1", session_factory=sf)
    args = PantryChangeArgs(items=[PantryEntry(name="rice", quantity=1, unit="kg")])
    assert build_registry().call("pantry_add", args, ctx, allowed={"pantry_add"}).ok
    assert stock(sf, "rice") == [(1000, "g")]
    monkeypatch.setenv("RECIPE_APPROVAL_TOOLS", "pantry_discard,pantry_add")
    res = build_registry().call("pantry_add", args, ctx, allowed={"pantry_add"})
    assert res.error_code == "needs_approval" and stock(sf, "rice") == [(1000, "g")]
    monkeypatch.setenv("RECIPE_APPROVAL_TOOLS", "none")
    assert approval_tools_from_env() == set()
    monkeypatch.setenv("RECIPE_APPROVAL_TOOLS", "no_such_tool")
    with pytest.raises(ValueError):
        build_registry()


def test_gated_tool_without_database_fails_closed():
    res = build_registry().call("profile_clear", {"what": "all"}, ToolContext(user_id="u1"), allowed={"profile_clear"})
    assert not res.ok and res.error_code == "failed"


def test_profile_clear_staged_then_confirmed(sf):
    ProfileService(sf(), memory).add_allergies("u1", ["milk"])
    llm = FakeLLM(RouteDecision=RouteDecision(intents=["profile"], language="en"),
                  ProfileRequest=ProfileRequest(action="clear", clear_what="allergies"))
    g = build_supervisor(extract=llm, session_factory=sf, memory=memory)
    first = g.invoke({"text": "forget my allergies", "user_id": "u1"})
    assert memory.get_exclusions("u1") == ["milk"] and '"yes"' in first["answer"]
    g.invoke({"text": "yes", "user_id": "u1"})
    assert memory.get_exclusions("u1") == []


def test_api_chat_two_step(sf, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from recipe_agent.api import server
    seed(sf)
    g = pantry_graph(sf, DISCARD)
    app = server.create_app(answer_fn=lambda t, u: g.invoke({"text": t, "user_id": u})["answer"], session_factory=sf)
    c = TestClient(app)
    r1 = c.post("/chat", json={"message": "امسحي الأرز", "user_id": "u1"})
    assert r1.status_code == 200 and stock(sf, "rice") == [(2000, "g")]
    c.post("/chat", json={"message": "تأكيد", "user_id": "u1"})
    assert stock(sf, "rice") == []
