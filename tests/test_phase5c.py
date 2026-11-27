"""Phase 5c: the supervisor wired into the Gradio app. Offline: fake gradio module, fake LLM, in-memory DB."""
import importlib
import sys
import types

import pytest

from recipe_agent import memory
from recipe_agent.db import init_db, make_engine, make_session_factory
from recipe_agent.multi import app as multi_app
from recipe_agent.multi import build_supervisor
from recipe_agent.multi.agents import (CostRequest, RouteDecision, keyword_topics)
from recipe_agent.multi.tools import ConvertArgs


class FakeLLM:
    def __init__(self, **by_schema):
        self.by_schema, self.calls = by_schema, []

    def __call__(self, schema, system, text):
        self.calls.append(schema.__name__)
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
    monkeypatch.setenv("RECIPE_MEMORY_PATH", str(tmp_path / "p.json"))


def graph(llm, sf, **kw):
    kw.setdefault("recipe_pipeline", lambda t, u: "RECIPES")
    return build_supervisor(extract=llm, session_factory=sf, memory=memory, **kw)


# ---------------- hybrid router ----------------
def test_keyword_topics():
    assert keyword_topics("عايز كشري") == []
    assert keyword_topics("تكلفة الكشري") == ["cost"]
    assert set(keyword_topics("تكلفة الكشري وقايمة شراء")) == {"cost", "shopping"}


def test_hybrid_router_uses_no_llm_for_no_topic_or_one_topic(sf):
    llm = FakeLLM(CostRequest=CostRequest(task="convert", convert=ConvertArgs(quantity=1, unit="kg", ingredient="rice")))
    g = graph(llm, sf, router="hybrid")
    out = g.invoke({"text": "عايز كشري", "user_id": "u"})
    assert out["answer"] == "RECIPES" and llm.calls == []                       # recipes: zero extra calls
    out = g.invoke({"text": "convert 1 kg rice to grams", "user_id": "u"})
    assert out["route_source"] == "keywords" and llm.calls == ["CostRequest"]   # one call: the extraction


def test_hybrid_router_asks_the_llm_only_for_several_topics(sf):
    llm = FakeLLM(RouteDecision=RouteDecision(intents=["cost", "shopping"], language="en"),
                  CostRequest=CostRequest(task="convert", convert=ConvertArgs(quantity=1, unit="kg", ingredient="rice")))
    out = graph(llm, sf, router="hybrid").invoke({"text": "cost and shopping list please", "user_id": "u"})
    assert llm.calls[0] == "RouteDecision" and out["route_source"] == "llm"


def test_keywords_router_never_calls_the_llm_router(sf):
    llm = FakeLLM()
    out = graph(llm, sf, router="keywords").invoke({"text": "cost and shopping list please", "user_id": "u"})
    assert "RouteDecision" not in llm.calls and out["route_source"] == "keywords"


# ---------------- recipes agent: quota / busy ----------------
def test_recipes_agent_reports_quota_in_plain_words(sf):
    def quota(t, u):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")
    out = graph(FakeLLM(), sf, router="keywords", recipe_pipeline=quota).invoke({"text": "عندي فراخ", "user_id": "u"})
    assert "الاستخدام المجاني" in out["answer"]


# ---------------- multi.app ----------------
def test_app_answer_uses_one_cached_supervisor(monkeypatch, tmp_path, sf):
    multi_app.get_supervisor.cache_clear()
    built = []
    monkeypatch.setattr(multi_app, "build_supervisor", lambda **kw: built.append(kw) or graph(FakeLLM(), sf, router=kw["router"]))
    assert multi_app.answer("عندي فراخ", "u1") == "RECIPES"
    assert multi_app.answer("عندي أرز", "u1") == "RECIPES"
    assert len(built) == 1 and built[0]["router"] == "hybrid"
    multi_app.get_supervisor.cache_clear()


def test_app_router_can_be_set_by_env(monkeypatch, sf):
    multi_app.get_supervisor.cache_clear()
    seen = []
    monkeypatch.setenv("RECIPE_AGENT_ROUTER", "keywords")
    monkeypatch.setattr(multi_app, "build_supervisor", lambda **kw: seen.append(kw["router"]) or graph(FakeLLM(), sf))
    multi_app.get_supervisor()
    assert seen == ["keywords"]
    multi_app.get_supervisor.cache_clear()


# ---------------- gradio_app (fake gradio) ----------------
@pytest.fixture
def gradio_app(monkeypatch):
    fake = types.ModuleType("gradio")
    fake.Request = type("Request", (), {})
    fake.ChatInterface = lambda **kw: types.SimpleNamespace(launch=lambda: None, kwargs=kw)
    monkeypatch.setitem(sys.modules, "gradio", fake)
    sys.modules.pop("gradio_app", None)
    import pathlib
    monkeypatch.syspath_prepend(str(pathlib.Path(__file__).resolve().parents[1] / "app"))
    mod = importlib.import_module("gradio_app")
    yield mod
    sys.modules.pop("gradio_app", None)


def test_gradio_respond_goes_through_the_supervisor(gradio_app, monkeypatch):
    calls = []
    monkeypatch.setattr(gradio_app.supervisor_app, "answer", lambda m, user_id: calls.append((m, user_id)) or "SUP")
    monkeypatch.setattr(gradio_app.graph, "answer", lambda *a, **k: "CLASSIC")
    req = types.SimpleNamespace(username="waad")
    assert gradio_app.respond("hi", [], req) == "SUP" and calls == [("hi", "waad")]
    assert gradio_app.respond("hi", [], types.SimpleNamespace(username=None)) == "SUP" and calls[-1][1] == "default"


def test_gradio_classic_mode_still_available(gradio_app, monkeypatch):
    monkeypatch.setattr(gradio_app, "MODE", "classic")
    monkeypatch.setattr(gradio_app.graph, "answer", lambda *a, **k: "CLASSIC")
    assert gradio_app.respond("hi", [], types.SimpleNamespace(username=None)) == "CLASSIC"


def test_gradio_quota_and_busy_messages_still_work(gradio_app, monkeypatch):
    def quota(*a, **k):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")
    monkeypatch.setattr(gradio_app.supervisor_app, "answer", quota)
    assert gradio_app.respond("hi", [], types.SimpleNamespace(username=None)) == gradio_app.QUOTA_MESSAGE
    def busy(*a, **k):
        raise RuntimeError("503 UNAVAILABLE")
    monkeypatch.setattr(gradio_app.supervisor_app, "answer", busy)
    monkeypatch.setattr(gradio_app.time, "sleep", lambda s: None)
    assert gradio_app.respond("hi", [], types.SimpleNamespace(username=None)) == gradio_app.BUSY_MESSAGE


def test_gradio_examples_and_title_updated(gradio_app):
    assert "عايز كشري" in gradio_app.EXAMPLES and "مطبخك" in gradio_app.TITLE
