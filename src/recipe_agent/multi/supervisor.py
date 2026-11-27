"""LangGraph supervisor: route -> specialist agent(s) -> finalize.

The router (LLM, with a keyword fallback) picks 1 or 2 agents. Agents run in order. An agent that
needs more information, or fails, stops the chain: the user gets its question or error, plus the
answers already produced. Nothing is retried blindly and nothing is guessed.
"""
import dataclasses
import logging
import re
from datetime import datetime, timezone
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import (tool_failure_text, tr, AgentResult, BudgetAgent, BusinessAgent, CostAgent, PantryAgent, RecipesAgent,
                     ROUTER_PROMPT, RouteDecision, ShoppingAgent, ProfileAgent, keyword_route, keyword_topics, with_context)
from recipe_agent.services import ApprovalService, HistoryService, is_no, is_yes

from .llm import llm_extract
from .tools import ToolContext, build_registry

log = logging.getLogger(__name__)

HISTORY_TURNS = 6
NODES = ["recipes", "pantry", "cost", "shopping", "business", "budget", "profile"]


class SupState(TypedDict, total=False):
    text: str
    user_id: str
    language: str
    pending: list
    results: list
    route_source: str
    answer: str
    history: list
    gated: bool


def build_supervisor(extract=None, session_factory=None, store_loader=None, recipe_pipeline=None,
                     registry=None, today=None, memory=None, save_history=True, router="llm", now=None):
    """Compile the supervisor graph. Everything external is injectable (tests use fakes).

    router: "llm" always asks the LLM; "keywords" never does; "hybrid" (used by the app, to save free
    quota) decides by keywords when the message matches one topic or none (none = recipes), and
    asks the LLM only when it matches two or more topics."""
    extract = extract or llm_extract
    registry = registry or build_registry()
    if session_factory is None:
        from recipe_agent.db import init_db, make_engine, make_session_factory
        engine = make_engine()
        init_db(engine)
        session_factory = make_session_factory(engine)
    agents = {
        "recipes": RecipesAgent(registry, extract, pipeline=recipe_pipeline),
        "pantry": PantryAgent(registry, extract),
        "cost": CostAgent(registry, extract),
        "shopping": ShoppingAgent(registry, extract),
        "business": BusinessAgent(registry, extract),
        "budget": BudgetAgent(registry, extract),
        "profile": ProfileAgent(registry, extract),
    }

    def make_ctx(state):
        return ToolContext(user_id=state.get("user_id") or "default", session_factory=session_factory,
                           store_loader=store_loader, today=today, message=state["text"],
                           history=state.get("history") or [], memory=memory)

    def load_history(state: SupState) -> dict:
        """Recent turns of this user (for follow-ups). A database problem never blocks the answer."""
        try:
            with session_factory() as s:
                return {"history": HistoryService(s).recent(state.get("user_id") or "default", HISTORY_TURNS)}
        except Exception:
            log.exception("could not load the chat history")
            return {"history": []}

    def gate(state: SupState) -> dict:
        """Sensitive actions: the user's next message answers a saved pending action.
        yes -> run exactly the saved action (approved); no -> cancel; anything else -> drop it, route normally."""
        uid = state.get("user_id") or "default"
        text = state["text"]
        try:
            with session_factory() as s:
                pend = ApprovalService(s).take(uid, now() if now else datetime.now(timezone.utc))
        except Exception:
            log.exception("could not read the pending action")
            return {"gated": False}
        if pend is None:
            return {"gated": False}
        lang = "ar" if re.search("[\u0600-\u06ff]", text) else "en"
        if is_no(text):
            res = AgentResult("approval", answer=tr(lang, "تمام، لغيت الإجراء ومفيش حاجة اتغيرت.",
                                                    "Okay, cancelled. Nothing was changed."))
        elif is_yes(text):
            ctx = dataclasses.replace(make_ctx(state), approved=True)
            name = "profile" if pend.tool == "profile_clear" else "pantry"
            res = AgentResult(name)
            out = registry.call(pend.tool, pend.args, ctx, allowed={pend.tool})
            res.tool_calls.append((pend.tool, out.ok))
            if out.ok:
                res.data = out.data
                res.answer = tr(lang, "تمام، اتنفذ.", "Done.")
            else:
                res.ok = False
                res.answer, res.needs_input = tool_failure_text(out, lang)
        else:
            return {"gated": False}   # a different message: the pending action is dropped, nothing runs
        return {"gated": True, "results": [res], "pending": [], "language": lang}

    def after_gate(state: SupState) -> str:
        return "finalize" if state.get("gated") else "route"

    def route(state: SupState) -> dict:
        text = state["text"]
        if router in ("keywords", "hybrid"):
            topics = keyword_topics(text)
            if router == "keywords" or len(topics) <= 1:
                d = keyword_route(text)
                return {"pending": d.intents, "language": d.language, "results": [], "route_source": "keywords"}
        try:
            decision = extract(RouteDecision, ROUTER_PROMPT, with_context(text, make_ctx(state)))
            if isinstance(decision, dict):
                decision = RouteDecision.model_validate(decision)
            source = "llm"
        except Exception as exc:
            log.warning("router LLM failed, using keyword routing: %s: %s", type(exc).__name__, exc)
            decision, source = keyword_route(text), "keywords"
        intents = list(dict.fromkeys(decision.intents))[:2]
        return {"pending": intents, "language": decision.language, "results": [], "route_source": source}

    def make_node(name):
        def node(state: SupState) -> dict:
            ctx = make_ctx(state)
            pending = list(state.get("pending") or [])
            if pending and pending[0] == name:
                pending.pop(0)
            res = agents[name].run(state["text"], state.get("language", "en"), ctx)
            return {"results": [*state.get("results", []), res], "pending": pending}
        return node

    def finalize(state: SupState) -> dict:
        results = state.get("results", [])
        answer = "\n\n".join(r.answer for r in results if r.answer)
        if save_history and answer:
            try:
                uid = state.get("user_id") or "default"
                with session_factory() as s:
                    h = HistoryService(s)
                    if not any(getattr(r, "agent", "") == "profile" and r.ok and r.tool_calls
                               and r.tool_calls[-1][0] == "profile_clear" for r in results):
                        h.add(uid, "user", state["text"])
                        h.add(uid, "assistant", answer, ",".join(r.agent for r in results))
            except Exception:
                log.exception("could not save the chat history")
        return {"answer": answer}

    def next_node(state: SupState) -> str:
        results = state.get("results", [])
        if results and (results[-1].needs_input or not results[-1].ok):
            return "finalize"
        pending = state.get("pending") or []
        return pending[0] if pending else "finalize"

    g = StateGraph(SupState)
    g.add_node("route", route)
    for n in NODES:
        g.add_node(n, make_node(n))
    g.add_node("finalize", finalize)
    g.add_node("history", load_history)
    g.add_edge(START, "history")
    g.add_node("gate", gate)
    g.add_edge("history", "gate")
    g.add_conditional_edges("gate", after_gate, {"finalize": "finalize", "route": "route"})
    targets = {n: n for n in NODES} | {"finalize": "finalize"}
    g.add_conditional_edges("route", next_node, targets)
    for n in NODES:
        g.add_conditional_edges(n, next_node, targets)
    g.add_edge("finalize", END)
    return g.compile()


def run_supervisor(text: str, user_id: str = "default", graph=None, **kwargs) -> dict:
    """Run one message through the supervisor. Returns the final state (answer, results, ...)."""
    graph = graph or build_supervisor(**kwargs)
    return graph.invoke({"text": text, "user_id": user_id})
