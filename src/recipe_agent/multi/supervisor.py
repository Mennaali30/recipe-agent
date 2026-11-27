"""LangGraph supervisor: route -> specialist agent(s) -> finalize.

The router (LLM, with a keyword fallback) picks 1 or 2 agents. Agents run in order. An agent that
needs more information, or fails, stops the chain: the user gets its question or error, plus the
answers already produced. Nothing is retried blindly and nothing is guessed.
"""
import logging
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import (AgentResult, BudgetAgent, BusinessAgent, CostAgent, PantryAgent, RecipesAgent,
                     ROUTER_PROMPT, RouteDecision, ShoppingAgent, ProfileAgent, keyword_route, with_context)
from recipe_agent.services import HistoryService

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


def build_supervisor(extract=None, session_factory=None, store_loader=None, recipe_pipeline=None,
                     registry=None, today=None, memory=None, save_history=True):
    """Compile the supervisor graph. Everything external is injectable (tests use fakes)."""
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

    def route(state: SupState) -> dict:
        text = state["text"]
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
    g.add_edge("history", "route")
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
