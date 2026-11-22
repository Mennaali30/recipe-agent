"""LangGraph workflow:
parse -> memory -> (clarify
                    | plan (several meals)
                    | search (widen once if few results) -> substitute)
      -> verify -> respond -> check.
"""
import logging
from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from recipe_agent import memory, tools, verifier
from recipe_agent.agents import (
    MAX_MEALS,
    parse_request,
    plan_meals,
    plan_substitutions,
    write_answer,
)

log = logging.getLogger(__name__)

TOP_K = 3
CANDIDATES = (40, 200)  # first search, then one wider retry
MIN_RESULTS = 2
PLAN_SHORTLIST = 12


class GraphState(TypedDict, total=False):
    text: str
    user_id: str
    request: object  # UserRequest
    saved_exclusions: list
    memory_event: str
    recipes: list
    shopping: list
    attempt: int
    answer: str
    verifier_report: dict
    fallback: str
    plan_info: dict


def parse_node(state: GraphState) -> dict:
    return {"request": parse_request(state["text"]), "attempt": 0}


def memory_node(state: GraphState) -> dict:
    """Code only (no LLM): save / forget when asked, then add saved allergies to this request."""
    req = state["request"]
    uid = state.get("user_id") or "default"
    event = None
    if req.forget_memory:
        memory.forget(uid)
        saved = []
        event = "forgotten"
    elif req.remember and req.exclude_ingredients:
        saved = memory.remember_exclusions(uid, req.exclude_ingredients)
        event = "saved"
    else:
        saved = memory.get_exclusions(uid)
    req.exclude_ingredients = list(dict.fromkeys([*req.exclude_ingredients, *saved]))
    return {"request": req, "saved_exclusions": saved, "memory_event": event}


def search_node(state: GraphState) -> dict:
    req = state["request"]
    attempt = state.get("attempt", 0)
    found = tools.find_recipes(
        req.ingredients,
        max_minutes=req.max_minutes,
        exclude=req.exclude_ingredients,  # never relaxed: allergies stay enforced
        top_k=TOP_K,
        candidates=CANDIDATES[min(attempt, len(CANDIDATES) - 1)],
    )
    for recipe in found:
        recipe["instructions"] = tools.get_instructions(recipe["recipe_id"])
    shopping = tools.shopping_list([r["missing"] for r in found])
    return {"recipes": found, "shopping": shopping, "attempt": attempt + 1}


def plan_node(state: GraphState) -> dict:
    """Several meals: code builds the shortlist, the planner agent picks and orders the days.

    Allergies are enforced by find_recipes, and the planner can only choose from its shortlist.
    """
    req = state["request"]
    n = min(req.meals, MAX_MEALS)
    shortlist = tools.find_recipes(
        req.ingredients,
        max_minutes=req.max_minutes,
        exclude=req.exclude_ingredients,
        top_k=PLAN_SHORTLIST,
        candidates=200,
    )
    planned, source = plan_meals(shortlist, n)
    recipes = []
    for day, recipe in enumerate(planned, 1):
        recipe = dict(recipe)
        recipe["day"] = day
        recipe["instructions"] = tools.get_instructions(recipe["recipe_id"])
        recipe["substitutions"] = []
        recipes.append(recipe)
    shopping = tools.shopping_list([r["missing"] for r in recipes])
    return {
        "recipes": recipes,
        "shopping": shopping,
        "plan_info": {"requested": req.meals, "planned": len(recipes), "source": source},
    }


def substitute_node(state: GraphState) -> dict:
    """Optional step: if it fails (busy server, quota), the answer is written without it."""
    try:
        plan = plan_substitutions(state["request"], state["recipes"])
    except Exception as exc:
        log.warning("substitution step skipped: %s", type(exc).__name__)
        plan = []
    by_recipe = {}
    for item in plan:
        by_recipe.setdefault(item["recipe_id"], []).append(
            {k: v for k, v in item.items() if k != "recipe_id"}
        )
    recipes = []
    for recipe in state["recipes"]:
        recipe = dict(recipe)
        recipe["substitutions"] = by_recipe.get(recipe["recipe_id"], [])
        recipes.append(recipe)
    return {"recipes": recipes}


def verify_node(state: GraphState) -> dict:
    """Code only (no LLM): re-check the final recipes against the allergies before writing."""
    kept, issues = verifier.verify_recipes(state["request"], state["recipes"])
    shopping = tools.shopping_list([r["missing"] for r in kept])
    update = {"recipes": kept, "shopping": shopping, "verifier_report": {"removed": issues}}
    if state.get("plan_info"):
        update["plan_info"] = {**state["plan_info"], "planned": len(kept)}
    return update


def _only_pantry(ingredients) -> bool:
    """True if every ingredient is something we assume everyone has (salt, water, oil...)."""
    norms = [tools.normalize(i) for i in ingredients]
    return bool(norms) and all(n in tools._PANTRY for n in norms)


def clarify_node(state: GraphState) -> dict:
    """The request is too thin to search: confirm any memory change and ask for ingredients.

    No extra LLM call: the question was written by the intent agent in the same call.
    When this node confirms a memory change itself, the fixed question is used instead,
    so the confirmation is never duplicated or claimed without being true.
    """
    req = state["request"]
    ar = req.language == "ar"
    saved_items = verifier.format_items(state.get("saved_exclusions") or [], req.language)
    event = state.get("memory_event")
    parts = []
    if event == "saved":
        parts.append(
            f"تمام، هفتكر إنك بتتجنبي: {saved_items}." if ar
            else f"Done, I will remember that you avoid: {saved_items}."
        )
    elif event == "forgotten":
        parts.append("تمام، مسحت الحساسيات المحفوظة." if ar else "Done, I cleared your saved allergies.")
    elif req.remember:  # asked to remember, but there was nothing to save
        parts.append(
            "مقدرتش أحدد إيه اللي أحفظه. قوليلي الحساسية بالاسم، مثلاً: افتكري إني عندي حساسية من اللبن."
            if ar
            else "I could not tell what to save. Tell me the allergy by name, for example: remember that I'm allergic to milk."
        )
    elif saved_items:
        parts.append(
            f"الحساسيات المحفوظة اللي هتتطبق: {saved_items}." if ar
            else f"Saved allergies that will be applied: {saved_items}."
        )

    if req.ingredients:  # only salt / water / oil, which we assume everyone has
        parts.append(
            "الملح والميّة والزيت بفترض إنهم موجودين عندك دايماً. إيه المكونات التانية اللي عندك؟" if ar
            else "I assume you always have salt, water and oil. Which other ingredients do you have?"
        )
    elif req.clarifying_question and not event and not req.remember:
        parts.append(req.clarifying_question)
    else:
        parts.append(
            "ممكن تقوليلي إيه المكونات اللي عندك؟ مثلاً: فراخ، أرز، طماطم." if ar
            else "Which ingredients do you have? For example: chicken, rice, tomatoes."
        )
    return {"answer": "\n\n".join(parts)}


def respond_node(state: GraphState) -> dict:
    """Write the answer with the LLM; if that fails, build a plain answer from the data."""
    try:
        text = write_answer(
            state["request"],
            state["recipes"],
            state["shopping"],
            saved_exclusions=state.get("saved_exclusions"),
            memory_event=state.get("memory_event"),
            plan=state.get("plan_info"),
        )
        return {"answer": text, "fallback": None}
    except Exception as exc:
        log.warning("response step failed, using the plain answer: %s", type(exc).__name__)
        return {
            "answer": verifier.fallback_answer(
                state["request"],
                state["recipes"],
                state["shopping"],
                saved_exclusions=state.get("saved_exclusions"),
                memory_event=state.get("memory_event"),
                reason="llm_error",
            ),
            "fallback": "llm_error",
        }


def check_node(state: GraphState) -> dict:
    """Code only (no LLM): scan the written answer; swap it for the plain one if it leaks."""
    if state.get("fallback"):
        return {}  # already built by code from verified data
    report = dict(state.get("verifier_report") or {})
    blocking, info = verifier.check_answer(state["request"], state["recipes"], state["answer"])
    report["answer_problems"] = blocking
    report["number_notes"] = info
    if blocking:
        log.warning("answer check failed (%s): using the plain answer", blocking)
        return {
            "answer": verifier.fallback_answer(
                state["request"],
                state["recipes"],
                state["shopping"],
                saved_exclusions=state.get("saved_exclusions"),
                memory_event=state.get("memory_event"),
                reason="check_failed",
            ),
            "fallback": "check_failed",
            "verifier_report": report,
        }
    return {"verifier_report": report}


def after_memory(state: GraphState) -> str:
    req = state["request"]
    if not req.ingredients or _only_pantry(req.ingredients):
        return "clarify"
    if req.meals and req.meals >= 2:
        return "plan"
    return "search"


def after_search(state: GraphState) -> str:
    if len(state["recipes"]) < MIN_RESULTS and state["attempt"] < len(CANDIDATES):
        return "search"
    return "substitute"


@lru_cache(maxsize=1)
def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("parse", parse_node)
    graph.add_node("memory", memory_node)
    graph.add_node("search", search_node)
    graph.add_node("plan", plan_node)
    graph.add_node("substitute", substitute_node)
    graph.add_node("verify", verify_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("respond", respond_node)
    graph.add_node("check", check_node)

    graph.add_edge(START, "parse")
    graph.add_edge("parse", "memory")
    graph.add_conditional_edges(
        "memory", after_memory, {"search": "search", "plan": "plan", "clarify": "clarify"}
    )
    graph.add_conditional_edges(
        "search", after_search, {"search": "search", "substitute": "substitute"}
    )
    graph.add_edge("plan", "verify")
    graph.add_edge("substitute", "verify")
    graph.add_edge("verify", "respond")
    graph.add_edge("respond", "check")
    graph.add_edge("check", END)
    graph.add_edge("clarify", END)
    return graph.compile()


def run_agent(text: str, user_id: str = "default") -> dict:
    """Run the whole workflow and return the final state (request, recipes, answer, ...)."""
    return build_graph().invoke({"text": text, "user_id": user_id})


def answer(text: str, user_id: str = "default") -> str:
    return run_agent(text, user_id)["answer"]
