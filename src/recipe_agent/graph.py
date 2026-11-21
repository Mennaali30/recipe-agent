"""LangGraph workflow:
parse -> memory -> (clarify | search (widen once if few results) -> substitute -> respond).
"""
import logging
from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from recipe_agent import memory, tools
from recipe_agent.agents import parse_request, plan_substitutions, write_answer

log = logging.getLogger(__name__)

TOP_K = 3
CANDIDATES = (40, 200)  # first search, then one wider retry
MIN_RESULTS = 2


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


def _only_pantry(ingredients) -> bool:
    """True if every ingredient is something we assume everyone has (salt, water, oil...)."""
    norms = [tools.normalize(i) for i in ingredients]
    return bool(norms) and all(n in tools._PANTRY for n in norms)


def clarify_node(state: GraphState) -> dict:
    """The request is too thin to search: confirm any memory change and ask for ingredients.

    No extra LLM call: the question was written by the intent agent in the same call.
    """
    req = state["request"]
    ar = req.language == "ar"
    saved = ", ".join(state.get("saved_exclusions") or [])
    event = state.get("memory_event")
    parts = []
    if event == "saved":
        parts.append(
            f"تمام، هفتكر إنك بتتجنبي: {saved}." if ar
            else f"Done, I will remember that you avoid: {saved}."
        )
    elif event == "forgotten":
        parts.append("تمام، مسحت الحساسيات المحفوظة." if ar else "Done, I cleared your saved allergies.")
    elif saved:
        parts.append(
            f"الحساسيات المحفوظة اللي هتتطبق: {saved}." if ar
            else f"Saved allergies that will be applied: {saved}."
        )

    if req.ingredients:  # only salt / water / oil, which we assume everyone has
        parts.append(
            "الملح والميّة والزيت بفترض إنهم موجودين عندك دايماً. إيه المكونات التانية اللي عندك؟" if ar
            else "I assume you always have salt, water and oil. Which other ingredients do you have?"
        )
    elif req.clarifying_question:
        parts.append(req.clarifying_question)
    else:
        parts.append(
            "ممكن تقوليلي إيه المكونات اللي عندك؟ مثلاً: فراخ، أرز، طماطم." if ar
            else "Which ingredients do you have? For example: chicken, rice, tomatoes."
        )
    return {"answer": "\n\n".join(parts)}


def respond_node(state: GraphState) -> dict:
    return {
        "answer": write_answer(
            state["request"],
            state["recipes"],
            state["shopping"],
            saved_exclusions=state.get("saved_exclusions"),
            memory_event=state.get("memory_event"),
        )
    }


def after_memory(state: GraphState) -> str:
    ingredients = state["request"].ingredients
    if not ingredients or _only_pantry(ingredients):
        return "clarify"
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
    graph.add_node("substitute", substitute_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "parse")
    graph.add_edge("parse", "memory")
    graph.add_conditional_edges("memory", after_memory, {"search": "search", "clarify": "clarify"})
    graph.add_conditional_edges(
        "search", after_search, {"search": "search", "substitute": "substitute"}
    )
    graph.add_edge("substitute", "respond")
    graph.add_edge("respond", END)
    graph.add_edge("clarify", END)
    return graph.compile()


def run_agent(text: str, user_id: str = "default") -> dict:
    """Run the whole workflow and return the final state (request, recipes, answer, ...)."""
    return build_graph().invoke({"text": text, "user_id": user_id})


def answer(text: str, user_id: str = "default") -> str:
    return run_agent(text, user_id)["answer"]
