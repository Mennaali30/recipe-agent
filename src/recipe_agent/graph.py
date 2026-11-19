"""LangGraph workflow: parse -> search (widen once if few results) -> respond."""
from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from recipe_agent import tools
from recipe_agent.agents import parse_request, write_answer

TOP_K = 3
CANDIDATES = (40, 200)  # first search, then one wider retry
MIN_RESULTS = 2


class GraphState(TypedDict, total=False):
    text: str
    request: object  # UserRequest
    recipes: list
    shopping: list
    attempt: int
    answer: str


def parse_node(state: GraphState) -> dict:
    return {"request": parse_request(state["text"]), "attempt": 0}


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


def clarify_node(state: GraphState) -> dict:
    if state["request"].language == "ar":
        msg = "ممكن تقوليلي إيه المكونات اللي عندك؟ مثلاً: فراخ، أرز، طماطم."
    else:
        msg = "Which ingredients do you have? For example: chicken, rice, tomatoes."
    return {"answer": msg}


def respond_node(state: GraphState) -> dict:
    return {"answer": write_answer(state["request"], state["recipes"], state["shopping"])}


def after_parse(state: GraphState) -> str:
    return "search" if state["request"].ingredients else "clarify"


def after_search(state: GraphState) -> str:
    if len(state["recipes"]) < MIN_RESULTS and state["attempt"] < len(CANDIDATES):
        return "search"
    return "respond"


@lru_cache(maxsize=1)
def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("parse", parse_node)
    graph.add_node("search", search_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "parse")
    graph.add_conditional_edges("parse", after_parse, {"search": "search", "clarify": "clarify"})
    graph.add_conditional_edges("search", after_search, {"search": "search", "respond": "respond"})
    graph.add_edge("respond", END)
    graph.add_edge("clarify", END)
    return graph.compile()


def run_agent(text: str) -> dict:
    """Run the whole workflow and return the final state (request, recipes, answer, ...)."""
    return build_graph().invoke({"text": text})


def answer(text: str) -> str:
    return run_agent(text)["answer"]
