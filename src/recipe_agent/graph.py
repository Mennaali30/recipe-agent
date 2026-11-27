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

from recipe_agent import cache, egypt_store, memory, tools, verifier
from recipe_agent.allergy import extract_allergies
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
    egypt_note: str
    dish_blocked: list


def parse_node(state: GraphState) -> dict:
    req = parse_request(state["text"])
    # Allergies named in the raw message are added by code, whatever the LLM extracted.
    req.exclude_ingredients = list(
        dict.fromkeys([*req.exclude_ingredients, *extract_allergies(state["text"])])
    )
    return {"request": req, "attempt": 0}


def memory_node(state: GraphState) -> dict:
    """Code only (no LLM): save / forget when asked, then add saved allergies to this request."""
    req = state["request"]
    uid = state.get("user_id") or "default"
    event = None
    saved = []
    try:
        if req.forget_memory:
            memory.forget(uid)
            event = "forgotten"
        elif req.remember and req.exclude_ingredients:
            saved = memory.remember_exclusions(uid, req.exclude_ingredients)
            event = "saved"
        else:
            saved = memory.get_exclusions(uid)
    except memory.MemoryReadError:
        # The file is left untouched. Allergies from this message still apply.
        log.exception("saved allergies could not be read")
        event = "memory_error"
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
        lang=req.language,
    )
    for recipe in found:
        recipe["instructions"] = tools.get_instructions(recipe["recipe_id"], lang=req.language)
    shopping = tools.priced_shopping(found, req.language)
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
        lang=req.language,
    )
    planned, source = plan_meals(shortlist, n)
    recipes = []
    for day, recipe in enumerate(planned, 1):
        recipe = dict(recipe)
        recipe["day"] = day
        recipe["instructions"] = tools.get_instructions(recipe["recipe_id"], lang=req.language)
        recipe["substitutions"] = []
        recipes.append(recipe)
    shopping = tools.priced_shopping(recipes, req.language)
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
        log.warning("substitution step skipped: %s: %s", type(exc).__name__, exc)
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


def dish_node(state: GraphState) -> dict:
    """Code only (no LLM): a message with no ingredients may name a dish ("عايز كشري").

    Looks the dish up in the curated Egyptian recipes. Allergies and the time limit are applied;
    if every match is removed by an allergy, the user is told that, not shown the recipe.
    """
    req = state["request"]
    try:
        found, blocked = egypt_store.find_by_name(
            state["text"], exclude=req.exclude_ingredients, max_minutes=req.max_minutes, lang=req.language
        )
    except Exception:
        log.exception("dish lookup skipped")
        return {"recipes": []}
    if not found and blocked:
        ar = req.language == "ar"
        names = "، ".join(blocked[:3]) if ar else ", ".join(blocked[:3])
        return {
            "recipes": [],
            "dish_blocked": blocked,
            "answer": (
                f"الوصفة اللي بتدوري عليها ({names}) فيها مكون بتتجنبيه بسبب الحساسية، فمش هعرضها."
                if ar else
                f"The dish you asked for ({names}) contains something you avoid because of your allergies, so I will not show it."
            ),
        }
    recipes = []
    for recipe in found:
        recipe = dict(recipe)
        recipe["instructions"] = tools.get_instructions(recipe["recipe_id"], lang=req.language)
        recipe["substitutions"] = []
        recipes.append(recipe)
    return {"recipes": recipes, "shopping": tools.priced_shopping(recipes, req.language)}


def after_dish(state: GraphState) -> str:
    if state.get("recipes"):
        return "verify"
    return "end" if state.get("dish_blocked") else "clarify"


def verify_node(state: GraphState) -> dict:
    """Code only (no LLM): re-check the final recipes against the allergies before writing."""
    kept, issues = verifier.verify_recipes(state["request"], state["recipes"])
    shopping = tools.priced_shopping(kept, state["request"].language)
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
    elif event == "memory_error":
        parts.append(
            "معرفتش أقرا الحساسيات المحفوظة (الملف ماتمسحش). اكتبي الحساسية في رسالتك وهتتطبق."
            if ar
            else "I could not read your saved allergies (the file was not changed). Write the allergy in your message and it will be applied."
        )
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


def _with_egypt_note(note, answer: str) -> str:
    """The no-Egyptian-match note always sits at the top of the answer."""
    return f"{note}\n\n{answer}" if note else answer


def respond_node(state: GraphState) -> dict:
    """Write the answer with the LLM; if that fails, build a plain answer from the data.

    When none of the verified recipes is Egyptian, a short code-written note (in the user's
    language) is prepended: it says the shown recipes are international alternatives and
    names the closest Egyptian recipe when only the time limit excluded it.
    """
    note = None
    try:
        note = egypt_store.no_match_note(
            state["recipes"],
            state["request"].ingredients,
            max_minutes=state["request"].max_minutes,
            exclude=state["request"].exclude_ingredients,
            lang=state["request"].language,
        )
    except Exception:  # the note is a nicety; never lose the answer over it
        log.exception("Egyptian no-match note skipped")
    try:
        text = write_answer(
            state["request"],
            state["recipes"],
            state["shopping"],
            saved_exclusions=state.get("saved_exclusions"),
            memory_event=state.get("memory_event"),
            plan=state.get("plan_info"),
        )
        # The approximate cost of an Egyptian recipe always reaches the answer,
        # even if the writer dropped it.
        text = egypt_store.ensure_cost_lines(text, state["recipes"], state["request"].language)
        return {"answer": _with_egypt_note(note, text), "fallback": None, "egypt_note": note}
    except Exception as exc:
        # Log the real error (quota, overload, key...) so it can be diagnosed,
        # while the user still gets the plain answer.
        log.warning(
            "response step failed, using the plain answer: %s: %s",
            type(exc).__name__, exc,
        )
        return {
            "answer": _with_egypt_note(
                note,
                verifier.fallback_answer(
                    state["request"],
                    state["recipes"],
                    state["shopping"],
                    saved_exclusions=state.get("saved_exclusions"),
                    memory_event=state.get("memory_event"),
                    reason="llm_error",
                ),
            ),
            "fallback": "llm_error",
            "egypt_note": note,
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
            "answer": _with_egypt_note(
                state.get("egypt_note"),
                verifier.fallback_answer(
                    state["request"],
                    state["recipes"],
                    state["shopping"],
                    saved_exclusions=state.get("saved_exclusions"),
                    memory_event=state.get("memory_event"),
                    reason="check_failed",
                ),
            ),
            "fallback": "check_failed",
            "verifier_report": report,
        }
    return {"verifier_report": report}


def after_memory(state: GraphState) -> str:
    req = state["request"]
    if not req.ingredients or _only_pantry(req.ingredients):
        # No ingredients: the message may name a dish. Not when a memory action was just done.
        if not req.ingredients and not state.get("memory_event") and not req.remember:
            return "dish"
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
    graph.add_node("dish", dish_node)
    graph.add_node("substitute", substitute_node)
    graph.add_node("verify", verify_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("respond", respond_node)
    graph.add_node("check", check_node)

    graph.add_edge(START, "parse")
    graph.add_edge("parse", "memory")
    graph.add_conditional_edges(
        "memory", after_memory,
        {"search": "search", "plan": "plan", "clarify": "clarify", "dish": "dish"},
    )
    graph.add_conditional_edges(
        "dish", after_dish, {"verify": "verify", "clarify": "clarify", "end": END}
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


def _cacheable(out: dict) -> bool:
    """Only plain successful answers are stored: not errors or plain fallbacks, not memory
    actions (they must run again to take effect), not clarification questions."""
    return bool(out.get("recipes")) and not out.get("fallback") and not out.get("memory_event")


def answer(text: str, user_id: str = "default", use_cache: bool = True) -> str:
    """Final answer for a message. A repeated request is served from the cache (no LLM call)."""
    try:
        exclusions = memory.get_exclusions(user_id)
    except memory.MemoryReadError:
        exclusions, use_cache = [], False  # never serve or store a cached answer blind
    if use_cache:
        hit = cache.get(text, exclusions)
        if hit:
            return hit
    out = run_agent(text, user_id)
    if use_cache and _cacheable(out):
        cache.put(text, exclusions, out["answer"])
    return out["answer"]
