"""LLM-powered agents. Each agent does one job and returns structured data."""
import json
import logging

from recipe_agent import tools
from recipe_agent.config import get_llm_chain
from recipe_agent.schemas import MealPlan, SubstitutionPlan, UserRequest
from recipe_agent.utils import message_text

log = logging.getLogger(__name__)

MAX_MEALS = 5
MAX_QUESTION_CHARS = 300

INTENT_PROMPT = """You extract a cooking request from the user's message. The message may be in
Arabic (including Egyptian dialect) or English.

Rules:
- ingredients: ONLY ingredients the user says they HAVE. Translate to simple singular English
  names, as they would appear in a recipe. Examples: فراخ = chicken, رز or أرز = rice,
  ثوم = garlic, بصل = onion, طماطم = tomato, لحمة = beef, بيض = egg, بطاطس = potato,
  جبنة = cheese, لبن or حليب = milk, زبادي = yogurt, عدس = lentils.
- exclude_ingredients: ingredients the user wants to avoid (allergy, dislike, diet). Same rules.
  For an allergy or avoidance of a whole food family, use the family word: milk (for dairy,
  لبن, ألبان), egg, peanut, nut (tree nuts), wheat, gluten, soy, fish, shellfish, sesame.
- max_minutes: a time limit in minutes only if the user mentions one (convert hours to minutes).
  Otherwise null.
- language: "ar" if the user wrote in Arabic, otherwise "en".
- remember: true ONLY if the user explicitly asks to remember or save their allergies / avoided
  ingredients for next time (for example "remember that I'm allergic to nuts", "افتكري إني عندي
  حساسية من اللبن"). Merely mentioning an allergy is NOT a request to remember: use false.
- forget_memory: true ONLY if the user explicitly asks to forget or clear saved allergies or
  preferences (for example "forget my allergies", "انسي الحساسيات اللي حفظتيها").
- meals: a number (2 to 7) ONLY if the user explicitly asks for several meals or a plan for
  several days (for example "وجبات لـ 3 أيام" = 3, "plan dinners for the week" = 7). For a
  single dish or one meal use null.
- clarifying_question: ONLY when the user gave NO ingredients they have. Then write ONE short
  reply (at most two sentences) in the user's language that asks which ingredients they have.
  Tailor it to what they said: if they mention a kind of food or a diet (light, vegetarian,
  spicy, quick...), acknowledge it in a few words and ask for the ingredients; if it is a
  greeting or a question about what you can do, say in one sentence that you suggest recipes
  from the ingredients they have, then ask for them. Do not suggest any recipe and do not give
  health or allergy advice. If the user gave ingredients, use null.
- Never invent ingredients or constraints the user did not mention."""


def _clean_list(items) -> list:
    seen, out = set(), []
    for item in items or []:
        value = str(item).strip().lower()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def parse_request(text: str, llm=None) -> UserRequest:
    """Intent agent: free text -> UserRequest (one LLM call)."""
    llm = llm or get_llm_chain(temperature=0, schema=UserRequest)
    req = llm.invoke([("system", INTENT_PROMPT), ("human", text)])
    req.ingredients = _clean_list(req.ingredients)
    req.exclude_ingredients = _clean_list(req.exclude_ingredients)
    if req.max_minutes is not None and req.max_minutes <= 0:
        req.max_minutes = None
    if req.meals is not None and req.meals < 2:
        req.meals = None
    # The clarifying question is only kept when there are no ingredients and it looks sane.
    question = (req.clarifying_question or "").strip()
    if req.ingredients or len(question) < 5 or len(question) > MAX_QUESTION_CHARS:
        question = ""
    req.clarifying_question = question or None
    return req


SUBSTITUTION_PROMPT = """You help choose ingredient substitutions for recipes.

You receive JSON: a list of recipes, each with "options". Every option has a "missing"
ingredient and a list of "candidates" (allowed substitutes, already checked for allergies).

For each missing ingredient:
- Decide whether one candidate is a sensible swap IN THAT RECIPE. Use the recipe name
  (for example egg replacers suit cakes and muffins, not omelets or fried eggs).
- Choose at most one candidate per missing ingredient. Prefer candidates with user_has = true.
- If no candidate fits, skip that ingredient. Skipping is fine.
- Copy recipe_id, missing and substitute EXACTLY as they appear in the input.
- Never invent a substitute that is not in the candidates."""


def plan_substitutions(request: UserRequest, recipes: list, llm=None) -> list:
    """Substitution agent: pick substitutes for missing ingredients.

    Code builds the allowed candidates, the LLM chooses among them, and code validates the
    choices (anything not in the candidate list is dropped). One LLM call for all recipes,
    and none at all when no missing ingredient has a candidate.
    """
    items, allowed = [], {}
    for recipe in recipes:
        options = []
        missing_norms = [tools.normalize(m) for m in recipe.get("missing", [])]
        for missing in recipe.get("missing", []):
            cands = tools.substitute_candidates(
                missing, request.ingredients, request.exclude_ingredients
            )
            # A substitute that the recipe is itself missing is no help
            # (for example lime juice <-> lemon juice when both are missing).
            cands = [
                c for c in cands
                if not any(
                    tools._matches(tools.normalize(c["substitute"]), m) for m in missing_norms
                )
            ]
            if cands:
                options.append({"missing": missing, "candidates": cands})
                allowed[(recipe["recipe_id"], missing.strip().lower())] = (
                    missing,
                    {c["substitute"].strip().lower(): c for c in cands},
                )
        if options:
            items.append(
                {"recipe_id": recipe["recipe_id"], "name": recipe["name"], "options": options}
            )
    if not items:
        return []

    llm = llm or get_llm_chain(temperature=0, schema=SubstitutionPlan)
    plan = llm.invoke(
        [("system", SUBSTITUTION_PROMPT), ("human", json.dumps(items, ensure_ascii=False))]
    )

    result, seen = [], set()
    for choice in plan.choices:
        key = (choice.recipe_id, choice.missing.strip().lower())
        entry = allowed.get(key)
        if entry is None or key in seen:
            continue
        original_missing, by_name = entry
        cand = by_name.get(choice.substitute.strip().lower())
        if cand is None:
            continue  # invented by the LLM: dropped
        seen.add(key)
        result.append(
            {
                "recipe_id": choice.recipe_id,
                "missing": original_missing,
                "substitute": cand["substitute"],
                "note": cand["note"],
                "user_has": cand["user_has"],
            }
        )
    return result


PLAN_PROMPT = """You build a short, varied meal plan from a shortlist of recipes.

You receive JSON with "meals" (how many recipes to choose) and "shortlist" (recipes already
checked for the user's allergies, best match first). Each one has recipe_id, name, category,
total_minutes, calories, used (how many of the user's ingredients it uses) and missing_count.

- Choose exactly "meals" DIFFERENT recipes from the shortlist.
- Prefer variety: different categories and different main ingredients.
- Prefer recipes that use more of the user's ingredients and have fewer missing items.
- Put them in a sensible order for the days (for example a lighter meal after a heavy one).
- Copy recipe_id exactly as it appears in the input. Never invent an id."""


def _pick_varied(candidates, n, already=()):
    """Simple deterministic choice: best ranked first, different categories before repeats.

    `candidates` are already ranked. Returns a list of recipe_ids (at most n).
    """
    chosen = list(already)
    by_id = {r["recipe_id"]: r for r in candidates}
    used_categories = {by_id[i]["category"] for i in chosen if i in by_id}
    for r in candidates:
        if len(chosen) >= n:
            break
        if r["recipe_id"] not in chosen and r["category"] not in used_categories:
            chosen.append(r["recipe_id"])
            used_categories.add(r["category"])
    for r in candidates:
        if len(chosen) >= n:
            break
        if r["recipe_id"] not in chosen:
            chosen.append(r["recipe_id"])
    return chosen


def plan_meals(shortlist: list, n: int, llm=None):
    """Meal-planner agent: choose and order n recipes from a code-built shortlist.

    The LLM picks and orders; code drops invented or repeated ids and fills any gap with a
    simple varied choice. If the LLM fails, the plan is built by code alone.
    Returns (recipes_in_day_order, source) where source is "llm", "llm+code", "code" or "all"
    ("all": the shortlist was not bigger than n, so there was nothing to choose).
    """
    if len(shortlist) <= n:
        return list(shortlist), "all"

    by_id = {r["recipe_id"]: r for r in shortlist}
    chosen = []
    try:
        llm = llm or get_llm_chain(temperature=0, schema=MealPlan)
        items = [
            {
                "recipe_id": r["recipe_id"],
                "name": r["name"],
                "category": r["category"],
                "total_minutes": r["total_minutes"],
                "calories": r["nutrition"]["calories"],
                "used": r["used"],
                "missing_count": len(r["missing"]),
            }
            for r in shortlist
        ]
        plan = llm.invoke(
            [
                ("system", PLAN_PROMPT),
                ("human", json.dumps({"meals": n, "shortlist": items}, ensure_ascii=False)),
            ]
        )
        for rid in plan.recipe_ids:
            if rid in by_id and rid not in chosen:
                chosen.append(rid)
            if len(chosen) == n:
                break
    except Exception as exc:
        log.warning("meal planner LLM step failed, using the simple choice: %s", type(exc).__name__)

    from_llm = len(chosen)
    chosen = _pick_varied(shortlist, n, already=chosen)
    source = "llm" if from_llm == n else "llm+code" if from_llm else "code"
    return [by_id[i] for i in chosen], source


RESPONSE_PROMPT = """You write the final answer of a recipe assistant, using ONLY the JSON data you receive.

Rules:
- Write the whole answer in the language given by request.language ("ar" = clear, simple Arabic;
  "en" = English).
- Use only the data provided. Never add ingredients, quantities, times, steps or nutrition values.
- For each recipe write: its name (when translating to Arabic keep the English name in parentheses),
  total time in minutes, servings, calories and protein per serving, what the user already has
  ("have"), what is missing ("missing"), and the numbered steps.
- Translate the steps faithfully and completely. Keep every quantity and temperature exactly as written.
  The dataset lists quantities only inside the steps, so do not make a separate quantities list.
- If you are not sure how to translate a food word or a cooking term, keep the English word in
  parentheses instead of guessing.
- Arabic food words, use exactly these: shortening (Crisco) = دهن نباتي (shortening), NEVER سمن
  (سمن means ghee, a dairy product); sour cream = كريمة حامضة (sour cream); baking powder = بيكنج
  بودر; flour = دقيق; butter = زبدة; margarine = مارجرين.
- If "plan" is not null, this is a meal plan: start with one short line saying how many meals were
  planned (plan.planned), and put each recipe under its day using the recipe's "day" (Arabic:
  اليوم N, English: Day N). If plan.requested is larger than plan.planned, say that fewer
  matching recipes were found or that the maximum is 5 meals. Add one line saying the dataset has
  no ingredient quantities, so the user should check that what they have is enough for all the
  meals. The shopping list covers all the meals together.
- If a recipe has a non-empty "substitutions" list, add under it a short section (Arabic: بدائل
  ممكنة, English: Possible substitutions). For each item write: the missing ingredient, the
  substitute, and the note translated faithfully. If "user_has" is true, say the user already has
  it. Use only what is in the data: never add other substitutes or quantities. Say these are
  suggestions from a small hand-written table that were not tested; never say they come from the
  recipe dataset.
- If "missing" is empty, say the user has everything they need.
- If request.exclude_ingredients is not empty, add one short note that the filtering is based on
  ingredient names and recipe steps only and the user must double-check labels for allergies.
- If "saved_allergies_applied" is not empty, say in one short line that the saved allergies
  (list them) were applied, and that the user can ask to forget them.
- If "memory_event" is "saved", say the allergies were saved for next time. If it is "forgotten",
  say the saved allergies were cleared.
- If "memory_event" is "memory_error", say the saved allergies could not be read, only the allergies
  in this message were applied, and the user should double-check.
- If a recipe's "total_minutes" is null, say its time is unknown. Never write 0 minutes.
- If "recipes" is empty, say no matching recipes were found and suggest adding more ingredients or
  allowing more time. Do not invent a recipe.
- End with a combined shopping list if "shopping_list" is not empty.
- Finish with one short line saying that times and nutrition values come from the recipe dataset
  and are approximate.
- Use short markdown: a heading per recipe and numbered steps. No long introductions."""


def write_answer(
    request: UserRequest,
    recipes: list,
    shopping: list,
    saved_exclusions=None,
    memory_event=None,
    plan=None,
) -> str:
    """Response agent: structured results -> final answer in the user's language."""
    payload = {
        "request": request.model_dump(),
        "recipes": recipes,
        "shopping_list": shopping,
        "saved_allergies_applied": list(saved_exclusions or []),
        "memory_event": memory_event,
        "plan": plan,
    }
    llm = get_llm_chain(temperature=0.3)
    message = llm.invoke(
        [("system", RESPONSE_PROMPT), ("human", json.dumps(payload, ensure_ascii=False))]
    )
    return message_text(message)
