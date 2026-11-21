"""LLM-powered agents. Each agent does one job and returns structured data."""
import json

from recipe_agent import tools
from recipe_agent.config import get_llm_chain
from recipe_agent.schemas import SubstitutionPlan, UserRequest
from recipe_agent.utils import message_text

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
- clarifying_question: ONLY when the user gave NO ingredients they have. Then write ONE short
  reply (at most two sentences) in the user's language that asks which ingredients they have.
  Tailor it to what they said: if they mention a kind of food or a diet (light, vegetarian,
  spicy, quick...), acknowledge it in a few words and ask for the ingredients; if it is a
  greeting or a question about what you can do, say in one sentence that you suggest recipes
  from the ingredients they have, then ask for them. Do not suggest any recipe and do not give
  health or allergy advice. If the user gave ingredients, use null.
- Never invent ingredients or constraints the user did not mention."""

MAX_QUESTION_CHARS = 300


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
        for missing in recipe.get("missing", []):
            cands = tools.substitute_candidates(
                missing, request.ingredients, request.exclude_ingredients
            )
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
- If a recipe has a non-empty "substitutions" list, add under it a short section (Arabic: بدائل
  ممكنة, English: Possible substitutions). For each item write: the missing ingredient, the
  substitute, and the note translated faithfully. If "user_has" is true, say the user already has
  it. Use only what is in the data: never add other substitutes or quantities. Say these are
  suggestions that were not tested.
- If "missing" is empty, say the user has everything they need.
- If request.exclude_ingredients is not empty, add one short note that the filtering is based on
  ingredient names and recipe steps only and the user must double-check labels for allergies.
- If "saved_allergies_applied" is not empty, say in one short line that the saved allergies
  (list them) were applied, and that the user can ask to forget them.
- If "memory_event" is "saved", say the allergies were saved for next time. If it is "forgotten",
  say the saved allergies were cleared.
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
) -> str:
    """Response agent: structured results -> final answer in the user's language."""
    payload = {
        "request": request.model_dump(),
        "recipes": recipes,
        "shopping_list": shopping,
        "saved_allergies_applied": list(saved_exclusions or []),
        "memory_event": memory_event,
    }
    llm = get_llm_chain(temperature=0.3)
    message = llm.invoke(
        [("system", RESPONSE_PROMPT), ("human", json.dumps(payload, ensure_ascii=False))]
    )
    return message_text(message)
