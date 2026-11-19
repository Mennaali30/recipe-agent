"""LLM-powered agents. Each agent does one job and returns structured data."""
import json

from recipe_agent.config import get_llm
from recipe_agent.schemas import UserRequest
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
- Never invent ingredients or constraints the user did not mention."""


def _clean_list(items) -> list[str]:
    seen, out = set(), []
    for item in items or []:
        value = str(item).strip().lower()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def parse_request(text: str) -> UserRequest:
    """Intent agent: free text -> UserRequest."""
    llm = get_llm(temperature=0).with_structured_output(UserRequest)
    req = llm.invoke([("system", INTENT_PROMPT), ("human", text)])
    req.ingredients = _clean_list(req.ingredients)
    req.exclude_ingredients = _clean_list(req.exclude_ingredients)
    if req.max_minutes is not None and req.max_minutes <= 0:
        req.max_minutes = None
    return req


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
- If "missing" is empty, say the user has everything they need.
- If request.exclude_ingredients is not empty, add one short note that the filtering is based on
  ingredient names and recipe steps only and the user must double-check labels for allergies.
- If "recipes" is empty, say no matching recipes were found and suggest adding more ingredients or
  allowing more time. Do not invent a recipe.
- End with a combined shopping list if "shopping_list" is not empty.
- Finish with one short line saying that times and nutrition values come from the recipe dataset
  and are approximate.
- Use short markdown: a heading per recipe and numbered steps. No long introductions."""


def write_answer(request: UserRequest, recipes: list, shopping: list) -> str:
    """Response agent: structured results -> final answer in the user's language."""
    payload = {
        "request": request.model_dump(),
        "recipes": recipes,
        "shopping_list": shopping,
    }
    llm = get_llm(temperature=0.3)
    message = llm.invoke(
        [("system", RESPONSE_PROMPT), ("human", json.dumps(payload, ensure_ascii=False))]
    )
    return message_text(message)
