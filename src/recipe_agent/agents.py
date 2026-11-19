"""LLM-powered agents. Each agent does one job and returns structured data."""
from recipe_agent.config import get_llm
from recipe_agent.schemas import UserRequest

INTENT_PROMPT = """You extract a cooking request from the user's message. The message may be in
Arabic (including Egyptian dialect) or English.

Rules:
- ingredients: ONLY ingredients the user says they HAVE. Translate to simple singular English
  names, as they would appear in a recipe. Examples: فراخ = chicken, رز or أرز = rice,
  ثوم = garlic, بصل = onion, طماطم = tomato, لحمة = beef, بيض = egg, بطاطس = potato,
  جبنة = cheese, لبن or حليب = milk, زبادي = yogurt, عدس = lentils.
- exclude_ingredients: ingredients the user wants to avoid (allergy, dislike, diet). Same rules.
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
