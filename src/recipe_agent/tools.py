"""Plain-Python tools: ingredient matching, shopping list, nutrition, recipe lookup."""
import re

from recipe_agent.retriever import get_recipe, search

# Assumed to be available in any kitchen, so they are never reported as missing.
PANTRY = {"salt", "pepper", "black pepper", "water", "oil", "vegetable oil"}

_DESCRIPTORS = {
    "fresh", "freshly", "chopped", "minced", "diced", "sliced", "large", "small",
    "medium", "boneless", "skinless", "unsalted", "salted", "whole", "finely",
    "cooked", "raw", "frozen",
}


def _stem(token: str) -> str:
    """Very small singularizer (tomatoes -> tomato, berries -> berry, eggs -> egg)."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("oes"):
        return token[:-2]
    if len(token) > 4 and token.endswith(("ches", "shes", "xes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize(text: str) -> frozenset:
    """Reduce an ingredient string to a set of stemmed keywords."""
    text = re.sub(r"\(.*?\)", " ", str(text).lower())
    tokens = re.findall(r"[a-z]+", text)
    return frozenset(_stem(t) for t in tokens if t not in _DESCRIPTORS)


_PANTRY = {normalize(p) for p in PANTRY}


def _matches(a: frozenset, b: frozenset) -> bool:
    """True if one keyword set contains the other (chicken ~ chicken breast)."""
    return bool(a) and bool(b) and (a <= b or b <= a)


def compare_ingredients(user_ingredients, recipe_ingredients, assume_pantry: bool = True) -> dict:
    """Split a recipe's ingredients into what the user has and what is missing."""
    user = [normalize(u) for u in user_ingredients]
    have, missing = [], []
    for ing in recipe_ingredients:
        norm = normalize(ing)
        if assume_pantry and norm in _PANTRY:
            continue
        if any(_matches(u, norm) for u in user):
            have.append(ing)
        else:
            missing.append(ing)
    total = len(have) + len(missing)
    coverage = len(have) / total if total else 0.0
    return {"have": have, "missing": missing, "coverage": round(coverage, 2)}


def nutrition_per_serving(recipe: dict) -> dict:
    """Nutrition values as given by the dataset (treated as per-serving)."""
    return {
        "calories": round(float(recipe["calories"])),
        "protein_g": round(float(recipe["protein_g"]), 1),
        "carbs_g": round(float(recipe["carbs_g"]), 1),
        "fat_g": round(float(recipe["fat_g"]), 1),
        "sugar_g": round(float(recipe["sugar_g"]), 1),
    }


def shopping_list(missing_lists) -> list:
    """Merge the missing ingredients of several recipes without duplicates."""
    seen, out = set(), []
    for items in missing_lists:
        for item in items:
            key = normalize(item) or frozenset({str(item).lower()})
            if key not in seen:
                seen.add(key)
                out.append(item)
    return out


def find_recipes(ingredients, max_minutes=None, top_k: int = 5, candidates: int = 30) -> list:
    """Search the index, compare ingredients, and rank by how much the user already has."""
    query = "Ingredients: " + ", ".join(ingredients)
    min_have = min(2, len(ingredients))
    results = []
    for hit in search(query, n_results=candidates, max_minutes=max_minutes):
        recipe = get_recipe(hit["recipe_id"])
        cmp = compare_ingredients(ingredients, list(recipe["ingredients"]))
        if len(cmp["have"]) < min_have:
            continue
        results.append(
            {
                "recipe_id": hit["recipe_id"],
                "name": recipe["name"],
                "category": recipe["category"],
                "total_minutes": hit["total_minutes"],
                "servings": int(recipe["servings"]),
                "nutrition": nutrition_per_serving(recipe),
                "have": cmp["have"],
                "missing": cmp["missing"],
                "coverage": cmp["coverage"],
                "distance": hit["distance"],
            }
        )
    results.sort(key=lambda r: (-r["coverage"], len(r["missing"]), r["distance"]))
    return results[:top_k]
