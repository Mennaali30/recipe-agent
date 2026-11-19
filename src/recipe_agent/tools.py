"""Plain-Python tools: ingredient matching, shopping list, nutrition, recipe lookup."""
import re

from recipe_agent.retriever import get_recipe, search

# Assumed to be available in any kitchen, so they are never reported as missing.
PANTRY = {
    "salt", "kosher salt", "sea salt", "table salt", "pepper", "black pepper",
    "ground black pepper", "water", "boiling water", "oil", "vegetable oil",
    "olive oil", "extra virgin olive oil", "canola oil", "cooking oil",
}

_DESCRIPTORS = {
    "fresh", "freshly", "chopped", "minced", "diced", "sliced", "large", "small",
    "medium", "boneless", "skinless", "unsalted", "salted", "whole", "finely",
    "cooked", "raw", "frozen", "of", "and", "ground", "dried",
    "table", "boiling", "all", "purpose", "extra", "virgin", "kosher", "sea",
}

# Words that turn an ingredient into a different product:
# "chicken broth" is not "chicken", "garlic powder" is not "garlic".
_PROCESSED = {
    "broth", "stock", "bouillon", "sauce", "powder", "soup", "paste", "oil",
    "butter", "cream", "milk", "cheese", "flour", "sugar", "vinegar",
    "dressing", "seasoning", "extract", "mix", "instant",
}


def _stem(token: str) -> str:
    """Very small singularizer (tomatoes -> tomato, berries -> berry, eggs -> egg)."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("oes"):
        return token[:-2]
    if len(token) > 4 and token.endswith(("ches", "shes", "xes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us")):
        return token[:-1]
    return token


def normalize(text: str) -> frozenset:
    """Reduce an ingredient string to a set of stemmed keywords."""
    text = re.sub(r"\(.*?\)", " ", str(text).lower())
    tokens = re.findall(r"[a-z]+", text)
    return frozenset(_stem(t) for t in tokens if t not in _DESCRIPTORS)


def _words(text) -> frozenset:
    """All stemmed words of a free text (used to scan recipe steps)."""
    return frozenset(_stem(t) for t in re.findall(r"[a-z]+", str(text).lower()))


_PANTRY = {normalize(p) for p in PANTRY}


def _group(*words) -> frozenset:
    return frozenset(_stem(w) for w in words)


# Allergen families. Matching is deliberately conservative (it may exclude harmless
# recipes such as ones with coconut milk) because a wrong inclusion is worse than a miss.
_DAIRY = _group(
    "milk", "cheese", "butter", "cream", "yogurt", "yoghurt", "whey", "ghee",
    "buttermilk", "casein", "custard", "mozzarella", "parmesan", "cheddar",
    "feta", "ricotta", "gruyere", "mascarpone", "halloumi", "paneer", "kefir",
)
_WHEAT = _group(
    "wheat", "flour", "bread", "breadcrumb", "pasta", "noodle", "couscous",
    "semolina", "barley", "rye", "bulgur", "cracker", "tortilla", "spaghetti",
    "macaroni", "pita",
)
ALLERGEN_GROUPS = {
    "milk": _DAIRY,
    "dairy": _DAIRY,
    "lactose": _DAIRY,
    "egg": _group("egg", "mayonnaise", "meringue"),
    "peanut": _group("peanut"),
    "nut": _group(
        "nut", "almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut",
        "macadamia", "peanut",
    ),
    "wheat": _WHEAT,
    "gluten": _WHEAT,
    "soy": _group("soy", "soya", "tofu", "edamame", "miso", "tempeh"),
    "fish": _group("fish", "salmon", "tuna", "cod", "anchovy", "tilapia", "sardine", "trout"),
    "shellfish": _group(
        "shellfish", "shrimp", "prawn", "crab", "lobster", "clam", "mussel",
        "oyster", "scallop",
    ),
    "sesame": _group("sesame", "tahini"),
}


def _matches(user: frozenset, ing: frozenset) -> bool:
    """Does what the user typed cover this recipe ingredient?

    chicken ~ chicken breast (extra words are fine),
    but chicken !~ chicken broth (extra word turns it into another product).
    """
    if not user or not ing:
        return False
    if user == ing:
        return True
    if user < ing:
        return not ((ing - user) & _PROCESSED)
    if ing < user:
        return not ((user - ing) & _PROCESSED)
    return False


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
    """Nutrition values as given by the dataset (treated as per-serving, approximate)."""
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


def _used_count(user_ingredients, recipe_ingredients) -> int:
    """How many of the USER's ingredients does this recipe actually use?"""
    user = [normalize(u) for u in user_ingredients]
    ings = [normalize(i) for i in recipe_ingredients]
    return sum(1 for u in user if any(_matches(u, i) for i in ings))


def get_instructions(recipe_id) -> list:
    """The recipe's steps, as plain strings."""
    return [str(s) for s in get_recipe(recipe_id)["instructions"]]


def _is_excluded(recipe_name, recipe_ingredients, exclude, instructions=None) -> bool:
    """True if the recipe's name, ingredients or steps contain an excluded ingredient.

    Allergen words (milk, egg, nut, wheat, ...) are expanded to their whole family,
    so 'milk' also excludes cheese, butter, cream, etc. The steps are scanned too,
    because some recipes mention an ingredient only there (e.g. 'serve with sour cream').
    Hidden sources (e.g. chocolate, bread) are still not detected.
    """
    if not exclude:
        return False
    texts = [normalize(recipe_name)] + [normalize(i) for i in recipe_ingredients]
    step_words = _words(" ".join(instructions)) if instructions else frozenset()
    for item in exclude:
        target = normalize(item)
        if not target:
            continue
        if any(target <= text for text in texts):
            return True
        if step_words and target <= step_words:
            return True
        family = frozenset().union(*(ALLERGEN_GROUPS.get(tok, frozenset()) for tok in target))
        if family and any(family & text for text in texts):
            return True
        if family and family & step_words:
            return True
    return False


def find_recipes(
    ingredients, max_minutes=None, exclude=None, top_k: int = 5, candidates: int = 40
) -> list:
    """Search the index, compare ingredients, and rank recipes.

    Ranking: uses the most of the user's ingredients first, then fewest missing items,
    then highest coverage, then closest semantic match.
    Recipes containing an excluded ingredient (or its allergen family) are dropped.
    """
    query = "Ingredients: " + ", ".join(ingredients)
    min_used = min(2, len(ingredients))
    results = []
    for hit in search(query, n_results=candidates, max_minutes=max_minutes):
        recipe = get_recipe(hit["recipe_id"])
        recipe_ings = list(recipe["ingredients"])
        steps = [str(s) for s in recipe["instructions"]]
        if _is_excluded(recipe["name"], recipe_ings, exclude, steps):
            continue
        used = _used_count(ingredients, recipe_ings)
        if used < min_used:
            continue
        cmp = compare_ingredients(ingredients, recipe_ings)
        results.append(
            {
                "recipe_id": hit["recipe_id"],
                "name": recipe["name"],
                "category": recipe["category"],
                "total_minutes": hit["total_minutes"],
                "servings": int(recipe["servings"]),
                "nutrition": nutrition_per_serving(recipe),
                "used": used,
                "have": cmp["have"],
                "missing": cmp["missing"],
                "coverage": cmp["coverage"],
                "distance": hit["distance"],
            }
        )
    results.sort(key=lambda r: (-r["used"], len(r["missing"]), -r["coverage"], r["distance"]))
    return results[:top_k]
