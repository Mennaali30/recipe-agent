"""Plain-Python tools: ingredient matching, shopping list, nutrition, recipe lookup,
allergy filtering and substitution candidates."""
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
    "coarse", "coarsely", "cracked",
}

# Words that turn an ingredient into a different product:
# "chicken broth" is not "chicken", "garlic powder" is not "garlic".
_PROCESSED = {
    "broth", "stock", "bouillon", "sauce", "powder", "soup", "paste", "oil",
    "butter", "cream", "milk", "cheese", "flour", "sugar", "vinegar",
    "dressing", "seasoning", "extract", "mix", "instant", "cider",
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
# Hidden sources not listed here (e.g. chocolate that may contain milk) are NOT detected.
_DAIRY = _group(
    "milk", "cheese", "butter", "cream", "yogurt", "yoghurt", "whey", "ghee",
    "buttermilk", "casein", "custard", "mozzarella", "parmesan", "cheddar",
    "feta", "ricotta", "gruyere", "mascarpone", "halloumi", "paneer", "kefir",
    "brie", "camembert", "gouda", "provolone", "gorgonzola", "romano",
    "pecorino", "creamer",
)
_WHEAT = _group(
    "wheat", "flour", "bread", "breadcrumb", "pasta", "noodle", "couscous",
    "semolina", "barley", "rye", "bulgur", "cracker", "tortilla", "spaghetti",
    "macaroni", "pita", "pastry", "dough", "crust", "bisquick", "biscuit",
    "pancake", "waffle", "muffin",
)
ALLERGEN_GROUPS = {
    "milk": _DAIRY,
    "dairy": _DAIRY,
    "lactose": _DAIRY,
    "egg": _group(
        "egg", "mayonnaise", "mayo", "meringue", "eggnog", "custard", "aioli",
        "hollandaise",
    ),
    "peanut": _group("peanut"),
    "nut": _group(
        "nut", "almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut",
        "macadamia", "peanut",
    ),
    "wheat": _WHEAT,
    "gluten": _WHEAT,
    "soy": _group("soy", "soya", "tofu", "edamame", "miso", "tempeh", "tamari", "soybean"),
    "fish": _group(
        "fish", "salmon", "tuna", "cod", "anchovy", "tilapia", "sardine", "trout",
        "catfish", "snapper", "haddock", "halibut", "mackerel", "herring",
        "worcestershire", "caesar",
    ),
    "shellfish": _group(
        "shellfish", "shrimp", "prawn", "crab", "lobster", "clam", "mussel",
        "oyster", "scallop", "seafood", "surimi",
    ),
    "sesame": _group("sesame", "tahini", "hummus", "halva", "halvah"),
}

# Multi-word forms that the word lists cannot catch (matched on the raw lowercase text).
_HALF_AND_HALF = re.compile(r"half[\s-]*(?:and|&)[\s-]*half")
ALLERGEN_PHRASES = {
    "milk": [_HALF_AND_HALF],
    "dairy": [_HALF_AND_HALF],
    "lactose": [_HALF_AND_HALF],
}

# Canned / packaged soup used as an ingredient is often dairy or wheat based (cream soups,
# onion soup mix) but it is frequently missing from the ingredient list of the dataset.
# This is a word-level heuristic: it ignores cookware and serving phrases ("soup pot",
# "as a soup", "whole soup") and may still produce false alarms or miss some cases.
_SOUP_FAMILIES = frozenset({"milk", "dairy", "lactose", "wheat", "gluten"})
_SOUP_STRONG = re.compile(r"\bcondensed\b|\bcream of (?:\w+ ){0,2}soup\b|\bsoup mix\b")
_SOUP_WORD = re.compile(r"\bsoup\b")
_SOUP_NOT_AFTER = {"pot", "pots", "pan", "bowl", "bowls", "spoon", "spoons", "kettle", "ladle", "mug"}
_SOUP_NOT_BEFORE = {"a", "as", "whole", "large", "big", "medium", "small", "for", "this", "your", "entire"}


def _canned_soup_in_steps(texts) -> bool:
    """True if the texts seem to use ready-made soup as an ingredient."""
    for text in texts:
        low = str(text).lower()
        if _SOUP_STRONG.search(low):
            return True
        for m in _SOUP_WORD.finditer(low):
            before = [re.sub(r"[^a-z]", "", w) for w in low[: m.start()].split()[-1:]]
            after = [re.sub(r"[^a-z]", "", w) for w in low[m.end():].split()[:1]]
            if before and before[0] in _SOUP_NOT_BEFORE:
                continue
            if after and after[0] in _SOUP_NOT_AFTER:
                continue
            return True
    return False


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


def _is_excluded(
    recipe_name, recipe_ingredients, exclude, instructions=None, check_soup=False
) -> bool:
    """True if the recipe's name, ingredients or steps contain an excluded ingredient.

    Allergen words (milk, egg, nut, wheat, ...) are expanded to their whole family,
    so 'milk' also excludes cheese, butter, cream, etc. The steps are scanned too,
    because some recipes mention an ingredient only there (e.g. 'serve with sour cream').
    With check_soup=True (used by the recipe search), a recipe that uses ready-made soup is
    also excluded for milk / dairy / wheat / gluten (see _canned_soup_in_steps).
    Hidden sources that are not in the lists (e.g. chocolate) are still not detected.
    """
    if not exclude:
        return False
    ings = [str(i) for i in recipe_ingredients]
    steps = [str(s) for s in instructions] if instructions else []
    texts = [normalize(recipe_name)] + [normalize(i) for i in ings]
    step_words = _words(" ".join(steps)) if steps else frozenset()
    raw = " ".join([str(recipe_name), *ings, *steps]).lower()
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
        if any(p.search(raw) for tok in target for p in ALLERGEN_PHRASES.get(tok, [])):
            return True
    if check_soup and not _SOUP_WORD.search(str(recipe_name).lower()):
        wanted = set()
        for item in exclude:
            wanted |= set(normalize(item))
        if wanted & _SOUP_FAMILIES and _canned_soup_in_steps(ings + steps):
            return True
    return False


# --------------------------------------------------------------------------- #
# Substitutions: a small hand-written table (NOT exhaustive, culinary notes are
# general and were not tested in real recipes). The LLM only chooses among these
# candidates; the code removes anything that conflicts with the user's allergies.
# --------------------------------------------------------------------------- #
_SUB_IGNORE = frozenset(
    {
        "reduced", "sodium", "low", "fat", "free", "plain", "unsweetened", "pure",
        "light", "clove", "granulated", "yellow",
    }
)


def _sub_key(text: str) -> frozenset:
    return normalize(text) - _SUB_IGNORE


_CREAM_SUBS = [
    ("coconut milk", "richer texture with a slight coconut flavor"),
    ("yogurt", "thicker and tangy; stir it in off the heat so it does not curdle"),
    ("milk", "thinner result; simmer to reduce if you need it thicker"),
]
_BROTH_SUBS = [
    ("vegetable broth", "same use; milder flavor"),
    ("water", "add a bit more salt and seasoning"),
]
_WINE_SUBS = [
    ("chicken broth", "add a splash of lemon juice or vinegar for acidity"),
]

_RAW_SUBSTITUTIONS = {
    "butter": [
        ("olive oil", "works for cooking and baking; the flavor is different"),
        ("vegetable oil", "works for cooking and baking; the flavor is milder"),
        ("margarine", "usually a direct swap"),
    ],
    "milk": [
        ("almond milk", "similar texture; unsweetened works best in savory dishes"),
        ("oat milk", "creamy; fine for most recipes"),
        ("soy milk", "closest in protein; fine for most recipes"),
        ("water", "only when the recipe needs a little liquid; the result is less rich"),
    ],
    "heavy cream": _CREAM_SUBS,
    "cream": _CREAM_SUBS,
    "whipping cream": _CREAM_SUBS,
    "heavy whipping cream": _CREAM_SUBS,
    "buttermilk": [
        ("yogurt", "thin it with a little milk or water"),
        ("milk", "add a spoonful of lemon juice or vinegar and let it sit a few minutes"),
    ],
    "sour cream": [("yogurt", "the closest swap; slightly more tangy")],
    "egg": [
        ("banana", "mashed; for sweet baking such as cakes and muffins"),
        ("applesauce", "unsweetened; for sweet baking such as cakes and muffins"),
    ],
    "garlic": [("garlic powder", "use much less than fresh garlic")],
    "onion": [
        ("shallot", "milder and slightly sweeter"),
        ("scallion", "milder; use the white and green parts"),
        ("onion powder", "use much less than fresh onion"),
    ],
    "lemon juice": [
        ("lime juice", "same acidity, slightly different flavor"),
        ("white vinegar", "sharper; use less"),
    ],
    "lime juice": [("lemon juice", "same acidity, slightly different flavor")],
    "honey": [
        ("maple syrup", "similar sweetness"),
        ("sugar", "dissolve it in a little warm water first"),
    ],
    "brown sugar": [("sugar", "slightly less depth of flavor")],
    "sugar": [("honey", "sweeter and liquid; reduce other liquids a little")],
    "soy sauce": [
        ("coconut aminos", "slightly sweeter; soy-free and gluten-free"),
        ("tamari", "similar flavor; check the label if you avoid gluten"),
    ],
    "chicken broth": _BROTH_SUBS,
    "chicken stock": _BROTH_SUBS,
    "baking powder": [
        ("baking soda", "not a direct swap: use less, plus a little lemon juice or vinegar"),
    ],
    "cornstarch": [
        ("flour", "use about twice as much to thicken"),
        ("potato starch", "use the same amount to thicken"),
    ],
    "tomato paste": [("tomato sauce", "use more and simmer to thicken")],
    "basil": [("oregano", "different flavor; use a little less")],
    "thyme": [
        ("oregano", "use a little less"),
        ("marjoram", "similar and slightly sweeter"),
    ],
    "rosemary": [("thyme", "milder")],
    "parsley": [("cilantro", "different flavor")],
    "cilantro": [("parsley", "milder and less citrusy")],
    "dry white wine": _WINE_SUBS,
    "white wine": _WINE_SUBS,
}

SUBSTITUTIONS = {_sub_key(k): v for k, v in _RAW_SUBSTITUTIONS.items()}

# Plant milks are fine for a dairy allergy (but still checked for nut / soy allergies).
_PLANT_MILKS = {
    normalize(m) for m in ("almond milk", "oat milk", "soy milk", "coconut milk", "rice milk")
}
_DAIRY_WORDS = frozenset({"milk", "dairy", "lactose"})

# Substitutes that may contain an allergen the word lists cannot see.
_HIDDEN_RISK = {
    normalize("margarine"): {"milk", "dairy", "lactose"},
    normalize("oat milk"): {"gluten", "wheat"},
    normalize("tamari"): {"gluten", "wheat"},
}


def _substitute_blocked(name: str, exclude) -> bool:
    """True if this substitute conflicts with something the user must avoid."""
    if not exclude:
        return False
    norm = normalize(name)
    risk = _HIDDEN_RISK.get(norm, set())
    effective = []
    for item in exclude:
        target = normalize(item)
        if not target:
            continue
        if target & risk:
            return True
        if norm in _PLANT_MILKS and target <= _DAIRY_WORDS:
            continue  # plant milk is not dairy
        effective.append(item)
    return _is_excluded(name, [name], effective)


def substitute_candidates(missing_item, user_ingredients, exclude=None) -> list:
    """Allowed substitutes for one missing ingredient (empty if the table has none)."""
    user = [normalize(u) for u in user_ingredients]
    out = []
    for name, note in SUBSTITUTIONS.get(_sub_key(missing_item), []):
        if _substitute_blocked(name, exclude):
            continue
        norm = normalize(name)
        user_match = any(_matches(u, norm) for u in user)
        out.append(
            {
                "substitute": name,
                "note": note,
                "user_has": user_match or norm in _PANTRY,
                "_user_match": user_match,
            }
        )
    out.sort(key=lambda c: not c["_user_match"])  # stable: what you own comes first
    for c in out:
        c.pop("_user_match")
    return out


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
        if _is_excluded(recipe["name"], recipe_ings, exclude, steps, check_soup=True):
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
