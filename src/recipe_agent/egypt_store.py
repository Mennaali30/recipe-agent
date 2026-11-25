"""Curated Egyptian recipes (fifi.cooking) inside the app.

Loads the 150 curated recipes from data/egypt/recipes/ (ids in data/egypt/curated.json),
matches them against the user's ingredients (English from the intent agent, or Arabic),
applies the allergy filter to the Arabic data (word scan + fifi's own allergen flags as a
cross-check), the under-N-minutes filter, and attaches the approximate EGP cost.

Runs offline once the data files exist. Keyword matching (no embedding model): with 150
recipes a small inverted index is deterministic and testable; the Chroma index still serves
the English Food.com recipes. Credit fifi.cooking (see data/egypt/DATA_NOTES.txt).

Known limits:
- User ingredients are matched through the price-map ingredient groups, so an ingredient
  that maps to no group (e.g. "molokhia" is fine, "Cheddar" is not) only matches by name.
- have/missing and the steps for Egyptian recipes are Arabic (English steps are used when
  the data has them, which is rare).
- Costs are approximate (see egypt_cost).
"""
import json
import re
from pathlib import Path

from recipe_agent.allergy import arabic_allergen_families, families_for_item
from recipe_agent.egypt_cost import (estimate_recipe_cost, format_cost,
                                     load_price_map, recipe_minutes)
from recipe_agent.egypt_parse import match as price_match
from recipe_agent.egypt_parse import norm, parse_servings

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "egypt"
ID_PREFIX = "egy:"

# fifi's allergen flags use the EU-14 names; these are the ones the app has families for.
# (celery, mustard, lupin, sulphites have no app family and are only word-scanned.)
FIFI_ALLERGEN_MAP = {
    "milk": {"milk", "dairy", "lactose"},
    "eggs": {"egg"},
    "peanuts": {"peanut", "nut"},
    "nuts": {"nut"},
    "sesame": {"sesame"},
    "cereals_gluten": {"wheat", "gluten"},
    "fish": {"fish"},
    "crustaceans": {"shellfish"},
    "molluscs": {"shellfish"},
    "soybeans": {"soy"},
}

# Assumed available in every kitchen (never reported as missing).
PANTRY_KEYS = {"salt", "spices_black_pepper", "water", "oil", "vinegar", "non_food"}

# English ingredient words -> price-map group key. The intent agent translates the user's
# ingredients to English, so this is the bridge back to the Arabic data.
_KEY_ENGLISH = {
    "chicken": ["chicken"], "buffalo_meat": ["buffalo"],
    "beef_generic": ["beef", "meat", "lamb", "mutton", "veal", "liver", "kidney"],
    "minced_beef": ["minced", "ground beef", "kofta", "kufta"],
    "shrimp": ["shrimp", "prawn", "prawns"], "tilapia": ["tilapia"],
    "fish_generic": ["fish"], "tuna": ["tuna"],
    "gizzards_offal": ["gizzard", "gizzards", "offal"],
    "rice": ["rice"], "pasta": ["pasta", "macaroni", "spaghetti", "noodles"],
    "lentils": ["lentils", "lentil"], "chickpeas": ["chickpeas", "chickpea", "hummus"],
    "fava_beans": ["fava", "broad beans", "foul", "beans"],
    "green_beans": ["green beans"], "peas": ["peas"],
    "tomato": ["tomato", "tomatoes"], "tomato_paste_sauce": ["tomato paste", "tomato sauce"],
    "potato": ["potato", "potatoes"], "onion": ["onion"], "garlic": ["garlic"],
    "carrot": ["carrot", "carrots"], "eggplant": ["eggplant", "aubergine"],
    "zucchini": ["zucchini", "courgette"], "okra": ["okra"],
    "molokhia": ["molokhia", "molokheya", "jute"], "spinach": ["spinach"],
    "cucumber": ["cucumber"], "lettuce": ["lettuce"], "arugula": ["arugula", "rocket"],
    "bell_pepper": ["bell pepper"], "green_pepper": ["pepper", "peppers", "capsicum"],
    "fresh_herbs": ["parsley", "coriander", "dill", "cilantro", "mint", "herbs"],
    "lemon": ["lemon"], "orange": ["orange"], "apple": ["apple"], "banana": ["banana"],
    "pineapple": ["pineapple"], "coconut": ["coconut"], "raisins": ["raisins"],
    "olives": ["olives"], "eggs": ["egg", "eggs"], "milk": ["milk"],
    "yogurt": ["yogurt", "yoghurt"], "cheese_generic": ["cheese"],
    "roumi_cheese": ["romi", "roumi"], "cream": ["cream"], "cream_cheese": ["cream cheese"],
    "butter": ["butter"], "ghee": ["ghee", "samna"], "industrial_ghee": ["shortening"],
    "oil": ["oil"], "flour": ["flour"], "bread": ["bread", "baladi bread"],
    "breadcrumbs": ["breadcrumbs"], "phyllo_pastry": ["phyllo", "filo", "goulash"],
    "sugar": ["sugar"], "honey_molasses": ["honey", "molasses"],
    "nuts": ["nuts", "almonds", "walnuts", "hazelnuts", "pistachios"],
    "peanuts": ["peanut", "peanuts", "peanut butter"],
    "salt": ["salt"], "spices_black_pepper": ["black pepper"],
    "cumin": ["cumin"], "coriander_dry": ["ground coriander"], "turmeric": ["turmeric"],
    "paprika_curry": ["paprika", "curry"], "cardamom": ["cardamom"],
    "saffron_special": ["saffron"], "bay_leaf": ["bay leaf", "bay leaves"],
    "spices_other": ["cinnamon", "cloves", "nutmeg", "spices", "baharat"],
    "ketchup": ["ketchup"], "mayonnaise": ["mayonnaise", "mayo"],
    "soy_sauce": ["soy sauce"], "vinegar": ["vinegar"], "broth": ["broth", "stock", "bouillon"],
    "water": ["water"], "yeast_baking": ["yeast", "baking powder", "baking soda"],
    "oats_bran_freekeh": ["oats", "freekeh", "bran"], "chocolate": ["chocolate", "cocoa"],
    "artichoke": ["artichoke"], "celery": ["celery"], "mallow_khobeiza": ["mallow"],
    "ice": ["ice cream", "icecream"], "bbq_sauce": ["bbq sauce", "barbecue sauce"],
    "garlic_powder": ["garlic powder"], "onion_powder": ["onion powder"],
    "kiri_cheese": ["kiri"], "koreish_cheese": ["qarish", "kareish"],
}

# Longest-first so "green beans" wins over "beans".
_EN_PHRASES = sorted(
    ((phrase, key) for key, phrases in _KEY_ENGLISH.items() for phrase in phrases),
    key=lambda item: -len(item[0]),
)

_store = None


def _user_key(ingredient: str, price_map):
    """Map one user ingredient (English or Arabic) to a price-map group key, or None."""
    text = str(ingredient)
    if re.search(r"[\u0600-\u06FF]", text):  # Arabic: the price map's own keywords
        row = price_match(text, price_map)
        return row["key"] if row else None
    low = " ".join(re.findall(r"[a-z]+", text.lower()))
    if not low:
        return None
    for phrase, key in _EN_PHRASES:
        if re.search(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", low):
            return key
    return None


def _recipe_families(raw: dict) -> set:
    """Allergen families for one recipe: fifi's own flags PLUS the word scans (both ways).

    fifi's flags (allergens.contains, gluten/lactose status) catch sources the word lists
    miss; the word scans catch anything fifi's analysis missed. Union, so either one excludes.
    """
    fams = set()
    for item in (raw.get("allergens") or {}).get("contains") or []:
        fams |= FIFI_ALLERGEN_MAP.get(str(item), set())
    if (raw.get("gluten") or {}).get("status") not in (None, "free"):
        fams |= {"wheat", "gluten"}
    if (raw.get("lactose") or {}).get("status") not in (None, "free"):
        fams |= {"milk", "dairy", "lactose"}
    rec = raw.get("recipe", raw)
    arabic = " ".join(
        [rec.get("title") or ""]
        + [m.get("name") or "" for m in rec.get("masterIngredients", [])]
        + [s.get("text") or "" for s in rec.get("uniqueInstructions", [])]
    )
    fams |= arabic_allergen_families(arabic)
    english = " ".join(
        [rec.get("titleEn") or ""]
        + [m.get("nameEn") or "" for m in rec.get("masterIngredients", [])]
        + [s.get("textEn") or "" for s in rec.get("uniqueInstructions", [])]
    )
    fams |= families_for_item(english)
    return fams


def _entry(recipe_id: str, raw: dict, bucket: str, price_map) -> dict:
    rec = raw.get("recipe", raw)
    est = estimate_recipe_cost(raw, price_map)
    ings = rec.get("masterIngredients", [])
    nutrition = {"calories": None, "protein_g": None, "carbs_g": None, "fat_g": None, "sugar_g": None}
    estimate = raw.get("estimate") or {}
    if estimate.get("kcal"):
        nutrition.update(
            calories=int(round(float(estimate["kcal"]))),
            protein_g=round(float(estimate.get("protein") or 0), 1),
            carbs_g=round(float(estimate.get("carbs") or 0), 1),
            fat_g=round(float(estimate.get("fat") or 0), 1),
            sugar_g=round(float(estimate.get("sugar") or 0), 1),
        )
    servings = parse_servings(rec.get("servings")) or (estimate.get("servings") or None)
    return {
        "recipe_id": ID_PREFIX + recipe_id,
        "name": rec.get("title") or recipe_id,
        "name_en": rec.get("titleEn") or "",
        "category": bucket,  # chicken / meat / soups / mahshi / breakfast / street
        "category_ar": rec.get("category") or "",
        "ingredients": [m.get("name") or "" for m in ings],
        "ingredients_en": [m.get("nameEn") or "" for m in ings],
        "ingredient_keys": [
            (price_match(m.get("name") or "", price_map) or {}).get("key") for m in ings
        ],
        "to_taste": [l["to_taste"] for l in est["lines"]],
        "instructions": [s.get("text") or "" for s in rec.get("uniqueInstructions", [])],
        "instructions_en": [
            (s.get("textEn") or s.get("text") or "") for s in rec.get("uniqueInstructions", [])
        ],
        "total_minutes": recipe_minutes(raw),
        "servings": int(servings) if servings else 4,
        "nutrition": nutrition,  # fifi's own estimate, approximate
        "cost": est,
        "families": _recipe_families(raw),
        "raw": raw,
    }


def load_store(data_dir=None, force: bool = False) -> dict:
    """recipe_id (egy:...) -> entry, plus an inverted index price-map key -> recipe ids."""
    global _store
    if _store is not None and not force and data_dir is None:
        return _store
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    curated = json.loads((data_dir / "curated.json").read_text(encoding="utf-8"))
    buckets = curated["buckets"]
    price_map = load_price_map()
    entries, by_key = {}, {}
    for bucket, ids in buckets.items():
        for rid in ids:
            path = data_dir / "recipes" / f"{rid}.json"
            raw = json.loads(path.read_text(encoding="utf-8"))
            e = _entry(rid, raw, bucket, price_map)
            entries[e["recipe_id"]] = e
            for key in set(e["ingredient_keys"]) - {None}:
                by_key.setdefault(key, set()).add(e["recipe_id"])
    store = {"entries": entries, "by_key": by_key, "buckets": buckets}
    if data_dir is None:
        _store = store
    return store


def is_egypt_id(recipe_id) -> bool:
    return str(recipe_id).startswith(ID_PREFIX)


def get_recipe(recipe_id: str) -> dict:
    return load_store()["entries"][str(recipe_id)]


def instructions(recipe_id: str, lang: str = "en") -> list:
    """Steps; Arabic for Arabic users, the dataset's English steps when they exist."""
    e = get_recipe(recipe_id)
    return e["instructions"] if lang == "ar" else e["instructions_en"]


def excluded(entry: dict, exclude) -> bool:
    """True if the recipe conflicts with one excluded item (allergen family or plain food)."""
    if not exclude:
        return False
    wanted, plain = set(), []
    for item in exclude:
        fams = families_for_item(item)
        if fams:
            wanted |= fams
        else:
            plain.append(str(item))
    if wanted & entry["families"]:
        return True
    for item in plain:
        key = _user_key(item, load_price_map())
        if key and key in entry["ingredient_keys"]:
            return True
        needle = norm(item)
        if len(needle) >= 3 and any(
            re.search(r"(?<![\u0600-\u06FF])" + re.escape(needle), norm(name))
            for name in entry["ingredients"]
        ):
            return True
    return False


def find(ingredients, max_minutes=None, exclude=None, top_k: int = 5, lang: str = "ar") -> list:
    """Match the curated Egyptian recipes against the user's ingredients.

    Ranking: uses the most of the user's ingredients, then fewest missing, then coverage,
    then the shorter recipe. Unknown-time recipes are excluded by a time limit.
    """
    store = load_store()
    price_map = load_price_map()
    user_keys = {}
    for ing in ingredients or []:
        key = _user_key(ing, price_map)
        if key:
            user_keys.setdefault(key, []).append(ing)
    results = []
    for e in store["entries"].values():
        if exclude and excluded(e, exclude):
            continue
        if max_minutes is not None:
            if e["total_minutes"] is None or e["total_minutes"] > max_minutes:
                continue
        used = 0
        for key in user_keys:
            if key in e["ingredient_keys"] or any(
                norm(u) and any(norm(u) in norm(n) for n in e["ingredients"])
                for u in user_keys[key]
            ):
                used += len(user_keys[key])
        have, missing = [], []
        names = e["ingredients"] if lang == "ar" else [
            en or ar for ar, en in zip(e["ingredients"], e["ingredients_en"])
        ]
        for name, key, to_taste in zip(names, e["ingredient_keys"], e["to_taste"]):
            if to_taste or key in PANTRY_KEYS:
                continue
            covered = key in user_keys or any(
                len(norm(u)) >= 3 and norm(u) in norm(name)
                for u in (ingredients or [])
            )
            (have if covered else missing).append(name)
        total = len(have) + len(missing)
        coverage = round(len(have) / total, 2) if total else 0.0
        results.append(
            {
                "recipe_id": e["recipe_id"],
                "name": e["name"],
                "name_en": e["name_en"],
                "category": e["category"],
                "total_minutes": e["total_minutes"],
                "servings": e["servings"],
                "nutrition": e["nutrition"],
                "used": used,
                "have": have,
                "missing": missing,
                "coverage": coverage,
                "distance": None,
                "egyptian": True,
                "bucket": e["category"],
                "cost_total_egp": e["cost"]["total_egp"],
                "price_source": e["cost"]["price_source"],
                "cost_text": format_cost(e["cost"], lang),
            }
        )
    min_used = min(2, len(ingredients or [])) or 0
    strong = [r for r in results if r["used"] >= min_used]
    if not strong and ingredients:
        strong = [r for r in results if r["used"] >= 1]  # small set: relax once
    strong.sort(
        key=lambda r: (-r["used"], len(r["missing"]), -r["coverage"], r["total_minutes"] or 9999)
    )
    return strong[:top_k]


def ensure_cost_lines(answer: str, recipes, lang: str = "ar") -> str:
    """Append any missing Egyptian cost line, so the cost always reaches the answer."""
    missing = []
    for r in recipes or []:
        text = r.get("cost_text")
        if r.get("egyptian") and text and text not in answer:
            missing.append(f"- {r['name']}: {text}")
    if not missing:
        return answer
    head = (
        "\n\n**التكلفة التقريبية بالجنيه:**\n" if lang == "ar"
        else "\n\n**Approximate cost in EGP:**\n"
    )
    return answer + head + "\n".join(missing)
