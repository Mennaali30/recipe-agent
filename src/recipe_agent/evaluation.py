"""Evaluation of the recipe agent: retrieval quality, allergy-filter audit, intent accuracy.

Run from a notebook:
    from recipe_agent import evaluation as ev
    df_ret = ev.eval_retrieval()
    df_all, leaks = ev.eval_allergy_audit()
    df_int = ev.eval_intent()
"""
import re
import time

import pandas as pd

from recipe_agent import tools
from recipe_agent.retriever import load_recipes

# --------------------------------------------------------------------------- #
# 1. Retrieval cases (structured input, no LLM involved)
# --------------------------------------------------------------------------- #
RETRIEVAL_CASES = [
    {"ingredients": ["chicken", "rice", "garlic"], "main": "chicken", "max_minutes": 30},
    {"ingredients": ["chicken", "rice", "garlic", "lemon"], "main": "chicken", "max_minutes": 40},
    {"ingredients": ["beef", "onion", "potato"], "main": "beef", "max_minutes": None},
    {"ingredients": ["egg", "tomato", "cheese"], "main": "egg", "max_minutes": 20},
    {"ingredients": ["potato", "egg"], "main": "potato", "max_minutes": None},
    {"ingredients": ["pasta", "tomato", "garlic", "basil"], "main": "pasta", "max_minutes": 30},
    {"ingredients": ["lentils", "onion", "carrot"], "main": "lentils", "max_minutes": None},
    {"ingredients": ["tuna", "rice"], "main": "tuna", "max_minutes": 25},
    {"ingredients": ["spinach", "egg", "onion"], "main": "spinach", "max_minutes": 30},
    {"ingredients": ["chicken", "potato", "carrot", "onion"], "main": "chicken", "max_minutes": 60},
    {"ingredients": ["shrimp", "garlic", "rice"], "main": "shrimp", "max_minutes": 30},
    {"ingredients": ["banana", "oats", "honey"], "main": "banana", "max_minutes": None},
    {"ingredients": ["zucchini", "tomato", "onion"], "main": "zucchini", "max_minutes": 45},
    {"ingredients": ["ground beef", "rice", "tomato"], "main": "beef", "max_minutes": 45},
    {"ingredients": ["chickpeas", "tomato", "garlic"], "main": "chickpeas", "max_minutes": None},
    {"ingredients": ["yogurt", "cucumber", "garlic"], "main": "yogurt", "max_minutes": 15},
    {"ingredients": ["salmon", "lemon"], "main": "salmon", "max_minutes": 30},
    {"ingredients": ["bread", "egg", "cheese"], "main": "bread", "max_minutes": 20},
    {"ingredients": ["cabbage", "carrot", "onion"], "main": "cabbage", "max_minutes": None},
    {"ingredients": ["turkey", "potato"], "main": "turkey", "max_minutes": None},
]


def _uses(main: str, recipe_ingredients) -> bool:
    m = tools.normalize(main)
    return any(tools._matches(m, tools.normalize(i)) for i in recipe_ingredients)


def eval_retrieval(cases=None, top_k: int = 3, candidates: int = 40) -> pd.DataFrame:
    """Run find_recipes on each case (same settings as the first search of the graph)."""
    rows = []
    for case in cases or RETRIEVAL_CASES:
        ings = case["ingredients"]
        limit = case["max_minutes"]
        res = tools.find_recipes(ings, max_minutes=limit, top_k=top_k, candidates=candidates)
        n = len(res)
        main_flags = [_uses(case["main"], r["have"]) for r in res]
        rows.append(
            {
                "ingredients": ", ".join(ings),
                "max_minutes": limit,
                "n_results": n,
                "needs_widening": n < 2,
                "main_top1": bool(main_flags and main_flags[0]),
                "main_share": sum(main_flags) / n if n else 0.0,
                "avg_used_ratio": sum(r["used"] for r in res) / (n * len(ings)) if n else 0.0,
                "any_full_use": any(r["used"] == len(ings) for r in res),
                "time_ok": all(r["total_minutes"] <= limit for r in res) if limit else True,
                "top1": res[0]["name"] if n else "",
            }
        )
    return pd.DataFrame(rows)


def summarize_retrieval(df: pd.DataFrame) -> dict:
    pct = lambda s: round(100 * float(s.mean()), 1)
    return {
        "cases": len(df),
        "pct_with_2plus_results": pct(df["n_results"] >= 2),
        "pct_needing_widening": pct(df["needs_widening"]),
        "pct_main_ingredient_in_top1": pct(df["main_top1"]),
        "pct_main_ingredient_in_top3_avg": pct(df["main_share"]),
        "pct_of_user_ingredients_used_avg": pct(df["avg_used_ratio"]),
        "pct_cases_with_a_full_use_recipe": pct(df["any_full_use"]),
        "pct_time_limit_respected": pct(df["time_ok"]),
    }


# --------------------------------------------------------------------------- #
# 2. Allergy-filter audit (no LLM): compare the filter with a wide word list
#    searched with a different method (regex on the raw text).
#    NOTE: the filter and this list now share most words, so a low leak rate means
#    "the filter implements our lists correctly", NOT "every allergen is caught".
#    Hidden sources absent from both lists (e.g. chocolate) are not measured.
# --------------------------------------------------------------------------- #
AUDIT_WORDS = {
    "milk": [
        "milk", "cheese", "butter", "cream", "yogurt", "yoghurt", "whey", "ghee",
        "buttermilk", "custard", "mozzarella", "parmesan", "cheddar", "ricotta",
        "feta", "gruyere", "mascarpone", "halloumi", "paneer", "half and half",
        "half-and-half", "ice cream", "gelato", "cream cheese", "sour cream",
        "provolone", "gouda", "brie", "cottage cheese", "casein", "lactose",
        "camembert", "gorgonzola", "romano", "pecorino", "creamer",
    ],
    "egg": [
        "egg", "mayonnaise", "mayo", "meringue", "eggnog", "aioli", "hollandaise",
        "custard", "egg white", "egg yolk",
    ],
    "peanut": ["peanut", "groundnut", "satay"],
    "nut": [
        "nut", "almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut",
        "macadamia", "peanut", "pine nut", "brazil nut",
    ],
    "wheat": [
        "wheat", "flour", "bread", "breadcrumb", "bread crumb", "pasta", "noodle",
        "couscous", "semolina", "barley", "rye", "bulgur", "cracker", "tortilla",
        "spaghetti", "macaroni", "pita", "pastry", "dough", "crust", "biscuit",
        "pancake", "waffle", "muffin", "bisquick",
    ],
    "soy": ["soy", "soya", "tofu", "edamame", "miso", "tempeh", "tamari", "soybean"],
    "fish": [
        "fish", "salmon", "tuna", "cod", "anchovy", "anchovies", "tilapia", "sardine",
        "trout", "haddock", "halibut", "mackerel", "snapper", "catfish", "herring",
        "worcestershire", "caesar",
    ],
    "shellfish": [
        "shellfish", "shrimp", "prawn", "crab", "lobster", "clam", "mussel", "oyster",
        "scallop", "crawfish", "crayfish", "seafood", "surimi",
    ],
    "sesame": ["sesame", "tahini", "halva", "halvah", "hummus"],
}

# "egg-free", "eggless", ... must not count as a mention of the allergen
_FREE_OF = re.compile(
    r"\b(?:egg|dairy|milk|nut|gluten|soy|wheat|fish|peanut|sesame|lactose)[\s-]*free\b|\beggless\b"
)


def _audit_pattern(words):
    alts = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"\b(?:{alts})(?:e?s)?\b")


def eval_allergy_audit(sample_size: int = 5000, seed: int = 42, show_examples: int = 6):
    """For every allergen: how many recipes does the audit flag, and how many of them
    does the filter miss ('leaked')? Also counts recipes the filter drops although the
    audit finds nothing ('over_excluded', harmless but reduces results)."""
    df = load_recipes()
    sample = df.sample(n=min(sample_size, len(df)), random_state=seed).to_dict("records")
    rows, examples = [], {}
    for allergen, words in AUDIT_WORDS.items():
        pattern = _audit_pattern(words)
        audit_hits = filter_hits = leaked = over = 0
        leak_examples = []
        for rec in sample:
            ings = [str(i) for i in rec["ingredients"]]
            steps = [str(s) for s in rec["instructions"]]
            text = " ".join([rec["name"], *ings, *steps]).lower()
            text = _FREE_OF.sub(" ", text)
            matches = [m.group(0) for m in pattern.finditer(text)]
            audit = bool(matches)
            flagged = tools._is_excluded(rec["name"], ings, [allergen], steps)
            audit_hits += audit
            filter_hits += flagged
            if audit and not flagged:
                leaked += 1
                if len(leak_examples) < show_examples:
                    leak_examples.append((rec["name"], sorted(set(matches))[:4]))
            elif flagged and not audit:
                over += 1
        rows.append(
            {
                "allergen": allergen,
                "recipes_checked": len(sample),
                "audit_flagged": audit_hits,
                "filter_flagged": filter_hits,
                "leaked": leaked,
                "leak_rate_%": round(100 * leaked / audit_hits, 1) if audit_hits else 0.0,
                "over_excluded": over,
            }
        )
        examples[allergen] = leak_examples
    return pd.DataFrame(rows), examples


# --------------------------------------------------------------------------- #
# 3. Intent agent accuracy (uses the LLM, one call per case)
# --------------------------------------------------------------------------- #
INTENT_CASES = [
    {"text": "عندي فراخ وأرز وثوم وعايزة أكلة تخلص في نص ساعة",
     "ingredients": ["chicken", "rice", "garlic"], "exclude": [], "max_minutes": 30, "language": "ar"},
    {"text": "عندي بطاطس وبيض وعندي حساسية من اللبن",
     "ingredients": ["potato", "egg"], "exclude": ["milk"], "max_minutes": None, "language": "ar"},
    {"text": "I have eggs, tomatoes and cheese. No onions please.",
     "ingredients": ["egg", "tomato", "cheese"], "exclude": ["onion"], "max_minutes": None, "language": "en"},
    {"text": "I have chicken, rice and garlic. Something quick, under 30 minutes.",
     "ingredients": ["chicken", "rice", "garlic"], "exclude": [], "max_minutes": 30, "language": "en"},
    {"text": "عندي لحمة مفرومة وطماطم وبصل ومعنديش وقت، ربع ساعة بس",
     "ingredients": ["beef", "tomato", "onion"], "exclude": [], "max_minutes": 15, "language": "ar"},
    {"text": "عندي عدس وجزر وبصل",
     "ingredients": ["lentils", "carrot", "onion"], "exclude": [], "max_minutes": None, "language": "ar"},
    {"text": "عندي سمك وليمون",
     "ingredients": ["fish", "lemon"], "exclude": [], "max_minutes": None, "language": "ar"},
    {"text": "عندي فراخ وأرز وعندي حساسية من المكسرات",
     "ingredients": ["chicken", "rice"], "exclude": ["nut"], "max_minutes": None, "language": "ar"},
    {"text": "I have rice, tomatoes and basil. I'm allergic to gluten.",
     "ingredients": ["rice", "tomato", "basil"], "exclude": ["gluten"], "max_minutes": None, "language": "en"},
    {"text": "I have potatoes and eggs. I have about an hour.",
     "ingredients": ["potato", "egg"], "exclude": [], "max_minutes": 60, "language": "en"},
    {"text": "I have salmon and lemon, 20 minutes max, I'm allergic to nuts.",
     "ingredients": ["salmon", "lemon"], "exclude": ["nut"], "max_minutes": 20, "language": "en"},
    {"text": "عندي زبادي وخيار وثوم",
     "ingredients": ["yogurt", "cucumber", "garlic"], "exclude": [], "max_minutes": None, "language": "ar"},
    {"text": "عندي كوسة وطماطم وبصل وعايزة حاجة تتعمل في ساعة",
     "ingredients": ["zucchini", "tomato", "onion"], "exclude": [], "max_minutes": 60, "language": "ar"},
    {"text": "عندي تونة وأرز وبيض، ومعنديش حساسية من حاجة",
     "ingredients": ["tuna", "rice", "egg"], "exclude": [], "max_minutes": None, "language": "ar"},
    {"text": "عندي كبدة وبصل",
     "ingredients": ["liver", "onion"], "exclude": [], "max_minutes": None, "language": "ar"},
    {"text": "I have shrimp and garlic and I want something in 2 hours",
     "ingredients": ["shrimp", "garlic"], "exclude": [], "max_minutes": 120, "language": "en"},
    {"text": "إزيك عامل إيه",
     "ingredients": [], "exclude": [], "max_minutes": None, "language": "ar"},
    {"text": "Hello, what can you do?",
     "ingredients": [], "exclude": [], "max_minutes": None, "language": "en"},
]


def _related(a: frozenset, b: frozenset) -> bool:
    return bool(a) and bool(b) and (a <= b or b <= a)


def _score_lists(expected, predicted):
    exp = [tools.normalize(x) for x in expected]
    pred = [tools.normalize(x) for x in predicted]
    recall = 1.0 if not exp else sum(any(_related(e, p) for p in pred) for e in exp) / len(exp)
    precision = 1.0 if not pred else sum(any(_related(p, e) for e in exp) for p in pred) / len(pred)
    return recall, precision


def _parse_with_retry(text: str, tries: int = 1, base_wait: float = 20.0):
    """Call the intent agent; on failure (rate limit, 503...) wait and retry."""
    from recipe_agent.agents import parse_request  # needs the LLM env vars

    last = None
    for attempt in range(tries):
        try:
            return parse_request(text), None
        except Exception as exc:
            last = exc
            if attempt < tries - 1:
                time.sleep(base_wait * (attempt + 1))
    return None, f"{type(last).__name__}: {str(last)[:160]}"


def eval_intent(cases=None, pause: float = 13.0, tries: int = 1, base_wait: float = 20.0) -> pd.DataFrame:
    """One LLM call per case. pause=13s keeps us under 5 requests per minute (free tier)."""
    rows = []
    for case in cases or INTENT_CASES:
        req, error = _parse_with_retry(case["text"], tries=tries, base_wait=base_wait)
        if req is None:
            rows.append({"text": case["text"], "error": error, "all_ok": False})
            time.sleep(pause)
            continue
        ing_r, ing_p = _score_lists(case["ingredients"], req.ingredients)
        exc_r, exc_p = _score_lists(case["exclude"], req.exclude_ingredients)
        time_ok = req.max_minutes == case["max_minutes"]
        lang_ok = req.language == case["language"]
        rows.append(
            {
                "text": case["text"],
                "error": None,
                "pred_ingredients": req.ingredients,
                "pred_exclude": req.exclude_ingredients,
                "pred_minutes": req.max_minutes,
                "pred_language": req.language,
                "ing_recall": ing_r,
                "ing_precision": ing_p,
                "exc_recall": exc_r,
                "exc_precision": exc_p,
                "time_ok": time_ok,
                "lang_ok": lang_ok,
                "all_ok": ing_r == 1 and ing_p == 1 and exc_r == 1 and exc_p == 1 and time_ok and lang_ok,
            }
        )
        time.sleep(pause)
    return pd.DataFrame(rows)


def summarize_intent(df: pd.DataFrame) -> dict:
    valid = df[df["error"].isna()]
    out = {
        "cases": len(df),
        "valid_cases": len(valid),
        "errors": int(df["error"].notna().sum()),
    }
    if len(valid) == 0:
        out["warning"] = "no valid case: rerun before using any number"
        return out
    for col in ["ing_recall", "ing_precision", "exc_recall", "exc_precision"]:
        out[col + "_%"] = round(100 * float(valid[col].mean()), 1)
    for col in ["time_ok", "lang_ok", "all_ok"]:
        out[col + "_%"] = round(100 * float(valid[col].astype(float).mean()), 1)
    out["all_ok_errors_count_as_fail_%"] = round(100 * float(df["all_ok"].astype(float).mean()), 1)
    if out["errors"]:
        out["warning"] = "some calls failed: the % above cover the valid cases only"
    return out
