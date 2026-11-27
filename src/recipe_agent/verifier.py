"""Verifier: deterministic safety checks and a plain fallback answer (no model calls here)."""
import re

from recipe_agent import tools
from recipe_agent.allergy import arabic_allergen_families, families_for_item
from recipe_agent.retriever import get_recipe

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# Lines about allergies are skipped by the answer scan: the allergen names legitimately
# appear there (for example "saved allergies applied: milk").
_NOTE_MARKERS = ("allerg", "حساس")

ALLERGEN_AR = {
    "milk": "اللبن",
    "dairy": "الألبان",
    "lactose": "اللاكتوز",
    "egg": "البيض",
    "peanut": "الفول السوداني",
    "nut": "المكسرات",
    "wheat": "القمح",
    "gluten": "الجلوتين",
    "soy": "الصويا",
    "fish": "السمك",
    "shellfish": "المحار والجمبري",
    "sesame": "السمسم",
}


def format_items(items, language="en") -> str:
    """Readable list of avoided ingredients (allergen words shown in Arabic for Arabic users)."""
    items = [str(i) for i in items]
    if language == "ar":
        return "، ".join(ALLERGEN_AR.get(i.strip().lower(), i) for i in items)
    return ", ".join(items)


def verify_recipes(request, recipes):
    """Re-check the final recipes (and their substitutes) against the user's allergies.

    Returns (kept_recipes, issues).
    """
    exclude = list(request.exclude_ingredients)
    kept, issues = [], []
    for recipe in recipes:
        full = get_recipe(recipe["recipe_id"])
        ingredients = [str(i) for i in full["ingredients"]]
        steps = [str(s) for s in full["instructions"]]
        if tools._is_excluded(full["name"], ingredients, exclude, steps):
            issues.append(
                {"recipe_id": recipe["recipe_id"], "problem": "recipe conflicts with the allergies"}
            )
            continue
        recipe = dict(recipe)
        safe = []
        for sub in recipe.get("substitutions", []):
            if tools._substitute_blocked(sub["substitute"], exclude):
                issues.append(
                    {
                        "recipe_id": recipe["recipe_id"],
                        "problem": f"substitute removed: {sub['substitute']}",
                    }
                )
            else:
                safe.append(sub)
        recipe["substitutions"] = safe
        kept.append(recipe)
    return kept, issues


def check_answer(request, recipes, answer):
    """Check the final text. Returns (blocking, info).

    blocking: an allergen word (English) appears in the answer outside the allergy note.
    info: calories / minutes from the data that are not found in the answer (logged only).
    """
    blocking, info = [], []

    lines = [ln for ln in answer.splitlines() if not any(m in ln.lower() for m in _NOTE_MARKERS)]
    text = "\n".join(lines).lower()
    for recipe in recipes:  # allowed substitutes (e.g. almond milk) are not a leak
        for sub in recipe.get("substitutions", []):
            text = text.replace(str(sub["substitute"]).lower(), " ")
    if request.exclude_ingredients and tools._is_excluded(
        "", [], request.exclude_ingredients, [text]
    ):
        blocking.append("an allergen word appears in the answer")
    elif request.exclude_ingredients:
        wanted = set()
        for item in request.exclude_ingredients:
            wanted |= families_for_item(item)
        arabic_hits = arabic_allergen_families(text) & wanted
        if arabic_hits:
            blocking.append("an Arabic allergen word appears in the answer")

    digits = re.sub(r"(?<=\d),(?=\d)", "", answer.translate(_ARABIC_DIGITS))
    numbers = set(re.findall(r"\d+(?:\.\d+)?", digits))
    for recipe in recipes:
        for label, value in (
            ("calories", recipe["nutrition"]["calories"]),
            ("minutes", recipe["total_minutes"]),
        ):
            if value is None:
                continue  # unknown time: nothing to compare
            if str(value) not in numbers:
                info.append(f"{recipe['name']}: {label} {value} not found in the answer")
    return blocking, info


_TEXT = {
    "en": {
        "intro_llm_error": "The answer writer is unavailable right now, so here are the recipes in plain form (steps in English).",
        "intro_check_failed": "To be safe, here are the recipes in plain form (steps in English).",
        "time": "Total time",
        "min": "minutes",
        "servings": "Servings",
        "cal": "Calories per serving",
        "protein": "Protein per serving",
        "have": "You have",
        "missing": "Missing",
        "none": "nothing",
        "steps": "Steps",
        "subs": "Possible substitution (not tested)",
        "shopping": "Shopping list",
        "saved": "Saved allergies applied: {items}.",
        "saved_now": "Your allergies were saved for next time.",
        "forgotten": "Your saved allergies were cleared.",
        "allergy": "Allergy filtering checks ingredient names and recipe steps only; always check product labels.",
        "footer": "Times and nutrition values come from the recipe dataset and are approximate.",
        "unknown_time": "unknown",
        "memory_error": "Your saved allergies could not be read, so only the allergies in this message were applied. Please check them.",
        "empty": "No matching recipes were found. Try adding more ingredients or allowing more time.",
    },
    "ar": {
        "intro_llm_error": "خدمة كتابة الرد مش متاحة دلوقتي، فده عرض بسيط للوصفات (الخطوات بالإنجليزي).",
        "intro_check_failed": "للأمان، ده عرض بسيط للوصفات (الخطوات بالإنجليزي).",
        "time": "الوقت الإجمالي",
        "min": "دقيقة",
        "servings": "عدد الحصص",
        "cal": "السعرات لكل حصة",
        "protein": "البروتين لكل حصة",
        "have": "المتوفر عندك",
        "missing": "الناقص",
        "none": "مفيش",
        "steps": "الخطوات",
        "subs": "بديل ممكن (مش مجرّب)",
        "shopping": "قايمة التسوق",
        "saved": "الحساسيات المحفوظة المتطبقة: {items}.",
        "saved_now": "اتحفظت حساسيتك للمرات الجاية.",
        "forgotten": "اتمسحت الحساسيات المحفوظة.",
        "allergy": "فلتر الحساسية بيعتمد على أسماء المكونات وخطوات الوصفة فقط، راجعي ملصقات المنتجات دايماً.",
        "footer": "الأوقات والقيم الغذائية من قاعدة بيانات الوصفات وهي تقريبية.",
        "unknown_time": "غير معروف",
        "memory_error": "معرفتش أقرا الحساسيات المحفوظة، فاتطبقت بس الحساسيات المذكورة في الرسالة دي. راجعيها.",
        "empty": "مفيش وصفات مطابقة. جرّبي تضيفي مكونات أو تسمحي بوقت أكبر.",
    },
}


def fallback_answer(
    request, recipes, shopping, saved_exclusions=None, memory_event=None, reason="llm_error"
) -> str:
    """Plain answer built by code from verified data (no LLM). Steps stay in English."""
    t = _TEXT["ar" if request.language == "ar" else "en"]
    intro = t["intro_check_failed"] if reason == "check_failed" else t["intro_llm_error"]
    lines = [intro, ""]
    if memory_event == "saved":
        lines += [t["saved_now"], ""]
    elif memory_event == "forgotten":
        lines += [t["forgotten"], ""]
    elif memory_event == "memory_error":
        lines += [t["memory_error"], ""]
    if not recipes:
        lines += [t["empty"], ""]
    for r in recipes:
        n = r["nutrition"]
        lines.append(f"### {r['name']}")
        minutes = r["total_minutes"]
        time_text = f"{minutes} {t['min']}" if minutes else t["unknown_time"]
        lines.append(f"- {t['time']}: {time_text} | {t['servings']}: {r['servings']}")
        lines.append(f"- {t['cal']}: {n['calories']} | {t['protein']}: {n['protein_g']} g")
        lines.append(f"- {t['have']}: {', '.join(r['have']) or t['none']}")
        lines.append(f"- {t['missing']}: {', '.join(r['missing']) or t['none']}")
        for sub in r.get("substitutions", []):
            lines.append(f"- {t['subs']}: {sub['missing']} -> {sub['substitute']} ({sub['note']})")
        lines.append(f"**{t['steps']}:**")
        lines.extend(f"{i}. {step}" for i, step in enumerate(r.get("instructions", []), 1))
        lines.append("")
    if shopping:
        lines.append(f"**{t['shopping']}:**")
        lines.extend(f"- {item}" for item in shopping)
        lines.append("")
    if saved_exclusions:
        lines.append(t["saved"].format(items=format_items(saved_exclusions, request.language)))
    if request.exclude_ingredients:
        lines.append(t["allergy"])
    lines.append(t["footer"])
    return "\n".join(lines)
