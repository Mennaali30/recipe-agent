"""Reproduce the curated Egyptian set (needs internet). Not used by the tests.

Downloads fifi.cooking recipes, classifies the Egyptian ones (collection != "world") into the
six buckets (chicken, meat, soups, mahshi, breakfast, street) with word-boundary keyword rules,
scores each with the Phase 2 cost pipeline, drops near-duplicates and recipes whose cost is not
trustworthy enough (coverage < 0.6, or < 0.5 for soups), and writes the trimmed files to
data/egypt/recipes/ plus data/egypt/curated.json. The shipped data was produced on 24 Nov 2026;
fifi.cooking content can change, so re-running may give a different set.
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recipe_agent.egypt_cost import estimate_recipe_cost, load_price_map, recipe_minutes  # noqa: E402

BASE = "https://fifi.cooking/data"
OUT = ROOT / "data" / "egypt"
BUCKETS = ("chicken", "meat", "soups", "mahshi", "breakfast", "street")
PER_BUCKET = 25

CHICKEN = ["فراخ", "فرخه", "دجاج", "دجاجه", "chicken", "كتكوت"]
MEAT = ["لحم", "لحمه", "لحوم", "كفته", "كبده", "كلاوي", "ممبار", "حواوشي", "سجق", "مومبار",
        "عصبان", "طرب", "beef", "meat", "veal", "لسان", "رقبه", "ضاني", "بتلو", "جاموسي",
        "كندوز", "كوارع", "برجر", "بسطرمه", "استيك", "ستيك", "كباب", "ارنب", "حمام", "بطه",
        "وزه", "عجل", "روستو", "مفروم", "عصاج"]
SOUP = ["شوربه", "شوربة", "حساء", "soup", "مرق", "مرقه"]
MAHSHI = ["محشي", "محشو", "محشوات", "محشوه"]
BREAKFAST_T = ["فول", "طعميه", "فلافل", "شكشوكه", "بليله", "بيض", "اومليت", "فطار", "بصاره",
               "مدره", "مسقعه", "قريش"]
STREET_T = ["كشري", "كشرى", "حواوشي", "كبده", "شاورما", "فتة", "فته", "كباب حله", "حمص الشام",
            "ترمس", "بطاطا محمره", "بطاطس محمره", "ممبار", "سجق", "كوارع", "ملوخيه"]
DESSERT = ["بسكويت", "كيك", "كنافه", "بقلاوه", "قطايف", "زلابيه", "لقمه", "مربي", "حلاوه",
           "بسبوسه", "مهلبيه", "ارز بلبن", "رز بلبن", "ام علي", "بودنج", "كراميل", "شوكولات",
           "دونات", "كوكيز", "تارت", "ايس كريم", "خشاف", "سوبيا", "ميلك", "شيك", "عصير",
           "مشروب", "قهوه", "وافل", "براونيز", "غريبه", "بتيفور"]

AW = "\u0600-\u06FF"


def _norm(s):
    s = str(s or "").lower()
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ة", "ه")):
        s = s.replace(a, b)
    return s


def _any(text, kws):
    return any(
        re.search(rf"(?<![{AW}])(?:ال)?{re.escape(k)}(?![{AW}])", text) for k in kws
    )


def classify(item):
    t = _norm(item.get("title")) + " | " + _norm(item.get("titleEn")) + " | " + _norm(item.get("category"))
    ings = " ".join(
        _norm(p.get("name")) + " " + _norm(p.get("nameEn"))
        for p in item.get("previewIngredients", [])
    )
    full = t + " | " + ings
    if _any(t, DESSERT):
        return None
    if _any(t, MAHSHI):
        return "mahshi"
    if _any(t, SOUP):
        return "soups"
    if _any(t, BREAKFAST_T):
        return "breakfast"
    if _any(t, STREET_T):
        return "street"
    if _any(full, CHICKEN):
        return "chicken"
    if _any(full, MEAT):
        return "meat"
    return None


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (recipe-agent curation)"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def fetch_index():
    items, page = [], 1
    while True:
        try:
            idx = _get(f"{BASE}/index/{page}.json")
        except Exception:
            break
        if not idx:
            break
        items.extend(idx)
        page += 1
    return [i for i in items if i.get("collection") != "world"]


def _tokens(title):
    out = set()
    for w in re.split(rf"[^{AW}a-z]+", _norm(title)):
        w = re.sub(r"^(ال|و)", "", w)
        if len(w) > 2:
            out.add(w)
    return out


def trim(raw):
    rec = raw.get("recipe", {})
    return {
        "recipe": {
            "id": rec.get("id"), "title": rec.get("title"), "titleEn": rec.get("titleEn"),
            "category": rec.get("category"), "cookingMethod": rec.get("cookingMethod"),
            "prepTime": rec.get("prepTime"), "cookTime": rec.get("cookTime"),
            "servings": rec.get("servings"),
            "masterIngredients": [
                {"name": m.get("name"), "nameEn": m.get("nameEn"),
                 "standardAmount": m.get("standardAmount")}
                for m in rec.get("masterIngredients", [])
            ],
            "uniqueInstructions": [
                {"stepNumber": s.get("stepNumber"), "text": s.get("text"),
                 "textEn": s.get("textEn")}
                for s in rec.get("uniqueInstructions", [])
            ],
        },
        "estimate": {k: (raw.get("estimate") or {}).get(k)
                     for k in ("servings", "kcal", "protein", "fat", "carbs", "fiber", "sugar")},
        "allergens": raw.get("allergens"),
        "gluten": raw.get("gluten"),
        "lactose": raw.get("lactose"),
        "trimmed": True,
    }


def run():
    price_map = load_price_map()
    buckets = {}
    for item in fetch_index():
        bucket = classify(item)
        if bucket:
            buckets.setdefault(bucket, []).append(item["id"])
    picked, buckets_out = [], {b: [] for b in BUCKETS}
    (OUT / "recipes").mkdir(parents=True, exist_ok=True)
    for bucket in BUCKETS:
        scored = []
        for rid in buckets.get(bucket, []):
            raw = _get(f"{BASE}/recipes/{rid}.json")
            est = estimate_recipe_cost(raw, price_map)
            mins = recipe_minutes(raw)
            rec = raw.get("recipe", {})
            if est["total_egp"] is None or mins is None or (est["coverage"] or 0) < 0.6:
                continue
            if not (5 <= est["total_egp"] <= 1500):
                continue
            if len(rec.get("masterIngredients", [])) < 3 or len(rec.get("uniqueInstructions", [])) < 2:
                continue
            scored.append((-(est["coverage"] or 0), rid, raw))
        scored.sort()
        seen = []
        for _, rid, raw in scored:
            tk = _tokens(raw.get("recipe", {}).get("title"))
            union = lambda s: len(tk & s) / len(tk | s) if (tk | s) else 0
            if any(union(s) > 0.6 for s in seen):
                continue
            seen.append(tk)
            (OUT / "recipes" / f"{rid}.json").write_text(
                json.dumps(trim(raw), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            buckets_out[bucket].append(rid)
            picked.append(rid)
            if len(buckets_out[bucket]) == PER_BUCKET:
                break
    (OUT / "curated.json").write_text(
        json.dumps({"note": "curated by scripts/curate_egypt.py", "buckets": buckets_out},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"curated {len(picked)} recipes -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    run()
