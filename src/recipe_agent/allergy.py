"""Deterministic allergy handling that does not depend on the LLM.

- ``extract_allergies(text)``: allergies / avoided foods named in the RAW user message
  (Arabic, Egyptian dialect, English). Code only, so a missed LLM extraction cannot drop one.
- ``arabic_allergen_families(text)``: allergen families whose Arabic words appear in a text
  (used by the verifier to scan Arabic answers).
- ``EXTRA_ENGLISH_TERMS``: extra English words per allergen family, merged into the filter
  lists in ``tools.py``.

Limits: word lists only. Hidden sources (for example chocolate that may contain milk) are not
detected, and an allergy written in a way these patterns do not cover is still missed.
"""
import re

# --------------------------------------------------------------------------- #
# English: extra words per family (merged into tools.ALLERGEN_GROUPS)
# --------------------------------------------------------------------------- #
EXTRA_ENGLISH_TERMS = {
    "dairy": ["curd", "labneh", "labna", "quark", "skyr", "gelato", "lactose", "milkshake"],
    "egg": ["yolk", "albumen", "omelet", "omelette", "frittata", "quiche", "souffle"],
    "peanut": ["groundnut", "arachis"],
    "nut": [
        "chestnut", "marzipan", "praline", "nutella", "frangipane", "pinenut", "brazil",
        "nougat",
    ],
    "wheat": [
        "farina", "spelt", "kamut", "seitan", "freekeh", "orzo", "lasagna", "lasagne",
        "ravioli", "gnocchi", "udon", "ramen", "baguette", "ciabatta", "focaccia", "naan",
        "chapati", "phyllo", "filo", "panko", "fettuccine", "linguine", "penne", "rigatoni",
        "vermicelli", "tortellini", "kunafa", "semolina", "malt", "croissant", "cake",
        "cookie", "brownie", "pizza",
    ],
    "soy": ["natto", "shoyu", "teriyaki", "soybean"],
    "fish": [
        "pollock", "swordfish", "flounder", "sole", "perch", "bonito", "carp", "eel",
        "caviar", "roe", "sashimi", "bass", "pike",
    ],
    "shellfish": [
        "squid", "calamari", "octopus", "crawfish", "crayfish", "langoustine", "cockle",
        "abalone", "krill", "crustacean", "mollusk", "mollusc",
    ],
    "sesame": ["gomasio", "benne"],
}

# --------------------------------------------------------------------------- #
# English words -> allergen key, for reading the raw message
# --------------------------------------------------------------------------- #
_EN_KEYS = {
    "milk": "milk", "dairy": "dairy", "lactose": "lactose", "cheese": "milk",
    "butter": "milk", "cream": "milk", "yogurt": "milk", "yoghurt": "milk",
    "egg": "egg", "eggs": "egg",
    "peanut": "peanut", "peanuts": "peanut", "groundnut": "peanut", "groundnuts": "peanut",
    "nut": "nut", "nuts": "nut", "almond": "nut", "almonds": "nut", "walnut": "nut",
    "walnuts": "nut", "cashew": "nut", "cashews": "nut", "pecan": "nut", "pecans": "nut",
    "pistachio": "nut", "pistachios": "nut", "hazelnut": "nut", "hazelnuts": "nut",
    "wheat": "wheat", "gluten": "gluten", "flour": "wheat", "bread": "wheat",
    "soy": "soy", "soya": "soy", "soybean": "soy", "soybeans": "soy", "tofu": "soy",
    "fish": "fish", "salmon": "fish", "tuna": "fish", "anchovy": "fish", "anchovies": "fish",
    "shellfish": "shellfish", "shrimp": "shellfish", "prawn": "shellfish",
    "prawns": "shellfish", "crab": "shellfish", "lobster": "shellfish",
    "seafood": "shellfish", "sesame": "sesame", "tahini": "sesame",
}

# --------------------------------------------------------------------------- #
# Arabic lexicon (written without diacritics; matched after normalization and after
# removing the definite article "ال" and common one-letter prefixes)
# --------------------------------------------------------------------------- #
AR_LEXICON = {
    "milk": [
        "لبن", "حليب", "جبن", "جبنه", "زبده", "قشطه", "كريمه", "كريم", "زبادي", "زبدي",
        "روب", "سمن", "قريشه", "رومي", "موزاريلا", "شيدر", "بارميزان", "ريكوتا", "فيتا",
        "مثلثات", "قشده", "لاكتوز", "مصل", "كازين", "ايس كريم", "كيك", "مهلبيه", "ارز بلبن",
    ],
    "dairy": ["البان", "ألبان", "منتجات الالبان", "منتجات البان"],
    "lactose": ["لاكتوز", "لاكتوس"],
    "egg": ["بيض", "بيضه", "بيضات", "مايونيز", "ميونيز", "عجه", "اومليت", "اوملت", "صفار", "بياض"],
    "peanut": ["فول سوداني", "سوداني", "فستق عبيد", "زبده فول سوداني", "فولسوداني"],
    "nut": [
        "مكسرات", "لوز", "جوز", "بندق", "كاجو", "فستق", "عين جمل", "بيكان", "كاشو",
        "مكسر", "مارزيبان", "نوتيلا", "كستناء",
    ],
    "wheat": [
        "قمح", "دقيق", "عيش", "خبز", "مكرونه", "شعير", "فريك", "برغل", "كسكسي", "بقسماط",
        "شعريه", "سميد", "لازانيا", "فطير", "جلاش", "كنافه", "بسكوت", "معجنات", "عجين",
        "بيتزا", "رقاق", "بقصماط", "باستا", "نودلز", "اسباجتي",
    ],
    "gluten": ["جلوتين", "غلوتين", "جلوتن", "غلوتن", "قمح", "دقيق", "شعير", "سيلياك"],
    "soy": ["صويا", "فول صويا", "توفو", "ميزو", "ادامامي", "صوص صويا"],
    "fish": ["سمك", "سمكه", "اسماك", "تونه", "سلمون", "سردين", "انشوجه", "بلطي", "بوري", "ماكريل", "فيليه", "رنجه", "فسيخ", "تونا", "قاروص", "دينس", "موسي", "ماكرل"],
    "shellfish": [
        "جمبري", "جمبرى", "كابوريا", "استاكوزا", "محار", "سي فود", "روبيان", "قريدس",
        "بلح بحر", "كلاماري", "حبار", "اخطبوط", "جندوفلي", "كالاماري", "مأكولات بحريه",
        "ماكولات بحريه", "فواكه البحر", "سمك قشري",
    ],
    "sesame": ["سمسم", "طحينه", "طحينيه", "حلاوه", "حلوه طحينيه", "حمص بطحينه", "سيسم"],
}
# Words that look like an allergen but are not (fava beans, coconut...).
_AR_NOT_ALLERGEN_PAIRS = {("جوز", "هند")}

# --------------------------------------------------------------------------- #
# Normalization helpers
# --------------------------------------------------------------------------- #
_TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_PUNCT = re.compile(r"[.,;:!?\n\r()\[\]\"'،؛؟\-/\\+*|]|(?<=\s)&(?=\s)")


def norm_ar(text) -> str:
    """Lowercase, drop diacritics, unify alef / yaa / taa marbuta / hamza forms."""
    t = _TASHKEEL.sub("", str(text).lower().translate(_AR_DIGITS))
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ٱ", "ا"), ("ى", "ي"), ("ة", "ه"),
                 ("ؤ", "و"), ("ئ", "ي")):
        t = t.replace(a, b)
    return t


def _variants(token: str) -> list:
    """The token plus forms with a definite article / one-letter prefix / plural suffix removed."""
    out = [token]
    for prefix in ("وبال", "وال", "بال", "كال", "فال", "لل", "ال", "ول", "و", "ب", "ل", "ف"):
        if token.startswith(prefix) and len(token) - len(prefix) >= 2:
            rest = token[len(prefix):]
            out.append(rest)
            if prefix == "لل":
                out.append("ل" + rest)
            if rest.startswith("ال") and len(rest) > 3:
                out.append(rest[2:])
    extra = []
    for v in out:
        for suffix in ("ات", "ين", "ون", "ها", "ه", "ي", "ا"):
            if v.endswith(suffix) and len(v) - len(suffix) >= 3:
                extra.append(v[: -len(suffix)])
    return list(dict.fromkeys(out + extra))


_AR_INDEX = {}
for _key, _terms in AR_LEXICON.items():
    for _term in _terms:
        _AR_INDEX.setdefault(norm_ar(_term), set()).add(_key)

def _ar_tokens(text) -> list:
    return re.findall(r"[\u0600-\u06FF]+", norm_ar(text))


def arabic_allergen_families(text) -> set:
    """Allergen keys (milk, egg, ...) whose Arabic words appear in the text."""
    tokens = _ar_tokens(text)
    found, i = set(), 0
    while i < len(tokens):
        forms = _variants(tokens[i])
        nxt = _variants(tokens[i + 1]) if i + 1 < len(tokens) else []
        # coconut ("جوز الهند") is not a tree nut
        if any((a, b) in _AR_NOT_ALLERGEN_PAIRS for a in forms for b in nxt):
            i += 2
            continue
        # two-word terms first
        hit = False
        for a in forms:
            for b in nxt:
                keys = _AR_INDEX.get(f"{a} {b}")
                if keys:
                    found |= keys
                    hit = True
        if hit:
            i += 2
            continue
        for a in forms:
            found |= _AR_INDEX.get(a, set())
        i += 1
    return found


def families_for_item(item) -> set:
    """Allergen keys that one excluded item stands for (English or Arabic word)."""
    text = str(item).lower().strip()
    keys = set()
    for word in re.findall(r"[a-z]+", text):
        if word in _EN_KEYS:
            keys.add(_EN_KEYS[word])
    keys |= arabic_allergen_families(text)
    return keys


# --------------------------------------------------------------------------- #
# Reading the raw message
# --------------------------------------------------------------------------- #
# Strong triggers: the user says they cannot eat something.
_AR_STRONG = {
    "حساسيه", "حساسيت", "حساسه", "حساس", "تحسس", "بتحسس", "ممنوع", "ممنوعه",
    "مبكلش", "مباكلش", "ماباكلش", "مبااكلش", "متحمل", "بيعمللي", "بتعمللي", "مينفعش",
    "منفعش", "ينفعش", "اتجنب", "تجنب", "تجنبي", "ابعد", "ابعدي", "بعيد", "ماكلش",
    "ماكلهاش", "باكلش", "اكلش", "ماينفعنيش", "ينفعنيش", "ينفعنيش", "نباتي",
}
_AR_WEAK = {"بدون", "من", "غير", "ماعدا", "بلا"}  # "من غير X" / "بدون X"
_AR_NEG_BEFORE = {"مش", "مفيش", "ماعنديش", "معنديش", "عنديش", "لا", "ما", "ليس", "مفيهاش"}
_AR_STOP = {
    "بس", "لكن", "عايز", "عايزه", "عاوز", "عاوزه", "عايزين", "اعمل", "اعملي", "ابغي",
    "طيب", "وعندي", "عندي", "معايا", "وعايز", "وعايزه", "وعاوز", "وعاوزه", "هاتلي",
    "هاتي", "ادني", "قولي", "اقترح", "اقترحي", "ورشحلي", "رشحلي", "ورشحي", "عاملي",
    "اعملنا", "وده", "ومعايا", "ومعاي", "وكمان", "دلوقتي",
}
_EN_STRONG = re.compile(
    r"\b(?:allergic to|allergy to|allergies to|allergy|allergies|allergic|intolerant to|"
    r"intolerance to|intolerance|can'?t eat|cannot eat|can not eat|don'?t eat|do not eat|"
    r"doesn'?t eat|avoid|avoiding|vegan|dairy[- ]free|gluten[- ]free|nut[- ]free)\b"
)
_EN_WEAK = re.compile(r"\b(?:without|no|free of|free from|except|excluding|exclude)\b")
_EN_NEG = re.compile(
    r"(?:\bno\s+(?:known\s+)?(?:food\s+)?(?:allergy|allergies)\b|\bnot\s+allergic\b|"
    r"\bdon'?t\s+have\s+(?:any\s+)?(?:food\s+)?allerg|\bdo\s+not\s+have\s+(?:any\s+)?allerg|"
    r"\bwithout\s+(?:any\s+)?allerg)"
)
_EN_STOP = re.compile(
    r"\b(?:but|and i (?:have|want|need)|i (?:have|want|need)|i'?ve got|want|need|please|"
    r"make|give|suggest|show|using|with)\b"
)
_EN_FILLER = {
    "to", "the", "any", "all", "of", "a", "an", "from", "in", "my", "food", "foods", "and",
    "or", "also", "any", "kind", "type", "types", "products", "product", "anything", "made",
    "with", "that", "contains", "contain", "containing", "like", "is", "am", "i", "me", "no",
    "not", "dont", "don't", "do", "eat", "ingredients", "ingredient", "please", "because",
    "since", "so", "it", "its", "he", "she", "they", "my", "our", "child", "kid", "son",
    "daughter", "husband", "wife", "mom", "dad", "free", "vegan", "just", "only", "too",
    "very", "really", "severely", "severe", "mild", "slightly", "bit", "little", "lot",
    "ok", "okay", "thanks", "thank", "you", "can", "cannot", "cant", "can't",
}


def _english_part(text: str) -> list:
    low = text.lower().replace("’", "'")
    low = _EN_NEG.sub(" ", low)
    found = []
    for m in list(_EN_STRONG.finditer(low)) + list(_EN_WEAK.finditer(low)):
        strong = m.re is _EN_STRONG
        label = m.group(0)
        if label in ("vegan",):
            continue
        if label.endswith("-free") or label.replace("-", " ") in ("dairy free", "gluten free", "nut free"):
            word = re.split(r"[\s-]", label)[0]
            found.append({"dairy": "dairy", "gluten": "gluten", "nut": "nut"}[word])
            continue
        rest = low[m.end():]
        stop = _EN_STOP.search(rest)
        punct = re.search(r"[.;!?\n]", rest)
        end = min([x.start() for x in (stop, punct) if x] or [len(rest)])
        clause = rest[:end]
        pieces = [p for p in re.split(r",|\band\b|\bor\b|&|/", clause) if p.strip()]
        for piece in pieces:
            words = [w for w in re.findall(r"[a-z]+", piece) if w not in _EN_FILLER]
            if not words:
                continue
            mapped = [(_EN_KEYS[w]) for w in words if w in _EN_KEYS]
            if mapped:
                found.extend(mapped)
            elif strong and len(words) <= 3:
                found.append(" ".join(words))  # unlisted food: kept as a plain ingredient name
    return found


def _arabic_part(text: str) -> list:
    norm = norm_ar(text)
    norm = _PUNCT.sub(" | ", norm)
    tokens = norm.split()
    found = []
    for i, tok in enumerate(tokens):
        forms = set(_variants(tok))
        is_strong = bool(forms & _AR_STRONG)
        is_weak = tok in _AR_WEAK and not is_strong
        if not (is_strong or is_weak):
            continue
        if is_weak and tok == "من":  # plain "من" is too common; only "من غير X"
            if not (i + 1 < len(tokens) and tokens[i + 1] == "غير"):
                continue
        if is_strong and forms & {"حساسيه", "حساسيت", "حساسه", "حساس", "تحسس", "بتحسس"}:
            before = tokens[max(0, i - 3): i]
            if any(b in _AR_NEG_BEFORE for b in before):
                continue
        j = i + 1
        clause = []
        while j < len(tokens) and tokens[j] != "|" and tokens[j] not in _AR_STOP \
                and not any(f in _AR_STOP for f in _variants(tokens[j])[:2]):
            clause.append(tokens[j])
            j += 1
        found.extend(sorted(arabic_allergen_families(" ".join(clause))))
    return found


def extract_allergies(text) -> list:
    """Allergy / avoid items named in the raw user message, as English allergen keys
    (milk, egg, peanut, nut, wheat, gluten, soy, fish, shellfish, sesame, dairy, lactose)
    or plain English ingredient names. Order is preserved, duplicates removed.

    It looks only at the part of a sentence after a phrase such as "عندي حساسية من",
    "مبكلش", "من غير", "allergic to", "can't eat" or "avoid". A negated phrase
    ("مفيش حساسية", "no allergies") adds nothing.
    """
    text = str(text or "")
    items = _arabic_part(text) + _english_part(text)
    return list(dict.fromkeys(i for i in items if i))
