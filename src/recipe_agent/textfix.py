"""Known Arabic wording fixes (no LLM, no network). Two small tables the owner can edit like the price file.

data/egypt/text_fixes.csv    typos in the Egyptian recipe text itself: wrong,right,recipe_id,note
                             (recipe_id empty = every recipe). Applied when the recipes are loaded.
data/egypt/answer_fixes.csv  words / stiff phrases the answer writer (LLM) is known to get wrong:
                             wrong,right,note. Applied to the final Arabic answer.
A row replaces only the whole word or phrase (never inside a longer Arabic word). Nothing else is rewritten.
"""
import csv
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)
DATA = Path(__file__).resolve().parents[2] / "data" / "egypt"
_AR = "\u0600-\u06FF"
_cache = {}


def _rows(path):
    path = str(path)
    if path not in _cache:
        try:
            with open(path, encoding="utf-8-sig", newline="") as fh:
                _cache[path] = [r for r in csv.DictReader(fh) if (r.get("wrong") or "").strip()]
        except FileNotFoundError:
            _cache[path] = []
    return _cache[path]


def _sub(text: str, wrong: str, right: str) -> str:
    # whole word; one conjunction / preposition letter in front is allowed ("والسماء" -> "والسماق")
    pat = r"(?<![%s])([وبفلك]?)%s(?![%s])" % (_AR, re.escape(wrong), _AR)
    return re.sub(pat, lambda m: m.group(1) + right, text)


def fix_recipe_text(text: str, recipe_id: str = "", path=None) -> str:
    """Fix known typos in one piece of recipe text (a step or a name)."""
    if not text:
        return text
    rid = (recipe_id or "").replace("egy:", "")
    for r in _rows(path or DATA / "text_fixes.csv"):
        only = (r.get("recipe_id") or "").strip()
        if only and only != rid:
            continue
        text = _sub(text, r["wrong"].strip(), (r.get("right") or "").strip())
    return text


def fix_answer(text: str, lang: str = "ar", path=None) -> str:
    """Fix known writer mistakes in the final answer (Arabic only). Logs when Latin words sit in Arabic text."""
    if not text or lang != "ar":
        return text
    for r in _rows(path or DATA / "answer_fixes.csv"):
        wrong, right = r["wrong"].strip(), (r.get("right") or "").strip()
        text = _sub(text, wrong, right)
    text = re.sub(r"^[\s\u060C,.]+", "", text)       # a removed opening phrase can leave a comma
    text = re.sub(r"[ \t]{2,}", " ", text)
    if re.search(r"[%s]\s+[A-Za-z]{4,}(?:\s+[A-Za-z]+)*\s+[%s]" % (_AR, _AR), text):
        log.warning("answer mixes a Latin word into Arabic text outside parentheses")
    return text
