"""Answer cache: an identical request is answered without calling the LLM again.

Two files are read: the user cache (data/cache/answers.json, local, never committed) and a
shipped file (examples/answers_cache.json, committed, used for the demo examples).
Only plain successful answers are stored; see graph._cacheable.
"""
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHIPPED_PATH = ROOT / "examples" / "answers_cache.json"

CACHE_VERSION = "v1"  # change it when prompts or tools change, so old answers are not reused
MAX_ENTRIES = 200


def _user_path() -> Path:
    return Path(os.getenv("RECIPE_CACHE_PATH") or ROOT / "data" / "cache" / "answers.json")


def make_key(text, exclusions) -> str:
    """Same text (ignoring case, extra spaces and trailing punctuation) + same saved allergies."""
    norm = re.sub(r"\s+", " ", str(text).strip().lower())
    norm = re.sub(r"[\s.!?؟،,]+$", "", norm)
    excl = ",".join(sorted(str(e).strip().lower() for e in exclusions))
    return f"{CACHE_VERSION}|{norm}|{excl}"


def _read(path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def get(text, exclusions):
    """Cached answer for this request, or None."""
    key = make_key(text, exclusions)
    for path in (_user_path(), SHIPPED_PATH):
        entry = _read(path).get(key)
        if entry and entry.get("answer"):
            return entry["answer"]
    return None


def put(text, exclusions, answer) -> None:
    """Store an answer in the user cache (the oldest entries are dropped past MAX_ENTRIES)."""
    path = _user_path()
    data = _read(path)
    key = make_key(text, exclusions)
    data.pop(key, None)  # re-insert at the end
    data[key] = {"answer": answer, "time": int(time.time())}
    while len(data) > MAX_ENTRIES:
        data.pop(next(iter(data)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
