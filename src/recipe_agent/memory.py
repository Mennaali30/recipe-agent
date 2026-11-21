"""Tiny persistent user memory: allergies / avoided ingredients, stored in a JSON file.

Nothing is saved unless the user explicitly asks. Saved items are always applied to the
allergy filter. The file is local and must never be committed (see .gitignore).
Set RECIPE_MEMORY_PATH to keep the file somewhere persistent (e.g. Google Drive in Colab).
"""
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _path() -> Path:
    return Path(os.getenv("RECIPE_MEMORY_PATH") or ROOT / "data" / "memory" / "profiles.json")


def _safe_id(user_id) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", str(user_id or "default"))[:64]


def _read() -> dict:
    try:
        return json.loads(_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write(data: dict) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_exclusions(user_id="default") -> list:
    return list(_read().get(_safe_id(user_id), {}).get("exclude", []))


def remember_exclusions(user_id, items) -> list:
    data = _read()
    current = data.setdefault(_safe_id(user_id), {}).setdefault("exclude", [])
    for item in items:
        item = str(item).strip().lower()
        if item and item not in current:
            current.append(item)
    _write(data)
    return list(current)


def forget(user_id="default") -> None:
    data = _read()
    key = _safe_id(user_id)
    if key in data:
        del data[key]
        _write(data)
