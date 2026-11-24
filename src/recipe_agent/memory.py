"""Tiny persistent user memory: allergies / avoided ingredients, stored in a JSON file.

Nothing is saved unless the user explicitly asks. Saved items are always applied to the
allergy filter. The file is local and must never be committed (see .gitignore).
Set RECIPE_MEMORY_PATH to keep the file somewhere persistent (e.g. Google Drive in Colab).
"""
import json
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


def _path() -> Path:
    return Path(os.getenv("RECIPE_MEMORY_PATH") or ROOT / "data" / "memory" / "profiles.json")


def _safe_id(user_id) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", str(user_id or "default"))[:64]


class MemoryReadError(RuntimeError):
    """The memory file exists but cannot be read. Nothing is overwritten when this happens."""


def _read() -> dict:
    """A missing file is an empty memory. Any other problem raises MemoryReadError, because
    returning {} here would let the next write erase the user's saved allergies."""
    path = _path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise MemoryReadError(f"cannot read {path}: {exc}") from exc
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError as exc:
        _keep_corrupt_copy(path)
        raise MemoryReadError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        _keep_corrupt_copy(path)
        raise MemoryReadError(f"{path} does not contain a JSON object")
    return data


def _keep_corrupt_copy(path: Path) -> None:
    try:
        shutil.copy2(path, path.with_name(f"{path.name}.corrupt-{int(time.time())}"))
    except OSError:
        log.warning("could not back up unreadable memory file %s", path)


def _write(data: dict) -> None:
    """Atomic write: temp file in the same folder, fsync, then rename over the old file."""
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


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
