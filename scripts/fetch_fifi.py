"""Download the Egyptian recipes from fifi.cooking to data/egypt/recipes/ (needs internet).
Egyptian = collection != "world". Credit fifi.cooking. The recipe text license is not stated:
ask the maintainers before any public release. Not used by the tests."""
import json, sys, urllib.request
from pathlib import Path

BASE = "https://fifi.cooking/data"
out = Path(__file__).resolve().parents[1] / "data" / "egypt" / "recipes"
out.mkdir(parents=True, exist_ok=True)
get = lambda u: json.load(urllib.request.urlopen(u, timeout=30))
n, page = 0, 1
while True:
    try:
        idx = get(f"{BASE}/index/{page}.json")
    except Exception:
        break
    for item in idx:
        if item.get("collection") == "world":
            continue
        (out / f"{item['id']}.json").write_text(
            json.dumps(get(f"{BASE}/recipes/{item['id']}.json"), ensure_ascii=False), encoding="utf-8")
        n += 1
    page += 1
print(f"saved {n} Egyptian recipes to {out}", file=sys.stderr)
