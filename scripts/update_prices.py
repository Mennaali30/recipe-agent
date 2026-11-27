"""Update data/egypt/ingredient_price_map.csv from a CSV of sourced prices.

  python scripts/update_prices.py my_updates.csv            check only, shows what would change
  python scripts/update_prices.py my_updates.csv --write    check, then write the map
  add --force to allow a price more than 5x away from the current one

Update file columns: key,price_egp_per_kg,source_url,date,type,reliability,note
(see data/egypt/price_updates_template.csv). One bad row stops the run and nothing is written.
After a --write, run: python -m pytest -q
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recipe_agent.price_update import update_file  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("updates")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--map", default=None)
    a = ap.parse_args(argv)
    try:
        changes = update_file(a.updates, a.map, write=a.write, force=a.force)
    except (ValueError, OSError) as exc:
        print("NOT UPDATED. Fix these first:\n" + str(exc))
        return 1
    for c in changes:
        print(f"{c['key']}: {c['old_price']} ({c['old_source']}) -> {c['new_price']} EGP/kg")
    print(f"{len(changes)} price(s) " + ("written." if a.write else "checked only. Add --write to save."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
