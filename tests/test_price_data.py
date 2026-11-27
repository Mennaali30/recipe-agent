"""Price plan, step 1: the most-used estimated groups now have a source, a URL and a date."""
import csv
from datetime import date

from recipe_agent.egypt_cost import DEFAULT_MAP, load_price_map
from recipe_agent.egypt_parse import match

SOURCED = ["fish_generic", "chickpeas", "yogurt", "tomato_paste_sauce", "lemon", "yeast_baking", "honey_molasses"]


def _rows():
    return {r["key"]: r for r in csv.DictReader(open(DEFAULT_MAP, encoding="utf-8-sig"))}


def test_step1_groups_are_sourced_with_url_date_and_reliability():
    rows = _rows()
    for k in SOURCED:
        r = rows[k]
        assert r["price_source"] == "current_web", k
        assert r["current_price_source_url"].startswith("https://"), k
        d = date.fromisoformat(r["current_price_date"])
        assert date(2026, 11, 1) <= d <= date(2026, 11, 30), k
        assert r["current_price_reliability"] in {"high", "medium", "low"}, k
        assert r["current_price_type"] in {"retail", "wholesale", "festival retail"}, k
        assert float(r["price_egp_per_kg"]) > 0, k


def test_every_current_web_row_has_https_url_and_iso_date():
    for k, r in _rows().items():
        if r["price_source"] == "current_web":
            assert r["current_price_source_url"].startswith("http"), k
            date.fromisoformat(r["current_price_date"])


def test_mozzarella_is_not_priced_as_banana():
    pm = load_price_map()
    assert match("موزاريلا", pm)["key"] == "cheese_generic"
    assert match("جبنة موتزاريلا", pm)["key"] == "cheese_generic"
    assert match("موز", pm)["key"] == "banana" and match("موزة", pm)["key"] == "banana"
