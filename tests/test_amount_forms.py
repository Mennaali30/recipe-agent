"""Amounts written with the Arabic dual ("بصلتان"), a bare "قرن", and en-dash ranges are priced. Offline."""
import pytest

from recipe_agent import egypt_parse as p
from recipe_agent import egypt_store
from recipe_agent.egypt_cost import load_price_map


@pytest.mark.parametrize("name,amount,grams", [
    ("بصل", "بصلتان متوسطتان", 220.0), ("بصل", "بصلتين", 220.0), ("بصل", "بصلة", 110.0),
    ("فلفل حار", "قرن", 15.0), ("فلفل حار", "قرنين", 30.0), ("فلفل حار", "2 قرن", 30.0),
    ("زيت", "معلقتان", 28.0),
])
def test_amounts_become_grams(name, amount, grams):
    pm = load_price_map()
    assert p.grams(p.parse_amount(amount), p.match(name, pm), name) == grams


def test_en_dash_range_is_a_midpoint():
    assert p.parse_amount("3–5")["qty"] == 4.0 and p.parse_amount("3-5")["qty"] == 4.0


def test_koshari_style_pasta_tagine_prices_onion_and_hot_pepper():
    e = egypt_store.load_store()["entries"]["egy:fah-217"]
    lines = {l["name"]: l for l in e["cost"]["lines"]}
    assert lines["بصل"]["cost_egp"] and lines["فلفل حار"]["cost_egp"]
    assert lines["بصل"]["price_source"] == "current_web"
