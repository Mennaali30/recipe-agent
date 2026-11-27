"""Price plan, step 2: the update script. Offline, works on a copy of the map."""
import csv
import shutil
from datetime import date

import pytest

from recipe_agent import price_update as pu
from recipe_agent.egypt_cost import DEFAULT_MAP

TODAY = date(2026, 11, 28)
HDR = "key,price_egp_per_kg,source_url,date,type,reliability,note\n"


@pytest.fixture
def mp(tmp_path):
    p = tmp_path / "map.csv"
    shutil.copy(DEFAULT_MAP, p)
    return p


def upd(tmp_path, *lines):
    f = tmp_path / "u.csv"
    f.write_text(HDR + "\n".join(lines) + "\n", encoding="utf-8")
    return f


GOOD = "cream,160,https://example.eg/prices,2026-11-26,retail,medium,cooking cream 200ml pack"


def row(path, key):
    return next(r for r in csv.DictReader(open(path, encoding="utf-8-sig")) if r["key"] == key)


def test_check_only_does_not_write(tmp_path, mp):
    before = mp.read_bytes()
    ch = pu.update_file(upd(tmp_path, GOOD), mp, write=False, today=TODAY)
    assert ch[0]["key"] == "cream" and ch[0]["new_price"] == "160" and mp.read_bytes() == before


def test_write_updates_row_and_keeps_old_price_in_note(tmp_path, mp):
    pu.update_file(upd(tmp_path, GOOD), mp, write=True, today=TODAY)
    r = row(mp, "cream")
    assert (r["price_source"], r["price_egp_per_kg"], r["current_price_date"]) == ("current_web", "160", "2026-11-26")
    assert r["current_price_source_url"] == "https://example.eg/prices" and r["current_price_reliability"] == "medium"
    assert "was 150 EGP/kg, estimated" in r["notes"]


def test_other_rows_and_file_format_untouched(tmp_path, mp):
    fields, before = pu.read_rows(mp)
    pu.update_file(upd(tmp_path, GOOD), mp, write=True, today=TODAY)
    fields2, after = pu.read_rows(mp)
    assert fields == fields2 and len(before) == len(after)
    assert [a for a, b in zip(after, before) if a != b] == [next(a for a in after if a["key"] == "cream")]


@pytest.mark.parametrize("line,frag", [
    ("nosuchkey,10,https://x.eg/a,2026-11-26,retail,high,", "unknown key"),
    ("cream,0,https://x.eg/a,2026-11-26,retail,high,", "price"),
    ("cream,abc,https://x.eg/a,2026-11-26,retail,high,", "price"),
    ("cream,-5,https://x.eg/a,2026-11-26,retail,high,", "price"),
    ("cream,nan,https://x.eg/a,2026-11-26,retail,high,", "price"),
    ("cream,160,not a url,2026-11-26,retail,high,", "source_url"),
    ("cream,160,,2026-11-26,retail,high,", "source_url"),
    ("cream,160,https://x.eg/a,26/11/2026,retail,high,", "date must be"),
    ("cream,160,https://x.eg/a,2026-12-30,retail,high,", "future"),
    ("cream,160,https://x.eg/a,2026-01-05,retail,high,", "older than"),
    ("cream,160,https://x.eg/a,2026-11-26,cheap,high,", "type"),
    ("cream,160,https://x.eg/a,2026-11-26,retail,great,", "reliability"),
    ("cream,5000,https://x.eg/a,2026-11-26,retail,high,", "away from"),
])
def test_bad_rows_are_refused(tmp_path, mp, line, frag):
    before = mp.read_bytes()
    with pytest.raises(ValueError, match=frag):
        pu.update_file(upd(tmp_path, line), mp, write=True, today=TODAY)
    assert mp.read_bytes() == before


def test_one_bad_row_stops_everything(tmp_path, mp):
    before = mp.read_bytes()
    with pytest.raises(ValueError):
        pu.update_file(upd(tmp_path, GOOD, "nosuchkey,10,https://x.eg/a,2026-11-26,retail,high,"), mp, write=True, today=TODAY)
    assert mp.read_bytes() == before


def test_force_allows_a_big_jump_and_duplicates_are_refused(tmp_path, mp):
    big = "cream,5000,https://x.eg/a,2026-11-26,retail,high,"
    pu.update_file(upd(tmp_path, big), mp, write=False, today=TODAY, force=True)
    with pytest.raises(ValueError, match="twice"):
        pu.update_file(upd(tmp_path, GOOD, GOOD), mp, today=TODAY)


def test_missing_columns_and_empty_file(tmp_path, mp):
    f = tmp_path / "bad.csv"
    f.write_text("key,price\ncream,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        pu.update_file(f, mp, today=TODAY)
    assert pu.update_file(upd(tmp_path), mp, today=TODAY) == []


def test_age_and_stale_helpers():
    assert pu.price_age_days({"current_price_date": "2026-11-18"}, TODAY) == 10
    assert pu.price_age_days({"current_price_date": ""}, TODAY) is None
    assert not pu.is_stale({"current_price_date": "2026-11-01"}, TODAY)       # 27 days
    assert pu.is_stale({"current_price_date": "2026-10-20"}, TODAY)           # 39 days
    assert not pu.is_stale({"current_price_date": ""}, TODAY)


def test_cli_dry_run_and_error_exit_code(tmp_path, mp, capsys):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("update_prices", Path(__file__).resolve().parents[1] / "scripts" / "update_prices.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    good = upd(tmp_path, "cream,160,https://x.eg/a," + date.today().isoformat() + ",retail,medium,")
    assert mod.main([str(good), "--map", str(mp)]) == 0 and "checked only" in capsys.readouterr().out
    assert row(mp, "cream")["price_source"] == "estimated"
    assert mod.main([str(good), "--map", str(mp), "--write"]) == 0 and row(mp, "cream")["price_source"] == "current_web"
    bad = upd(tmp_path, "zzz,1,https://x.eg/a,2026-11-26,retail,high,")
    assert mod.main([str(bad), "--map", str(mp)]) == 1 and "NOT UPDATED" in capsys.readouterr().out


def test_template_has_the_expected_header():
    from recipe_agent.egypt_cost import DEFAULT_MAP as m
    t = m.parent / "price_updates_template.csv"
    assert t.read_text(encoding="utf-8").strip() == HDR.strip()
