"""Known Arabic wording fixes: dataset typos and the writer's known mistakes. Offline."""
from recipe_agent import egypt_store, textfix


def test_dataset_typos_are_fixed_where_listed():
    store = egypt_store.load_store()["entries"]
    assert "غلوتين" not in " ".join(store["egy:osool-550"]["instructions"]) and "غليتين" in " ".join(store["egy:osool-550"]["instructions"])
    assert "السماق" in " ".join(store["egy:fah-096"]["instructions"]) and "السماء" not in " ".join(store["egy:fah-096"]["instructions"])
    assert "يذبل" in " ".join(store["egy:fah-082"]["instructions"])


def test_a_fix_is_limited_to_its_recipe_and_whole_words(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("wrong,right,recipe_id,note\nالسماء,السماق,fah-096,x\nقط,قطه,,y\n", encoding="utf-8")
    assert textfix.fix_recipe_text("رش السماء", "egy:fah-096", path=p) == "رش السماق"
    assert textfix.fix_recipe_text("رش السماء", "egy:fah-001", path=p) == "رش السماء"      # another recipe
    assert textfix.fix_recipe_text("قط وقطع", "", path=p) == "قطه وقطع"     # whole word only: "قطع" is untouched


def test_known_writer_mistakes_and_stiff_opening_are_fixed():
    out = textfix.fix_answer("الحمد لله، أستطيع أن أقترح وصفة لديدة جدا.")
    assert out == "أستطيع أن أقترح وصفة لذيذة جدا." or out.startswith("أستطيع")
    assert "لذيذة" in out and "لديدة" not in out and "الحمد" not in out


def test_english_answers_and_clean_text_are_untouched():
    assert textfix.fix_answer("Alhamdulillah, tasty", "en") == "Alhamdulillah, tasty"
    assert textfix.fix_answer("كشري لذيذ مع صلصة") == "كشري لذيذ مع صلصة"
    assert textfix.fix_answer("") == ""


def test_latin_word_in_arabic_text_only_logs(caplog):
    with caplog.at_level("WARNING"):
        out = textfix.fix_answer("جربي Bondage Chicken النهارده")
    assert out == "جربي Bondage Chicken النهارده" and "mixes a Latin word" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING"):
        textfix.fix_answer("جربي فراخ (Bondage Chicken) النهارده")
    assert "mixes" not in caplog.text


def test_respond_node_applies_the_answer_fixes(monkeypatch):
    from recipe_agent import graph
    from recipe_agent.schemas import UserRequest
    monkeypatch.setattr(graph, "parse_request", lambda t: UserRequest(language="ar", ingredients=["فراخ", "رز"]))
    monkeypatch.setattr(graph, "write_answer", lambda *a, **k: "الحمد لله، وصفة لديدة")
    monkeypatch.setattr(graph.memory, "get_exclusions", lambda uid="default": [])
    monkeypatch.setattr(graph.tools, "search", lambda *a, **k: [])      # no English index in tests
    graph.build_graph.cache_clear()
    out = graph.run_agent("عندي فراخ ورز", "t")
    assert "لذيذة" in out["answer"] and "لديدة" not in out["answer"] and "الحمد" not in out["answer"]
