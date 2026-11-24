"""Phase 1 tests: LLM retry/backoff, real error logging, name cleaning,
Fahrenheit to Celsius, marinating-time cap.

Run with:  pip install pytest && pytest -q
No dataset, Chroma index, GPU or API key is needed.
"""
import logging

import pytest

from recipe_agent import ingest, tools
from recipe_agent.llm_call import invoke_with_backoff, is_transient
from recipe_agent.utils import clean_name, fahrenheit_to_celsius


class FlakyRunnable:
    """Fake LLM runnable: fails with the given errors, then returns a value."""

    def __init__(self, errors, value="ok"):
        self.errors = list(errors)
        self.value = value
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.value


class Recorder:
    def __init__(self):
        self.waits = []

    def __call__(self, seconds):
        self.waits.append(seconds)


# --------------------------------------------------------------------------- #
# 1. Transient error detection (429 quota, 503 overload)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "message",
    [
        "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generate_requests",
        "503 UNAVAILABLE. The model is overloaded. Please try again later.",
        "Error code: 429 - rate limit",
        "Error code: 503",
    ],
)
def test_transient_errors_are_detected(message):
    assert is_transient(Exception(message))


@pytest.mark.parametrize(
    "message",
    [
        "API key not valid. Please pass a valid API key.",
        "400 Invalid argument: model name",
        "The recipe is ready in 15 minutes",
    ],
)
def test_non_transient_errors_are_not_retried(message):
    assert not is_transient(Exception(message))


def test_status_code_attribute_marks_transient():
    exc = Exception("whatever")
    exc.status_code = 503
    assert is_transient(exc)
    exc2 = Exception("whatever")
    exc2.status_code = 400
    assert not is_transient(exc2)


# --------------------------------------------------------------------------- #
# 2. Retry with backoff
# --------------------------------------------------------------------------- #
def test_transient_errors_are_retried_with_backoff():
    rec = Recorder()
    llm = FlakyRunnable(
        [Exception("503 UNAVAILABLE"), Exception("429 RESOURCE_EXHAUSTED")], value="done"
    )
    assert invoke_with_backoff(llm, ["hi"], attempts=4, base_delay=2, sleep=rec) == "done"
    assert llm.calls == 3
    assert rec.waits == [2, 4]  # exponential: 2, then 2*2


def test_non_transient_error_is_not_retried():
    rec = Recorder()
    llm = FlakyRunnable([Exception("API key not valid")])
    with pytest.raises(Exception, match="API key"):
        invoke_with_backoff(llm, ["hi"], attempts=4, base_delay=2, sleep=rec)
    assert llm.calls == 1
    assert rec.waits == []


def test_gives_up_after_the_last_attempt_and_raises():
    rec = Recorder()
    llm = FlakyRunnable([Exception("503 UNAVAILABLE")] * 10)
    with pytest.raises(Exception, match="503"):
        invoke_with_backoff(llm, ["hi"], attempts=3, base_delay=1, sleep=rec)
    assert llm.calls == 3
    assert rec.waits == [1, 2]


# --------------------------------------------------------------------------- #
# 3. The real LLM error is logged, not swallowed
# --------------------------------------------------------------------------- #
def test_respond_node_logs_the_real_error(caplog):
    pytest.importorskip("langgraph")
    from recipe_agent import graph
    from recipe_agent.schemas import UserRequest

    def boom(*args, **kwargs):
        raise Exception("429 RESOURCE_EXHAUSTED: quota exceeded")

    caplog.set_level(logging.WARNING)
    old = graph.write_answer
    graph.write_answer = boom
    try:
        state = {
            "request": UserRequest(ingredients=["chicken"], language="en"),
            "recipes": [],
            "shopping": [],
        }
        out = graph.respond_node(state)
    finally:
        graph.write_answer = old
    assert out["fallback"] == "llm_error"
    assert "429 RESOURCE_EXHAUSTED" in caplog.text
    assert "answer writer is unavailable" in out["answer"].lower()


def test_substitute_node_logs_the_real_error(caplog):
    pytest.importorskip("langgraph")
    from recipe_agent import graph
    from recipe_agent.schemas import UserRequest

    def boom(*args, **kwargs):
        raise Exception("503 UNAVAILABLE: model overloaded")

    caplog.set_level(logging.WARNING)
    old = graph.plan_substitutions
    graph.plan_substitutions = boom
    try:
        state = {
            "request": UserRequest(ingredients=["chicken"], language="en"),
            "recipes": [{"recipe_id": 1, "name": "Chicken"}],
        }
        out = graph.substitute_node(state)
    finally:
        graph.plan_substitutions = old
    assert "503 UNAVAILABLE" in caplog.text
    assert out["recipes"][0]["substitutions"] == []


# --------------------------------------------------------------------------- #
# 4. Recipe name cleaning
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  GRANDMA'S  CHICKEN   SOUP ", "Grandma's Chicken Soup"),
        ('"Easy Pie"', "Easy Pie"),
        ("Chicken   With  Rice", "Chicken With Rice"),
        ("Pasta , Quick", "Pasta, Quick"),
        ("Ordinary Name Stays", "Ordinary Name Stays"),
    ],
)
def test_clean_name(raw, expected):
    assert clean_name(raw) == expected


# --------------------------------------------------------------------------- #
# 5. Fahrenheit to Celsius
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Bake at 350 degrees F for 20 minutes.", "Bake at 175°C (350°F) for 20 minutes."),
        ("Preheat oven to 375°F.", "Preheat oven to 190°C (375°F)."),
        ("Preheat oven to 375 degrees Fahrenheit.", "Preheat oven to 190°C (375°F)."),
        ("Chill at 40 F.", "Chill at 5°C (40°F)."),
    ],
)
def test_fahrenheit_is_converted(raw, expected):
    assert fahrenheit_to_celsius(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "Cook for 20 minutes.",
        "Bake at 180°C.",
        "Add 2 tablespoons of flour.",
    ],
)
def test_text_without_fahrenheit_is_untouched(raw):
    assert fahrenheit_to_celsius(raw) == raw


def test_get_instructions_converts_fahrenheit(monkeypatch):
    monkeypatch.setattr(
        tools, "get_recipe",
        lambda rid: {"instructions": ["Bake at 350 F.", "Serve warm."]},
    )
    assert tools.get_instructions(1) == ["Bake at 175°C (350°F).", "Serve warm."]


# --------------------------------------------------------------------------- #
# 6. Marinating / chilling time cap
# --------------------------------------------------------------------------- #
def test_marinating_total_is_capped_to_active_time():
    steps = ["Mix the sauce.", "Marinate the chicken overnight in the fridge.", "Grill 10 minutes."]
    assert ingest.cap_waiting_time(720, 20, 10, steps) == 30


def test_chilling_total_is_capped():
    steps = ["Make the dough.", "Refrigerate for 12 hours.", "Bake."]
    assert ingest.cap_waiting_time(780, 15, 45, steps) == 60


def test_normal_recipe_time_is_not_capped():
    steps = ["Chop the onions.", "Simmer for 2 hours, stirring now and then."]
    assert ingest.cap_waiting_time(150, 30, 120, steps) == 150


def test_unknown_time_stays_unknown():
    assert ingest.cap_waiting_time(None, 20, 10, ["Marinate overnight."]) is None


def test_total_equal_to_active_time_is_not_capped():
    steps = ["Marinate quickly for 5 minutes.", "Fry."]
    assert ingest.cap_waiting_time(30, 20, 10, steps) == 30
