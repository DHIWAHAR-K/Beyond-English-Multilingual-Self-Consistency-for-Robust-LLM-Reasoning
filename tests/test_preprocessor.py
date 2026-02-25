"""Integration tests for src/data/preprocessor.py."""

import json

import pytest

from src.data.preprocessor import (
    FORMATTERS,
    _parse_choices,
    format_commonsense,
    format_example,
    format_extractive_qa,
    format_math,
    format_mcq,
    format_nli,
)


# ── _parse_choices ────────────────────────────────────────────────────────────

def test_parse_choices_none():
    assert _parse_choices(None) is None


def test_parse_choices_list():
    assert _parse_choices(["A", "B"]) == ["A", "B"]


def test_parse_choices_json_string():
    s = json.dumps(["choice1", "choice2"])
    assert _parse_choices(s) == ["choice1", "choice2"]


# ── format_math ───────────────────────────────────────────────────────────────

def test_format_math_contains_input(one_per_task):
    row = one_per_task["math_reasoning"]
    prompt = format_math(row)
    assert row["input"] in prompt
    assert "Answer:" in prompt
    assert "step" in prompt.lower()


def test_format_math_via_dispatch(one_per_task):
    row = one_per_task["math_reasoning"]
    assert format_example(row) == format_math(row)


# ── format_commonsense ────────────────────────────────────────────────────────

def test_format_commonsense_contains_choices(one_per_task):
    row = one_per_task["commonsense"]
    prompt = format_commonsense(row)
    choices = _parse_choices(row["choices"])
    assert choices[0] in prompt
    assert choices[1] in prompt
    assert "A)" in prompt
    assert "B)" in prompt


def test_format_commonsense_missing_choices_raises():
    row = {"input": "Some premise.", "choices": None, "task_type": "commonsense"}
    with pytest.raises(ValueError):
        format_commonsense(row)


def test_format_commonsense_via_dispatch(one_per_task):
    row = one_per_task["commonsense"]
    assert format_example(row) == format_commonsense(row)


# ── format_nli ────────────────────────────────────────────────────────────────

def test_format_nli_contains_input(one_per_task):
    row = one_per_task["nli"]
    prompt = format_nli(row)
    assert row["input"] in prompt
    assert "Entailment" in prompt
    assert "Contradiction" in prompt
    assert "Neutral" in prompt


def test_format_nli_via_dispatch(one_per_task):
    row = one_per_task["nli"]
    assert format_example(row) == format_nli(row)


# ── format_mcq ────────────────────────────────────────────────────────────────

def test_format_mcq_contains_abcd(one_per_task):
    row = one_per_task["mcqa"]
    prompt = format_mcq(row)
    for letter in ("A)", "B)", "C)", "D)"):
        assert letter in prompt


def test_format_mcq_contains_choices(one_per_task):
    row = one_per_task["mcqa"]
    choices = _parse_choices(row["choices"])
    prompt = format_mcq(row)
    for c in choices:
        assert c in prompt


def test_format_mcq_missing_choices_raises():
    row = {"input": "A question?", "choices": json.dumps(["A", "B"]), "task_type": "mcqa"}
    with pytest.raises(ValueError):
        format_mcq(row)


def test_format_mcq_via_dispatch(one_per_task):
    row = one_per_task["mcqa"]
    assert format_example(row) == format_mcq(row)


# ── format_extractive_qa ──────────────────────────────────────────────────────

def test_format_extractive_qa_contains_input(one_per_task):
    row = one_per_task["extractive_qa"]
    prompt = format_extractive_qa(row)
    assert row["input"] in prompt
    assert "passage" in prompt.lower() or "context" in prompt.lower()


def test_format_extractive_qa_via_dispatch(one_per_task):
    row = one_per_task["extractive_qa"]
    assert format_example(row) == format_extractive_qa(row)


# ── format_example dispatch ───────────────────────────────────────────────────

def test_format_example_all_task_types(one_per_task):
    for task_type, row in one_per_task.items():
        prompt = format_example(row)
        assert isinstance(prompt, str)
        assert len(prompt) > 20, f"Suspiciously short prompt for {task_type}"


def test_format_example_unknown_raises():
    row = {"input": "test", "choices": None, "task_type": "unknown_task"}
    with pytest.raises(ValueError, match="unknown task_type"):
        format_example(row)


def test_formatters_dict_has_all_task_types():
    expected = {"math_reasoning", "commonsense", "nli", "mcqa", "extractive_qa"}
    assert set(FORMATTERS.keys()) == expected
