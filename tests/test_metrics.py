"""Unit tests for src/evaluation/metrics.py — no disk I/O required."""

import json

import pytest

from src.evaluation.metrics import (
    compute_f1,
    eval_abcd,
    eval_binary,
    eval_numeric,
    eval_span,
    eval_three_way,
    get_evaluator,
    normalize_answer,
)


# ── normalize_answer ──────────────────────────────────────────────────────────

def test_normalize_lowercase():
    assert normalize_answer("Hello World") == "hello world"


def test_normalize_removes_articles():
    assert normalize_answer("the cat") == "cat"
    assert normalize_answer("a dog") == "dog"
    assert normalize_answer("an apple") == "apple"


def test_normalize_removes_punctuation():
    result = normalize_answer("hello, world!")
    assert "," not in result
    assert "!" not in result


def test_normalize_collapses_whitespace():
    assert normalize_answer("  multiple   spaces  ") == "multiple spaces"


# ── compute_f1 ────────────────────────────────────────────────────────────────

def test_f1_exact_match():
    assert compute_f1("New York City", "New York City") == pytest.approx(1.0)


def test_f1_partial_overlap():
    f1 = compute_f1("New York", "New York City")
    assert 0.0 < f1 < 1.0


def test_f1_no_overlap():
    assert compute_f1("Paris", "London") == pytest.approx(0.0)


def test_f1_empty_strings():
    assert compute_f1("", "") == pytest.approx(1.0)


def test_f1_one_empty():
    assert compute_f1("answer", "") == pytest.approx(0.0)


# ── eval_numeric ──────────────────────────────────────────────────────────────

def test_eval_numeric_correct():
    result = eval_numeric("The answer is 42.", "42")
    assert result["correct"] is True
    assert result["prediction"] == pytest.approx(42.0)


def test_eval_numeric_correct_with_steps():
    result = eval_numeric("Step 1: 10 + 8 = 18\nStep 2: 18 × 2 = 36\nAnswer: 36", "36")
    assert result["correct"] is True


def test_eval_numeric_wrong():
    result = eval_numeric("The answer is 40.", "42")
    assert result["correct"] is False


def test_eval_numeric_no_number():
    result = eval_numeric("I don't know.", "42")
    assert result["correct"] is False
    assert result["prediction"] is None


def test_eval_numeric_comma_separated():
    result = eval_numeric("The total is 1,234", "1234")
    assert result["correct"] is True


# ── eval_binary ───────────────────────────────────────────────────────────────

def test_eval_binary_correct_zero():
    assert eval_binary("0", "0")["correct"] is True


def test_eval_binary_correct_one():
    assert eval_binary("The answer is 1", "1")["correct"] is True


def test_eval_binary_letter_a():
    assert eval_binary("A", "0")["correct"] is True


def test_eval_binary_letter_b():
    assert eval_binary("B", "1")["correct"] is True


def test_eval_binary_wrong():
    assert eval_binary("1", "0")["correct"] is False


def test_eval_binary_no_match():
    result = eval_binary("I cannot determine the answer.", "0")
    assert result["correct"] is False
    assert result["prediction"] is None


# ── eval_three_way ────────────────────────────────────────────────────────────

def test_eval_three_way_entailment():
    result = eval_three_way("This is entailment.", "entailment")
    assert result["correct"] is True
    assert result["prediction"] == "entailment"


def test_eval_three_way_neutral():
    result = eval_three_way("The answer is neutral.", "neutral")
    assert result["correct"] is True


def test_eval_three_way_contradiction():
    result = eval_three_way("This is a contradiction.", "contradiction")
    assert result["correct"] is True


def test_eval_three_way_wrong():
    result = eval_three_way("entailment", "contradiction")
    assert result["correct"] is False


def test_eval_three_way_no_match():
    result = eval_three_way("I have no idea.", "neutral")
    assert result["correct"] is False
    assert result["prediction"] is None


def test_eval_three_way_case_insensitive_label():
    result = eval_three_way("Entailment.", "Entailment")
    assert result["correct"] is True


# ── eval_abcd ────────────────────────────────────────────────────────────────

def test_eval_abcd_correct_a():
    assert eval_abcd("A", "A")["correct"] is True


def test_eval_abcd_correct_c():
    assert eval_abcd("The answer is C.", "C")["correct"] is True


def test_eval_abcd_wrong():
    assert eval_abcd("B", "C")["correct"] is False


def test_eval_abcd_no_letter():
    result = eval_abcd("I don't know.", "A")
    assert result["correct"] is False
    assert result["prediction"] is None


def test_eval_abcd_case_insensitive_prediction():
    assert eval_abcd("the answer is c", "C")["correct"] is True


# ── eval_span ────────────────────────────────────────────────────────────────

def test_eval_span_exact_match():
    result = eval_span("New York City", "New York City")
    assert result["exact_match"] is True
    assert result["f1"] == pytest.approx(1.0)
    assert result["correct"] is True


def test_eval_span_partial_f1():
    result = eval_span("New York", "New York City")
    assert result["exact_match"] is False
    assert 0.0 < result["f1"] < 1.0


def test_eval_span_no_match():
    result = eval_span("Paris", "London")
    assert result["exact_match"] is False
    assert result["f1"] == pytest.approx(0.0)


def test_eval_span_json_list_label():
    # label as JSON list of acceptable answers
    label = json.dumps(["New York City", "NYC", "New York"])
    result = eval_span("NYC", label)
    assert result["exact_match"] is True


def test_eval_span_normalisation():
    # articles and case should not affect EM
    result = eval_span("the cat", "cat")
    assert result["exact_match"] is True


def test_eval_span_returns_required_keys():
    result = eval_span("answer", "answer")
    for key in ("correct", "exact_match", "f1", "prediction", "reference"):
        assert key in result


# ── get_evaluator ─────────────────────────────────────────────────────────────

def test_get_evaluator_all_types():
    for task_type in ("math_reasoning", "commonsense", "nli", "mcqa", "extractive_qa"):
        fn = get_evaluator(task_type)
        assert callable(fn)


def test_get_evaluator_returns_correct_function():
    assert get_evaluator("math_reasoning") is eval_numeric
    assert get_evaluator("commonsense") is eval_binary
    assert get_evaluator("nli") is eval_three_way
    assert get_evaluator("mcqa") is eval_abcd
    assert get_evaluator("extractive_qa") is eval_span


def test_get_evaluator_unknown_raises():
    with pytest.raises(ValueError, match="unknown task_type"):
        get_evaluator("unknown_task")
