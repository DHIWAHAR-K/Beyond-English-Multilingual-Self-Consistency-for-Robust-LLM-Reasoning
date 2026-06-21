from unittest.mock import MagicMock

import pytest

from src.tasks.reasoning import (
    build_cot_prompt,
    extract_answer_fallback,
    extract_answer_regex,
    majority_vote,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Step by step... The answer is 42", "42"),
        ("Therefore \\boxed{7}", "7"),
        ("No pattern here but final value 99", "99"),
    ],
)
def test_extract_answer_regex(text, expected):
    assert extract_answer_regex(text) == expected


def test_majority_vote():
    assert majority_vote(["1", "1", "2"]) == "1"
    assert majority_vote([None, None]) is None


def test_build_cot_prompt_contains_language():
    prompt = build_cot_prompt("2+2?", "es")
    assert "Spanish" in prompt
    assert "2+2?" in prompt


def test_extract_answer_fallback_uses_runner():
    runner = MagicMock()
    runner.generate.return_value = {"text": "The answer is 15"}
    assert extract_answer_fallback("some reasoning", runner) == "15"
