"""
Prompt builders for each benchmark task type.

Each builder takes a corpus row (a dict with keys: input, choices, task_type, ...)
and returns a plain string prompt ready to be wrapped in a chat template.

Public API
----------
build_prompt(row: dict) -> str
"""

from __future__ import annotations

import json


def _parse_choices(choices_val) -> list[str]:
    """Parse choices from a corpus row value into a list of strings."""
    if choices_val is None:
        return []
    # NaN check (pandas float NaN)
    if isinstance(choices_val, float):
        return []
    if isinstance(choices_val, list):
        return [str(c) for c in choices_val]
    if isinstance(choices_val, str):
        try:
            parsed = json.loads(choices_val)
            if isinstance(parsed, list):
                return [str(c) for c in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
        return [choices_val]
    return []


def _math_reasoning(row: dict) -> str:
    return (
        f"{row['input']}\n\n"
        "Solve step by step. Write the final numeric answer on the last line as:\n"
        "Answer: <number>"
    )


def _commonsense(row: dict) -> str:
    choices = _parse_choices(row.get("choices"))
    opts = "\n".join(f"Option {i}: {c}" for i, c in enumerate(choices))
    return (
        f"{row['input']}\n\n"
        f"{opts}\n\n"
        "Answer with just the number 0 or 1."
    )


def _nli(row: dict) -> str:
    return (
        f"{row['input']}\n\n"
        "Does the premise entail, contradict, or is neutral with respect to the hypothesis?\n"
        "Answer with exactly one word: entailment, neutral, or contradiction."
    )


def _mcqa(row: dict) -> str:
    choices = _parse_choices(row.get("choices"))
    letters = ["A", "B", "C", "D"]
    opts = "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(choices) if i < 4)
    return (
        f"{row['input']}\n\n"
        f"{opts}\n\n"
        "Answer with just the letter A, B, C, or D."
    )


def _extractive_qa(row: dict) -> str:
    return (
        f"{row['input']}\n\n"
        "Answer with a short phrase extracted directly from the passage. "
        "Do not explain."
    )


_BUILDERS = {
    "math_reasoning": _math_reasoning,
    "commonsense":    _commonsense,
    "nli":            _nli,
    "mcqa":           _mcqa,
    "extractive_qa":  _extractive_qa,
}


def build_prompt(row: dict) -> str:
    """Build a prompt string for a corpus row.

    Parameters
    ----------
    row : dict
        A corpus row with at minimum keys: task_type, input, choices.

    Returns
    -------
    str : The prompt text (not yet wrapped in a chat template).

    Raises
    ------
    ValueError for unrecognised task_type.
    """
    task_type = row.get("task_type", "")
    builder = _BUILDERS.get(task_type)
    if builder is None:
        raise ValueError(
            f"build_prompt: unknown task_type={task_type!r}. "
            f"Expected one of {sorted(_BUILDERS)}"
        )
    return builder(row)
