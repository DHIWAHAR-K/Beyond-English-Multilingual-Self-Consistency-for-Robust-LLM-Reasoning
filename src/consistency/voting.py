"""
Self-consistency majority voting.

For each example the model produces N raw response strings. We extract a
structured answer from each response, then pick the most frequent answer
(majority vote) as the final prediction.

Public API
----------
majority_vote(responses: list[str], task_type: str) -> dict
"""

from __future__ import annotations

import re
from collections import Counter

from src.evaluation.metrics import extract_number, normalize_answer


# ── Per-task extractors ────────────────────────────────────────────────────────

def _extract_math(response: str):
    """Return the last number in the response, or None."""
    return extract_number(response)


def _extract_binary(response: str):
    """Return 0 or 1 from the response, or None."""
    digit = re.search(r"\b([01])\b", response)
    if digit:
        return int(digit.group(1))
    upper = response.upper()
    if re.search(r"\bA\b", upper):
        return 0
    if re.search(r"\bB\b", upper):
        return 1
    return None


def _extract_three_way(response: str):
    """Return 'entailment', 'neutral', or 'contradiction', or None."""
    lower = response.lower()
    if "entail" in lower:
        return "entailment"
    if "contradict" in lower:
        return "contradiction"
    if "neutral" in lower:
        return "neutral"
    return None


def _extract_abcd(response: str):
    """Return A, B, C, or D, or None."""
    match = re.search(r"\b([ABCD])\b", response.upper())
    return match.group(1) if match else None


def _extract_span(response: str):
    """Normalize and return the response as a span answer, or None."""
    norm = normalize_answer(response)
    return norm if norm else None


_EXTRACTORS = {
    "math_reasoning": _extract_math,
    "commonsense":    _extract_binary,
    "nli":            _extract_three_way,
    "mcqa":           _extract_abcd,
    "extractive_qa":  _extract_span,
}


# ── Public API ─────────────────────────────────────────────────────────────────

def majority_vote(responses: list[str], task_type: str) -> dict:
    """Apply majority voting to a list of raw model responses.

    Parameters
    ----------
    responses : list[str]
        Raw response strings from the model (length == n_samples).
    task_type : str
        One of: math_reasoning, commonsense, nli, mcqa, extractive_qa.

    Returns
    -------
    dict with keys:
        prediction : the majority-voted answer (or None if all responses invalid)
        votes      : dict mapping each extracted answer -> vote count
        valid_n    : number of responses that yielded a parseable answer
        total_n    : total number of responses
    """
    extractor = _EXTRACTORS.get(task_type)
    if extractor is None:
        raise ValueError(
            f"majority_vote: unknown task_type={task_type!r}. "
            f"Expected one of {sorted(_EXTRACTORS)}"
        )

    extracted = [extractor(r) for r in responses]
    valid = [e for e in extracted if e is not None]

    if not valid:
        return {
            "prediction": None,
            "votes": {},
            "valid_n": 0,
            "total_n": len(responses),
        }

    counter: Counter = Counter(valid)
    prediction = counter.most_common(1)[0][0]

    return {
        "prediction": prediction,
        "votes": dict(counter),
        "valid_n": len(valid),
        "total_n": len(responses),
    }
