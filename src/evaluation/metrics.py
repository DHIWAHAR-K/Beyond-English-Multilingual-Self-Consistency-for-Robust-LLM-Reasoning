"""
Evaluation metrics for all benchmark task types.

Each evaluator takes (prediction: str, label: str) and returns a dict that
always contains at least ``{"correct": bool}``.

Span evaluators also return ``{"exact_match": bool, "f1": float}``.

Public API
----------
eval_numeric(prediction, label)     -> dict   (math_reasoning / MGSM)
eval_binary(prediction, label)      -> dict   (commonsense / XCOPA, XStoryCloze)
eval_three_way(prediction, label)   -> dict   (nli / XNLI, ML-NLI-26lang)
eval_abcd(prediction, label)        -> dict   (mcqa / Global-MMLU-Lite)
eval_span(prediction, label)        -> dict   (extractive_qa / TyDiQA, MLQA)
get_evaluator(task_type)            -> Callable
"""

from __future__ import annotations

import json
import re
import string
from typing import Callable


# ── Text normalisation ────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    """Lowercase, remove articles, punctuation, and collapse whitespace."""
    s = s.lower()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    return " ".join(s.split())


def extract_number(text: str) -> float | None:
    """Return the last number found in *text*, or None."""
    # Remove commas used as thousands separators before matching
    cleaned = text.replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    return float(matches[-1]) if matches else None


def compute_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between normalised prediction and reference strings."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common_tokens = set(pred_tokens) & set(ref_tokens)
    num_common = sum(
        min(pred_tokens.count(t), ref_tokens.count(t)) for t in common_tokens
    )
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ── Evaluators ────────────────────────────────────────────────────────────────

def eval_numeric(prediction: str, label: str) -> dict:
    """Evaluate a numeric answer (MGSM math_reasoning).

    Extracts the last number from *prediction* and compares to *label*.
    """
    pred_num = extract_number(prediction)
    try:
        true_num = float(label)
    except (ValueError, TypeError):
        return {"correct": False, "prediction": pred_num, "reference": label}

    correct = pred_num is not None and abs(pred_num - true_num) < 1e-6
    return {"correct": correct, "prediction": pred_num, "reference": true_num}


def eval_binary(prediction: str, label: str) -> dict:
    """Evaluate a binary choice answer (XCOPA / XStoryCloze commonsense).

    Accepts ``"0"``/``"1"`` digits or ``"A"``/``"B"`` letters in *prediction*.
    """
    pred: int | None = None

    # Prefer explicit 0/1
    digit_match = re.search(r"\b([01])\b", prediction)
    if digit_match:
        pred = int(digit_match.group(1))
    else:
        upper = prediction.upper()
        if re.search(r"\bA\b", upper):
            pred = 0
        elif re.search(r"\bB\b", upper):
            pred = 1

    try:
        true_label = int(label)
    except (ValueError, TypeError):
        return {"correct": False, "prediction": pred, "reference": label}

    correct = pred is not None and pred == true_label
    return {"correct": correct, "prediction": pred, "reference": true_label}


def eval_three_way(prediction: str, label: str) -> dict:
    """Evaluate a three-way NLI label (XNLI / ML-NLI-26lang).

    Matches ``entailment``, ``neutral``, or ``contradiction`` substrings.
    """
    lower = prediction.lower()
    if "entail" in lower:
        pred = "entailment"
    elif "contradict" in lower:
        pred = "contradiction"
    elif "neutral" in lower:
        pred = "neutral"
    else:
        pred = None

    correct = pred is not None and pred == label.lower().strip()
    return {"correct": correct, "prediction": pred, "reference": label}


def eval_abcd(prediction: str, label: str) -> dict:
    """Evaluate an A/B/C/D multiple-choice answer (Global-MMLU-Lite).

    Extracts the first standalone A, B, C, or D letter from *prediction*.
    """
    match = re.search(r"\b([ABCD])\b", prediction.upper())
    pred = match.group(1) if match else None

    correct = pred is not None and pred == label.upper().strip()
    return {"correct": correct, "prediction": pred, "reference": label}


def eval_span(prediction: str, label: str) -> dict:
    """Evaluate an extractive span answer (TyDiQA-GoldP / MLQA).

    *label* may be a JSON-encoded list of acceptable answer strings or a plain
    string.  Reports both Exact Match and token-level F1 against the best match.
    """
    # Parse label into a list of reference strings
    references: list[str]
    if isinstance(label, str):
        try:
            parsed = json.loads(label)
            references = parsed if isinstance(parsed, list) else [label]
        except (json.JSONDecodeError, TypeError):
            references = [label]
    else:
        references = [str(label)]

    em = any(
        normalize_answer(prediction) == normalize_answer(r) for r in references
    )
    f1 = max(compute_f1(prediction, r) for r in references)

    return {
        "correct": em,
        "exact_match": em,
        "f1": f1,
        "prediction": prediction,
        "reference": references,
    }


# ── Factory ───────────────────────────────────────────────────────────────────

_EVALUATORS: dict[str, Callable] = {
    "math_reasoning": eval_numeric,
    "commonsense":    eval_binary,
    "nli":            eval_three_way,
    "mcqa":           eval_abcd,
    "extractive_qa":  eval_span,
}


def get_evaluator(task_type: str) -> Callable:
    """Return the evaluation function for *task_type*.

    Parameters
    ----------
    task_type:
        One of ``math_reasoning``, ``commonsense``, ``nli``, ``mcqa``,
        ``extractive_qa``.

    Returns
    -------
    A callable ``(prediction: str, label: str) -> dict``.

    Raises
    ------
    ValueError for unrecognised task types.
    """
    if task_type not in _EVALUATORS:
        raise ValueError(
            f"get_evaluator: unknown task_type={task_type!r}. "
            f"Expected one of {sorted(_EVALUATORS)}"
        )
    return _EVALUATORS[task_type]
