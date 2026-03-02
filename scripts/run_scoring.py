#!/usr/bin/env python3
"""
Score completed inference runs using self-consistency majority voting.

Reads raw_outputs.jsonl from a run directory, applies majority voting across
the N sampled responses, evaluates each prediction against the gold label,
and writes scores.jsonl + summary.json to the same directory.

Usage
-----
Score one run:
    conda run -n scr python scripts/run_scoring.py \\
        --run-dir experiments/runs/qwen2_5_32b_mgsm_en_20260227_120000

Score all unscored runs at once:
    conda run -n scr python scripts/run_scoring.py --all

Re-score an already-scored run:
    conda run -n scr python scripts/run_scoring.py \\
        --run-dir experiments/runs/qwen2_5_32b_mgsm_en_20260227_120000 \\
        --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.consistency.voting import majority_vote
from src.evaluation.metrics import get_evaluator

EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "runs"


def score_run(run_dir: Path, force: bool = False) -> dict:
    """Score a single run directory.

    Parameters
    ----------
    run_dir : Path
        Directory containing raw_outputs.jsonl (written by run_inference.py).
    force : bool
        Re-score even if summary.json already exists.

    Returns
    -------
    dict : Summary statistics, or empty dict if the run was skipped.
    """
    raw_path = run_dir / "raw_outputs.jsonl"
    scores_path = run_dir / "scores.jsonl"
    summary_path = run_dir / "summary.json"

    if not raw_path.exists():
        print(f"[skip]    {run_dir.name}  — no raw_outputs.jsonl")
        return {}

    if summary_path.exists() and not force:
        existing = json.loads(summary_path.read_text())
        acc = existing.get("accuracy", "?")
        print(f"[already] {run_dir.name}  accuracy={float(acc):.1%}  (use --force to re-score)")
        return existing

    # ── Load records ──────────────────────────────────────────────────────────
    records: list[dict] = []
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        print(f"[skip]    {run_dir.name}  — raw_outputs.jsonl is empty")
        return {}

    task_type = records[0]["task_type"]
    evaluator = get_evaluator(task_type)

    # ── Vote + evaluate ───────────────────────────────────────────────────────
    score_records: list[dict] = []
    for rec in records:
        vote = majority_vote(rec["responses"], task_type)
        prediction = vote["prediction"]
        eval_result = evaluator(
            str(prediction) if prediction is not None else "",
            rec["label"],
        )

        score_rec = {
            "id":          rec["id"],
            "model":       rec["model"],
            "dataset":     rec["dataset"],
            "language":    rec["language"],
            "task_type":   task_type,
            "prediction":  prediction,
            "label":       rec["label"],
            "correct":     eval_result["correct"],
            "votes":       vote["votes"],
            "valid_n":     vote["valid_n"],
            "total_n":     vote["total_n"],
        }
        # Span tasks also carry f1
        if "f1" in eval_result:
            score_rec["f1"] = eval_result["f1"]
        if "exact_match" in eval_result:
            score_rec["exact_match"] = eval_result["exact_match"]

        score_records.append(score_rec)

    # ── Write scores.jsonl ────────────────────────────────────────────────────
    with open(scores_path, "w") as f:
        for rec in score_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── Compute summary ───────────────────────────────────────────────────────
    total = len(score_records)
    n_correct = sum(1 for r in score_records if r["correct"])
    accuracy = n_correct / total if total > 0 else 0.0
    avg_valid = sum(r["valid_n"] for r in score_records) / total if total > 0 else 0.0

    summary: dict = {
        "run_dir":        str(run_dir),
        "model":          records[0]["model"],
        "dataset":        records[0]["dataset"],
        "language":       records[0]["language"],
        "task_type":      task_type,
        "n_examples":     total,
        "n_correct":      n_correct,
        "accuracy":       round(accuracy, 4),
        "avg_valid_votes": round(avg_valid, 2),
    }
    if task_type == "extractive_qa":
        avg_f1 = sum(r.get("f1", 0.0) for r in score_records) / total if total > 0 else 0.0
        summary["avg_f1"] = round(avg_f1, 4)

    summary_path.write_text(json.dumps(summary, indent=2))

    extra = f"  f1={summary['avg_f1']:.4f}" if "avg_f1" in summary else ""
    print(
        f"[scored]  {run_dir.name}  "
        f"accuracy={accuracy:.1%}  ({n_correct}/{total}){extra}"
    )
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score inference runs using self-consistency majority voting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-dir", type=str,
        help="Path to a specific run directory to score",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Score all unscored runs under experiments/runs/",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-score runs that already have a summary.json",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.run_dir:
        result = score_run(Path(args.run_dir), force=args.force)
        if result:
            print(f"\nSummary written to: {Path(args.run_dir) / 'summary.json'}")
        return

    # --all
    if not EXPERIMENTS_DIR.exists():
        print(f"No runs directory found at {EXPERIMENTS_DIR}")
        sys.exit(0)

    run_dirs = sorted(d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir())
    if not run_dirs:
        print(f"No run directories found in {EXPERIMENTS_DIR}")
        sys.exit(0)

    summaries: list[dict] = []
    for run_dir in run_dirs:
        s = score_run(run_dir, force=args.force)
        if s:
            summaries.append(s)

    if summaries:
        print(f"\n{'─' * 60}")
        print(f"Scored {len(summaries)} run(s).")


if __name__ == "__main__":
    main()
