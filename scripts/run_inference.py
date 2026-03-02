#!/usr/bin/env python3
"""
Run MLX inference for multilingual self-consistency experiments.

Each run processes one (model, dataset, language) combination and writes
N sampled responses per example to a JSONL file. Score the outputs
afterwards with run_scoring.py.

Usage
-----
Single run (10 samples, temperature 0.7):
    conda run -n scr python scripts/run_inference.py \\
        --model qwen2_5_32b \\
        --dataset mgsm \\
        --language en

Smoke-test with 5 examples and 3 samples:
    conda run -n scr python scripts/run_inference.py \\
        --model phi4_14b \\
        --dataset mgsm \\
        --language en \\
        --n-samples 3 \\
        --max-examples 5

Resume an interrupted run:
    conda run -n scr python scripts/run_inference.py \\
        --model qwen2_5_32b \\
        --dataset mgsm \\
        --language en \\
        --run-dir experiments/runs/qwen2_5_32b_mgsm_en_20260227_120000

Validate inputs without running:
    conda run -n scr python scripts/run_inference.py \\
        --model qwen2_5_32b --dataset mgsm --language en --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import yaml

from src.inference.runner import DEFAULT_CORPUS, DEFAULT_MODELS_CONFIG, InferenceRunner


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run MLX inference for multilingual self-consistency experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model", required=True,
        help="Model key from configs/models.yaml (e.g. qwen2_5_32b)",
    )
    p.add_argument(
        "--dataset", required=True,
        help="Dataset key from configs/benchmarks.yaml (e.g. mgsm)",
    )
    p.add_argument(
        "--language", required=True,
        help="Language code matching the corpus (e.g. en, zh, fr)",
    )
    p.add_argument(
        "--n-samples", type=int, default=10,
        help="Self-consistency samples per example (default: 10)",
    )
    p.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    p.add_argument(
        "--max-tokens", type=int, default=None,
        help="Override max new tokens per response (default: auto per task type)",
    )
    p.add_argument(
        "--max-examples", type=int, default=None,
        help="Cap number of examples — useful for smoke tests",
    )
    p.add_argument(
        "--run-dir", type=str, default=None,
        help="Output directory (auto-named if not set; pass existing dir to resume)",
    )
    p.add_argument(
        "--corpus", type=str, default=str(DEFAULT_CORPUS),
        help=f"Path to corpus.parquet (default: {DEFAULT_CORPUS})",
    )
    p.add_argument(
        "--models-config", type=str, default=str(DEFAULT_MODELS_CONFIG),
        help=f"Path to models.yaml (default: {DEFAULT_MODELS_CONFIG})",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate inputs and exit without running inference",
    )
    return p.parse_args()


def _validate(args: argparse.Namespace) -> None:
    """Check model/dataset/language before loading any model. Exits on error."""
    models_config = Path(args.models_config)

    with open(models_config) as f:
        models_cfg = yaml.safe_load(f)
    models = models_cfg.get("models", {})

    if args.model not in models:
        print(f"ERROR: model {args.model!r} not in {models_config}. Available: {sorted(models)}")
        sys.exit(1)

    entry = models[args.model]
    if entry.get("requires_conversion"):
        print(
            f"ERROR: {args.model} has requires_conversion=true. "
            "Convert it first with mlx_lm.convert before running inference."
        )
        sys.exit(1)
    if entry.get("requires_offloading"):
        print(
            f"ERROR: {args.model} has requires_offloading=true "
            f"(estimated {entry.get('estimated_ram_gb')} GB). "
            "Cannot cold-load into 128 GB unified memory."
        )
        sys.exit(1)

    benchmarks_config = models_config.parent / "benchmarks.yaml"
    with open(benchmarks_config) as f:
        bench_cfg = yaml.safe_load(f)
    benchmarks = bench_cfg.get("benchmarks", {})

    if args.dataset not in benchmarks:
        print(f"ERROR: dataset {args.dataset!r} not in {benchmarks_config}. Available: {sorted(benchmarks)}")
        sys.exit(1)

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"ERROR: corpus not found at {corpus_path}")
        sys.exit(1)

    print(f"[validate] model      = {args.model!r}")
    print(f"[validate] dataset    = {args.dataset!r}")
    print(f"[validate] language   = {args.language!r}")
    print(f"[validate] n_samples  = {args.n_samples}")
    print(f"[validate] temperature= {args.temperature}")
    if args.max_examples:
        print(f"[validate] max_examples = {args.max_examples}")
    print(f"[validate] corpus     = {corpus_path}  ({corpus_path.stat().st_size // 1_048_576} MB)")
    print("[validate] OK")


def main() -> None:
    args = _parse_args()
    _validate(args)

    if args.dry_run:
        print("[dry-run] Validation passed. Exiting without running inference.")
        return

    runner = InferenceRunner(
        model_key=args.model,
        config_path=Path(args.models_config),
        n_samples=args.n_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    run_dir = runner.run(
        dataset=args.dataset,
        language=args.language,
        corpus_path=Path(args.corpus),
        run_dir=Path(args.run_dir) if args.run_dir else None,
        max_examples=args.max_examples,
    )

    runner.unload()

    print(f"\n[done] Results saved to:  {run_dir}")
    print(f"[done] Score with:        conda run -n scr python scripts/run_scoring.py --run-dir {run_dir}")


if __name__ == "__main__":
    main()
