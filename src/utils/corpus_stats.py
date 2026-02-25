"""
Corpus statistics and integrity verification utilities.

Usage (CLI)
-----------
    python -m src.utils.corpus_stats

Public API
----------
print_corpus_summary(corpus_path)    — formatted table of per-dataset stats
verify_corpus_integrity(data_dir)    — checks manifest + corpus shape
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_EXPECTED_ROWS = 129_253
_EXPECTED_COLS = 7
_CORPUS_COLUMNS = ["dataset", "language", "task_type", "id", "input", "choices", "label"]


def print_corpus_summary(corpus_path: str = "data/corpus.parquet") -> None:
    """Load corpus.parquet and print a formatted per-dataset statistics table."""
    path = Path(corpus_path)
    if not path.exists():
        print(f"ERROR: corpus file not found: {path}", file=sys.stderr)
        return

    df = pd.read_parquet(path, engine="pyarrow")

    # Per-dataset aggregation
    stats = (
        df.groupby(["dataset", "task_type"])
        .agg(
            languages=("language", "nunique"),
            total=("id", "count"),
        )
        .reset_index()
    )
    stats["approx_per_lang"] = (stats["total"] / stats["languages"]).round(0).astype(int)

    # Compute label_type from task_type
    label_map = {
        "math_reasoning": "numeric",
        "commonsense":    "binary (0/1)",
        "nli":            "three-way",
        "mcqa":           "ABCD",
        "extractive_qa":  "span",
    }
    stats["label_type"] = stats["task_type"].map(label_map)

    # Print header
    sep = "-" * 90
    col_w = [20, 18, 10, 12, 14, 14]
    headers = ["dataset", "task_type", "langs", "ex/lang", "total", "label_type"]
    header_line = "  ".join(f"{h:<{w}}" for h, w in zip(headers, col_w))
    print()
    print("=" * 90)
    print("  CORPUS SUMMARY")
    print("=" * 90)
    print(header_line)
    print(sep)

    grand_total = 0
    for _, row in stats.iterrows():
        grand_total += row["total"]
        line = "  ".join([
            f"{row['dataset']:<{col_w[0]}}",
            f"{row['task_type']:<{col_w[1]}}",
            f"{row['languages']:<{col_w[2]}}",
            f"{row['approx_per_lang']:<{col_w[3]}}",
            f"{row['total']:<{col_w[4]},}",
            f"{row['label_type']:<{col_w[5]}}",
        ])
        print(line)

    print(sep)
    print(f"  {'TOTAL':<{col_w[0]}}  {'':>{col_w[1]}}  "
          f"{'':>{col_w[2]}}  {'':>{col_w[3]}}  {grand_total:<{col_w[4]},}")
    print("=" * 90)
    print(f"\n  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    print()


def verify_corpus_integrity(data_dir: str = "data/") -> None:
    """Verify that every file in manifest.json exists on disk and that
    corpus.parquet has the expected shape.

    Raises
    ------
    AssertionError on the first integrity violation encountered.
    """
    base = Path(data_dir)
    manifest_path = base / "manifest.json"
    corpus_path = base / "corpus.parquet"

    print("\n=== CORPUS INTEGRITY CHECK ===\n")

    # ── 1. manifest.json exists ───────────────────────────────────────────────
    assert manifest_path.exists(), f"manifest.json not found: {manifest_path}"
    manifest = json.loads(manifest_path.read_text())
    total_files = manifest["total_files"]
    print(f"[OK] manifest.json found  ({total_files} file entries)")

    # ── 2. Every file in manifest exists on disk ──────────────────────────────
    missing: list[str] = []
    for entry in manifest["files"]:
        fpath = Path(entry["path"])
        if not fpath.exists():
            missing.append(str(fpath))

    if missing:
        for m in missing:
            print(f"[MISSING] {m}")
        raise AssertionError(
            f"{len(missing)} file(s) listed in manifest.json are missing on disk."
        )
    print(f"[OK] All {total_files} parquet splits exist on disk")

    # ── 3. corpus.parquet exists ──────────────────────────────────────────────
    assert corpus_path.exists(), f"corpus.parquet not found: {corpus_path}"
    print(f"[OK] corpus.parquet found  ({corpus_path.stat().st_size / 1_048_576:.1f} MB)")

    # ── 4. corpus.parquet shape ───────────────────────────────────────────────
    df = pd.read_parquet(corpus_path, engine="pyarrow")
    rows, cols = df.shape

    assert cols == _EXPECTED_COLS, (
        f"corpus.parquet has {cols} columns, expected {_EXPECTED_COLS}.\n"
        f"  Got: {list(df.columns)}"
    )
    print(f"[OK] {cols} columns: {list(df.columns)}")

    assert rows == _EXPECTED_ROWS, (
        f"corpus.parquet has {rows:,} rows, expected {_EXPECTED_ROWS:,}."
    )
    print(f"[OK] {rows:,} rows")

    print("\n=== ALL CHECKS PASSED ===\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_corpus_summary()
    verify_corpus_integrity()
