#!/usr/bin/env python3
"""
Download HuggingFace datasets to local Parquet files.

Saves each language split under:
    data/{dataset_name}/{language}/dataset.parquet

Usage:
    conda run -n scr python download_datasets.py
    conda run -n scr python download_datasets.py --dry-run

Requirements:
    datasets==2.19.0   (trust_remote_code still supported for script-based datasets)
    tqdm
    pyarrow
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset, get_dataset_split_names
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration — change DATA_DIR here if you want a different output root
# ---------------------------------------------------------------------------
DATA_DIR = Path("./data")

# ---------------------------------------------------------------------------
# Dataset registry
#
# config_template  : Python format string for the HF config name.
#                    Use "{lang}" as the placeholder.
#                    Set to None for datasets with no config (ML-NLI-26lang).
# trust_remote_code: True for datasets that still use a loading script
#                    (juletxara/mgsm, khalidalt/tydiqa-goldp, facebook/mlqa).
# subsample        : Keep only the first N rows after loading; None = keep all.
# ---------------------------------------------------------------------------
DATASETS = [
    {
        "name": "mgsm",
        "hf_id": "juletxara/mgsm",
        "languages": ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn", "te"],
        "split": "test",
        "subsample": None,
        "trust_remote_code": True,
        "config_template": "{lang}",
    },
    {
        "name": "xcopa",
        "hf_id": "cambridgeltl/xcopa",
        "languages": ["et", "ht", "id", "it", "qu", "sw", "ta", "th", "tr", "vi", "zh"],
        "split": "test",
        "subsample": None,
        "trust_remote_code": False,
        "config_template": "{lang}",
    },
    {
        "name": "xstorycloze",
        "hf_id": "juletxara/xstory_cloze",
        "languages": ["en", "zh", "es", "ar", "hi", "id", "te", "sw", "eu", "my", "ru"],
        "split": "eval",
        "subsample": 500,
        "trust_remote_code": False,
        "config_template": "{lang}",
    },
    {
        "name": "xnli",
        "hf_id": "facebook/xnli",
        "languages": ["en", "fr", "es", "de", "el", "bg", "ru", "tr", "ar", "vi", "th", "zh", "hi", "sw", "ur"],
        "split": "test",
        "subsample": 500,
        "trust_remote_code": False,
        "config_template": "{lang}",
    },
    {
        "name": "global_mmlu_lite",
        "hf_id": "CohereLabs/Global-MMLU-Lite",
        # Actual available configs (verified from HF): nl/pl/ro/ru/sv/tr/vi/uk do NOT exist
        "languages": ["ar", "bn", "cy", "de", "en", "es", "fr", "hi", "id", "it", "ja", "ko", "my", "pt", "sq", "sw", "yo", "zh"],
        "split": "test",
        "subsample": None,
        "trust_remote_code": False,
        "config_template": "{lang}",
    },
    {
        "name": "tydiqa",
        "hf_id": "khalidalt/tydiqa-goldp",
        "languages": ["english", "arabic", "bengali", "finnish", "indonesian", "japanese", "swahili", "korean", "russian", "telugu", "thai"],
        "split": "train",
        "subsample": None,
        "trust_remote_code": True,
        "config_template": "{lang}",
    },
    {
        "name": "mlqa",
        "hf_id": "facebook/mlqa",
        "languages": ["en", "ar", "de", "es", "hi", "vi", "zh"],
        "split": "test",
        "subsample": None,
        "trust_remote_code": True,
        "config_template": "mlqa.{lang}.{lang}",   # mlqa uses compound config names
    },
    {
        # No language config; all splits are <lang>_<source> (26 non-English languages).
        # We load the first available split and subsample 500 representative rows.
        "name": "ml_nli_26lang",
        "hf_id": "MoritzLaurer/multilingual-NLI-26lang-2mil7",
        "languages": ["all"],
        "split": None,          # resolved dynamically
        "subsample": 500,
        "trust_remote_code": False,
        "config_template": None,    # no config arg
    },
]


# ---------------------------------------------------------------------------
# Core loading logic
# ---------------------------------------------------------------------------

def _load(cfg: dict, lang: str):
    """Return a datasets.Dataset for one (config, language) pair."""
    kwargs = {"trust_remote_code": True} if cfg["trust_remote_code"] else {}

    if cfg["config_template"] is None:
        # ML-NLI-26lang: pick the first available split (ar_anli, 25k rows)
        # to avoid pulling 2.7 M rows when we only need 500.
        available = get_dataset_split_names(cfg["hf_id"])
        ds = load_dataset(cfg["hf_id"], split=available[0], **kwargs)
    else:
        config = cfg["config_template"].format(lang=lang)
        ds = load_dataset(cfg["hf_id"], config, split=cfg["split"], **kwargs)

    if cfg["subsample"] and len(ds) > cfg["subsample"]:
        ds = ds.select(range(cfg["subsample"]))

    return ds


# ---------------------------------------------------------------------------
# Per-entry processor (one language split)
# ---------------------------------------------------------------------------

def _process(cfg: dict, lang: str, dry_run: bool) -> dict:
    """Download (or skip) one split. Always returns a result dict."""
    name = cfg["name"]
    out_path = DATA_DIR / name / lang / "dataset.parquet"

    result = {
        "dataset": name,
        "language": lang,
        "path": str(out_path),
        "status": None,
        "examples": None,
        "size_mb": None,
        "error": None,
    }

    # ── Idempotency check ────────────────────────────────────────────────────
    if out_path.exists():
        size_mb = round(out_path.stat().st_size / 1_048_576, 3)
        result.update(status="skipped", size_mb=size_mb)
        tqdm.write(f"  ⏭  {name}/{lang}  — already exists ({size_mb:.2f} MB), skipping")
        return result

    # ── Dry-run shortcut ─────────────────────────────────────────────────────
    if dry_run:
        tqdm.write(f"  📋 [DRY RUN] {name}/{lang}  → {out_path}")
        result["status"] = "dry_run"
        return result

    # ── Download & save ──────────────────────────────────────────────────────
    try:
        tqdm.write(f"  ⬇  {name}/{lang}  downloading…")
        ds = _load(cfg, lang)
        n = len(ds)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_parquet(str(out_path))

        size_mb = round(out_path.stat().st_size / 1_048_576, 3)
        tqdm.write(f"     ✅ {n} examples → {out_path}  ({size_mb:.2f} MB)")
        result.update(status="success", examples=n, size_mb=size_mb)

    except Exception as exc:
        tqdm.write(f"     ❌ {name}/{lang}  — {exc}")
        result.update(status="failed", error=str(exc))

    return result


# ---------------------------------------------------------------------------
# Summary & manifest
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict]) -> None:
    bucket: dict[str, list] = {}
    for r in results:
        bucket.setdefault(r["status"], []).append(r)

    labels = {
        "success":  "✅ Downloaded",
        "skipped":  "⏭  Skipped (already existed)",
        "failed":   "❌ Failed",
        "dry_run":  "📋 Dry-run (not downloaded)",
    }

    print("\n" + "═" * 80)
    print("SUMMARY")
    print("═" * 80)

    for status in ("success", "skipped", "dry_run", "failed"):
        items = bucket.get(status, [])
        if not items:
            continue
        print(f"\n{labels.get(status, status)}  ({len(items)}):")
        for r in items:
            line = f"  {r['dataset']}/{r['language']}"
            if r["examples"] is not None:
                line += f"  {r['examples']:>6} examples"
            if r["size_mb"] is not None:
                line += f"  {r['size_mb']:.2f} MB"
            if r["error"]:
                line += f"  — {r['error'][:70]}"
            print(line)

    ok_count = len(bucket.get("success", [])) + len(bucket.get("skipped", []))
    fail_count = len(bucket.get("failed", []))
    print(f"\nTotal splits: {len(results)}  |  OK: {ok_count}  |  Failed: {fail_count}")


def _write_manifest(results: list[dict]) -> None:
    saved = [r for r in results if r["status"] in ("success", "skipped")]
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_files": len(saved),
        "files": saved,
    }
    manifest_path = DATA_DIR / "manifest.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n📋 Manifest written → {manifest_path}  ({len(saved)} files)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download multilingual HuggingFace datasets to Parquet files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading anything.",
    )
    args = parser.parse_args()

    # Expand all (dataset, language) pairs into a flat task list
    tasks: list[tuple[dict, str]] = [
        (cfg, lang)
        for cfg in DATASETS
        for lang in cfg["languages"]
    ]

    if args.dry_run:
        print(f"🔍 DRY RUN — {len(tasks)} splits would be downloaded.\n")

    results: list[dict] = []
    with tqdm(tasks, desc="Overall", unit="split", file=sys.stdout, dynamic_ncols=True) as pbar:
        for cfg, lang in pbar:
            pbar.set_postfix_str(f"{cfg['name']}/{lang}")
            results.append(_process(cfg, lang, dry_run=args.dry_run))

    _print_summary(results)

    if not args.dry_run:
        _write_manifest(results)


if __name__ == "__main__":
    main()
