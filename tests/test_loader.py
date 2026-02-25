"""Integration tests for src/data/loader.py — uses actual data/ parquet files."""

from pathlib import Path

import pytest

from src.data.loader import (
    CORPUS_COLUMNS,
    load_benchmark,
    load_corpus,
    load_split,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(REPO_ROOT / "data")
CORPUS_PATH = str(REPO_ROOT / "data" / "corpus.parquet")


# ── load_corpus ───────────────────────────────────────────────────────────────

def test_load_corpus_shape(corpus_df):
    assert corpus_df.shape == (129_253, 7)


def test_load_corpus_columns(corpus_df):
    assert list(corpus_df.columns) == CORPUS_COLUMNS


def test_load_corpus_task_types(corpus_df):
    expected = {"math_reasoning", "commonsense", "nli", "mcqa", "extractive_qa"}
    assert set(corpus_df["task_type"].unique()) == expected


def test_load_corpus_no_null_required_cols(corpus_df):
    # choices is intentionally null for open-ended tasks — skip it
    for col in CORPUS_COLUMNS:
        if col == "choices":
            continue
        assert corpus_df[col].isnull().sum() == 0, f"Unexpected nulls in column '{col}'"


def test_load_corpus_missing_file_raises():
    with pytest.raises(ValueError, match="not found"):
        load_corpus(corpus_path="/nonexistent/corpus.parquet")


# ── load_split ────────────────────────────────────────────────────────────────

def test_load_split_mgsm_en():
    df = load_split("mgsm", "en", data_dir=DATA_DIR)
    assert df.shape == (250, 2)
    assert set(df.columns) == {"question", "answer_number"}


def test_load_split_xcopa_id():
    df = load_split("xcopa", "id", data_dir=DATA_DIR)
    assert df.shape[0] == 500
    assert "premise" in df.columns
    assert "choice1" in df.columns
    assert "label" in df.columns


def test_load_split_xnli_en():
    df = load_split("xnli", "en", data_dir=DATA_DIR)
    assert df.shape[0] == 500
    assert set(df.columns) == {"premise", "hypothesis", "label"}


def test_load_split_tydiqa_english():
    df = load_split("tydiqa", "english", data_dir=DATA_DIR)
    assert df.shape[0] > 0
    assert "passage_text" in df.columns
    assert "question_text" in df.columns
    assert "answers" in df.columns


def test_load_split_missing_file_raises():
    with pytest.raises(ValueError, match="not found"):
        load_split("mgsm", "xx", data_dir=DATA_DIR)


def test_load_split_bad_dataset_raises():
    with pytest.raises(ValueError, match="not found"):
        load_split("nonexistent_dataset", "en", data_dir=DATA_DIR)


# ── load_benchmark ────────────────────────────────────────────────────────────

def test_load_benchmark_mgsm():
    splits = load_benchmark("mgsm", data_dir=DATA_DIR)
    assert len(splits) == 11
    expected_langs = {"en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn", "te"}
    assert set(splits.keys()) == expected_langs
    for lang, df in splits.items():
        assert df.shape == (250, 2), f"mgsm/{lang} unexpected shape {df.shape}"


def test_load_benchmark_xcopa():
    splits = load_benchmark("xcopa", data_dir=DATA_DIR)
    assert len(splits) == 11
    for df in splits.values():
        assert df.shape[0] == 500


def test_load_benchmark_mlqa():
    splits = load_benchmark("mlqa", data_dir=DATA_DIR)
    assert len(splits) == 7
    expected_langs = {"en", "ar", "de", "es", "hi", "vi", "zh"}
    assert set(splits.keys()) == expected_langs


def test_load_benchmark_missing_raises():
    with pytest.raises(ValueError, match="not found"):
        load_benchmark("nonexistent", data_dir=DATA_DIR)
