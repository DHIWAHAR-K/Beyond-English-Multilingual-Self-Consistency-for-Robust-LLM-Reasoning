"""Integration tests for src/utils/corpus_stats.py."""

import pytest

from src.utils.corpus_stats import print_corpus_summary, verify_corpus_integrity


def test_verify_corpus_integrity_passes():
    """Full integrity check should complete without raising."""
    verify_corpus_integrity(data_dir="data/")


def test_print_corpus_summary_runs(capsys):
    """Summary should print without error and include expected labels."""
    print_corpus_summary(corpus_path="data/corpus.parquet")
    captured = capsys.readouterr()
    assert "mgsm" in captured.out
    assert "129,253" in captured.out
    assert "math_reasoning" in captured.out
    assert "extractive_qa" in captured.out


def test_verify_integrity_bad_path_raises():
    with pytest.raises((AssertionError, FileNotFoundError, Exception)):
        verify_corpus_integrity(data_dir="/nonexistent/path/")
