import sys
from pathlib import Path

import pytest

# Ensure the repo root is on sys.path so `src` is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_corpus  # noqa: E402


@pytest.fixture(scope="session")
def corpus_df():
    return load_corpus(corpus_path=str(REPO_ROOT / "data" / "corpus.parquet"))


@pytest.fixture(scope="session")
def one_per_task(corpus_df):
    task_types = ["math_reasoning", "commonsense", "nli", "mcqa", "extractive_qa"]
    return {tt: corpus_df[corpus_df["task_type"] == tt].iloc[0] for tt in task_types}
