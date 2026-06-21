from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.mlc_runner import MLCRunner
from src.stats.bootstrap import bootstrap_ci, meets_significance_threshold
from src.tasks.base import BaseTask


def compute_compression_parity(
    target_logprobs: List[float],
    reference_logprobs: List[float],
) -> float:
    """Compute CP = -sum(log P(T|S)) / -sum(log P(E|S))."""
    target_nll = -float(np.sum(target_logprobs))
    reference_nll = -float(np.sum(reference_logprobs))
    if reference_nll == 0:
        return 0.0
    return target_nll / reference_nll


def negative_log_likelihood(logprobs: List[float]) -> float:
    return -float(np.sum(logprobs))


class ParityTask(BaseTask):
    """Compression Parity evaluation on translation samples."""

    def __init__(
        self,
        runner,
        output_dir: Path,
        model_slug: str,
        precision: str,
        cache_dir: Path = Path("data/cache"),
    ):
        super().__init__(runner, output_dir, model_slug, precision)
        self.cache_dir = Path(cache_dir)

    def _load_samples(self, dataset_name: str, split: str, max_samples: int) -> List[Dict[str, str]]:
        from datasets import load_dataset

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(dataset_name, split=split, cache_dir=str(self.cache_dir))

        samples: List[Dict[str, str]] = []
        for row in dataset:
            source = row.get("source") or row.get("sentence_eng_Latn") or row.get("en")
            target = row.get("target") or row.get("sentence_deu_Latn") or row.get("de")
            reference_en = row.get("reference_en") or row.get("en") or source
            if not source or not target:
                continue
            samples.append(
                {
                    "source": str(source),
                    "target": str(target),
                    "reference_en": str(reference_en),
                }
            )
            if len(samples) >= max_samples:
                break
        return samples

    def run(
        self,
        dataset_name: str = "google/fleurs",
        split: str = "train[:50]",
        max_samples: int = 50,
        **kwargs,
    ) -> Dict[str, Any]:
        if isinstance(self.runner, MLCRunner):
            payload = {
                "skipped": True,
                "reason": "MLC backend does not support token-level logprob extraction.",
            }
            self.save_results(payload, "metrics.json")
            return payload

        samples = self._load_samples(dataset_name, split, max_samples)
        per_sample_cp: List[float] = []
        records: List[Dict[str, Any]] = []

        for sample in samples:
            source = sample["source"]
            target = sample["target"]
            reference_en = sample["reference_en"]

            target_logprobs = self.runner.get_logprobs(source, target)
            reference_logprobs = self.runner.get_logprobs(source, reference_en)
            cp = compute_compression_parity(target_logprobs, reference_logprobs)

            per_sample_cp.append(cp)
            records.append(
                {
                    "source": source[:120],
                    "cp": cp,
                    "target_nll": negative_log_likelihood(target_logprobs),
                    "reference_nll": negative_log_likelihood(reference_logprobs),
                }
            )

        bootstrap = bootstrap_ci(per_sample_cp) if per_sample_cp else {}
        payload = {
            "dataset": dataset_name,
            "split": split,
            "n_samples": len(records),
            "mean_cp": bootstrap.get("mean", 0.0),
            "bootstrap": bootstrap,
            "significant": meets_significance_threshold(bootstrap.get("margin_pct", 100.0)),
            "records": records,
        }
        self.save_results(payload, "metrics.json")
        return payload
