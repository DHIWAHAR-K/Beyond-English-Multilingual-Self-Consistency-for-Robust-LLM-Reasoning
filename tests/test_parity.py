from unittest.mock import MagicMock

import pytest

from src.core.mlc_runner import MLCRunner
from src.tasks.parity import ParityTask, compute_compression_parity


def test_compute_compression_parity():
    target = [-1.0, -2.0]
    reference = [-1.0, -1.0]
    assert compute_compression_parity(target, reference) == pytest.approx(1.5)


def test_parity_task_skips_mlc_runner(tmp_results_dir):
    runner = MLCRunner()
    task = ParityTask(
        runner=runner,
        output_dir=tmp_results_dir / "parity",
        model_slug="test-model",
        precision="int4",
    )
    result = task.run()
    assert result["skipped"] is True


def test_parity_task_computes_cp_with_mocked_logprobs(tmp_results_dir):
    runner = MagicMock()
    runner.get_logprobs.side_effect = [
        [-1.0, -2.0],
        [-1.0, -1.0],
    ]

    task = ParityTask(
        runner=runner,
        output_dir=tmp_results_dir / "parity",
        model_slug="test-model",
        precision="int4",
    )

    samples = [
        {"source": "Hello", "target": "Hallo", "reference_en": "Hello there"},
    ]
    cp_values = []
    for sample in samples:
        target_logprobs = runner.get_logprobs(sample["source"], sample["target"])
        reference_logprobs = runner.get_logprobs(sample["source"], sample["reference_en"])
        cp_values.append(compute_compression_parity(target_logprobs, reference_logprobs))

    assert cp_values[0] == pytest.approx(1.5)
