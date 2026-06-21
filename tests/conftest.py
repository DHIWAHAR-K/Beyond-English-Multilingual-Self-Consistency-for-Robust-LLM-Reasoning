import pytest


@pytest.fixture
def tmp_results_dir(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    return results


@pytest.fixture
def sample_latency_results():
    return {
        128: {"ttft": 0.12, "latencies": [0.01] * 64, "peak_rss_gb": 4.5, "peak_gpu_gb": 3.2},
        256: {"ttft": 0.18, "latencies": [0.011] * 64, "peak_rss_gb": 5.0, "peak_gpu_gb": 3.5},
    }
