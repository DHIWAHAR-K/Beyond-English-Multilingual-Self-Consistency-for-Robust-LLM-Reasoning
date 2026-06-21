from pathlib import Path

from src.visualization.plots import plot_latency_cliff, plot_metric_bars, write_alt_text_md


def test_plot_latency_cliff_writes_png_and_md(tmp_path, sample_latency_results):
    out_path = tmp_path / "latency.png"
    plot_latency_cliff(sample_latency_results, out_path)

    assert out_path.exists()
    assert out_path.with_suffix(".md").exists()


def test_plot_metric_bars_writes_png_and_md(tmp_path):
    out_path = tmp_path / "metrics.png"
    metrics = {"chrF": 45.2, "COMET": 0.81}
    ci_bounds = {
        "chrF": {"mean": 45.2, "lower": 44.0, "upper": 46.4},
        "COMET": {"mean": 0.81, "lower": 0.79, "upper": 0.83},
    }
    plot_metric_bars(metrics, ci_bounds, out_path)

    assert out_path.exists()
    assert out_path.with_suffix(".md").exists()


def test_write_alt_text_md(tmp_path):
    png_path = tmp_path / "plot.png"
    png_path.write_bytes(b"fake")
    md_path = write_alt_text_md(
        png_path,
        {
            "visual_attributes": {"x_axis": "tokens"},
            "statistics": {"mean": 1.0},
            "conclusion": "Test conclusion",
        },
    )
    content = Path(md_path).read_text(encoding="utf-8")
    assert "Test conclusion" in content
