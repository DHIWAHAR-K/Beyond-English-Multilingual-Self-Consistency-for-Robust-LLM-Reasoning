from unittest.mock import MagicMock, patch

import pytest

from src.tasks.translation import compute_chrf, compute_comet


def test_compute_chrf():
    hypotheses = ["hello world", "good morning"]
    references = ["hello world", "good morning"]
    score = compute_chrf(hypotheses, references)
    assert score > 0


@patch("src.tasks.translation.load_comet_model")
def test_compute_comet(mock_load):
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.system_score = 0.87
    mock_model.predict.return_value = mock_output
    mock_load.return_value = mock_model

    score = compute_comet(["hi"], ["hello"], ["say hi"])
    assert score == pytest.approx(0.87)
