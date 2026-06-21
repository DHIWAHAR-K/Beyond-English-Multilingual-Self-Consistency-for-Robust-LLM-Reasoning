from unittest.mock import MagicMock, patch

import pytest

from src.core.mlc_runner import MLCRunner
from src.core.mlx_runner import MLXRunner
from src.core.torch_runner import TorchMPSRunner


@pytest.fixture
def mock_mlx_stream():
    class FakeResponse:
        def __init__(self, text):
            self.text = text

    with patch("src.core.mlx_runner.mlx_lm.load") as mock_load, patch(
        "src.core.mlx_runner.mlx_lm.stream_generate"
    ) as mock_stream, patch("src.core.mlx_runner.mx") as mock_mx:
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = None
        mock_tokenizer.encode.side_effect = lambda text, add_special_tokens=True: list(range(len(text)))
        mock_tokenizer.decode.return_value = "hello"

        mock_model = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer, {})
        mock_stream.return_value = [FakeResponse("a"), FakeResponse("b")]

        logits = MagicMock()
        logits.__sub__ = MagicMock(return_value=logits)
        mock_model.return_value = logits
        mock_mx.array.return_value = MagicMock()
        mock_mx.logsumexp.return_value = logits
        logits.__getitem__.return_value.item.return_value = -0.5

        yield mock_load, mock_stream


def test_mlx_runner_generate_schema(mock_mlx_stream):
    runner = MLXRunner()
    runner.load_model("test-model", "fp16")
    result = runner.generate("prompt", max_tokens=2)

    assert set(result.keys()) == {"text", "num_tokens", "latencies", "ttft"}
    assert result["num_tokens"] == 2
    assert len(result["latencies"]) == 1


def test_mlx_runner_get_logprobs_shape(mock_mlx_stream):
    runner = MLXRunner()
    runner.load_model("test-model", "fp16")
    logprobs = runner.get_logprobs("prompt", "completion")
    assert isinstance(logprobs, list)


@patch("src.core.torch_runner.AutoModelForCausalLM")
@patch("src.core.torch_runner.AutoTokenizer")
def test_torch_runner_generate_schema(mock_tokenizer_cls, mock_model_cls):
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda text, add_special_tokens=True: [1, 2, 3, 4]
    mock_tokenizer.decode.return_value = "answer"
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

    mock_logits = MagicMock()
    mock_logits.__truediv__ = MagicMock(return_value=mock_logits)
    mock_logits.size.return_value = 1

    mock_outputs = MagicMock()
    mock_outputs.logits = MagicMock()
    mock_outputs.logits.__getitem__.return_value = mock_logits
    mock_outputs.past_key_values = None

    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([MagicMock(device="cpu")])
    mock_model.eval.return_value = None
    mock_model.return_value = mock_outputs
    mock_model_cls.from_pretrained.return_value = mock_model

    with patch("torch.backends.mps.is_available", return_value=False), patch(
        "torch.argmax", return_value=MagicMock(item=lambda: 5, keepdim=lambda **_: MagicMock())
    ), patch("torch.multinomial", return_value=MagicMock(item=lambda: 5, keepdim=lambda **_: MagicMock())), patch(
        "torch.softmax", return_value=MagicMock()
    ), patch(
        "torch.tensor", return_value=MagicMock(to=lambda _: MagicMock())
    ), patch(
        "torch.no_grad", return_value=MagicMock()
    ):
        runner = TorchMPSRunner()
        runner.load_model("test-model", "fp16")
        result = runner.generate("prompt", max_tokens=2)

    assert set(result.keys()) == {"text", "num_tokens", "latencies", "ttft"}


def test_mlc_runner_get_logprobs_not_implemented():
    runner = MLCRunner()
    with pytest.raises(NotImplementedError):
        runner.get_logprobs("prompt", "completion")
