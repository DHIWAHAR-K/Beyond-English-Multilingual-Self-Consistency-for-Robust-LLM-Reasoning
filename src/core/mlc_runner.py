import gc
import time
from typing import Any, Dict, List, Optional

try:
    from mlc_llm import MLCEngine
except (ImportError, OSError):
    # Handle environment where mlc_llm is not installed or failed to load
    MLCEngine = None

from src.core.base import BaseModelRunner


class MLCRunner(BaseModelRunner):
    """MLC-LLM execution backend for running and profiling models using TVM on Apple Silicon."""

    def __init__(self):
        self.engine = None
        self.model_id = None

    def load_model(
        self, model_id: str, precision: str, quantization_path: Optional[str] = None
    ) -> None:
        """Loads model weights into memory under specified precision & quantization

        using MLCEngine.
        """
        if MLCEngine is None:
            raise ImportError(
                "MLC-LLM package could not be imported. Please ensure it is installed correctly "
                "with TVM dependencies."
            )

        # Cleanup existing engine
        if self.engine is not None:
            try:
                self.engine.terminate()
            except Exception:
                pass
            self.engine = None
        gc.collect()

        self.model_id = quantization_path if quantization_path else model_id

        # Initialize MLCEngine targeting Metal (default on macOS Apple Silicon)
        self.engine = MLCEngine(self.model_id, device="metal")

    def generate(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0
    ) -> Dict[str, Any]:
        """Generates text using the MLCEngine and profiles per-token latencies.

        Returns a dictionary containing:
        - "text": str (generated response)
        - "num_tokens": int (tokens generated)
        - "latencies": List[float] (wall-clock latency per generated token)
        - "ttft": float (time to first token in seconds)
        """
        if self.engine is None:
            raise ValueError("Model is not loaded. Call load_model first.")

        latencies = []
        tokens = []

        # Use ChatCompletions stream to measure TTFT and decode latencies
        messages = [{"role": "user", "content": prompt}]

        start_time = time.perf_counter()
        prev_time = start_time

        response = self.engine.chat.completions.create(
            messages=messages,
            model=self.model_id,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        for chunk in response:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    now = time.perf_counter()
                    latencies.append(now - prev_time)
                    tokens.append(content)
                    prev_time = now

        if not latencies:
            return {
                "text": "",
                "num_tokens": 0,
                "latencies": [],
                "ttft": 0.0,
            }

        ttft = latencies[0]
        decode_latencies = latencies[1:]
        text = "".join(tokens)

        return {
            "text": text,
            "num_tokens": len(latencies),
            "latencies": decode_latencies,
            "ttft": ttft,
        }

    def get_logprobs(self, prompt: str, completion: str) -> List[float]:
        """MLC-LLM backend abstracts decoding and does not support logprobs."""
        raise NotImplementedError(
            "MLC-LLM backend does not support token-level logprob extraction."
        )
