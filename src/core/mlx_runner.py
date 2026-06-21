import gc
import time
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import make_sampler
from mlx_lm.utils import quantize_model

from src.core.base import BaseModelRunner


class MLXRunner(BaseModelRunner):
    """MLX execution backend for running and evaluating models on Apple Silicon."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None

    def load_model(
        self, model_id: str, precision: str, quantization_path: Optional[str] = None
    ) -> None:
        """Loads model weights into memory under specified precision & quantization.

        If quantization_path is provided, it loads the model from that path.
        Otherwise, it loads the model from model_id and performs in-memory
        quantization if a quantized precision (int8, int4, int2) is specified.
        """
        # Clean up existing model and clear cache to free memory
        self.model = None
        self.tokenizer = None
        self.config = None
        gc.collect()
        mx.clear_cache()

        path_to_load = quantization_path if quantization_path else model_id

        # Load the model, tokenizer, and config using mlx_lm
        self.model, self.tokenizer, self.config = mlx_lm.load(path_to_load, return_config=True)

        # Apply in-memory quantization if quantization_path was not used and precision is quantized
        if not quantization_path:
            bits = {"int8": 8, "int4": 4, "int2": 2}.get(precision.lower())
            if bits is not None:
                self.model, self.config = quantize_model(
                    self.model, self.config, group_size=64, bits=bits
                )
                # Ensure quantized weights are evaluated and cache is cleared
                mx.eval(self.model.parameters())
                mx.clear_cache()

    def generate(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0
    ) -> Dict[str, Any]:
        """Generates text and tracks per-token latency and TTFT.

        Returns a dictionary containing:
        - "text": str (generated response)
        - "num_tokens": int (tokens generated)
        - "latencies": List[float] (wall-clock latency per generated token)
        - "ttft": float (time to first token in seconds)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model is not loaded. Call load_model first.")

        # Create the appropriate sampler
        sampler = make_sampler(temp=temperature, top_p=top_p)

        latencies = []
        tokens = []
        start_time = time.perf_counter()
        prev_time = start_time

        # Generate tokens step-by-step
        for response in mlx_lm.stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            now = time.perf_counter()
            latencies.append(now - prev_time)
            tokens.append(response.text)
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
        generated_text = "".join(tokens)

        return {
            "text": generated_text,
            "num_tokens": len(latencies),
            "latencies": decode_latencies,
            "ttft": ttft,
        }

    def get_logprobs(self, prompt: str, completion: str) -> List[float]:
        """Computes the log probabilities of each token in the completion,

        conditioned on the prompt.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model is not loaded. Call load_model first.")

        # Match tokenizer encoding settings used in stream_generate
        add_special_tokens = self.tokenizer.bos_token is None or not prompt.startswith(
            self.tokenizer.bos_token
        )

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        full_ids = self.tokenizer.encode(prompt + completion, add_special_tokens=add_special_tokens)

        prompt_len = len(prompt_ids)
        if prompt_len >= len(full_ids):
            return []

        # Run forward pass to get logits
        inputs = mx.array(full_ids)[None]
        logits = self.model(inputs)

        # Compute logprobs: log(softmax(logits))
        # shape of logits: (1, seq_len, vocab_size)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        # Extract log probability for completion tokens
        # Target token full_ids[i] is predicted at step i-1
        completion_logprobs = []
        for i in range(prompt_len, len(full_ids)):
            target_token_id = full_ids[i]
            lp = logprobs[0, i - 1, target_token_id].item()
            completion_logprobs.append(lp)

        return completion_logprobs
