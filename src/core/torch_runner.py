import gc
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.base import BaseModelRunner


class TorchMPSRunner(BaseModelRunner):
    """PyTorch MPS execution backend for running models on Apple Silicon."""

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(self, model_id: str, precision: str, quantization_path: Optional[str] = None) -> None:
        """Loads model weights into memory under specified precision & quantization.

        If quantization_path is provided, it loads the model from that path.
        Otherwise, it loads the model from model_id. Dynamic quantization is not
        supported on MPS, so it requires pre-quantized paths for quantized precision.
        """
        # Clean up existing model and clear cache to free memory
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        path_to_load = quantization_path if quantization_path else model_id

        # Map precision to PyTorch dtype
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(precision.lower(), torch.float16)

        # Handle quantization requirements
        if not quantization_path and precision.lower() in ["int8", "int4", "int2"]:
            raise ValueError(
                f"Dynamic/on-the-fly quantization to '{precision}' is not supported on PyTorch MPS. "
                "Please run with a pre-quantized model by specifying the 'quantization_path' config."
            )

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(path_to_load)
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load model with correct device placement
        # For pre-quantized models (e.g. GPTQ/AWQ), we rely on device_map="auto" to map to MPS
        if quantization_path:
            self.model = AutoModelForCausalLM.from_pretrained(
                path_to_load,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                path_to_load,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            ).to(device)

        self.model.eval()

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0) -> Dict[str, Any]:
        """Generates text and tracks per-token latency and TTFT.

        Returns a dictionary containing:
        - "text": str (generated response)
        - "num_tokens": int (tokens generated)
        - "latencies": List[float] (wall-clock latency per generated token)
        - "ttft": float (time to first token in seconds)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model is not loaded. Call load_model first.")

        device = next(self.model.parameters()).device

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        latencies = []
        generated_ids = []
        past_key_values = None

        start_time = time.perf_counter()
        prev_time = start_time

        # Generate tokens step-by-step
        with torch.no_grad():
            for step in range(max_tokens):
                if step == 0:
                    # Prefill step
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                    )
                else:
                    # Decode step - only pass the last generated token
                    outputs = self.model(
                        input_ids=input_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                # Sample next token
                if temperature == 0.0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                        # Shift to keep first token above top_p
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        for idx in range(logits.size(0)):
                            to_remove = sorted_indices[idx][sorted_indices_to_remove[idx]]
                            logits[idx, to_remove] = -float("inf")

                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                now = time.perf_counter()
                latencies.append(now - prev_time)
                prev_time = now

                token_id = next_token.item()
                generated_ids.append(token_id)

                if token_id == self.tokenizer.eos_token_id:
                    break

                # Prepare input for next iteration
                input_ids = next_token
                attention_mask = None

        if not latencies:
            return {
                "text": "",
                "num_tokens": 0,
                "latencies": [],
                "ttft": 0.0,
            }

        ttft = latencies[0]
        decode_latencies = latencies[1:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "text": text,
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

        device = next(self.model.parameters()).device

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        full_ids = self.tokenizer.encode(prompt + completion, add_special_tokens=True)

        prompt_len = len(prompt_ids)
        if prompt_len >= len(full_ids):
            return []

        inputs = torch.tensor([full_ids]).to(device)
        with torch.no_grad():
            outputs = self.model(inputs)
            logits = outputs.logits
            logprobs = F.log_softmax(logits, dim=-1)

        completion_logprobs = []
        for i in range(prompt_len, len(full_ids)):
            target_token_id = full_ids[i]
            lp = logprobs[0, i - 1, target_token_id].item()
            completion_logprobs.append(lp)

        return completion_logprobs
