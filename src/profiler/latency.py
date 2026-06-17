import gc
import threading
import time
from typing import Any, Dict, List, Optional

import psutil
import torch
from transformers import AutoTokenizer

try:
    import mlx.core as mx
except ImportError:
    mx = None


class MemoryMonitor:
    """Background thread to poll host RSS and GPU memory usage at high frequency."""

    def __init__(self, interval: float = 0.005):
        self.interval = interval
        self.peak_rss = 0.0
        self.peak_gpu = 0.0
        self.stop_event = threading.Event()
        self.thread = None
        self.process = psutil.Process()

    def _monitor(self):
        while not self.stop_event.is_set():
            # Track host Resident Set Size (RSS) memory
            try:
                rss = self.process.memory_info().rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
            except Exception:
                pass

            # Track PyTorch MPS allocated memory
            try:
                if torch.backends.mps.is_available():
                    gpu = torch.mps.current_allocated_memory()
                    if gpu > self.peak_gpu:
                        self.peak_gpu = gpu
            except Exception:
                pass

            time.sleep(self.interval)

    def start(self):
        self.peak_rss = self.process.memory_info().rss
        self.peak_gpu = 0.0
        if torch.backends.mps.is_available():
            try:
                self.peak_gpu = torch.mps.current_allocated_memory()
            except Exception:
                pass
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self) -> tuple[float, float]:
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        return self.peak_rss, self.peak_gpu


def get_dummy_prompt(tokenizer, target_len: int) -> str:
    """Generates a repeating text prompt that tokenizes to exactly target_len tokens."""
    try:
        # Get token ID for a simple, common word like " hello"
        token_ids = tokenizer.encode(" hello", add_special_tokens=False)
        if not token_ids:
            token_ids = tokenizer.encode("hello", add_special_tokens=False)
        token_id = token_ids[0]
    except Exception:
        # Fallback if encoding parameters are different or fail
        token_id = 9999

    # Iteratively adjust token count to account for BOS or any tokenizer-specific additions
    count = target_len
    for _ in range(5):
        ids = [token_id] * count
        text = tokenizer.decode(ids)
        encoded_ids = tokenizer.encode(text)
        current_len = len(encoded_ids)
        if current_len == target_len:
            return text
        count = count + (target_len - current_len)
        if count <= 0:
            count = 1

    return tokenizer.decode([token_id] * target_len)


def profile_latency(runner, model_id: str, prompt_lengths: List[int] = [128, 256, 512, 1024, 2048, 4096], decode_budget: int = 64) -> Dict[int, Dict[str, Any]]:
    """Runs latency and memory profiling on the loaded model runner.

    Evaluates TTFT, decode latency per token, and peak UMA memory usage.
    """
    # Load tokenizer to accurately construct prompt lengths
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        # Fallback to runner's tokenizer if available
        tokenizer = getattr(runner, "tokenizer", None)

    if tokenizer is None:
        raise ValueError(
            f"Could not initialize tokenizer for model_id: {model_id}"
        )

    # Clean cache and run warm-up to initialize kernels and caching graph
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if mx is not None:
        mx.clear_cache()

    print("Running warm-up generation step...")
    warmup_prompt = get_dummy_prompt(tokenizer, 128)
    runner.generate(warmup_prompt, max_tokens=16)

    results = {}

    for length in prompt_lengths:
        print(f"Profiling prompt length: {length}...")
        prompt = get_dummy_prompt(tokenizer, length)

        # Clear cache and reset hardware counters
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if mx is not None:
            mx.clear_cache()
            mx.reset_peak_memory()

        # Start background memory monitor thread
        monitor = MemoryMonitor()
        monitor.start()

        # Execute generation
        gen_out = runner.generate(prompt, max_tokens=decode_budget)

        # Stop monitoring and collect peak values
        peak_rss, peak_gpu_torch = monitor.stop()

        # Retrieve MLX hardware peak memory if using MLX
        peak_gpu_mlx = mx.get_peak_memory() if mx is not None else 0.0

        # Convert peak memories to Gigabytes (GB)
        peak_rss_gb = peak_rss / 1e9
        peak_gpu_gb = max(peak_gpu_torch, peak_gpu_mlx) / 1e9

        results[length] = {
            "ttft": gen_out["ttft"],
            "latencies": gen_out["latencies"],
            "peak_rss_gb": peak_rss_gb,
            "peak_gpu_gb": peak_gpu_gb,
            "text": gen_out["text"],
        }
        print(
            f"  TTFT: {gen_out['ttft']:.4f}s | Peak RSS: {peak_rss_gb:.2f} GB | Peak GPU: {peak_gpu_gb:.2f} GB"
        )

    return results
