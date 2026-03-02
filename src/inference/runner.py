"""
MLX inference runner for multilingual self-consistency experiments.

Loads a model once, generates N independent responses per corpus example,
and streams results to a JSONL file for later scoring.

Public API
----------
InferenceRunner(model_key, config_path, n_samples, temperature, max_tokens)
  .load()
  .run(dataset, language, corpus_path, run_dir, max_examples) -> Path
  .unload()
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from src.inference.prompts import build_prompt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO_ROOT / "data" / "corpus.parquet"
DEFAULT_MODELS_CONFIG = REPO_ROOT / "configs" / "models.yaml"

# Sensible max-token budgets per task type
_MAX_TOKENS: dict[str, int] = {
    "math_reasoning": 512,
    "commonsense":    64,
    "nli":            32,
    "mcqa":           16,
    "extractive_qa":  128,
}


class InferenceRunner:
    """Run inference for one model over a (dataset, language) corpus slice.

    Parameters
    ----------
    model_key : str
        Key in configs/models.yaml (e.g. "qwen2_5_32b").
    config_path : Path
        Path to models.yaml.
    n_samples : int
        Number of independent samples per example (self-consistency width).
    temperature : float
        Sampling temperature. Should be > 0 for self-consistency.
    max_tokens : int or None
        Override max new tokens per response; None uses per-task defaults.
    """

    def __init__(
        self,
        model_key: str,
        config_path: Path = DEFAULT_MODELS_CONFIG,
        n_samples: int = 10,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        self.model_key = model_key
        self.n_samples = n_samples
        self.temperature = temperature
        self.max_tokens = max_tokens

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        models = cfg.get("models", {})
        if model_key not in models:
            raise ValueError(
                f"Model key {model_key!r} not found in {config_path}. "
                f"Available: {sorted(models)}"
            )

        self.model_cfg = models[model_key]
        self.mlx_id = self.model_cfg["mlx_id"]

        self._model = None
        self._tokenizer = None

    # ── Model lifecycle ────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the MLX model and tokenizer into unified memory."""
        if self._model is not None:
            return
        from mlx_lm import load as mlx_load
        print(f"[runner] Loading {self.mlx_id} ...")
        self._model, self._tokenizer = mlx_load(self.mlx_id)
        print("[runner] Model loaded.")

    def unload(self) -> None:
        """Release model from unified memory."""
        try:
            del self._model
            del self._tokenizer
        except AttributeError:
            pass
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass

    # ── Generation ─────────────────────────────────────────────────────────────

    def _generate_one(self, prompt_text: str, max_tokens: int) -> str:
        """Generate a single response string for the given prompt."""
        from mlx_lm import generate as mlx_generate

        messages = [{"role": "user", "content": prompt_text}]
        try:
            formatted = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            formatted = prompt_text

        response = mlx_generate(
            self._model,
            self._tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            temp=self.temperature,
            verbose=False,
        )
        return response.strip()

    # ── Main run loop ──────────────────────────────────────────────────────────

    def run(
        self,
        dataset: str,
        language: str,
        corpus_path: Path = DEFAULT_CORPUS,
        run_dir: Optional[Path] = None,
        max_examples: Optional[int] = None,
    ) -> Path:
        """Run inference for one (dataset, language) slice of the corpus.

        Parameters
        ----------
        dataset : str
            Benchmark key (e.g. "mgsm").
        language : str
            Language code (e.g. "en").
        corpus_path : Path
            Path to corpus.parquet.
        run_dir : Path or None
            Output directory. Auto-named as
            ``experiments/runs/{model}_{dataset}_{lang}_{timestamp}`` if None.
        max_examples : int or None
            Cap the number of examples processed (useful for smoke tests).

        Returns
        -------
        Path : The run directory containing raw_outputs.jsonl and run_config.json.
        """
        # ── Output directory ──────────────────────────────────────────────────
        if run_dir is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            run_name = f"{self.model_key}_{dataset}_{language}_{ts}"
            run_dir = REPO_ROOT / "experiments" / "runs" / run_name
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        output_path = run_dir / "raw_outputs.jsonl"
        config_path = run_dir / "run_config.json"

        # ── Persist run config ────────────────────────────────────────────────
        run_config = {
            "model_key": self.model_key,
            "mlx_id": self.mlx_id,
            "dataset": dataset,
            "language": language,
            "n_samples": self.n_samples,
            "temperature": self.temperature,
            "max_tokens_override": self.max_tokens,
            "corpus_path": str(corpus_path),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        config_path.write_text(json.dumps(run_config, indent=2))

        # ── Load corpus slice ─────────────────────────────────────────────────
        df = pd.read_parquet(corpus_path)
        mask = (df["dataset"] == dataset) & (df["language"] == language)
        subset = df[mask].reset_index(drop=True)

        if max_examples is not None:
            subset = subset.head(max_examples)

        if len(subset) == 0:
            raise ValueError(
                f"No rows found for dataset={dataset!r}, language={language!r} "
                f"in {corpus_path}."
            )

        # ── Resume: skip already-done IDs ─────────────────────────────────────
        completed_ids: set[str] = set()
        if output_path.exists():
            with open(output_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            completed_ids.add(json.loads(line)["id"])
                        except (json.JSONDecodeError, KeyError):
                            pass
            if completed_ids:
                print(f"[runner] Resume: {len(completed_ids)} examples already done, skipping.")

        # ── Load model ────────────────────────────────────────────────────────
        self.load()

        # ── Generate ──────────────────────────────────────────────────────────
        task_type = subset["task_type"].iloc[0]
        max_tokens = self.max_tokens or _MAX_TOKENS.get(task_type, 256)
        total = len(subset)
        new_count = 0

        with open(output_path, "a") as out_f:
            for _, row in subset.iterrows():
                row_id = str(row["id"])
                if row_id in completed_ids:
                    continue

                prompt_text = build_prompt(row.to_dict())
                responses: list[str] = []
                for s in range(self.n_samples):
                    resp = self._generate_one(prompt_text, max_tokens)
                    responses.append(resp)
                    done_samples = s + 1
                    done_examples = len(completed_ids) + new_count
                    print(
                        f"[runner] [{done_examples + 1}/{total}] "
                        f"sample {done_samples}/{self.n_samples}  id={row_id}",
                        end="\r",
                        flush=True,
                    )

                label_val = row["label"]
                record = {
                    "id": row_id,
                    "model": self.model_key,
                    "dataset": dataset,
                    "language": language,
                    "task_type": task_type,
                    "prompt": prompt_text,
                    "responses": responses,
                    "label": str(label_val) if pd.notna(label_val) else "",
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                new_count += 1

        print(
            f"\n[runner] Done. "
            f"{new_count} new examples written → {output_path}"
        )
        return run_dir
