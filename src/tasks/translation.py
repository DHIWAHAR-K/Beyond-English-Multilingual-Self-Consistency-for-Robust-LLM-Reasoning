from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import sacrebleu
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.stats.bootstrap import bootstrap_ci, meets_significance_threshold
from src.tasks.base import BaseTask

CalibrationMethod = Literal["autoround", "awq", "gptq"]

_COMET_MODEL = None


def load_comet_model():
    """Lazy-load COMET wmt22-comet-da model."""
    global _COMET_MODEL
    if _COMET_MODEL is None:
        from comet import download_model, load_from_checkpoint

        model_path = download_model("Unbabel/wmt22-comet-da")
        _COMET_MODEL = load_from_checkpoint(model_path)
    return _COMET_MODEL


def compute_chrf(hypotheses: List[str], references: List[str]) -> float:
    score = sacrebleu.corpus_chrf(hypotheses, [references])
    return float(score.score)


def compute_comet(hypotheses: List[str], references: List[str], sources: List[str]) -> float:
    model = load_comet_model()
    data = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(sources, hypotheses, references)]
    output = model.predict(data, batch_size=8, gpus=0)
    return float(output.system_score)


def calibrate_model(
    model_id: str,
    method: CalibrationMethod,
    calib_texts: List[str],
    lang: str,
) -> Dict[str, Any]:
    """Calibrate a model using target-language text on CPU/torch path.

    MPS does not support all quantization kernels; calibration always runs on CPU.
    """
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model.to(device)
    model.eval()

    if method == "autoround":
        from auto_round import AutoRoundConfig, AutoRoundQuantizer

        quantizer = AutoRoundQuantizer(model, tokenizer, AutoRoundConfig(device=str(device)))
        quantizer.quantize(calib_texts[:32])
        output_path = f"results/translation/calibrated/{lang}/autoround"
    elif method == "awq":
        from awq import AutoAWQForCausalLM

        awq_model = AutoAWQForCausalLM.from_pretrained(model_id)
        awq_model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128})
        output_path = f"results/translation/calibrated/{lang}/awq"
    elif method == "gptq":
        from auto_gptq import BaseQuantizeConfig, AutoGPTQForCausalLM

        quant_config = BaseQuantizeConfig(bits=4, group_size=128)
        gptq_model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config=quant_config)
        gptq_model.quantize(calib_texts[:32])
        output_path = f"results/translation/calibrated/{lang}/gptq"
    else:
        raise ValueError(f"Unsupported calibration method: {method}")

    Path(output_path).mkdir(parents=True, exist_ok=True)
    return {
        "method": method,
        "language": lang,
        "n_calibration_samples": min(len(calib_texts), 32),
        "output_path": output_path,
        "device": str(device),
    }


class TranslationTask(BaseTask):
    """Machine translation evaluation with chrF++ and COMET metrics."""

    def __init__(
        self,
        runner,
        output_dir: Path,
        model_slug: str,
        precision: str,
        cache_dir: Path = Path("data/cache"),
    ):
        super().__init__(runner, output_dir, model_slug, precision)
        self.cache_dir = Path(cache_dir)

    def _load_samples(
        self,
        dataset_name: str,
        config: Optional[str],
        split: str,
        max_samples: int,
    ) -> List[Dict[str, str]]:
        from datasets import load_dataset

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if config:
            dataset = load_dataset(dataset_name, config, split=split, cache_dir=str(self.cache_dir))
        else:
            dataset = load_dataset(dataset_name, split=split, cache_dir=str(self.cache_dir))

        samples: List[Dict[str, str]] = []
        for row in dataset:
            source = row.get("source") or row.get("en") or row.get("sentence_eng_Latn")
            reference = row.get("target") or row.get("de") or row.get("sentence_deu_Latn")
            if not source or not reference:
                continue
            samples.append({"source": str(source), "reference": str(reference)})
            if len(samples) >= max_samples:
                break
        return samples

    def _translate_batch(self, sources: List[str], target_lang: str) -> List[str]:
        hypotheses = []
        for source in sources:
            prompt = (
                f"Translate the following text into {target_lang}. "
                "Return only the translation.\n\n"
                f"{source}"
            )
            result = self.runner.generate(prompt, max_tokens=256, temperature=0.0)
            hypotheses.append(result["text"].strip())
        return hypotheses

    def run(
        self,
        dataset_name: str = "google/fleurs",
        config: Optional[str] = "de_de",
        split: str = "train[:20]",
        max_samples: int = 20,
        target_lang: str = "German",
        calibrate: Optional[CalibrationMethod] = None,
        calibration_lang: str = "de",
        model_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        samples = self._load_samples(dataset_name, config, split, max_samples)
        sources = [sample["source"] for sample in samples]
        references = [sample["reference"] for sample in samples]
        hypotheses = self._translate_batch(sources, target_lang)

        chrf_score = compute_chrf(hypotheses, references)
        per_sample_chrf = [
            float(sacrebleu.sentence_chrf(hyp, [ref]).score) for hyp, ref in zip(hypotheses, references)
        ]
        bootstrap = bootstrap_ci(per_sample_chrf)

        comet_score = compute_comet(hypotheses, references, sources)

        calibration_en = None
        calibration_target = None
        if calibrate and model_id:
            calibration_en = calibrate_model(model_id, calibrate, sources, lang="en")
            calibration_target = calibrate_model(
                model_id,
                calibrate,
                references,
                lang=calibration_lang,
            )

        payload = {
            "dataset": dataset_name,
            "config": config,
            "split": split,
            "target_lang": target_lang,
            "n_samples": len(samples),
            "chrf": chrf_score,
            "comet": comet_score,
            "bootstrap": bootstrap,
            "significant": meets_significance_threshold(bootstrap.get("margin_pct", 100.0)),
            "calibration": {
                "english": calibration_en,
                "target_language": calibration_target,
            },
            "samples": [
                {"source": src, "hypothesis": hyp, "reference": ref}
                for src, hyp, ref in zip(sources, hypotheses, references)
            ],
        }
        self.save_results(payload, "mt_scores.json")
        return payload
