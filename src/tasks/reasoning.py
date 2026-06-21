import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.base import BaseModelRunner
from src.core.mlx_runner import MLXRunner
from src.stats.bootstrap import bootstrap_ci, meets_significance_threshold
from src.tasks.base import BaseTask

CLC_LANGUAGES = ["zh", "en", "bn", "es", "ru", "th"]

ANSWER_PATTERNS = [
    re.compile(r"(?:the answer is|answer:?)\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\\boxed\{([+-]?\d+(?:\.\d+)?)\}"),
    re.compile(r"(?:final answer|result)\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE),
]


def extract_answer_regex(cot_text: str) -> Optional[str]:
    """Extract numeric answer from chain-of-thought text using regex patterns."""
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(cot_text)
        if match:
            return match.group(1)

    numeric_tokens = re.findall(r"[+-]?\d+(?:\.\d+)?", cot_text)
    if numeric_tokens:
        return numeric_tokens[-1]
    return None


def extract_answer_fallback(cot_text: str, mlx_runner: BaseModelRunner) -> Optional[str]:
    """Parse answer using a lightweight local MLX model when regex fails."""
    prompt = (
        "Extract only the final numeric answer from the reasoning below. "
        "Reply with just the number.\n\n"
        f"{cot_text}"
    )
    response = mlx_runner.generate(prompt, max_tokens=16, temperature=0.0)
    return extract_answer_regex(response.get("text", ""))


def majority_vote(answers: List[Optional[str]]) -> Optional[str]:
    """Return deterministic majority vote over extracted answers."""
    valid = [answer for answer in answers if answer is not None]
    if not valid:
        return None
    counts = Counter(valid)
    return counts.most_common(1)[0][0]


def build_cot_prompt(question: str, language: str) -> str:
    language_names = {
        "zh": "Chinese",
        "en": "English",
        "bn": "Bengali",
        "es": "Spanish",
        "ru": "Russian",
        "th": "Thai",
    }
    lang_name = language_names.get(language, language)
    return (
        f"Solve the following math problem step by step in {lang_name}. "
        "End with a line: The answer is <number>.\n\n"
        f"Question: {question}"
    )


class ReasoningTask(BaseTask):
    """Cross-Lingual Consistency reasoning evaluation on MGSM."""

    def __init__(
        self,
        runner: BaseModelRunner,
        output_dir: Path,
        model_slug: str,
        precision: str,
        fallback_runner: Optional[BaseModelRunner] = None,
        cache_dir: Path = Path("data/cache"),
    ):
        super().__init__(runner, output_dir, model_slug, precision)
        self.fallback_runner = fallback_runner
        self.cache_dir = Path(cache_dir)

    def _load_mgsm(self, max_samples: int) -> List[Dict[str, Any]]:
        from datasets import load_dataset

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, Any]] = []

        for language in CLC_LANGUAGES:
            config = f"{language}"
            dataset = load_dataset(
                "juletxara/mgsm", config, split="test", cache_dir=str(self.cache_dir)
            )
            for item in dataset:
                rows.append(
                    {
                        "language": language,
                        "question": item["question"],
                        "answer": str(item["answer_number"]),
                    }
                )
                if len([r for r in rows if r["language"] == language]) >= max_samples:
                    break
        return rows

    def _extract_answer(self, cot_text: str) -> Optional[str]:
        answer = extract_answer_regex(cot_text)
        if answer is not None:
            return answer
        if self.fallback_runner is not None and isinstance(self.fallback_runner, MLXRunner):
            return extract_answer_fallback(cot_text, self.fallback_runner)
        return None

    def run(
        self,
        max_samples: int = 5,
        temperature: float = 0.5,
        baseline_temperature: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        temp = 0.0 if baseline_temperature else temperature
        samples = self._load_mgsm(max_samples=max_samples)

        grouped: Dict[str, List[Dict[str, Any]]] = {lang: [] for lang in CLC_LANGUAGES}
        for sample in samples:
            grouped[sample["language"]].append(sample)

        question_results: List[Dict[str, Any]] = []
        correct_flags: List[float] = []

        questions = {}
        for sample in samples:
            questions.setdefault(sample["question"], sample["answer"])

        for question, ground_truth in questions.items():
            per_language: Dict[str, Any] = {}
            extracted_answers: List[Optional[str]] = []

            for language in CLC_LANGUAGES:
                prompt = build_cot_prompt(question, language)
                generation = self.runner.generate(prompt, max_tokens=512, temperature=temp)
                answer = self._extract_answer(generation["text"])
                extracted_answers.append(answer)
                per_language[language] = {
                    "prompt": prompt,
                    "response": generation["text"],
                    "extracted_answer": answer,
                }

            consensus = majority_vote(extracted_answers)
            is_correct = consensus == ground_truth
            correct_flags.append(1.0 if is_correct else 0.0)
            question_results.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "consensus": consensus,
                    "correct": is_correct,
                    "languages": per_language,
                }
            )

        bootstrap = bootstrap_ci(correct_flags) if correct_flags else {}
        accuracy = bootstrap.get("mean", 0.0)
        payload = {
            "temperature": temp,
            "languages": CLC_LANGUAGES,
            "accuracy": accuracy,
            "bootstrap": bootstrap,
            "significant": meets_significance_threshold(bootstrap.get("margin_pct", 100.0)),
            "questions": question_results,
        }
        self.save_results(payload, "clc_results.json")
        return payload
