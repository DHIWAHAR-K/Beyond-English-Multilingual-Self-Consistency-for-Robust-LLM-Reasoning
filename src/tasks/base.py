import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.base import BaseModelRunner


class BaseTask(ABC):
    """Abstract base class for evaluation tasks."""

    def __init__(
        self,
        runner: BaseModelRunner,
        output_dir: Path,
        model_slug: str,
        precision: str,
    ):
        self.runner = runner
        self.output_dir = Path(output_dir)
        self.model_slug = model_slug
        self.precision = precision
        self.run_id = self._generate_run_id()

    @staticmethod
    def _generate_run_id() -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}_{uuid.uuid4().hex[:8]}"

    @property
    def task_output_dir(self) -> Path:
        return self.output_dir / self.model_slug / self.precision

    def save_results(self, payload: Dict[str, Any], filename: str) -> Path:
        """Persist task results as JSON under the task output directory."""
        self.task_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.task_output_dir / filename

        serializable = {
            "run_id": self.run_id,
            "model_slug": self.model_slug,
            "precision": self.precision,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2, default=str)

        return output_path

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """Execute the evaluation task and return result payload."""
        pass
