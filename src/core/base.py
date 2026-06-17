from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseModelRunner(ABC):
    """Abstract base class for all model runner backends (MLX, PyTorch MPS, MLC)."""
    @abstractmethod
    def load_model(self, model_id: str, precision: str, quantization_path: Optional[str] = None) -> None:
        """Load model weights into memory under specified precision and quantization"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0) -> Dict[str, Any]:
        """
        Generate text using the loaded model. Returns a dict with keys:
            - "text":       str         — generated response string
            - "num_tokens": int         — number of tokens generated
            - "latencies":  List[float] — wall-clock seconds per decoded token
            - "ttft":       float       — time to first token in seconds
        """

        pass
    
    @abstractmethod
    def get_logprobs(self, prompt: str, completion: str) -> List[float]:
        """Return the log probabilities of the completion"""
        pass


