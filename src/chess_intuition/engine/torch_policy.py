"""PyTorch adapter for inference; legal filtering remains in the engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class TorchPolicy:
    def __init__(self, model: Any, device: Any, value_trained: bool = False) -> None:
        self.model = model
        self.device = device
        self.value_trained = value_trained
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: str | None = None) -> "TorchPolicy":
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - requires optional extra
            raise ImportError("Install the [train] dependencies to load a model.") from exc

        from ..model import ChessResNet

        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError("No trained checkpoint at " + str(path))
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        target = torch.device(device)
        checkpoint = torch.load(path, map_location=target, weights_only=True)
        config = checkpoint.get("model_config", {})
        model = ChessResNet(
            blocks=int(config.get("blocks", 2)),
            channels=int(config.get("channels", 16)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(target)
        return cls(model, target, value_trained=bool(checkpoint.get("value_trained", False)))

    def predict_policy(self, features: np.ndarray) -> np.ndarray:
        import torch

        tensor = torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0)
        tensor = tensor.to(self.device)
        with torch.inference_mode():
            policy_logits, _ = self.model(tensor)
        return policy_logits[0].detach().to("cpu").numpy()

    def predict_value(self, features: np.ndarray) -> float:
        """Return the current value-head output (not trained by puzzle-only runs)."""
        import torch

        tensor = torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0)
        tensor = tensor.to(self.device)
        with torch.inference_mode():
            _, values = self.model(tensor)
        return float(values[0].detach().to("cpu").item())

