"""Evaluate a saved policy once on the untouched test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..model import ChessResNet
from .train import choose_device, run_epoch


def evaluate_test(
    data_dir: str | Path,
    checkpoint_path: str | Path,
    batch_size: int = 256,
    device_name: str = "auto",
) -> dict:
    data_path = Path(data_dir)
    checkpoint_file = Path(checkpoint_path)
    manifest_path = data_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Missing processed data manifest: " + str(manifest_path))
    if not checkpoint_file.is_file():
        raise FileNotFoundError("Missing model checkpoint: " + str(checkpoint_file))
    if not list((data_path / "test").glob("part-*.npz")):
        raise FileNotFoundError("No shards found for the held-out test split.")

    device = choose_device(device_name)
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
    config = checkpoint["model_config"]
    model = ChessResNet(
        blocks=int(config["blocks"]),
        channels=int(config["channels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = run_epoch(
        model,
        data_path / "test",
        batch_size,
        np.random.default_rng(0),
        device,
        optimizer=None,
        split_name="test",
    )
    report = {
        "checkpoint": str(checkpoint_file.resolve()),
        "checkpoint_epoch": checkpoint.get("training", {}).get("epoch"),
        "device": str(device),
        "value_trained": bool(checkpoint.get("value_trained", False)),
        "test": metrics,
        "interpretation": "Prediction agreement across every recorded solver decision position in the held-out puzzle lines; not chess strength or Elo.",
    }
    destination = checkpoint_file.parent / "test_metrics.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--checkpoint", default="checkpoints/puzzle-policy.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = evaluate_test(args.data, args.checkpoint, args.batch_size, args.device)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

