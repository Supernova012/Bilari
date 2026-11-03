"""Train the shared policy/value network from preprocessed puzzle policy labels."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Iterator, Tuple

import numpy as np

try:
    import torch
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - exercised in minimal installs
    raise ImportError("Install the [train] dependencies to run training.") from exc

from ..board import FEATURE_CHANNELS
from ..model import ChessResNet


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compact_batch_to_features(states: np.ndarray, meta: np.ndarray) -> np.ndarray:
    """Vectorized conversion of compact shard rows to [B,19,8,8] float inputs."""
    batch_size = states.shape[0]
    features = np.zeros((batch_size, FEATURE_CHANNELS, 8, 8), dtype=np.float32)
    board = states.reshape(batch_size, 8, 8)
    for code in range(1, 13):
        features[:, code - 1] = board == code
    for index in range(4):
        features[:, 12 + index] = meta[:, index, None, None]
    ep = meta[:, 4].astype(np.int64)
    rows = np.flatnonzero(ep < 64)
    if len(rows):
        ranks, files = np.divmod(ep[rows], 8)
        features[rows, 16, ranks, files] = 1.0
    features[:, 17] = meta[:, 6, None, None]
    features[:, 18] = meta[:, 5, None, None] / 100.0
    return features


def _batch_candidates(
    selected_rows: np.ndarray,
    policy_ids: np.ndarray,
    legal_offsets: np.ndarray,
    legal_actions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates_by_row = [
        legal_actions[int(legal_offsets[row]) : int(legal_offsets[row + 1])]
        for row in selected_rows
    ]
    width = max(len(row) for row in candidates_by_row)
    candidates = np.zeros((len(selected_rows), width), dtype=np.int64)
    mask = np.zeros((len(selected_rows), width), dtype=np.bool_)
    target_positions = np.empty(len(selected_rows), dtype=np.int64)
    for batch_index, (row, legal) in enumerate(zip(selected_rows, candidates_by_row)):
        if len(legal) == 0:
            raise ValueError("Preprocessed shard contains a position without legal moves.")
        candidates[batch_index, : len(legal)] = legal
        mask[batch_index, : len(legal)] = True
        matches = np.flatnonzero(legal == policy_ids[row])
        if len(matches) == 0:
            raise ValueError("Target policy action is absent from its legal move mask.")
        target_positions[batch_index] = int(matches[0])
    return candidates, mask, target_positions


def _iterate_split(
    split_dir: Path,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
    shuffle: bool,
    limit_examples: int | None = None,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    shard_paths = sorted(split_dir.glob("part-*.npz"))
    if shuffle:
        rng.shuffle(shard_paths)
    yielded = 0
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as archive:
            states = archive["states"]
            meta = archive["meta"]
            policy_ids = archive["policy"]
            offsets = archive["legal_offsets"]
            legal_actions = archive["legal_actions"]
            order = rng.permutation(len(policy_ids)) if shuffle else np.arange(len(policy_ids))
            for start in range(0, len(order), batch_size):
                selected = order[start : start + batch_size]
                if limit_examples is not None:
                    remaining = limit_examples - yielded
                    if remaining <= 0:
                        return
                    selected = selected[:remaining]
                if len(selected) == 0:
                    continue
                features = compact_batch_to_features(states[selected], meta[selected])
                candidates, mask, targets = _batch_candidates(
                    selected, policy_ids, offsets, legal_actions
                )
                yield (
                    torch.from_numpy(features).to(device),
                    torch.from_numpy(candidates).to(device),
                    torch.from_numpy(mask).to(device),
                    torch.from_numpy(targets).to(device),
                )
                yielded += len(selected)
                if limit_examples is not None and yielded >= limit_examples:
                    return


def run_epoch(
    model: ChessResNet,
    split_dir: Path,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    limit_examples: int | None = None,
    split_name: str = "data",
) -> dict:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    top1 = 0
    top5 = 0
    last_progress = time.perf_counter()
    examples_at_last_progress = 0
    grad_context = torch.enable_grad if training else torch.inference_mode
    with grad_context():
        for features, candidates, mask, targets in _iterate_split(
            split_dir,
            batch_size,
            rng,
            device,
            shuffle=training,
            limit_examples=limit_examples,
        ):
            if training:
                optimizer.zero_grad(set_to_none=True)
            policy_logits, _ = model(features)
            candidate_logits = policy_logits.gather(1, candidates)
            candidate_logits = candidate_logits.masked_fill(~mask, -torch.inf)
            loss = F.cross_entropy(candidate_logits, targets)
            if training:
                loss.backward()
                optimizer.step()

            count = int(features.shape[0])
            total_loss += float(loss.detach().item()) * count
            total_examples += count
            predicted = candidate_logits.argmax(dim=1)
            top1 += int((predicted == targets).sum().item())
            k = min(5, candidate_logits.shape[1])
            topk = candidate_logits.topk(k, dim=1).indices
            top5 += int((topk == targets[:, None]).any(dim=1).sum().item())
            now = time.perf_counter()
            if now - last_progress >= 30.0:
                elapsed = now - last_progress
                print(
                    json.dumps(
                        {
                            "split": split_name,
                            "examples_seen": total_examples,
                            "policy_loss_so_far": total_loss / total_examples,
                            "legal_top1_accuracy_so_far": top1 / total_examples,
                            "examples_per_second_window": (total_examples - examples_at_last_progress) / elapsed,
                        }
                    ),
                    flush=True,
                )
                last_progress = now
                examples_at_last_progress = total_examples
    if total_examples == 0:
        raise ValueError("No examples found in " + str(split_dir))
    return {
        "examples": total_examples,
        "policy_loss": total_loss / total_examples,
        "legal_top1_accuracy": top1 / total_examples,
        "legal_top5_accuracy": top5 / total_examples,
    }


def train(
    data_dir: str | Path,
    checkpoint_path: str | Path,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    blocks: int = 2,
    channels: int = 16,
    device_name: str = "auto",
    seed: int = 42,
    limit_examples: int | None = None,
    resume_from: str | Path | None = None,
    additional_epochs: int | None = None,
) -> dict:
    data_path = Path(data_dir)
    manifest_path = data_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run the preprocessing command first; missing " + str(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("format_version", 0)) != 2:
        raise ValueError("Unsupported preprocessed dataset format.")
    for split in ("train", "validation"):
        if not list((data_path / split).glob("part-*.npz")):
            raise FileNotFoundError("No shards found for split " + split)

    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device(device_name)
    model = ChessResNet(blocks=blocks, channels=channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    best_epoch = 0
    history = []
    start_epoch = 1
    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError("Resume checkpoint not found: " + str(resume_path))
        saved = torch.load(resume_path, map_location=device, weights_only=True)
        if int(saved.get("format_version", 0)) != 2:
            raise ValueError("Resume requires a version-2 all-moves checkpoint.")
        saved_config = saved.get("model_config", {})
        if saved_config != model.config:
            raise ValueError("Resume checkpoint model config does not match this training run.")
        model.load_state_dict(saved["model_state_dict"])
        saved_training = saved.get("training", {})
        if saved_training.get("dataset_examples") != manifest.get("training_examples"):
            raise ValueError("Resume checkpoint was trained on a different puzzle dataset.")
        start_epoch = int(saved_training.get("epoch", 0)) + 1
        best_accuracy = float(saved_training.get("validation_legal_top1_accuracy", -1.0))
        best_epoch = int(saved_training.get("best_checkpoint_epoch", start_epoch - 1))

        prior_history = saved_training.get("history")
        if isinstance(prior_history, list):
            history = prior_history
        else:
            prior_metrics = resume_path.parent / "training_metrics.json"
            if prior_metrics.is_file():
                try:
                    prior_report = json.loads(prior_metrics.read_text(encoding="utf-8"))
                    if prior_report.get("checkpoint_epoch", start_epoch - 1) == start_epoch - 1:
                        history = prior_report.get("history", [])
                except (json.JSONDecodeError, OSError):
                    history = []

        if "optimizer_state_dict" in saved and "scheduler_state_dict" in saved:
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            scheduler.load_state_dict(saved["scheduler_state_dict"])
        else:
            # The epoch-1 checkpoint predates optimizer-state saving. Continue its
            # cosine learning-rate schedule with fresh AdamW moments.
            completed_epochs = start_epoch - 1
            resumed_lr = learning_rate * 0.5 * (
                1.0 + math.cos(math.pi * completed_epochs / epochs)
            )
            for group in optimizer.param_groups:
                group["lr"] = resumed_lr
            scheduler.last_epoch = completed_epochs

    if start_epoch > epochs:
        raise ValueError(
            "Resume checkpoint is already at epoch %d, beyond the requested total of %d."
            % (start_epoch - 1, epochs)
        )
    if additional_epochs is not None and additional_epochs < 1:
        raise ValueError("additional_epochs must be positive when provided")
    end_epoch = epochs if additional_epochs is None else min(epochs, start_epoch + additional_epochs - 1)

    seed_for_run = seed + start_epoch - 1
    random.seed(seed_for_run)
    np.random.seed(seed_for_run)
    torch.manual_seed(seed_for_run)
    rng = np.random.default_rng(seed_for_run)

    for epoch in range(start_epoch, end_epoch + 1):
        started = time.perf_counter()
        learning_rate_used = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(
            model,
            data_path / "train",
            batch_size,
            rng,
            device,
            optimizer=optimizer,
            limit_examples=limit_examples,
            split_name="train",
        )
        validation_metrics = run_epoch(
            model,
            data_path / "validation",
            batch_size,
            rng,
            device,
            optimizer=None,
            limit_examples=limit_examples,
            split_name="validation",
        )
        scheduler.step()
        epoch_metrics = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "learning_rate": learning_rate_used,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics), flush=True)
        is_best = validation_metrics["legal_top1_accuracy"] > best_accuracy
        if is_best:
            best_accuracy = validation_metrics["legal_top1_accuracy"]
            best_epoch = epoch
        checkpoint = {
            "format_version": 2,
            "model_config": model.config,
            "model_state_dict": model.state_dict(),
            "value_trained": False,
            "training": {
                "dataset_manifest": str(manifest_path.resolve()),
                "dataset_puzzles": manifest.get("puzzle_rows_kept"),
                "dataset_examples": manifest.get("training_examples"),
                "policy_objective": "legal-masked cross-entropy on every solver decision in each recorded puzzle line",
                "value_objective": None,
                "epoch": epoch,
                "validation_legal_top1_accuracy": best_accuracy,
                "best_checkpoint_epoch": best_epoch,
                "seed": seed,
                "history": history,
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        last_path = destination.with_name(destination.stem + ".last" + destination.suffix)
        for save_path in (last_path, destination) if is_best else (last_path,):
            temporary = save_path.with_suffix(save_path.suffix + ".tmp")
            torch.save(checkpoint, temporary)
            os.replace(temporary, save_path)

    report = {
        "device": str(device),
        "checkpoint": str(destination.resolve()),
        "checkpoint_epoch": best_epoch,
        "dataset_puzzles": manifest.get("puzzle_rows_kept"),
        "dataset_examples": manifest.get("training_examples"),
        "value_trained": False,
        "completed_through_epoch": end_epoch,
        "note": "Each example is a solver-to-move position after applying the opponent trigger and any intervening opponent replies. Puzzle records have no position-value labels; the value head is intentionally left untrained.",
        "history": history,
    }
    metrics_path = destination.parent / "training_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--checkpoint", default="checkpoints/puzzle-policy.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-examples", type=int, help="Optional cap per split for a quick smoke run")
    parser.add_argument("--resume-from", help="Resume weights and optimizer from a version-2 checkpoint")
    parser.add_argument("--additional-epochs", type=int, help="Number of epochs to run after --resume-from")
    args = parser.parse_args()
    report = train(
        args.data,
        args.checkpoint,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.blocks,
        args.channels,
        args.device,
        args.seed,
        args.limit_examples,
        args.resume_from,
        args.additional_epochs,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

