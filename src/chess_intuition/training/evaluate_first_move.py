"""Measure top-1 agreement on each puzzle's first solver decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import chess
import numpy as np
import pandas as pd
import torch

from ..board import encode_action, encode_position, split_for_fen
from ..engine.torch_policy import TorchPolicy
from .train import choose_device


def evaluate_first_moves(
    csv_path: str | Path,
    checkpoint_path: str | Path,
    split: str = "validation",
    batch_size: int = 512,
    device_name: str = "auto",
    output_path: str | Path | None = None,
) -> dict:
    """Evaluate the first solver move on every puzzle in one deterministic split."""
    source = Path(csv_path).expanduser().resolve()
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not source.is_file():
        raise FileNotFoundError("Puzzle CSV not found: " + str(source))
    if not checkpoint_file.is_file():
        raise FileNotFoundError("Checkpoint not found: " + str(checkpoint_file))

    device = choose_device(device_name)
    policy = TorchPolicy.from_checkpoint(checkpoint_file, device=str(device))
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
    checkpoint_epoch = checkpoint.get("training", {}).get("epoch")

    hits = 0
    top5_hits = 0
    positions = 0
    rejected = 0
    pending_features: list[np.ndarray] = []
    pending_legal: list[np.ndarray] = []
    pending_targets: list[int] = []
    started = time.perf_counter()

    def flush() -> None:
        nonlocal hits, top5_hits, positions
        if not pending_features:
            return
        features = torch.from_numpy(np.stack(pending_features)).to(device)
        with torch.inference_mode():
            logits, _ = policy.model(features)
        for row, legal_ids in enumerate(pending_legal):
            candidates = torch.as_tensor(legal_ids.astype(np.int64), device=device)
            legal_logits = logits[row].index_select(0, candidates)
            ranks = legal_logits.topk(min(5, len(legal_ids))).indices
            ranked_candidates = candidates.index_select(0, ranks)
            predicted = int(ranked_candidates[0].item())
            hits += int(predicted == pending_targets[row])
            top5_hits += int(pending_targets[row] in ranked_candidates.tolist())
        positions += len(pending_features)
        pending_features.clear()
        pending_legal.clear()
        pending_targets.clear()

    reader = pd.read_csv(
        source,
        usecols=["FEN", "Moves"],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        on_bad_lines="error",
        chunksize=100_000,
    )
    rows_scanned = 0
    for frame in reader:
        for fen, moves_text in frame[["FEN", "Moves"]].itertuples(index=False, name=None):
            rows_scanned += 1
            if split_for_fen(fen) != split:
                continue
            try:
                moves = moves_text.split()
                if len(moves) < 2:
                    raise ValueError("solution has no solver reply")
                board = chess.Board(fen)
                trigger = chess.Move.from_uci(moves[0])
                target = chess.Move.from_uci(moves[1])
                if not board.is_legal(trigger):
                    raise ValueError("trigger move is illegal")
                board.push(trigger)
                legal_moves = list(board.legal_moves)
                if target not in legal_moves:
                    raise ValueError("first solver move is illegal after the trigger")
                pending_features.append(encode_position(board))
                pending_legal.append(
                    np.asarray([encode_action(move, board.turn) for move in legal_moves], dtype=np.uint16)
                )
                pending_targets.append(encode_action(target, board.turn))
            except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
                rejected += 1
                continue

            if len(pending_features) >= batch_size:
                flush()
            if rows_scanned % 1_000_000 == 0:
                print(
                    "Scanned %d CSV rows; scored %d first moves" % (rows_scanned, positions),
                    flush=True,
                )
    flush()
    if positions == 0:
        raise ValueError("No valid puzzles found for split " + split)

    result = {
        "checkpoint": str(checkpoint_file),
        "checkpoint_epoch": checkpoint_epoch,
        "source_csv": str(source),
        "split": split,
        "positions": positions,
        "top1_hits": hits,
        "first_move_top1_accuracy": hits / positions,
        "top5_hits": top5_hits,
        "first_move_top5_accuracy": top5_hits / positions,
        "rejected_rows_in_split": rejected,
        "elapsed_seconds": time.perf_counter() - started,
        "target": "Moves[1] after applying the opponent trigger Moves[0] to the CSV FEN",
        "interpretation": "First-solver-move agreement against the recorded line; this is not chess strength or Elo.",
    }
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else checkpoint_file.parent / ("first_move_%s_epoch%s.json" % (split, checkpoint_epoch))
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["results_file"] = str(destination)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to lichess_db_puzzle.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", help="Optional JSON output path")
    args = parser.parse_args()
    report = evaluate_first_moves(
        args.csv,
        args.checkpoint,
        args.split,
        args.batch_size,
        args.device,
        args.out,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
