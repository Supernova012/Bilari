"""Convert the Lichess puzzle CSV into compact, legal-masked training shards."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Dict, List, Optional

import chess
import numpy as np
import pandas as pd

from ..board import encode_action, encode_compact_position, split_for_fen


SPLITS = ("train", "validation", "test")


class ShardWriter:
    def __init__(self, output_dir: Path, shard_size: int) -> None:
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.buffers: Dict[str, dict] = {
            name: {"states": [], "meta": [], "policy": [], "rating": [], "legal": []}
            for name in SPLITS
        }
        self.shard_numbers = Counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            (self.output_dir / split).mkdir(exist_ok=True)

    def add(
        self,
        split: str,
        state: np.ndarray,
        meta: np.ndarray,
        policy: int,
        rating: int,
        legal_actions: List[int],
    ) -> None:
        buf = self.buffers[split]
        buf["states"].append(state)
        buf["meta"].append(meta)
        buf["policy"].append(policy)
        buf["rating"].append(rating)
        buf["legal"].append(np.asarray(legal_actions, dtype=np.uint16))
        if len(buf["policy"]) >= self.shard_size:
            self.flush(split)

    def flush(self, split: str) -> None:
        buf = self.buffers[split]
        count = len(buf["policy"])
        if count == 0:
            return

        offsets = np.zeros(count + 1, dtype=np.uint32)
        lengths = np.fromiter((len(row) for row in buf["legal"]), dtype=np.uint32, count=count)
        np.cumsum(lengths, out=offsets[1:])
        legal_actions = np.concatenate(buf["legal"]).astype(np.uint16, copy=False)
        shard = int(self.shard_numbers[split])
        destination = self.output_dir / split / ("part-%05d.npz" % shard)
        np.savez_compressed(
            destination,
            states=np.stack(buf["states"]).astype(np.uint8, copy=False),
            meta=np.stack(buf["meta"]).astype(np.uint8, copy=False),
            policy=np.asarray(buf["policy"], dtype=np.uint16),
            rating=np.asarray(buf["rating"], dtype=np.uint16),
            legal_offsets=offsets,
            legal_actions=legal_actions,
        )
        self.shard_numbers[split] += 1
        self.buffers[split] = {"states": [], "meta": [], "policy": [], "rating": [], "legal": []}

    def finish(self) -> None:
        for split in SPLITS:
            self.flush(split)


def preprocess_csv(
    csv_path: str | Path,
    output_dir: str | Path,
    shard_size: int = 50_000,
    max_rows: Optional[int] = None,
    overwrite: bool = False,
) -> dict:
    """Replay puzzle lines and write one example for every solver decision."""
    source = Path(csv_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("Puzzle CSV not found: " + str(source))
    if shard_size < 1:
        raise ValueError("shard_size must be positive")

    # Confirm the source can actually be opened before clearing generated output.
    # This avoids losing a previous dataset if macOS denies access to Downloads.
    with source.open("rb") as source_check:
        source_check.read(1)

    existing = (
        [path for path in destination.iterdir() if path.name != ".gitkeep"]
        if destination.exists()
        else []
    )
    if existing:
        if not overwrite:
            raise FileExistsError(
                "Output directory is not empty. Pass overwrite=True to replace its generated shards."
            )
        for path in destination.rglob("*.npz"):
            path.unlink()
        manifest = destination / "manifest.json"
        if manifest.exists():
            manifest.unlink()
    destination.mkdir(parents=True, exist_ok=True)

    writer = ShardWriter(destination, shard_size)
    split_counts = Counter()
    puzzle_split_counts = Counter()
    rejection_counts = Counter()
    rows_read = 0
    legal_move_total = 0
    solver_move_counts = Counter()
    chunk_size = min(max(shard_size, 10_000), 100_000)
    reader = pd.read_csv(
        source,
        usecols=["FEN", "Moves", "Rating"],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        on_bad_lines="error",
        chunksize=chunk_size,
    )

    for frame in reader:
        for fen, moves_text, rating_text in frame[["FEN", "Moves", "Rating"]].itertuples(
            index=False, name=None
        ):
            if max_rows is not None and rows_read >= max_rows:
                break
            rows_read += 1
            try:
                moves = moves_text.split()
                board = chess.Board(fen)
                rating = int(float(rating_text))
                if len(moves) < 2:
                    raise ValueError("solution has no move for the puzzle solver")

                # Lichess stores the position before the opponent's blunder. Moves[0]
                # is that trigger; the solver's decisions are Moves[1], Moves[3], ...
                # Replaying the complete line also validates every forced reply.
                examples = []
                for ply, move_text in enumerate(moves):
                    move = chess.Move.from_uci(move_text)
                    if not board.is_legal(move):
                        role = "solver" if ply % 2 else "trigger/reply"
                        raise ValueError(
                            "%s move at Moves[%d] is illegal in the replayed position"
                            % (role, ply)
                        )
                    if ply % 2 == 1:
                        legal_moves = list(board.legal_moves)
                        legal_actions = [encode_action(candidate, board.turn) for candidate in legal_moves]
                        target = encode_action(move, board.turn)
                        if target not in legal_actions:
                            raise ValueError("solver target is absent from its legal move mask")
                        state, meta = encode_compact_position(board)
                        examples.append((state, meta, target, legal_actions))
                    board.push(move)
                if not examples:
                    raise ValueError("solution has no solver decision")
            except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError) as exc:
                rejection_counts[type(exc).__name__ + ": " + str(exc)[:100]] += 1
                continue

            # Keep every decision from one puzzle together in a single split.
            split = split_for_fen(fen)
            for state, meta, target, legal_actions in examples:
                writer.add(split, state, meta, target, rating, legal_actions)
                split_counts[split] += 1
                legal_move_total += len(legal_actions)
            puzzle_split_counts[split] += 1
            solver_move_counts[len(examples)] += 1

        if max_rows is not None and rows_read >= max_rows:
            break
        if rows_read and rows_read % 100_000 == 0:
            print("processed %d rows" % rows_read, flush=True)

    writer.finish()
    kept = sum(split_counts.values())
    manifest = {
        "format_version": 2,
        "source_csv": str(source),
        "source_bytes": source.stat().st_size,
        "split_method": "stable BLAKE2b hash of first four source-FEN fields; 90/5/5; all decisions from a puzzle remain together",
        "position_identity": "each solver decision position after replaying the trigger and all intervening replies",
        "target": "every solver UCI move in the recorded line (Moves[1], Moves[3], Moves[5], ...); Moves[0] is applied as the opponent trigger",
        "legal_mask": "all legal move action ids stored as per-row ragged arrays",
        "rows_read": rows_read,
        "puzzle_rows_kept": int(sum(puzzle_split_counts.values())),
        "puzzle_rows_rejected": rows_read - int(sum(puzzle_split_counts.values())),
        "rows_kept": int(sum(puzzle_split_counts.values())),
        "rows_rejected": rows_read - int(sum(puzzle_split_counts.values())),
        "training_examples": kept,
        "split_counts": {split: int(split_counts[split]) for split in SPLITS},
        "puzzle_counts_by_split": {split: int(puzzle_split_counts[split]) for split in SPLITS},
        "shards_per_split": {split: int(writer.shard_numbers[split]) for split in SPLITS},
        "mean_legal_moves": float(legal_move_total / kept) if kept else 0.0,
        "solver_decisions_per_puzzle": {
            "mean": float(sum(count * puzzles for count, puzzles in solver_move_counts.items()) / sum(puzzle_split_counts.values()))
            if puzzle_split_counts
            else 0.0,
            "distribution": {str(count): int(puzzles) for count, puzzles in sorted(solver_move_counts.items())},
        },
        "rejection_reasons": dict(rejection_counts),
        "value_labels": False,
        "note": "Moves[0] is the opponent trigger and is never a policy target. Every solver decision after the trigger is a separate policy example. Puzzle ratings are difficulty labels, not position evaluations; the value head is untrained.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to lichess_db_puzzle.csv")
    parser.add_argument("--out", default="data/processed", help="Output directory for compact shards")
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument("--max-rows", type=int, help="Optional row limit for a quick smoke run")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated shards in --out")
    args = parser.parse_args()
    manifest = preprocess_csv(args.csv, args.out, args.shard_size, args.max_rows, args.overwrite)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

