"""Benchmark all move modes on a reproducible sample of held-out Lichess puzzles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import chess

from ..board import split_for_fen
from ..engine.intuitive import PureIntuitionEngine
from ..engine.torch_policy import TorchPolicy
from ..search.alphabeta import PolicyGuidedAlphaBeta


SAMPLE_FIELDS = (
    "PuzzleId",
    "FEN",
    "Moves",
    "Rating",
    "RatingDeviation",
    "Themes",
)
MODE_BUDGETS_MS = {"low": 250, "medium": 1_000, "high": 3_000}


def select_sample(csv_path: Path, sample_path: Path, count: int, seed: int) -> dict:
    """Uniform reservoir sample from rows in the existing hash-based test split."""
    rng = random.Random(seed)
    reservoir: list[dict[str, str]] = []
    test_rows = 0
    rows_read = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(SAMPLE_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError("Puzzle CSV is missing columns: " + ", ".join(sorted(missing)))
        for row in reader:
            rows_read += 1
            if split_for_fen(row["FEN"]) != "test":
                continue
            test_rows += 1
            sample_row = {field: row[field] for field in SAMPLE_FIELDS}
            if len(reservoir) < count:
                reservoir.append(sample_row)
            else:
                slot = rng.randrange(test_rows)
                if slot < count:
                    reservoir[slot] = sample_row
            if rows_read % 1_000_000 == 0:
                print(f"CSV rows scanned: {rows_read:,}; reserved test rows seen: {test_rows:,}", flush=True)
    if len(reservoir) < count:
        raise ValueError(f"Only {len(reservoir)} rows found in the reserved test split; need {count}.")
    reservoir.sort(key=lambda row: row["PuzzleId"])
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(reservoir)
    info = {
        "source_csv": str(csv_path.resolve()),
        "rows_read": rows_read,
        "reserved_test_rows": test_rows,
        "sample_count": len(reservoir),
        "seed": seed,
        "sample_csv": str(sample_path.resolve()),
        "sampling": "uniform reservoir sample from the test partition",
        "split_method": "split_for_fen over the first four fields of the source CSV FEN",
    }
    sample_path.with_suffix(sample_path.suffix + ".json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    return info


def load_sample(sample_path: Path, expected_count: int) -> list[dict[str, str]]:
    with sample_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_count:
        raise ValueError(f"Sample file contains {len(rows)} rows; expected {expected_count}.")
    if any(split_for_fen(row["FEN"]) != "test" for row in rows):
        raise ValueError("The sample file contains a row outside the reserved test split.")
    return rows


def puzzle_position(row: dict[str, str]) -> tuple[chess.Board, chess.Move, set[chess.Move]]:
    line = [chess.Move.from_uci(text) for text in row["Moves"].split()]
    if len(line) < 2:
        raise ValueError(f"Puzzle {row['PuzzleId']} has no solver response in Moves[1].")
    board = chess.Board(row["FEN"])
    if not board.is_legal(line[0]):
        raise ValueError(f"Puzzle {row['PuzzleId']} has an illegal triggering move.")
    board.push(line[0])
    target = line[1]
    if not board.is_legal(target):
        raise ValueError(f"Puzzle {row['PuzzleId']} has an illegal solution move.")

    accepted = {target}
    trial = board.copy(stack=False)
    trial.push(target)
    if trial.is_checkmate():
        for move in list(board.legal_moves):
            board.push(move)
            if board.is_checkmate():
                accepted.add(move)
            board.pop()
    return board, target, accepted


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    rate = successes / total
    denom = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denom
    return [max(0.0, center - radius), min(1.0, center + radius)]


def rating_band(rating: int) -> str:
    if rating < 1200:
        return "<1200"
    if rating < 1600:
        return "1200-1599"
    if rating < 2000:
        return "1600-1999"
    if rating < 2400:
        return "2000-2399"
    return "2400+"


def benchmark(
    sample: list[dict[str, str]],
    checkpoint: Path,
    results_path: Path,
    sample_info: dict,
    modes: tuple[str, ...] = ("pure", "low", "medium", "high"),
    runtime_budget_seconds: float | None = None,
) -> dict:
    policy = TorchPolicy.from_checkpoint(checkpoint)
    pure = PureIntuitionEngine(policy)
    searchers = {
        mode: PolicyGuidedAlphaBeta(policy, time_ms=budget)
        for mode, budget in MODE_BUDGETS_MS.items()
        if mode in modes
    }
    mode_names = modes
    records: list[dict] = []
    aggregate = {
        mode: {
            "top1": 0,
            "top5": 0,
            "thinking_time_ms": 0.0,
            "nodes": 0,
            "completed_depth": 0,
            "by_rating_band": defaultdict(lambda: {"count": 0, "top1": 0, "top5": 0}),
        }
        for mode in mode_names
    }
    started = time.perf_counter()

    stopped_early = False
    for index, row in enumerate(sample, start=1):
        if runtime_budget_seconds is not None and index > 1 and time.perf_counter() - started >= runtime_budget_seconds:
            stopped_early = True
            break
        board, target, accepted = puzzle_position(row)
        rating = int(row["Rating"])
        per_puzzle = {
            "puzzle_id": row["PuzzleId"],
            "rating": rating,
            "rating_deviation": int(row["RatingDeviation"]),
            "themes": row["Themes"],
            "pre_blunder_fen": row["FEN"],
            "puzzle_fen": board.fen(),
            "trigger_move_uci": row["Moves"].split()[0],
            "solution_move_uci": target.uci(),
            "accepted_mate_in_one_moves_uci": sorted(move.uci() for move in accepted),
            "modes": {},
        }

        ranked_moves: dict[str, list[chess.Move]] = {}
        for mode in mode_names:
            if mode == "pure":
                ranked, elapsed_ms, _ = pure.rank_moves(board, count=5)
                moves = [candidate.move for candidate in ranked]
                nodes, depth = 0, 0
            else:
                result = searchers[mode].rank_moves(board, count=5)
                moves = [candidate.move for candidate in result.moves]
                elapsed_ms = result.thinking_time_ms
                nodes, depth = result.nodes, result.completed_depth
            ranked_moves[mode] = moves
            hit1 = bool(moves and moves[0] in accepted)
            hit5 = any(move in accepted for move in moves[:5])
            mode_record = {
                "top5_uci": [move.uci() for move in moves],
                "top1_hit": hit1,
                "top5_hit": hit5,
                "thinking_time_ms": round(elapsed_ms, 3),
                "nodes": nodes,
                "completed_depth": depth,
            }
            per_puzzle["modes"][mode] = mode_record
            stats = aggregate[mode]
            stats["top1"] += int(hit1)
            stats["top5"] += int(hit5)
            stats["thinking_time_ms"] += elapsed_ms
            stats["nodes"] += nodes
            stats["completed_depth"] += depth
            band_stats = stats["by_rating_band"][rating_band(rating)]
            band_stats["count"] += 1
            band_stats["top1"] += int(hit1)
            band_stats["top5"] += int(hit5)
        records.append(per_puzzle)

        if index % 25 == 0 or index == len(sample):
            elapsed = time.perf_counter() - started
            print(
                f"Benchmarked {index}/{len(sample)} puzzles; elapsed {elapsed / 60:.1f} min",
                flush=True,
            )

    summary = {}
    evaluated_count = len(records)
    if not evaluated_count:
        raise ValueError("The runtime budget expired before any puzzle was evaluated.")
    for mode in mode_names:
        stats = aggregate[mode]
        count = evaluated_count
        by_band = {}
        for band, values in sorted(stats["by_rating_band"].items()):
            band_count = values["count"]
            by_band[band] = {
                "puzzles": band_count,
                "top1_hits": values["top1"],
                "top1_rate": values["top1"] / band_count,
                "top5_hits": values["top5"],
                "top5_rate": values["top5"] / band_count,
            }
        summary[mode] = {
            "puzzles": count,
            "top1_hits": stats["top1"],
            "top1_rate": stats["top1"] / count,
            "top1_wilson_95_ci": wilson_interval(stats["top1"], count),
            "top5_hits": stats["top5"],
            "top5_rate": stats["top5"] / count,
            "top5_wilson_95_ci": wilson_interval(stats["top5"], count),
            "mean_thinking_time_ms": stats["thinking_time_ms"] / count,
            "mean_nodes": stats["nodes"] / count,
            "mean_completed_depth": stats["completed_depth"] / count,
            "by_rating_band": by_band,
        }

    ratings = [int(row["Rating"]) for row in sample]
    result = {
        "benchmark": "current-checkpoint Lichess puzzle solving baseline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint.resolve()),
        "sample": {**sample_info, "requested_count": len(sample), "evaluated_count": evaluated_count},
        "evaluation": {
            "modes": list(mode_names),
            "runtime_budget_seconds": runtime_budget_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "stopped_early": stopped_early,
        },
        "sample_rating_summary": {
            "minimum": min(ratings),
            "maximum": max(ratings),
            "mean": sum(ratings) / len(ratings),
            "median": sorted(ratings)[len(ratings) // 2],
        },
        "puzzle_interpretation": (
            "Apply Moves[0] to the CSV FEN; score moves from that position against Moves[1]. "
            "Alternative mate-in-one moves are accepted when the recorded response is mate."
        ),
        "limitations": [
            "This is recorded-solution agreement, not a chess Elo estimate.",
            "Bilari is the corrected epoch-3 policy trained on every solver decision in each recorded puzzle line.",
            "Lichess ratings describe puzzle difficulty; they are not engine strength ratings.",
            "Search modes use different time budgets and a material/piece-square leaf evaluator.",
            "A time-limited run can finish the current puzzle after the wall-clock budget is crossed; the overrun is at most approximately one move's search budget.",
        ],
        "mode_summary": summary,
        "per_puzzle": records,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Path to lichess_db_puzzle.csv")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/puzzle-policy.pt"))
    parser.add_argument("--sample", type=Path, default=Path("outputs/puzzle_benchmark_500_sample.csv"))
    parser.add_argument("--results", type=Path, default=Path("outputs/puzzle_benchmark_500_results.json"))
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--modes", default="pure,low,medium,high", help="Comma-separated modes: pure,low,medium,high")
    parser.add_argument("--runtime-budget-seconds", type=float, help="Stop after this much wall time, finishing the current puzzle first")
    parser.add_argument("--order-seed", type=int, help="Shuffle the saved sample deterministically before timing")
    parser.add_argument("--new-sample", action="store_true", help="Draw a fresh random sample from the reserved split.")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    modes = tuple(dict.fromkeys(part.strip().lower() for part in args.modes.split(",") if part.strip()))
    allowed_modes = {"pure", "low", "medium", "high"}
    if not modes or any(mode not in allowed_modes for mode in modes):
        parser.error("--modes must contain one or more of pure,low,medium,high")
    if args.runtime_budget_seconds is not None and args.runtime_budget_seconds <= 0:
        parser.error("--runtime-budget-seconds must be positive")
    if args.new_sample or not args.sample.is_file():
        sample_info = select_sample(args.csv, args.sample, args.count, args.seed)
    else:
        metadata_path = args.sample.with_suffix(args.sample.suffix + ".json")
        if metadata_path.is_file():
            sample_info = json.loads(metadata_path.read_text(encoding="utf-8"))
            sample_info["reused_existing_sample"] = True
        else:
            sample_info = {
                "sample_count": args.count,
                "seed": None,
                "sample_csv": str(args.sample.resolve()),
                "reused_existing_sample": True,
            }
    sample = load_sample(args.sample, args.count)
    if args.order_seed is not None:
        random.Random(args.order_seed).shuffle(sample)
        sample_info = {**sample_info, "evaluation_order_seed": args.order_seed}
    result = benchmark(
        sample,
        args.checkpoint,
        args.results,
        sample_info,
        modes=modes,
        runtime_budget_seconds=args.runtime_budget_seconds,
    )
    print(json.dumps(result["mode_summary"], indent=2))
    print("Full report: " + str(args.results.resolve()))


if __name__ == "__main__":
    main()
