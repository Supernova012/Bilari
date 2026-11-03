"""Rank candidate moves for a FEN using the puzzle-trained policy and search."""

from __future__ import annotations

import argparse
import json

import chess

from .engine.intuitive import PureIntuitionEngine
from .engine.torch_policy import TorchPolicy


MODE_TIME_MS = {"low": 250, "medium": 1_000, "high": 3_000}


def position_warning(board: chess.Board) -> str | None:
    """Allow analyzable FENs where the non-moving king is already checked."""
    status = board.status()
    unsupported = status & ~chess.STATUS_OPPOSITE_CHECK
    if unsupported:
        raise ValueError(f"FEN is not a usable standard chess position (status={status}).")
    if status & chess.STATUS_OPPOSITE_CHECK:
        moving = chess.COLOR_NAMES[board.turn].title()
        checked = chess.COLOR_NAMES[not board.turn].title()
        return (
            f"{checked} king is already in check while {moving} is to move. "
            "This position is not reachable by legal play, but move suggestions are generated."
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    parser.add_argument("--device", help="auto, cpu, mps, or cuda")
    parser.add_argument("--mode", choices=("pure", "low", "medium", "high"), default="high")
    parser.add_argument("--top", type=int, default=5, help="Number of candidate moves to return")
    parser.add_argument("--time-ms", type=int, help="Override search time for low/medium/high")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    board = chess.Board(args.fen)
    try:
        warning = position_warning(board)
    except ValueError as error:
        parser.error(str(error))
    policy = TorchPolicy.from_checkpoint(args.checkpoint, device=args.device)
    if args.mode == "pure":
        ranked, thinking_ms, legal_count = PureIntuitionEngine(policy).rank_moves(board, args.top)
        result = {
            "mode": "Pure Intuition — 0 depth",
            "fen": board.fen(),
            "side_to_move": chess.COLOR_NAMES[board.turn].title(),
            "evaluator": "puzzle-trained policy only",
            "moves": [
                {
                    "rank": index,
                    "uci": candidate.move.uci(),
                    "san": board.san(candidate.move),
                    "policy_probability": candidate.policy_probability,
                    "policy_logit": candidate.policy_logit,
                }
                for index, candidate in enumerate(ranked, start=1)
            ],
            "thinking_time_ms": thinking_ms,
            "legal_moves_considered": legal_count,
            "policy_calls": 1 if legal_count else 0,
            "search_calls": 0,
            "nodes_searched": 0,
        }
    else:
        from .search.alphabeta import PolicyGuidedAlphaBeta

        time_ms = args.time_ms if args.time_ms is not None else MODE_TIME_MS[args.mode]
        search = PolicyGuidedAlphaBeta(policy, time_ms=time_ms)
        search_result = search.rank_moves(board, args.top)
        result = {
            "mode": args.mode,
            "fen": board.fen(),
            "side_to_move": chess.COLOR_NAMES[board.turn].title(),
            "evaluator": search_result.evaluator,
            "moves": [
                {
                    "rank": index,
                    "uci": candidate.move.uci(),
                    "san": board.san(candidate.move),
                    "score_cp": candidate.score_cp,
                    "policy_probability": candidate.policy_probability,
                    "principal_variation": list(candidate.principal_variation),
                }
                for index, candidate in enumerate(search_result.moves, start=1)
            ],
            "thinking_time_ms": search_result.thinking_time_ms,
            "time_limit_ms": time_ms,
            "completed_depth": search_result.completed_depth,
            "nodes_searched": search_result.nodes,
            "nodes_per_second": search_result.nodes_per_second,
            "policy_calls": search_result.policy_calls,
            "search_calls": search_result.search_calls,
            "timed_out": search_result.timed_out,
            "value_head_trained": policy.value_trained,
        }
    if warning:
        result["position_warning"] = warning
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

