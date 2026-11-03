"""Pure learned-policy move selection with legal masking and no look-ahead."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Protocol

import chess
import numpy as np

from ..board import encode_action, encode_position


class PolicyScorer(Protocol):
    def predict_policy(self, features: np.ndarray) -> np.ndarray:
        """Return one logit per policy action for the supplied position."""


@dataclass(frozen=True)
class EngineDecision:
    move: chess.Move
    confidence: float
    thinking_time_ms: float
    legal_moves_considered: int
    policy_calls: int
    search_calls: int = 0
    nodes_searched: int = 0
    mode: str = "Pure Intuition — 0 depth"


@dataclass(frozen=True)
class RankedMove:
    move: chess.Move
    policy_logit: float
    policy_probability: float


class PureIntuitionEngine:
    """Choose from legal current-position moves using one policy inference.

    This class deliberately has no search dependency, recursion, board pushes,
    or opponent-response simulation. Its only use of the rules library is to
    enumerate legal moves for the current position and mask policy scores.
    """

    def __init__(self, policy: PolicyScorer) -> None:
        self.policy = policy

    def rank_moves(self, board: chess.Board, count: int = 5) -> tuple[list[RankedMove], float, int]:
        """Rank legal moves from one current-position policy inference."""
        if count < 1:
            raise ValueError("count must be positive")
        if board.is_game_over(claim_draw=False):
            return [], 0.0, 0

        start = time.perf_counter()
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return [], (time.perf_counter() - start) * 1000.0, 0

        logits = np.asarray(self.policy.predict_policy(encode_position(board)), dtype=np.float64)
        if logits.shape != (4672,):
            raise ValueError("Policy scorer must return exactly 4,672 logits.")
        action_ids = np.fromiter(
            (encode_action(move, board.turn) for move in legal_moves),
            dtype=np.int64,
            count=len(legal_moves),
        )
        legal_logits = logits[action_ids]
        legal_logits = np.where(np.isfinite(legal_logits), legal_logits, -np.inf)
        if not np.isfinite(legal_logits).any():
            raise ValueError("Policy scorer returned no finite score for a legal move.")

        best_logit = float(np.max(legal_logits))
        probabilities = np.exp(legal_logits - best_logit)
        probabilities /= probabilities.sum()
        order = np.argsort(-legal_logits, kind="stable")[: min(count, len(legal_moves))]
        ranked = [
            RankedMove(
                move=legal_moves[int(index)],
                policy_logit=float(legal_logits[int(index)]),
                policy_probability=float(probabilities[int(index)]),
            )
            for index in order
        ]
        return ranked, (time.perf_counter() - start) * 1000.0, len(legal_moves)

    def choose_move(self, board: chess.Board) -> EngineDecision:
        ranked, thinking_ms, legal_count = self.rank_moves(board, count=1)
        if not ranked:
            raise ValueError("The game is over; there is no legal move to choose.")
        return EngineDecision(
            move=ranked[0].move,
            confidence=ranked[0].policy_probability,
            thinking_time_ms=thinking_ms,
            legal_moves_considered=legal_count,
            policy_calls=1,
        )

