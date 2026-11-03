"""Policy-ordered iterative-deepening negamax with alpha-beta and quiescence."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Dict, Optional

import chess
import numpy as np

from ..board import encode_action, encode_position
from ..engine.intuitive import PolicyScorer

INF = 1_000_000
MATE = 100_000
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 335,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


@dataclass(frozen=True)
class SearchMove:
    move: chess.Move
    score_cp: int
    policy_probability: float
    principal_variation: tuple[str, ...]


@dataclass(frozen=True)
class SearchResult:
    moves: tuple[SearchMove, ...]
    thinking_time_ms: float
    completed_depth: int
    nodes: int
    policy_calls: int
    nodes_per_second: float
    timed_out: bool
    evaluator: str
    search_calls: int = 1


@dataclass
class _TTEntry:
    depth: int
    score: int
    bound: str
    best_move: Optional[str]


class _SearchTimeout(Exception):
    pass


def _static_white_score(board: chess.Board) -> int:
    """Small material/PST fallback, used because puzzle labels lack a value target."""
    non_king_material = 0
    for piece in board.piece_map().values():
        if piece.piece_type != chess.KING:
            non_king_material += PIECE_VALUES[piece.piece_type]
    endgame = non_king_material <= 2_600

    score = 0
    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        file_ = chess.square_file(square)
        rank = chess.square_rank(square)
        relative_rank = rank if piece.color == chess.WHITE else 7 - rank
        center = 7.0 - abs(3.5 - file_) - abs(3.5 - rank)
        positional = 0.0
        if piece.piece_type == chess.PAWN:
            positional = relative_rank * 6.0 + max(center, 0.0) * 1.5
        elif piece.piece_type == chess.KNIGHT:
            positional = center * 8.0
        elif piece.piece_type == chess.BISHOP:
            positional = center * 4.0
        elif piece.piece_type == chess.ROOK:
            positional = 18.0 if relative_rank == 6 else relative_rank * 1.5
        elif piece.piece_type == chess.QUEEN:
            positional = center * 2.0
        elif piece.piece_type == chess.KING:
            positional = center * (4.0 if endgame else -3.0)
        score += sign * (PIECE_VALUES[piece.piece_type] + int(positional))
    return score


class PolicyGuidedAlphaBeta:
    """A separate search mode; the base Pure Intuition engine never calls it.

    The neural policy orders all legal moves and supplies root probabilities.
    Alpha-beta explores responses. Leaf scores use the network value head only
    if checkpoint metadata says it was trained; puzzle-only checkpoints use a
    small material and piece-square evaluator instead.
    """

    def __init__(
        self,
        policy: PolicyScorer,
        time_ms: int = 1_000,
        max_depth: int = 64,
        quiescence_plies: int = 6,
        table_limit: int = 100_000,
        policy_cache_limit: int = 20_000,
    ) -> None:
        if time_ms < 1:
            raise ValueError("time_ms must be positive")
        self.policy = policy
        self.time_ms = time_ms
        self.max_depth = max_depth
        self.quiescence_plies = quiescence_plies
        self.table_limit = table_limit
        self.policy_cache_limit = policy_cache_limit
        self.value_trained = bool(getattr(policy, "value_trained", False))
        self.evaluator = "trained neural value head" if self.value_trained else "material + piece-square heuristic"
        self._policy_cache: Dict[str, np.ndarray] = {}
        self._table: Dict[str, _TTEntry] = {}
        self._deadline = 0.0
        self._nodes = 0
        self._policy_calls = 0

    def _check_time(self) -> None:
        if self._nodes & 31 == 0 and time.perf_counter() >= self._deadline:
            raise _SearchTimeout

    def _position_key(self, board: chess.Board) -> str:
        return board.fen(en_passant="fen")

    def _policy_logits(self, board: chess.Board) -> np.ndarray:
        key = self._position_key(board)
        cached = self._policy_cache.get(key)
        if cached is not None:
            return cached
        logits = np.asarray(self.policy.predict_policy(encode_position(board)), dtype=np.float32)
        if logits.shape != (4672,):
            raise ValueError("Policy scorer must return exactly 4,672 logits.")
        self._policy_calls += 1
        if len(self._policy_cache) >= self.policy_cache_limit:
            self._policy_cache.pop(next(iter(self._policy_cache)))
        self._policy_cache[key] = logits
        return logits

    def _ordered_moves(self, board: chess.Board, legal: list[chess.Move]) -> list[chess.Move]:
        logits = self._policy_logits(board)
        scored = []
        for move in legal:
            action = encode_action(move, board.turn)
            value = float(logits[action])
            scored.append((value if math.isfinite(value) else -math.inf, move))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [move for _, move in scored]

    def _root_probabilities(self, board: chess.Board, legal: list[chess.Move]) -> dict[chess.Move, float]:
        logits = self._policy_logits(board)
        move_scores = np.asarray(
            [logits[encode_action(move, board.turn)] for move in legal], dtype=np.float64
        )
        move_scores = np.where(np.isfinite(move_scores), move_scores, -np.inf)
        if not np.isfinite(move_scores).any():
            move_scores.fill(0.0)
        exp_scores = np.exp(move_scores - np.max(move_scores))
        probabilities = exp_scores / exp_scores.sum()
        return {move: float(probabilities[index]) for index, move in enumerate(legal)}

    def _leaf_score(self, board: chess.Board) -> int:
        if self.value_trained and hasattr(self.policy, "predict_value"):
            value = float(self.policy.predict_value(encode_position(board)))
            return int(max(-1.0, min(1.0, value)) * 1_000)
        white_score = _static_white_score(board)
        return white_score if board.turn == chess.WHITE else -white_score

    @staticmethod
    def _terminal_score(board: chess.Board, legal: list[chess.Move], ply: int) -> Optional[int]:
        if not legal:
            return -MATE + ply if board.is_check() else 0
        if (
            board.is_fivefold_repetition()
            or board.is_seventyfive_moves()
            or board.is_insufficient_material()
        ):
            return 0
        return None

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self._nodes += 1
        self._check_time()
        key = self._position_key(board)
        original_alpha = alpha
        cached = self._table.get(key)
        tt_move = None
        if cached is not None:
            tt_move = cached.best_move
            if cached.depth >= depth:
                if cached.bound == "exact":
                    return cached.score
                if cached.bound == "lower":
                    alpha = max(alpha, cached.score)
                elif cached.bound == "upper":
                    beta = min(beta, cached.score)
                if alpha >= beta:
                    return cached.score

        legal = list(board.legal_moves)
        terminal = self._terminal_score(board, legal, ply)
        if terminal is not None:
            return terminal
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply, 0, legal)

        ordered = self._ordered_moves(board, legal)
        if tt_move:
            tt_move_obj = chess.Move.from_uci(tt_move)
            if tt_move_obj in ordered:
                ordered.remove(tt_move_obj)
                ordered.insert(0, tt_move_obj)

        best_score = -INF
        best_move = None
        for move in ordered:
            self._check_time()
            board.push(move)
            try:
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if score > best_score:
                best_score = score
                best_move = move.uci()
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        bound = "exact"
        if best_score <= original_alpha:
            bound = "upper"
        elif best_score >= beta:
            bound = "lower"
        if len(self._table) >= self.table_limit:
            self._table.clear()
        self._table[key] = _TTEntry(depth, best_score, bound, best_move)
        return best_score

    def _quiescence(
        self,
        board: chess.Board,
        alpha: int,
        beta: int,
        ply: int,
        qply: int,
        legal: Optional[list[chess.Move]] = None,
    ) -> int:
        self._nodes += 1
        self._check_time()
        if legal is None:
            legal = list(board.legal_moves)
        terminal = self._terminal_score(board, legal, ply)
        if terminal is not None:
            return terminal

        in_check = board.is_check()
        if not in_check:
            stand_pat = self._leaf_score(board)
            if stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)
            if qply >= self.quiescence_plies:
                return alpha
            forcing = [move for move in legal if board.is_capture(move) or move.promotion]
        else:
            if qply >= self.quiescence_plies + 4:
                return self._leaf_score(board)
            forcing = legal

        def tactical_order(move: chess.Move) -> tuple[int, int]:
            captured = board.piece_at(move.to_square)
            if board.is_en_passant(move):
                captured_value = PIECE_VALUES[chess.PAWN]
            else:
                captured_value = PIECE_VALUES[captured.piece_type] if captured else 0
            attacker = board.piece_at(move.from_square)
            attacker_value = PIECE_VALUES[attacker.piece_type] if attacker else 0
            promotion_value = PIECE_VALUES[move.promotion] if move.promotion else 0
            return (captured_value * 10 - attacker_value + promotion_value, -move.from_square)

        forcing.sort(key=tactical_order, reverse=True)
        for move in forcing:
            self._check_time()
            board.push(move)
            try:
                score = -self._quiescence(board, -beta, -alpha, ply + 1, qply + 1)
            finally:
                board.pop()
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def _principal_variation(self, board: chess.Board, first: chess.Move, depth: int) -> tuple[str, ...]:
        copy = board.copy(stack=True)
        pv = [first.uci()]
        copy.push(first)
        for _ in range(max(0, depth - 1)):
            entry = self._table.get(self._position_key(copy))
            if entry is None or not entry.best_move:
                break
            move = chess.Move.from_uci(entry.best_move)
            if not copy.is_legal(move):
                break
            pv.append(move.uci())
            copy.push(move)
        return tuple(pv)

    def rank_moves(self, board: chess.Board, count: int = 5) -> SearchResult:
        if count < 1:
            raise ValueError("count must be positive")
        start = time.perf_counter()
        self._deadline = start + self.time_ms / 1000.0
        self._nodes = 0
        self._policy_calls = 0
        self._policy_cache.clear()
        self._table.clear()

        legal = list(board.legal_moves)
        if not legal:
            return SearchResult((), 0.0, 0, 0, 0, 0.0, False, self.evaluator)
        if board.is_game_over(claim_draw=False):
            return SearchResult((), 0.0, 0, 0, 0, 0.0, False, self.evaluator)

        probabilities = self._root_probabilities(board, legal)
        ordered_root = self._ordered_moves(board, legal)
        # Initial policy order is also a safe legal fallback if a tiny time
        # budget expires before the first complete iteration.
        best_complete = [
            SearchMove(move, self._leaf_score_after(board, move), probabilities[move], (move.uci(),))
            for move in ordered_root
        ]
        completed_depth = 0
        timed_out = False

        for depth in range(1, self.max_depth + 1):
            iteration: list[SearchMove] = []
            try:
                for move in ordered_root:
                    self._check_time()
                    board.push(move)
                    try:
                        score = -self._negamax(board, depth - 1, -INF, INF, 1)
                    finally:
                        board.pop()
                    iteration.append(
                        SearchMove(
                            move,
                            score,
                            probabilities[move],
                            self._principal_variation(board, move, depth),
                        )
                    )
            except _SearchTimeout:
                timed_out = True
                break
            if len(iteration) != len(ordered_root):
                break
            best_complete = sorted(iteration, key=lambda result: result.score_cp, reverse=True)
            completed_depth = depth
            ordered_root = [candidate.move for candidate in best_complete]
            if time.perf_counter() >= self._deadline:
                timed_out = True
                break

        elapsed = time.perf_counter() - start
        return SearchResult(
            moves=tuple(best_complete[:count]),
            thinking_time_ms=elapsed * 1000.0,
            completed_depth=completed_depth,
            nodes=self._nodes,
            policy_calls=self._policy_calls,
            nodes_per_second=self._nodes / elapsed if elapsed > 0 else 0.0,
            timed_out=timed_out,
            evaluator=self.evaluator,
        )

    def _leaf_score_after(self, board: chess.Board, move: chess.Move) -> int:
        board.push(move)
        try:
            return -self._leaf_score(board)
        finally:
            board.pop()

