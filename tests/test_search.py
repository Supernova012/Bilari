import unittest

import chess
import numpy as np

from chess_intuition.board import ACTION_SIZE
from chess_intuition.search.alphabeta import PolicyGuidedAlphaBeta


class FlatPolicy:
    value_trained = False

    def __init__(self):
        self.calls = 0

    def predict_policy(self, features):
        self.calls += 1
        return np.zeros(ACTION_SIZE, dtype=np.float32)


class SearchTests(unittest.TestCase):
    def test_search_returns_legal_top_five_and_stats(self):
        board = chess.Board()
        original_fen = board.fen()
        policy = FlatPolicy()
        search = PolicyGuidedAlphaBeta(policy, time_ms=1_000, max_depth=2)
        result = search.rank_moves(board, count=5)

        self.assertEqual(len(result.moves), 5)
        self.assertTrue(all(candidate.move in list(board.legal_moves) for candidate in result.moves))
        self.assertGreaterEqual(result.completed_depth, 1)
        self.assertGreater(result.nodes, 0)
        self.assertEqual(result.search_calls, 1)
        self.assertGreater(result.policy_calls, 0)
        self.assertEqual(board.fen(), original_fen)
        self.assertFalse(policy.value_trained)
        self.assertEqual(result.evaluator, "material + piece-square heuristic")

    def test_terminal_position_returns_no_candidates(self):
        board = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
        result = PolicyGuidedAlphaBeta(FlatPolicy(), time_ms=100).rank_moves(board)
        self.assertEqual(result.moves, ())
        self.assertEqual(result.nodes, 0)


if __name__ == "__main__":
    unittest.main()

