import unittest
from unittest import mock

import chess
import numpy as np

from chess_intuition.board import ACTION_SIZE, encode_action
from chess_intuition.engine.intuitive import PureIntuitionEngine


class ConstantPolicy:
    def __init__(self, scores=None):
        self.scores = np.zeros(ACTION_SIZE, dtype=np.float32) if scores is None else scores
        self.calls = 0

    def predict_policy(self, features):
        self.calls += 1
        self.features = features
        return self.scores


class PureIntuitionTests(unittest.TestCase):
    def test_one_policy_call_legal_move_and_zero_search(self):
        board = chess.Board()
        original_fen = board.fen()
        policy = ConstantPolicy()
        with mock.patch.object(board, "push", side_effect=AssertionError("look-ahead push")):
            decision = PureIntuitionEngine(policy).choose_move(board)
        self.assertIn(decision.move, list(board.legal_moves))
        self.assertEqual(policy.calls, 1)
        self.assertEqual(decision.policy_calls, 1)
        self.assertEqual(decision.search_calls, 0)
        self.assertEqual(decision.nodes_searched, 0)
        self.assertEqual(board.fen(), original_fen)
        self.assertEqual(policy.features.shape, (19, 8, 8))

    def test_illegal_high_score_is_masked_out(self):
        board = chess.Board()
        scores = np.zeros(ACTION_SIZE, dtype=np.float32)
        illegal = chess.Move.from_uci("a1a8")
        scores[encode_action(illegal, board.turn)] = 1000.0
        decision = PureIntuitionEngine(ConstantPolicy(scores)).choose_move(board)
        self.assertIn(decision.move, list(board.legal_moves))
        self.assertNotEqual(decision.move, illegal)

    def test_confidence_is_legal_policy_probability(self):
        board = chess.Board()
        decision = PureIntuitionEngine(ConstantPolicy()).choose_move(board)
        self.assertAlmostEqual(decision.confidence, 1.0 / board.legal_moves.count())

    def test_top_five_are_ranked_by_the_single_policy_inference(self):
        board = chess.Board()
        scores = np.zeros(ACTION_SIZE, dtype=np.float32)
        first = chess.Move.from_uci("e2e4")
        second = chess.Move.from_uci("d2d4")
        scores[encode_action(first, board.turn)] = 4.0
        scores[encode_action(second, board.turn)] = 3.0
        policy = ConstantPolicy(scores)
        ranked, _, legal_count = PureIntuitionEngine(policy).rank_moves(board, 5)
        self.assertEqual(len(ranked), 5)
        self.assertEqual(ranked[0].move, first)
        self.assertEqual(ranked[1].move, second)
        self.assertEqual(policy.calls, 1)
        self.assertGreaterEqual(legal_count, 5)

    def test_terminal_position_has_no_move(self):
        board = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
        self.assertTrue(board.is_checkmate())
        with self.assertRaises(ValueError):
            PureIntuitionEngine(ConstantPolicy()).choose_move(board)


if __name__ == "__main__":
    unittest.main()

