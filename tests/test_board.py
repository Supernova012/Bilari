import unittest

import chess
import numpy as np

from chess_intuition.board import (
    ACTION_SIZE,
    encode_action,
    encode_compact_position,
    encode_position,
    split_for_fen,
)


class BoardEncodingTests(unittest.TestCase):
    def test_compact_and_network_shapes(self):
        board = chess.Board()
        squares, meta = encode_compact_position(board)
        self.assertEqual(squares.shape, (64,))
        self.assertEqual(meta.shape, (7,))
        self.assertEqual(encode_position(board).shape, (19, 8, 8))
        self.assertEqual(int(squares[chess.E2]), chess.PAWN)
        self.assertEqual(int(squares[chess.E7]), chess.PAWN + 6)

    def test_black_to_move_is_rotated_to_mover_perspective(self):
        board = chess.Board()
        board.turn = chess.BLACK
        squares, _ = encode_compact_position(board)
        # The black e7 pawn rotates to d2 under a 180-degree orientation.
        self.assertEqual(int(squares[chess.D2]), chess.PAWN)
        self.assertEqual(int(squares[chess.D7]), chess.PAWN + 6)

    def test_all_initial_legal_actions_are_unique_and_in_range(self):
        board = chess.Board()
        actions = [encode_action(move, board.turn) for move in board.legal_moves]
        self.assertEqual(len(actions), len(set(actions)))
        self.assertTrue(all(0 <= action < ACTION_SIZE for action in actions))

    def test_underpromotion_actions_are_distinct(self):
        board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        promotion_moves = [move for move in board.legal_moves if move.promotion]
        actions = [encode_action(move, board.turn) for move in promotion_moves]
        self.assertEqual(len(actions), 4)
        self.assertEqual(len(set(actions)), 4)

    def test_duplicate_fens_with_different_move_counters_share_a_split(self):
        first = "8/8/8/8/8/8/4k3/4K3 w - - 0 1"
        second = "8/8/8/8/8/8/4k3/4K3 w - - 99 73"
        self.assertEqual(split_for_fen(first), split_for_fen(second))


if __name__ == "__main__":
    unittest.main()

