import unittest

import chess

from chess_intuition.play import position_warning


class FenInputTests(unittest.TestCase):
    def test_opposite_check_is_accepted_with_warning(self):
        fen = "3R2k1/5pp1/7p/4B3/4P2P/b4P2/1p4PK/2r5 w - - 0 1"
        board = chess.Board(fen)

        warning = position_warning(board)

        self.assertIn("Black king is already in check", warning)
        self.assertEqual(len(list(board.legal_moves)), 29)

    def test_positions_with_other_fen_status_errors_are_rejected(self):
        board_without_white_king = chess.Board("8/8/8/8/8/8/8/4k3 w - - 0 1")

        with self.assertRaisesRegex(ValueError, "not a usable standard chess position"):
            position_warning(board_without_white_king)

    def test_ordinary_starting_position_has_no_warning(self):
        self.assertIsNone(position_warning(chess.Board()))


if __name__ == "__main__":
    unittest.main()
