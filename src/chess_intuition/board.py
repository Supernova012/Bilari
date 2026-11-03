"""Position features and policy-action encoding for standard chess."""

from __future__ import annotations

from typing import Tuple

import chess
import numpy as np

BOARD_SIZE = 8
QUEEN_DIRECTIONS = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)
KNIGHT_DIRECTIONS = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)
QUEEN_PLANES = 8 * 7
KNIGHT_PLANES = 8
UNDERPROMOTION_PLANES = 3 * 3
ACTION_PLANES = QUEEN_PLANES + KNIGHT_PLANES + UNDERPROMOTION_PLANES
ACTION_SIZE = 64 * ACTION_PLANES
FEATURE_CHANNELS = 19
META_SIZE = 7

_UNDERPROMOTION_TYPES = (chess.KNIGHT, chess.BISHOP, chess.ROOK)


def _canonical_square(square: chess.Square, turn: chess.Color) -> chess.Square:
    """Rotate Black-to-move positions by 180 degrees so the mover faces north."""
    return square if turn == chess.WHITE else square ^ 63


def encode_compact_position(board: chess.Board) -> Tuple[np.ndarray, np.ndarray]:
    """Return square codes [64] and metadata [7], both compact uint8 arrays.

    Square codes are 0 for empty, 1..6 for the side to move's P/N/B/R/Q/K,
    and 7..12 for the opponent's pieces. Square index is rank * 8 + file after
    canonical orientation. Metadata is own K/Q castling, opponent K/Q
    castling, en-passant square (64 means absent), halfmove clock (capped at
    100), and whether the side to move is in check.
    """
    squares = np.zeros(64, dtype=np.uint8)
    for square, piece in board.piece_map().items():
        oriented = _canonical_square(square, board.turn)
        code = piece.piece_type if piece.color == board.turn else piece.piece_type + 6
        squares[chess.square_rank(oriented) * 8 + chess.square_file(oriented)] = code

    own = board.turn
    opponent = not own
    ep = 64 if board.ep_square is None else _canonical_square(board.ep_square, own)
    meta = np.asarray(
        [
            board.has_kingside_castling_rights(own),
            board.has_queenside_castling_rights(own),
            board.has_kingside_castling_rights(opponent),
            board.has_queenside_castling_rights(opponent),
            ep,
            min(board.halfmove_clock, 100),
            int(board.is_check()),
        ],
        dtype=np.uint8,
    )
    return squares, meta


def compact_to_features(squares: np.ndarray, meta: np.ndarray) -> np.ndarray:
    """Expand compact records into a float32 [19, 8, 8] network input."""
    if squares.shape != (64,) or meta.shape != (META_SIZE,):
        raise ValueError("Expected square codes [64] and metadata [7].")
    features = np.zeros((FEATURE_CHANNELS, 8, 8), dtype=np.float32)
    piece_map = squares.reshape(8, 8)
    for code in range(1, 13):
        features[code - 1] = piece_map == code
    for channel, flag in enumerate(meta[:4], start=12):
        features[channel].fill(float(flag))
    if int(meta[4]) < 64:
        rank, file_ = divmod(int(meta[4]), 8)
        features[16, rank, file_] = 1.0
    features[17].fill(float(meta[6]))
    features[18].fill(float(meta[5]) / 100.0)
    return features


def encode_position(board: chess.Board) -> np.ndarray:
    """Return the network input for a python-chess board."""
    squares, meta = encode_compact_position(board)
    return compact_to_features(squares, meta)


def encode_action(move: chess.Move, turn: chess.Color) -> int:
    """Encode a move into the 4,672-action policy space.

    The first 56 planes represent queen-like moves, the next 8 represent
    knight moves, and the final 9 represent underpromotions. Black positions
    are rotated by 180 degrees, matching the board feature orientation.
    """
    from_square = _canonical_square(move.from_square, turn)
    to_square = _canonical_square(move.to_square, turn)
    from_file, from_rank = chess.square_file(from_square), chess.square_rank(from_square)
    to_file, to_rank = chess.square_file(to_square), chess.square_rank(to_square)
    df, dr = to_file - from_file, to_rank - from_rank

    if move.promotion in _UNDERPROMOTION_TYPES:
        if dr != 1 or df not in (-1, 0, 1):
            raise ValueError("Underpromotion move is outside the canonical promotion rank.")
        plane = 64 + (df + 1) * 3 + _UNDERPROMOTION_TYPES.index(move.promotion)
    elif (df, dr) in KNIGHT_DIRECTIONS:
        plane = QUEEN_PLANES + KNIGHT_DIRECTIONS.index((df, dr))
    else:
        plane = -1
        for direction, (dx, dy) in enumerate(QUEEN_DIRECTIONS):
            for distance in range(1, 8):
                if (df, dr) == (dx * distance, dy * distance):
                    plane = direction * 7 + distance - 1
                    break
            if plane >= 0:
                break
        if plane < 0:
            raise ValueError("Move is not representable in the standard policy space.")

    square_index = from_rank * 8 + from_file
    return square_index * ACTION_PLANES + plane


def split_for_fen(fen: str) -> str:
    """Stable 90/5/5 split by board identity; duplicate states stay together."""
    import hashlib

    position_key = " ".join(fen.split()[:4]).encode("ascii")
    bucket = int.from_bytes(
        hashlib.blake2b(position_key, digest_size=8, person=b"chesspos").digest(),
        "big",
    ) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"

