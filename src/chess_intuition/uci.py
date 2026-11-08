"""Minimal UCI bridge for running Bilari's pure policy as a chess bot."""

from __future__ import annotations

import argparse
import sys

import chess

from .engine.intuitive import PureIntuitionEngine
from .engine.torch_policy import TorchPolicy


def _position(tokens: list[str]) -> chess.Board:
    if not tokens or tokens[0] != "position":
        raise ValueError("expected UCI position command")
    if len(tokens) < 2:
        raise ValueError("position command is incomplete")
    if tokens[1] == "startpos":
        board = chess.Board()
        rest = tokens[2:]
    elif tokens[1] == "fen":
        try:
            moves_at = tokens.index("moves", 2)
        except ValueError:
            moves_at = len(tokens)
        fen = " ".join(tokens[2:moves_at])
        board = chess.Board(fen)
        rest = tokens[moves_at + 1:] if moves_at < len(tokens) else []
    else:
        raise ValueError("position must use startpos or fen")
    if rest:
        if rest[0] != "moves":
            raise ValueError("unexpected text after position")
        for uci in rest[1:]:
            board.push_uci(uci)
    return board


def run_uci(policy: TorchPolicy) -> None:
    board = chess.Board()
    engine = PureIntuitionEngine(policy)
    for raw in sys.stdin:
        command = raw.strip().split()
        if not command:
            continue
        try:
            if command[0] == "uci":
                print("id name Bilari Intuition", flush=True)
                print("id author Bilari", flush=True)
                print("uciok", flush=True)
            elif command[0] == "isready":
                print("readyok", flush=True)
            elif command[0] == "ucinewgame":
                board = chess.Board()
            elif command[0] == "position":
                board = _position(command)
            elif command[0] == "go":
                decision = engine.choose_move(board)
                print(f"bestmove {decision.move.uci() if decision else '0000'}", flush=True)
            elif command[0] == "quit":
                return
        except Exception as error:
            print(f"info string Bilari error: {error}", flush=True)
            if command[0] == "go":
                print("bestmove 0000", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", help="auto, cpu, mps, or cuda")
    args = parser.parse_args()
    try:
        policy = TorchPolicy.from_checkpoint(args.checkpoint, device=args.device)
        run_uci(policy)
    except Exception as error:
        print(f"Bilari could not start: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
