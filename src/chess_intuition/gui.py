"""Local browser GUI for puzzle and self-play chess with Bilari policies."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
import json
import mimetypes
from pathlib import Path
import threading
import webbrowser
from datetime import datetime

import chess
import chess.pgn

from .engine.intuitive import PureIntuitionEngine
from .engine.torch_policy import TorchPolicy
from .play import position_warning
from .search.alphabeta import PolicyGuidedAlphaBeta


LOW_MED_HIGH_MS = {"low": 250, "medium": 1_000, "high": 3_000}
ULTRA_HARD_LIMIT_SECONDS = 30
WEB_ROOT = Path(__file__).with_name("web")


class ModelRegistry:
    """Load each local checkpoint once and reuse it for inference."""

    def __init__(self, bilari_path: Path, oldb_path: Path) -> None:
        self.paths = {"bilari": bilari_path.resolve(), "oldb": oldb_path.resolve()}
        self.labels = {"bilari": "Bilari", "oldb": "Old-B"}
        self.descriptions = {
            "bilari": "Corrected epoch-3 policy, trained on all solver decisions in the puzzle lines.",
            "oldb": "Retained pre-correction pilot checkpoint: one epoch on the first 20,000 puzzle rows.",
        }
        self._policies: dict[str, TorchPolicy] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.RLock()

    def get(self, name: str) -> TorchPolicy:
        if name not in self.paths:
            raise ValueError("Choose Bilari or Old-B as the model.")
        with self._lock:
            if name in self._policies:
                return self._policies[name]
            path = self.paths[name]
            if not path.is_file():
                raise FileNotFoundError(f"{self.labels[name]} checkpoint is missing: {path}")
            try:
                policy = TorchPolicy.from_checkpoint(path)
            except Exception as exc:
                self._errors[name] = str(exc)
                raise RuntimeError(f"Could not load {self.labels[name]}: {exc}") from exc
            self._policies[name] = policy
            self._errors.pop(name, None)
            return policy

    def describe(self) -> dict:
        models = {}
        for name, path in self.paths.items():
            models[name] = {
                "label": self.labels[name],
                "description": self.descriptions[name],
                "available": path.is_file() and name not in self._errors,
                "error": self._errors.get(name),
            }
        return {
            "models": models,
            "modes": ["intuition", "low", "medium", "high", "ultra"],
            "mode_budgets_ms": LOW_MED_HIGH_MS,
            "ultra_default_seconds": 10,
            "ultra_hard_limit_seconds": ULTRA_HARD_LIMIT_SECONDS,
            "starting_fen": chess.STARTING_FEN,
        }


def _load_board(fen: str) -> chess.Board:
    fen = fen.strip()
    if not fen:
        raise ValueError("Paste a FEN first, or use the starting position in Game mode.")
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"That FEN could not be read: {exc}") from exc
    try:
        position_warning(board)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    return board


def _load_position(payload: dict) -> chess.Board:
    initial_fen = str(payload.get("initial_fen") or payload.get("fen", "")).strip()
    board = _load_board(initial_fen)
    history = payload.get("moves", [])
    if not isinstance(history, list) or len(history) > 1_000:
        raise ValueError("The move history is invalid or too long.")
    for uci in history:
        board, _ = _move(board, str(uci))
    expected_fen = str(payload.get("fen", "")).strip()
    if expected_fen and board.fen() != _load_board(expected_fen).fen():
        raise ValueError("The move history does not match the current position.")
    return board


def _state(board: chess.Board, last_move: chess.Move | None = None) -> dict:
    legal_moves = list(board.legal_moves) if not board.is_game_over(claim_draw=True) else []
    outcome = board.outcome(claim_draw=True)
    pieces = []
    for square, piece in board.piece_map().items():
        pieces.append(
            {
                "square": chess.square_name(square),
                "color": "w" if piece.color else "b",
                "kind": piece.symbol().lower(),
            }
        )
    result = {
        "fen": board.fen(),
        "side_to_move": "w" if board.turn else "b",
        "side_name": chess.COLOR_NAMES[board.turn].title(),
        "in_check": board.is_check(),
        "pieces": pieces,
        "legal_moves": [
            {"uci": move.uci(), "san": board.san(move)} for move in legal_moves
        ],
        "last_move": last_move.uci() if last_move else None,
        "game_over": outcome is not None,
        "result": outcome.result() if outcome else None,
        "termination": outcome.termination.name.replace("_", " ").title() if outcome else None,
        "turn_number": board.fullmove_number,
    }
    if outcome is not None and outcome.winner is not None:
        result["winner"] = "w" if outcome.winner else "b"
    else:
        result["winner"] = None
    try:
        warning = position_warning(board)
        if warning:
            result["warning"] = warning
    except ValueError as exc:
        result["warning"] = str(exc)
    return result


def _move(board: chess.Board, uci: str) -> tuple[chess.Board, chess.Move]:
    try:
        move = chess.Move.from_uci(uci)
    except ValueError as exc:
        raise ValueError("Choose a valid chess move.") from exc
    if move not in board.legal_moves:
        raise ValueError("That move is not legal in this position.")
    board.push(move)
    return board, move


def _analyze(registry: ModelRegistry, payload: dict) -> dict:
    board = _load_position(payload)
    model_name = str(payload.get("model", "bilari")).lower()
    mode = str(payload.get("mode", "intuition")).lower()
    if mode not in {"intuition", "low", "medium", "high", "ultra"}:
        raise ValueError("Choose Intuition, Low, Medium, High, or Ultra mode.")
    policy = registry.get(model_name)
    count = max(1, min(5, int(payload.get("top", 5))))
    if mode == "intuition":
        ranked, thinking_ms, legal_count = PureIntuitionEngine(policy).rank_moves(board, count)
        moves = [
            {
                "rank": index,
                "uci": candidate.move.uci(),
                "san": board.san(candidate.move),
                "policy_probability": candidate.policy_probability,
                "score_cp": None,
                "principal_variation": [candidate.move.uci()],
            }
            for index, candidate in enumerate(ranked, start=1)
        ]
        result = {
            "mode": mode,
            "mode_label": "Intuition · no look-ahead",
            "evaluator": "Puzzle-trained policy only",
            "thinking_time_ms": thinking_ms,
            "time_limit_ms": None,
            "completed_depth": 0,
            "nodes_searched": 0,
            "search_calls": 0,
            "policy_calls": 1 if legal_count else 0,
            "moves": moves,
        }
    else:
        if mode == "ultra":
            try:
                requested = int(payload.get("ultra_max_seconds", 10))
            except (TypeError, ValueError):
                requested = 10
            seconds = min(ULTRA_HARD_LIMIT_SECONDS, max(1, requested))
            time_ms = seconds * 1_000
            mode_label = f"Ultra · up to {seconds}s"
        else:
            time_ms = LOW_MED_HIGH_MS[mode]
            mode_label = f"{mode.title()} · {time_ms / 1_000:g}s"
        search_result = PolicyGuidedAlphaBeta(policy, time_ms=time_ms).rank_moves(board, count)
        moves = [
            {
                "rank": index,
                "uci": candidate.move.uci(),
                "san": board.san(candidate.move),
                "policy_probability": candidate.policy_probability,
                "score_cp": candidate.score_cp,
                "principal_variation": list(candidate.principal_variation),
            }
            for index, candidate in enumerate(search_result.moves, start=1)
        ]
        result = {
            "mode": mode,
            "mode_label": mode_label,
            "evaluator": search_result.evaluator,
            "thinking_time_ms": search_result.thinking_time_ms,
            "time_limit_ms": time_ms,
            "completed_depth": search_result.completed_depth,
            "nodes_searched": search_result.nodes,
            "nodes_per_second": search_result.nodes_per_second,
            "search_calls": search_result.search_calls,
            "policy_calls": search_result.policy_calls,
            "timed_out": search_result.timed_out,
            "moves": moves,
        }
    result.update(
        {
            "model": model_name,
            "model_label": registry.labels[model_name],
            "fen": board.fen(),
            "side_to_move": chess.COLOR_NAMES[board.turn].title(),
        }
    )
    return result


def _export_pgn(payload: dict) -> str:
    final_board = _load_position(payload)
    initial_fen = str(payload.get("initial_fen") or payload.get("fen", "")).strip()
    board = _load_board(initial_fen)
    game = chess.pgn.Game()
    game.setup(board)
    node = game
    for uci in payload.get("moves", []):
        try:
            move = chess.Move.from_uci(str(uci))
        except ValueError as exc:
            raise ValueError("The move history contains an invalid move.") from exc
        if move not in board.legal_moves:
            raise ValueError("The move history contains an illegal move.")
        node = node.add_variation(move)
        board.push(move)

    def header(key: str, fallback: str) -> str:
        value = str(payload.get(key, fallback))
        clean = "".join(char for char in value if char.isprintable()).strip()
        return clean[:80] or fallback

    game.headers["Event"] = header("event", "Chess Study")
    game.headers["Site"] = "Local"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = "1"
    game.headers["White"] = header("white", "White")
    game.headers["Black"] = header("black", "Black")
    outcome = final_board.outcome(claim_draw=True)
    game.headers["Result"] = outcome.result() if outcome else "*"
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return game.accept(exporter) + "\n"


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, registry: ModelRegistry):
        self.registry = registry
        super().__init__(address, handler)


class RequestHandler(BaseHTTPRequestHandler):
    server: _Server
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        print("[chess-gui] " + fmt % args)

    def _send(self, status: int, content: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(content)

    def _json(self, status: int, data: dict) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/config":
            self._json(200, self.server.registry.describe())
            return
        assets = {"/": "index.html", "/app.css": "app.css", "/app.js": "app.js"}
        name = assets.get(self.path)
        if name is None:
            self._json(404, {"error": "Not found."})
            return
        target = WEB_ROOT / name
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript"}:
            content_type += "; charset=utf-8"
        try:
            self._send(200, target.read_bytes(), content_type)
        except OSError:
            self._json(500, {"error": "The app interface files could not be read."})

    def do_POST(self) -> None:  # noqa: N802
        if self.client_address[0] not in {"127.0.0.1", "::1"}:
            self._json(403, {"error": "This local app only accepts requests from this computer."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 100_000:
                raise ValueError("The request is empty or too large.")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("The request body must be a JSON object.")
            if self.path == "/api/state":
                board = _load_position(payload)
                self._json(200, _state(board))
            elif self.path == "/api/move":
                board = _load_position(payload)
                board, move = _move(board, str(payload.get("uci", "")))
                self._json(200, _state(board, move))
            elif self.path == "/api/analyze":
                self._json(200, _analyze(self.server.registry, payload))
            elif self.path == "/api/pgn":
                self._json(200, {"pgn": _export_pgn(payload)})
            else:
                self._json(404, {"error": "Not found."})
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json(400, {"error": "The request was not valid JSON."})
        except (ValueError, TypeError) as exc:
            self._json(400, {"error": str(exc)})
        except FileNotFoundError as exc:
            self._json(503, {"error": str(exc)})
        except Exception as exc:  # Keep errors visible in the local interface.
            self._json(500, {"error": f"The request could not be completed: {exc}"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bilari-checkpoint", type=Path, default=project_root / "models" / "bilari-epoch3.pt")
    parser.add_argument("--oldb-checkpoint", type=Path, default=project_root / "models" / "old-b-pilot.pt")
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser automatically.")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("The GUI must bind to a loopback address and is not a network server.")
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between 0 and 65535.")

    registry = ModelRegistry(args.bilari_checkpoint, args.oldb_checkpoint)
    server = None
    last_error = None
    ports = [args.port] if args.port == 0 else list(range(args.port, min(args.port + 11, 65536)))
    for port in ports:
        try:
            server = _Server((args.host, port), RequestHandler, registry)
            break
        except OSError as exc:
            last_error = exc
    if server is None:
        raise SystemExit(f"Could not start the local app on ports {args.port}–{ports[-1]}: {last_error}")

    address = server.server_address
    url_host = "127.0.0.1" if address[0] in {"0.0.0.0", ""} else address[0]
    url = f"http://{url_host}:{address[1]}/"
    print("Chess Study is running locally at " + url, flush=True)
    print("Press Ctrl+C in this window to close it.", flush=True)
    if not args.no_open:
        threading.Timer(0.7, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nClosing Chess Study…", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
