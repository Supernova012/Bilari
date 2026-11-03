# Bilari

Bilari is a puzzle-trained chess move suggester. Give it a FEN and it returns five candidate moves. It includes a no-look-ahead policy mode and policy-guided alpha-beta search at several time budgets, plus a local browser GUI.

Bilari was trained using Lichess puzzles only. It is an experimental puzzle model, not a general-purpose chess engine and not a substitute for Stockfish.

## Included models

- `models/bilari-epoch3.pt` — corrected epoch-3 policy, trained on all solver decisions in each puzzle line.
- `models/old-b-pilot.pt` — earlier one-epoch pilot trained on the first 20,000 puzzle rows. It is included only to reproduce the GUI's Bilari-versus-Old-B option.

The network has two residual blocks with 16 channels and a 4,672-action policy head. Its value head was not trained from puzzle ratings. Search uses the policy for move ordering and a material/piece-square leaf evaluator.

## Quick start

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[train]'
```

Ask for five moves from a FEN:

```bash
.venv/bin/python -m chess_intuition.play \
  --checkpoint models/bilari-epoch3.pt \
  --fen 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' \
  --mode pure \
  --top 5
```

Modes are `pure`, `low`, `medium`, and `high`. `pure` makes one policy inference and performs no search. Search budgets are 250 ms, 1 second, and 3 seconds. The FEN's active-color field determines whose moves are ranked.

Launch the browser GUI:

```bash
.venv/bin/python -m chess_intuition.gui \
  --bilari-checkpoint models/bilari-epoch3.pt \
  --oldb-checkpoint models/old-b-pilot.pt
```

The GUI offers Puzzle and Game modes, click or drag moves, PGN export, and Intuition/Low/Medium/High/Ultra settings. Ultra is limited to 30 seconds per move. The server listens on the local computer only.

## Training from Lichess puzzles

The puzzle CSV and generated training shards are not included. Download the current puzzle export from the [Lichess Open Database](https://database.lichess.org/), then preprocess it:

```bash
.venv/bin/chess-intuition-preprocess \
  --csv /path/to/lichess_db_puzzle.csv \
  --out data/processed
```

Preprocessing applies the initial trigger move and creates a training example for every solver decision in the solution line, keeping all examples from one puzzle in the same split.

Train a policy:

```bash
.venv/bin/chess-intuition-train \
  --data data/processed \
  --checkpoint checkpoints/puzzle-policy.pt \
  --epochs 3 \
  --batch-size 256 \
  --device auto
```

Lichess says its database exports, including puzzles, are released under CC0. This repository does not redistribute the puzzle CSV or processed puzzle positions. Source code and included model weights are licensed under MIT; see [LICENSE](LICENSE).

## Evaluation

On the held-out test split, Bilari's pure policy matched the recorded first solver move on 51.92% of 307,754 puzzles and placed it in the top five on 87.40%. This measures agreement with the published solution, not full puzzle completion or game strength.

The separate 500-puzzle benchmark estimates puzzle Glicko ratings from first-move correctness. It estimated 1523 for Intuition, 1885 for Low, 1989 for Medium, and 2008 for High. High processed 100 puzzles in its five-minute run; the estimate is therefore especially uncertain. These are sample-based puzzle-rating estimates, not chess-game Elo or official Lichess ratings. Details and limitations are in [results/BENCHMARKS.md](results/BENCHMARKS.md).

## Repository contents

- `src/chess_intuition/` — model, training, inference, search, and GUI.
- `models/` — Bilari and the small historical Old-B comparison checkpoint.
- `tests/` — unit tests for board encoding, preprocessing, inference, and search.
- `results/` — benchmark summary and methodology.

## License

Source code and included weights: MIT. The Lichess puzzle export is a separate CC0 work and is not included here.
