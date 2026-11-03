# Puzzle benchmark summary

All measurements use puzzle-only data and score the first solver move against the recorded solution, with the benchmark's accepted alternate-mate-in-one handling. A correct first move does not imply the full puzzle was solved.

## Held-out test split

The pure policy was evaluated on 307,754 held-out puzzles:

| Measure | Result |
| --- | ---: |
| Top-1 first-move agreement | 51.92% |
| Top-5 first-move agreement | 87.40% |

## Search-mode sample

Intuition, Low, and Medium used the same 500 random puzzles. High ran for a five-minute wall-clock budget on a shuffled subset of that same sample and completed 100 puzzles.

| Mode | Puzzles | Top-1 | Top-5 | Estimated puzzle Glicko-2 |
| --- | ---: | ---: | ---: | ---: |
| Intuition | 500 | 49.6% | 85.2% | 1523 |
| Low | 500 | 70.4% | 85.8% | 1885 |
| Medium | 500 | 76.2% | 90.2% | 1989 |
| High | 100 | 77.0% | 90.0% | 2008 |

The Glicko estimates replay top-1 hits/misses as puzzle outcomes using puzzle ratings and the documented Lichess-style Glicko-2 method. Values are sample-based estimates, sensitive to ordering and starting uncertainty; they are not chess-game Elo or official Lichess ratings. High has only 100 results, so its estimate is especially uncertain and is not strong evidence that High is better than Medium.

## Reproduction scope

The original random sample and per-puzzle benchmark outputs are intentionally not included. To reproduce the work, download the Lichess puzzle export, rebuild the deterministic position-grouped splits, and run the benchmark scripts in `src/chess_intuition/training/`. The Lichess export is CC0 and available from <https://database.lichess.org/>.
