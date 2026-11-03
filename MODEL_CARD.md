# Bilari model card

## Summary

Bilari is a compact chess policy network trained from Lichess puzzle solution lines. It ranks legal moves from a FEN. Search modes use the policy to order moves and use a handcrafted material/piece-square evaluator at leaf nodes.

## Model

- Architecture: compact residual CNN, 2 residual blocks, 16 channels.
- Input: 19 piece/side/castling/turn feature planes.
- Output: 4,672 policy actions, filtered to legal moves for inference.
- Training objective: cross-entropy over legal moves.
- Value head: present in the network, but not trained from puzzle ratings.
- Included checkpoint: corrected epoch 3, trained using every solver-to-move position in each puzzle line.

## Data

Training and evaluation used Lichess puzzle positions and recorded solution lines. The preprocessing code applies each puzzle's initial trigger move, then creates examples for all solver turns while replaying opponent replies. Splits are grouped by puzzle so one puzzle cannot appear in multiple splits.

The source CSV and processed shards are not included. Lichess publishes its puzzle database as CC0 at <https://database.lichess.org/>.

## Evaluation

On the held-out test split of 307,754 puzzles, the pure policy selected the recorded first solver move in 51.92% of cases and ranked it in the top five in 87.40% of cases. This is first-move agreement, not full puzzle solve rate.

On a fixed 500-puzzle sample, the estimated puzzle Glicko ratings were 1523 (Intuition), 1885 (Low), and 1989 (Medium). High mode was run for five minutes and covered 100 puzzles, yielding an estimate of 2008. The High estimate has a wider uncertainty and should not be treated as evidence of a meaningful improvement over Medium. These numbers are not chess-game Elo and are not official Lichess ratings. See [the benchmark details](results/BENCHMARKS.md).

## Limitations and intended use

Bilari learns move selection from tactical puzzle solutions. It does not receive general game-position value supervision, and its search evaluator is handcrafted. Puzzle accuracy and puzzle-rating estimates do not establish tournament or general-play strength. Treat the model as a research and study project; verify critical moves with a trusted chess engine.

## License

The source and included model checkpoints are licensed under MIT. The external Lichess puzzle dataset is a separate CC0 work and is not redistributed by this repository.
