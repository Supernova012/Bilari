"""Replay puzzle benchmark outcomes through Lichess-style Glicko-2 updates."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import median


SCALE = 173.7178
TAU = 0.75
INITIAL_RATING = 1500.0
INITIAL_RD = 500.0
INITIAL_VOLATILITY = 0.09
def g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def expected(mu: float, opponent_mu: float, opponent_phi: float) -> float:
    return 1.0 / (1.0 + math.exp(-g(opponent_phi) * (mu - opponent_mu)))


def next_volatility(mu: float, phi: float, sigma: float, variance: float, delta: float) -> float:
    a = math.log(sigma * sigma)

    def f(x: float) -> float:
        ex = math.exp(x)
        denom = phi * phi + variance + ex
        return ex * (delta * delta - phi * phi - variance - ex) / (2.0 * denom * denom) - (x - a) / (TAU * TAU)

    A = a
    if delta * delta > phi * phi + variance:
        B = math.log(delta * delta - phi * phi - variance)
    else:
        k = 1
        B = a - k * TAU
        while f(B) < 0.0:
            k += 1
            B = a - k * TAU

    fA, fB = f(A), f(B)
    iterations = 0
    while abs(B - A) > 1e-6 and iterations < 1000:
        iterations += 1
        C = A + ((A - B) * fA) / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA /= 2.0
        B, fB = C, fC
    if iterations >= 1000:
        raise ArithmeticError("Glicko volatility iteration failed to converge")
    return math.exp(A / 2.0)


def cap_glicko(rating: float, rd: float, volatility: float) -> tuple[float, float, float]:
    return min(4000.0, max(400.0, rating)), min(500.0, max(45.0, rd)), min(0.1, volatility)


def glicko_game(
    rating: float,
    rd: float,
    sigma: float,
    opponent_rating: float,
    opponent_rd: float,
    score: float,
) -> tuple[float, float, float]:
    """One single-game update, matching Scalachess GlickoCalculator defaults."""
    mu = (rating - INITIAL_RATING) / SCALE
    phi = rd / SCALE
    opponent_mu = (opponent_rating - INITIAL_RATING) / SCALE
    opponent_phi = opponent_rd / SCALE
    impact = g(opponent_phi)
    exp_score = expected(mu, opponent_mu, opponent_phi)
    variance = 1.0 / (impact * impact * exp_score * (1.0 - exp_score))
    outcome_delta = impact * (score - exp_score)
    delta = variance * outcome_delta
    new_sigma = next_volatility(mu, phi, sigma, variance, delta)
    phi_star = math.sqrt(phi * phi + new_sigma * new_sigma)  # one game = one rating period
    new_phi = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / variance)
    new_mu = mu + new_phi * new_phi * outcome_delta
    return INITIAL_RATING + SCALE * new_mu, SCALE * new_phi, new_sigma


def play_attempt(rating: float, rd: float, sigma: float, puzzle: dict, solved: bool) -> tuple[float, float, float]:
    opponent_rating = min(4000.0, max(400.0, float(puzzle["rating"])))
    opponent_rd = min(500.0, max(45.0, float(puzzle["rating_deviation"])))
    raw_rating, raw_rd, raw_sigma = glicko_game(
        rating,
        rd,
        sigma,
        opponent_rating,
        opponent_rd,
        1.0 if solved else 0.0,
    )

    # Lila's Perf.addOrReset and Glicko.cap limits.
    raw_rating = min(rating + 700.0, max(rating - 700.0, raw_rating))
    new_rating, new_rd, new_sigma = cap_glicko(raw_rating, raw_rd, raw_sigma)

    # PuzzleFinisher's ponder.player weighting for a mixed puzzle session.
    # Puzzle RD >= 110 is provisional: Lichess weights a win by .8 and a loss by .3.
    weight = 1.0
    if opponent_rd >= 110.0:
        weight = 0.8 if solved else 0.3
    rating = rating * (1.0 - weight) + new_rating * weight
    rd = rd * (1.0 - weight) + new_rd * weight
    sigma = sigma * (1.0 - weight) + new_sigma * weight
    return cap_glicko(rating, rd, sigma)


def replay(rows: list[dict], mode: str) -> dict:
    rating, rd, sigma = INITIAL_RATING, INITIAL_RD, INITIAL_VOLATILITY
    for puzzle in rows:
        rating, rd, sigma = play_attempt(
            rating,
            rd,
            sigma,
            puzzle,
            bool(puzzle["modes"][mode]["top1_hit"]),
        )
    return {"rating": rating, "rating_deviation": rd, "volatility": sigma}


def percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * proportion
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def estimate(input_path: Path, output_path: Path, permutations: int, seed: int) -> dict:
    benchmark = json.loads(input_path.read_text(encoding="utf-8"))
    rows = benchmark["per_puzzle"]
    if not rows:
        raise ValueError("Benchmark contains no puzzle records")
    modes = tuple(rows[0]["modes"])
    if any(not mode in row["modes"] for row in rows for mode in modes):
        raise ValueError("Puzzle records have inconsistent mode results")

    order_results = {mode: [] for mode in modes}
    for index in range(permutations):
        replay_order = list(benchmark["per_puzzle"])
        order_seed = seed + index
        random.Random(order_seed).shuffle(replay_order)
        for mode in modes:
            outcome = replay(replay_order, mode)
            outcome["order_seed"] = order_seed
            order_results[mode].append(outcome)

    summary = {}
    for mode in modes:
        outcomes = order_results[mode]
        ratings = [item["rating"] for item in outcomes]
        median_rating = median(ratings)
        representative = min(outcomes, key=lambda item: abs(item["rating"] - median_rating))
        final = representative
        summary[mode] = {
            "estimated_puzzle_glicko": round(final["rating"], 1),
            "rating_deviation": round(final["rating_deviation"], 1),
            "lichess_style_approx_95_interval": [
                int(final["rating"] - 2.0 * final["rating_deviation"]),
                int(final["rating"] + 2.0 * final["rating_deviation"]),
            ],
            "volatility": round(final["volatility"], 5),
            "representative_order_seed": final["order_seed"],
            "order_sensitivity": {
                "permutations": permutations,
                "rating_median": round(median_rating, 1),
                "rating_p2_5_to_p97_5": [
                    round(percentile(ratings, 0.025), 1),
                    round(percentile(ratings, 0.975), 1),
                ],
                "rating_min_to_max": [round(min(ratings), 1), round(max(ratings), 1)],
            },
        }

    result = {
        "estimate": "Lichess-style Glicko-2 replay of the benchmark top-1 outcomes",
        "input_benchmark": str(input_path.resolve()),
        "sample_size": len(rows),
        "rating_initialization": {
            "rating": INITIAL_RATING,
            "deviation": INITIAL_RD,
            "volatility": INITIAL_VOLATILITY,
        },
        "attempt_assumptions": [
            "Treat each mode as a new puzzle-rating account starting at the Lichess default 1500/500/0.09.",
            "Each top-1 hit is a win and each miss a loss; each puzzle is a separate single-game Glicko period.",
            "Use each puzzle's CSV rating and rating deviation as the opponent rating and RD; puzzle ratings are held fixed at the exported snapshot.",
            "Use the mixed-puzzle weighting in PuzzleFinisher, including its reduced weight for provisional puzzles (RD >= 110).",
            "Each mode is replayed in the same deterministic randomized order; separate permutations show order sensitivity.",
            "The benchmark records first-response move matches only, whereas Lichess rates completion of the full puzzle line.",
        ],
        "modes": summary,
        "warning": "These are first-move Glicko-equivalent estimates for the evaluated puzzle sample, not official Lichess ratings or chess-game Elo.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/puzzle_benchmark_500_results.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/puzzle_benchmark_500_glicko.json"))
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    if args.permutations < 1:
        parser.error("--permutations must be positive")
    result = estimate(args.input, args.output, args.permutations, args.seed)
    print(json.dumps(result["modes"], indent=2))
    print("Full estimate: " + str(args.output.resolve()))


if __name__ == "__main__":
    main()
