"""
generate_data.py
-----------------
Generates a synthetic "will I procrastinate on this task?" dataset.

There's no public dataset for this (obviously), so we simulate one with a
hand-crafted ground-truth function plus noise. The function encodes a few
intuitive, debatable-over-chai hypotheses about procrastination:

  * More hours until the deadline  -> more room to procrastinate.
  * More open browser tabs / social media checks -> more distraction.
  * Harder tasks get avoided more (task aversiveness).
  * Better sleep and a higher mood score -> more self-control.
  * A strong track record of finishing things on time -> less procrastination.
  * Weekends nudge things up a little (no judgment).

None of this is scientific. It's a believable simulator for a fun DL demo,
not a psychology paper.

Usage:
    python generate_data.py --n 5000 --out procrastination_data.csv
"""

import argparse
import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "hours_until_deadline",
    "task_difficulty",
    "hours_slept",
    "cups_of_coffee",
    "open_browser_tabs",
    "social_media_checks_per_hour",
    "mood_score",
    "previous_completion_rate",
    "is_weekend",
    "noise_level",
]


def _simulate(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    hours_until_deadline = rng.exponential(scale=20, size=n).clip(0.5, 168)       # up to a week
    task_difficulty = rng.integers(1, 11, size=n).astype(float)                   # 1-10
    hours_slept = rng.normal(6.5, 1.3, size=n).clip(2, 10)
    cups_of_coffee = rng.poisson(1.8, size=n).clip(0, 8).astype(float)
    open_browser_tabs = rng.poisson(12, size=n).clip(1, 60).astype(float)
    social_media_checks_per_hour = rng.poisson(4, size=n).clip(0, 30).astype(float)
    mood_score = rng.integers(1, 11, size=n).astype(float)                        # 1-10
    previous_completion_rate = rng.beta(5, 2, size=n)                             # skewed high, 0-1
    is_weekend = rng.integers(0, 2, size=n).astype(float)
    noise_level = rng.integers(1, 11, size=n).astype(float)                       # environment distraction

    # Standardize each raw feature to roughly comparable scale before weighting,
    # so no single feature with a big numeric range silently dominates.
    def z(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    logit = (
        0.9 * z(hours_until_deadline)
        + 0.5 * z(task_difficulty)
        + 0.6 * z(open_browser_tabs)
        + 0.7 * z(social_media_checks_per_hour)
        + 0.3 * z(noise_level)
        + 0.25 * z(is_weekend)
        - 0.5 * z(hours_slept)
        - 0.4 * z(mood_score)
        - 0.8 * z(previous_completion_rate)
        - 0.15 * z(cups_of_coffee)          # a little coffee helps; too much is its own feature if you want
        + rng.normal(0, 0.6, size=n)        # irreducible randomness, because humans
    )

    probability = 1 / (1 + np.exp(-logit))
    will_procrastinate = (rng.random(n) < probability).astype(int)

    df = pd.DataFrame(
        {
            "hours_until_deadline": hours_until_deadline.round(2),
            "task_difficulty": task_difficulty,
            "hours_slept": hours_slept.round(2),
            "cups_of_coffee": cups_of_coffee,
            "open_browser_tabs": open_browser_tabs,
            "social_media_checks_per_hour": social_media_checks_per_hour,
            "mood_score": mood_score,
            "previous_completion_rate": previous_completion_rate.round(3),
            "is_weekend": is_weekend,
            "noise_level": noise_level,
            "procrastination_probability": probability.round(4),
            "will_procrastinate": will_procrastinate,
        }
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic procrastination data.")
    parser.add_argument("--n", type=int, default=5000, help="Number of rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--out",
        type=str,
        default="data/procrastination_data.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    df = _simulate(args.n, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    print(f"Positive class rate (will_procrastinate=1): {df['will_procrastinate'].mean():.3f}")


if __name__ == "__main__":
    main()
