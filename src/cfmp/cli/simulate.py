"""Simulate a neutral-field game, or every FBS team against every other.

Uses the power rating plus the run/pass tilt. Scores come from the public
CollegeFootballData-derived EPA splits (rush and pass, per play). A
CFBD_KEY is not required for this file; the live API still needs one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cfmp.models.style import all_neutral, build_styles, ratings_from_schedule, simulate_pair

SPLIT_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "cfb_team_opponent_splits/cfb_team_opponent_splits_{year}.parquet"
)
SCHED_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "cfb_schedules/cfb_schedules_{year}.parquet"
)
FIRST = 2019


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def load_frames(current: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = []
    schedules = []
    for year in range(FIRST, current + 1):
        split_path = Path(f"data/style/{year}.parquet")
        sched_path = Path(f"data/schedules/{year}.parquet")
        _download(SPLIT_URL.format(year=year), split_path)
        _download(SCHED_URL.format(year=year), sched_path)
        splits.append(pd.read_parquet(split_path))
        schedules.append(pd.read_parquet(sched_path))
    return pd.concat(splits, ignore_index=True), pd.concat(schedules, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate neutral-field matchups from run/pass splits.")
    parser.add_argument("team", nargs="?", help="Team to price. Omit with --all.")
    parser.add_argument("opponent", nargs="?", help="Opponent.")
    parser.add_argument("--all", action="store_true", help="Write every FBS-vs-FBS neutral simulation.")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--out", default="data/neutral_matchups.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits, schedule = load_frames(args.season)
    ratings = ratings_from_schedule(schedule)
    styles = build_styles(splits, schedule, args.season)
    if args.all or not args.team:
        board = all_neutral(ratings, styles)
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        board.to_csv(dest, index=False)
        print(f"Wrote {len(board)} neutral matchups -> {dest}")
        return
    if not args.opponent:
        raise SystemExit("Give an opponent, or pass --all.")
    result = simulate_pair(args.team, args.opponent, ratings, styles)
    favored = args.team if result["spread"] >= 0 else args.opponent
    print(
        f"{args.team} {result['spread']:+.1f} vs {args.opponent} on a neutral field "
        f"({favored} by {abs(result['spread']):.1f})."
    )
    print(
        f"Score rating {result['base_spread']:+.1f}, run/pass tilt {result['style_adj']:+.1f}. "
        f"In {int(result['sims'])} sims, {args.team} won {result['win_prob']:.1%}. "
        f"Typical margin {result['p10']:+.0f} to {result['p90']:+.0f}."
    )


if __name__ == "__main__":
    main()
