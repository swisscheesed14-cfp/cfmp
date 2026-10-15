"""Price the weekly slate: neutral rating, site spread, and the Vegas number beside it."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cfmp.ingest.lines import LINES_PATH
from cfmp.models.params import load_params
from cfmp.models.power import predict_slate, top25

RUNS_DIR = Path("runs")
RATINGS = Path("data/power_ratings.parquet")


def find_latest_slate() -> Path:
    """Newest slate by file time, not folder name."""
    candidates = list(RUNS_DIR.glob("*/slate.csv"))
    if not candidates:
        raise SystemExit("No runs/*/slate.csv found. Generate a slate first.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _attach_lines(slate: pd.DataFrame) -> pd.DataFrame:
    if not LINES_PATH.exists() or "id" not in slate.columns:
        slate["vegas_spread"] = pd.NA
        slate["vegas_open"] = pd.NA
        return slate
    lines = pd.read_parquet(LINES_PATH)
    use = lines[["id", "vegas_spread", "vegas_open"]].drop_duplicates(subset=["id"], keep="last")
    out = slate.drop(columns=["vegas_spread", "vegas_open"], errors="ignore")
    return out.merge(use, on="id", how="left")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Price a weekly slate against a neutral-field rating.")
    parser.add_argument("--ratings", default=str(RATINGS), help="Path to power_ratings parquet.")
    parser.add_argument("--slate", help="Path to a slate CSV. Defaults to the newest run.")
    parser.add_argument("--params", help="Optional params JSON. Defaults to the package file.")
    parser.add_argument("--out", help="Output CSV. Defaults to predictions.csv next to the slate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratings_path = Path(args.ratings)
    if not ratings_path.exists():
        raise SystemExit(f"Ratings not found: {ratings_path}. Run python -m cfmp.cli.train first.")
    slate_path = Path(args.slate) if args.slate else find_latest_slate()
    if not slate_path.exists():
        raise SystemExit(f"Slate file not found: {slate_path}")

    params = load_params(Path(args.params) if args.params else None)
    ratings = pd.read_parquet(ratings_path)
    slate = _attach_lines(pd.read_csv(slate_path))
    preds = predict_slate(slate, ratings, params)
    keep = [
        "id", "year", "week", "kickoff", "away", "home", "neutral_site", "venue", "indoor",
        "rating_away", "rating_home", "neutral_spread", "hfa_points", "weather_adj",
        "rest_adj", "travel_adj", "qb_adj", "site_spread", "vegas_spread", "vegas_open",
        "edge", "win_prob_home", "wx_temp_f", "wx_wind_mph", "wx_pop", "wx_short",
    ]
    ordered = [column for column in keep if column in preds.columns]
    extras = [column for column in preds.columns if column not in ordered]
    preds = preds[ordered + extras]

    out_path = Path(args.out) if args.out else slate_path.parent / "predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_path, index=False)
    priced = preds["vegas_spread"].notna().sum() if "vegas_spread" in preds.columns else 0
    print(f"Wrote {len(preds)} games ({priced} with a market number) -> {out_path}")
    board = top25(ratings)
    board_path = out_path.parent / "top25.csv"
    board.to_csv(board_path, index=False)
    print(f"Wrote neutral-field top 25 -> {board_path}")
    print(board.to_string(index=False))


if __name__ == "__main__":
    main()
