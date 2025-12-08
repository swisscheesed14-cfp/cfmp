# src/cfmp/cli/predict_week.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cfmp.models.elo import predict_week as elo_predict

RUNS_DIR = Path("runs")


def find_latest_slate() -> Path:
    """Pick the newest runs/*/slate.csv by folder name."""
    candidates = sorted(RUNS_DIR.glob("*/slate.csv"))
    if not candidates:
        raise SystemExit("No runs/*/slate.csv found. Generate a slate first.")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Make matchup predictions for a weekly slate using Elo ratings."
    )
    p.add_argument("--elo", required=True, help="Path to elo_ratings parquet (from train_elo).")
    p.add_argument(
        "--slate",
        help="Path to runs/<YYYY-Www>/slate.csv. If omitted, the latest runs folder is used.",
    )
    p.add_argument(
        "--hfa",
        type=float,
        default=55.0,
        help="Home-field advantage in Elo points (default: 55.0).",
    )
    p.add_argument(
        "--out",
        help=(
            "Optional explicit output file path. "
            "If provided, writes to this file (csv or parquet by extension). "
            "If omitted, writes predictions.csv next to the slate."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Load inputs
    elo_path = Path(args.elo)
    if not elo_path.exists():
        raise SystemExit(f"Elo ratings file not found: {elo_path}")

    slate_path = Path(args.slate) if args.slate else find_latest_slate()
    if not slate_path.exists():
        raise SystemExit(f"Slate file not found: {slate_path}")

    ratings = pd.read_parquet(elo_path)
    slate = pd.read_csv(slate_path)

    # ---- Predict
    preds = elo_predict(slate, ratings, hfa=args.hfa)

    # ---- Decide output location
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".parquet":
            preds.to_parquet(out_path, index=False)
        else:
            # default to CSV for unknown/empty extensions as well
            if out_path.suffix.lower() != ".csv":
                out_path = out_path.with_suffix(".csv")
            preds.to_csv(out_path, index=False)
        print(f"Wrote predictions to {out_path}")
        return

    # Default behavior: write next to the slate (this is the requested change)
    out_dir = slate_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "predictions.csv"
    preds.to_csv(out_file, index=False)
    print(f"Wrote predictions to {out_file}")


if __name__ == "__main__":
    main()
