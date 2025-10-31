# cfmp/cli/predict_week.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Tuple

import pandas as pd

from cfmp.models.elo import predict_week as elo_predict


RUNS_DIR = Path("runs")
ELO_FILE = Path("data/elo_ratings.parquet")


def _pick_year_week() -> Tuple[int, int]:
    # env overrides
    ey = os.getenv("CFB_YEAR")
    ew = os.getenv("CFB_WEEK")
    if ey and ew:
        return int(ey), int(ew)

    # latest folder like 2025-W10
    if not RUNS_DIR.exists():
        raise SystemExit("runs/ does not exist. Generate slate first.")
    pats = []
    for p in RUNS_DIR.iterdir():
        m = re.match(r"^(\d{4})-W(\d+)$", p.name)
        if m:
            pats.append((int(m.group(1)), int(m.group(2)), p))
    if not pats:
        raise SystemExit("No runs/YYYY-W## folders found.")
    pats.sort()  # by year, week ascending
    y, w, _ = pats[-1]
    return y, w


def main():
    year, week = _pick_year_week()
    run_dir = RUNS_DIR / f"{year}-W{week}"
    slate_csv = run_dir / "slate.csv"
    if not slate_csv.exists():
        raise SystemExit(f"Slate not found: {slate_csv}")

    if not ELO_FILE.exists():
        raise SystemExit("Missing data/elo_ratings.parquet. Run: python -m cfmp.cli.train_elo")

    slate = pd.read_csv(slate_csv)
    ratings = pd.read_parquet(ELO_FILE)

    preds = elo_predict(slate, ratings)
    out = run_dir / "predictions.csv"
    preds.to_csv(out, index=False)
    print(f"Wrote predictions -> {out} ({len(preds)} rows)")


if __name__ == "__main__":
    main()
