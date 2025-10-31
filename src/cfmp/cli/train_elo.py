# cfmp/cli/train_elo.py
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from cfmp.ingest.results import main as fetch_results_main  # reuse if needed
from cfmp.models.elo import EloConfig, train_elo


DATA_DIR = Path("data")
RESULTS_ALL = DATA_DIR / "results_all.parquet"
ELO_OUT = DATA_DIR / "elo_ratings.parquet"


def _ensure_results():
    if RESULTS_ALL.exists():
        return
    # If not present, fetch all years (respects CFBD_KEY inside ingest module)
    fetch_results_main([])


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    _ensure_results()
    df = pd.read_parquet(RESULTS_ALL)
    ratings = train_elo(df, EloConfig())
    ratings["updated_at"] = pd.Timestamp.utcnow().isoformat()
    ratings.to_parquet(ELO_OUT, index=False)
    print(f"Saved {len(ratings)} team ratings -> {ELO_OUT}")


if __name__ == "__main__":
    main()
