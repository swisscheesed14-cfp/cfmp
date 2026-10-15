"""Train the neutral-field ratings on every completed game.

Also refreshes results, closing lines, and quarterback rows. The line is
stored beside the games and is not used to move a rating.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cfmp.ingest.dataset import load_games
from cfmp.ingest.lines import ensure_lines
from cfmp.ingest.qb import ensure_qb
from cfmp.ingest.results import FIRST_SEASON, ensure_results, season_year
from cfmp.models.params import load_params
from cfmp.models.power import final_ratings, top25
from cfmp.models.priors import load_priors
from cfmp.models.process import attach_process

RESULTS = Path("data/results_all.parquet")
RATINGS = Path("data/power_ratings.parquet")


def main() -> None:
    end = season_year(datetime.now(timezone.utc))
    years = list(range(FIRST_SEASON, end + 1))
    ensure_results(years, RESULTS, refresh_year=end)
    try:
        ensure_lines(years, refresh_year=end)
    except Exception as exc:
        print(f"Lines pull failed, backtest will have no market number: {exc}")
    ensure_qb(years, refresh_year=end)
    games = attach_process(load_games(RESULTS))
    ratings = final_ratings(games, load_params(), load_priors())
    ratings["updated_at"] = pd.Timestamp.utcnow().isoformat()
    RATINGS.parent.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(RATINGS, index=False)
    board = top25(ratings)
    board_path = Path("data/top25.csv")
    board.to_csv(board_path, index=False)
    print(f"Saved {len(ratings)} team ratings -> {RATINGS}")
    print(board.to_string(index=False))


if __name__ == "__main__":
    main()
