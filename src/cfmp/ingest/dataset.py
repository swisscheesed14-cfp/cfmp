"""Join results, venues, closing lines, and quarterbacks.

The closing line is attached so the backtest can grade it. The rating update
never reads vegas_spread.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cfmp.ingest.lines import LINES_PATH
from cfmp.ingest.qb import QB_PATH

RESULTS_PATH = Path("data/results_all.parquet")
VENUES_PATH = Path("data/venues_min.parquet")


def _attach_venues(games: pd.DataFrame) -> pd.DataFrame:
    if not VENUES_PATH.exists() or games.empty:
        games["venue_lat"] = games.get("venue_lat")
        games["venue_lon"] = games.get("venue_lon")
        return games
    venues = pd.read_parquet(VENUES_PATH)
    venues = venues.rename(columns={"id": "venue_id", "latitude": "venue_lat", "longitude": "venue_lon"})
    keep = [col for col in ("venue_id", "venue_lat", "venue_lon", "dome") if col in venues.columns]
    venues = venues[keep].drop_duplicates(subset=["venue_id"])
    if "venue_lat" in games.columns:
        games = games.drop(columns=["venue_lat", "venue_lon"], errors="ignore")
    if "venue_id" not in games.columns:
        games["venue_lat"] = None
        games["venue_lon"] = None
        return games
    return games.merge(venues, on="venue_id", how="left")


def _attach_optional(games: pd.DataFrame, path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists() or games.empty or "id" not in games.columns:
        return games
    extra = pd.read_parquet(path)
    use = ["id"] + [col for col in columns if col in extra.columns]
    extra = extra[use].drop_duplicates(subset=["id"], keep="last")
    return games.merge(extra, on="id", how="left")


def load_games(results_path: Path = RESULTS_PATH) -> pd.DataFrame:
    games = pd.read_parquet(results_path)
    games = _attach_venues(games)
    games = _attach_optional(games, LINES_PATH, ["vegas_spread", "vegas_open"])
    games = _attach_optional(games, QB_PATH, ["home_qb", "away_qb", "home_pass_rate", "away_pass_rate"])
    games["start_date"] = pd.to_datetime(games["start_date"], utc=True, errors="coerce")
    games = games.sort_values(["start_date", "season", "week", "id"], kind="stable")
    return games.reset_index(drop=True)
