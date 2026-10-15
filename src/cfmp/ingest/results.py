"""Historical results shaped for the Elo trainer.

Uses the same httpx client as the weekly slate. The old `cfbd` SDK is not a
dependency and its column names (`home_team`) did not match `train_elo`.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cfmp.ingest.cfbd import schedule

RESULTS_DIR = Path("data/results")
FIRST_SEASON = 2014


def season_year(now: datetime | None = None) -> int:
    now = now or datetime.now()
    return now.year if now.month >= 3 else now.year - 1


def _points(game: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in game and game[key] is not None:
            return game[key]
    return None


def games_to_results(games: list[dict[str, Any]], year: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for game in games:
        home = game.get("homeTeam") or game.get("home_team") or game.get("home")
        away = game.get("awayTeam") or game.get("away_team") or game.get("away")
        if not home or not away:
            continue
        neutral = game.get("neutralSite")
        if neutral is None:
            neutral = game.get("neutral_site")
        rows.append(
            {
                "season": game.get("season") or year,
                "week": game.get("week"),
                "start_date": game.get("startDate") or game.get("start_date"),
                "neutral_site": bool(neutral),
                "home": home,
                "away": away,
                "home_points": _points(game, "homePoints", "home_points"),
                "away_points": _points(game, "awayPoints", "away_points"),
                "id": game.get("id"),
                "venue": game.get("venue"),
                "venue_id": game.get("venueId") if "venueId" in game else game.get("venue_id"),
                "season_type": game.get("seasonType") or game.get("season_type"),
                "home_classification": game.get("homeClassification") or game.get("home_classification"),
                "away_classification": game.get("awayClassification") or game.get("away_classification"),
            }
        )
    return rows


def fetch_year(year: int) -> list[dict[str, Any]]:
    return games_to_results(schedule(year, season_type="both"), year)


def _jsonl_path(year: int) -> Path:
    return RESULTS_DIR / f"{year}.jsonl"


def write_year(year: int, rows: list[dict[str, Any]]) -> None:
    path = _jsonl_path(year)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def read_year(year: int) -> list[dict[str, Any]]:
    path = _jsonl_path(year)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_results(years: list[int], dest: Path, refresh_year: int | None = None) -> Path:
    """Fetch missing seasons, always refresh `refresh_year`, write one parquet."""
    frames: list[pd.DataFrame] = []
    for year in years:
        path = _jsonl_path(year)
        if year == refresh_year or not path.exists():
            rows = fetch_year(year)
            write_year(year, rows)
            print(f"Fetched {len(rows)} games for {year}")
        else:
            rows = read_year(year)
            if rows and "home_classification" not in rows[0]:
                rows = fetch_year(year)
                write_year(year, rows)
                print(f"Refetched {len(rows)} games for {year} to record division")
        frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    print(f"Wrote {len(df)} games -> {dest}")
    return dest


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv if argv is None else argv)
    if len(argv) > 1:
        years = [int(arg) for arg in argv[1:]]
    else:
        end = season_year()
        years = list(range(FIRST_SEASON, end + 1))
    dest = Path("data/results_all.parquet")
    ensure_results(years, dest, refresh_year=max(years))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
