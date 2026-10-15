"""Refresh schedules and snap stats from CFBD.

Historical seasons are downloaded once. Later runs only replace the current
season, which is the part that changes during the week.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

from cfmp.config import SETTINGS
from cfmp.ingest.results import season_year

BASE = "https://api.collegefootballdata.com"
SNAPS_PATH = Path("data/cfbd/game_model.parquet")
SCHEDULE_DIR = Path("data/schedules")
FIRST_SEASON = 2014


def _get(path: str, params: dict) -> list:
    if not SETTINGS.cfbd_key:
        raise SystemExit("CFBD_KEY is not set.")
    headers = {"Authorization": f"Bearer {SETTINGS.cfbd_key}", "Accept": "application/json"}
    for attempt in range(4):
        response = httpx.get(BASE + path, params=params, headers=headers, timeout=120)
        if response.status_code == 200:
            return response.json()
        if response.status_code in (429, 500, 502, 503) and attempt < 3:
            time.sleep(1.5 * (attempt + 1))
            continue
        response.raise_for_status()
    return []


def _ratio(text: object) -> float | None:
    if text is None or "-" not in str(text):
        return None
    made, tries = str(text).split("-", 1)
    try:
        tries_n = float(tries)
    except ValueError:
        return None
    if tries_n <= 0:
        return None
    return float(made) / tries_n


def _num(text: object) -> float | None:
    try:
        return float(text)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _schedule_rows(year: int) -> list[dict]:
    rows = []
    for game in _get("/games", {"year": year, "seasonType": "regular"}):
        rows.append(
            {
                "game_id": game.get("id"),
                "season": game.get("season", year),
                "week": game.get("week"),
                "start_date": game.get("startDate"),
                "home_team": game.get("homeTeam"),
                "away_team": game.get("awayTeam"),
                "home_points": game.get("homePoints"),
                "away_points": game.get("awayPoints"),
                "neutral_site": bool(game.get("neutralSite")),
                "home_division": str(game.get("homeClassification") or "").lower(),
                "away_division": str(game.get("awayClassification") or "").lower(),
            }
        )
    return rows
