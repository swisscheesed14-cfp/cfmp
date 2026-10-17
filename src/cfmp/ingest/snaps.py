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


def _pause(response: httpx.Response, attempt: int) -> None:
    """Wait out a rate limit. CollegeFootballData asks for longer than a couple of seconds."""
    header = response.headers.get("Retry-After")
    try:
        pause = float(header) if header else 20 * (attempt + 1)
    except ValueError:
        pause = 20 * (attempt + 1)
    time.sleep(min(max(pause, 5), 90))


def _get(path: str, params: dict) -> list:
    if not SETTINGS.cfbd_key:
        raise SystemExit("CFBD_KEY is not set.")
    headers = {"Authorization": f"Bearer {SETTINGS.cfbd_key}", "Accept": "application/json"}
    for attempt in range(5):
        response = httpx.get(BASE + path, params=params, headers=headers, timeout=120)
        if response.status_code == 200:
            return response.json()
        if response.status_code in (429, 500, 502, 503) and attempt < 4:
            _pause(response, attempt)
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
                "home_conference": game.get("homeConference"),
                "away_conference": game.get("awayConference"),
                "conference_game": bool(game.get("conferenceGame")),
                "venue": game.get("venue"),
                "venue_id": game.get("venueId"),
            }
        )
    return rows


def _advanced_rows(year: int) -> list[dict]:
    rows = []
    for row in _get("/stats/game/advanced", {"year": year}):
        off, de = row.get("offense") or {}, row.get("defense") or {}
        rush, pas = off.get("rushingPlays") or {}, off.get("passingPlays") or {}
        dr, dp = de.get("rushingPlays") or {}, de.get("passingPlays") or {}
        std, pdn = off.get("standardDowns") or {}, off.get("passingDowns") or {}
        dstd, dpdn = de.get("standardDowns") or {}, de.get("passingDowns") or {}
        rows.append(
            {
                "season": row.get("season", year),
                "week": row.get("week"),
                "game_id": row.get("gameId"),
                "team": row.get("team"),
                "rush_ppa": rush.get("totalPPA"),
                "def_rush_ppa": dr.get("totalPPA"),
                "pass_ppa": pas.get("totalPPA"),
                "def_pass_ppa": dp.get("totalPPA"),
                "line_yards": off.get("lineYards"),
                "def_line": de.get("lineYards"),
                "second_yards": off.get("secondLevelYards"),
                "def_second": de.get("secondLevelYards"),
                "open_yards": off.get("openFieldYards"),
                "def_open": de.get("openFieldYards"),
                "stuff_rate": off.get("stuffRate"),
                "def_stuff": de.get("stuffRate"),
                "off_success": off.get("successRate"),
                "def_success": de.get("successRate"),
                "explosiveness": off.get("explosiveness"),
                "def_explosiveness": de.get("explosiveness"),
                "power_success": off.get("powerSuccess"),
                "def_power": de.get("powerSuccess"),
                "standard_success": std.get("successRate"),
                "def_standard_success": dstd.get("successRate"),
                "passing_success": pdn.get("successRate"),
                "def_passing_success": dpdn.get("successRate"),
            }
        )
    return rows


def _box(year: int) -> dict[tuple, dict]:
    found: dict[tuple, dict] = {}
    for week in range(1, 17):
        for game in _get("/games/teams", {"year": year, "week": week, "seasonType": "regular"}):
            sides = []
            for side in game.get("teams") or []:
                stats = {item["category"]: item["stat"] for item in side.get("stats") or []}
                attempts = _num(stats.get("rushingAttempts")) or 0.0
                passes = 0.0
                completion = None
                comp = stats.get("completionAttempts")
                if comp and "-" in str(comp):
                    made, tried = str(comp).split("-", 1)
                    try:
                        passes = float(tried)
                        if passes > 0:
                            completion = float(made) / passes
                    except ValueError:
                        passes = 0.0
                plays = attempts + passes
                yards = _num(stats.get("totalYards"))
                sides.append(
                    {
                        "team": side.get("team"),
                        "third_down": _ratio(stats.get("thirdDownEff")),
                        "fourth_down": _ratio(stats.get("fourthDownEff")),
                        "ypp": None if not plays or yards is None else yards / plays,
                        "ypr": _num(stats.get("yardsPerRushAttempt")),
                        "ypa": _num(stats.get("yardsPerPass")),
                        "turnovers": _num(stats.get("turnovers")),
                        "sacks": _num(stats.get("sacks")),
                        "tfl": _num(stats.get("tacklesForLoss")),
                        "completion": completion,
                    }
                )
            if len(sides) != 2:
                continue
            first, second = sides
            for own, opp in ((first, second), (second, first)):
                found[(game.get("id"), own["team"])] = {
                    "third_down": own["third_down"],
                    "def_third": opp["third_down"],
                    "fourth_down": own["fourth_down"],
                    "def_fourth": opp["fourth_down"],
                    "ypp": own["ypp"],
                    "def_ypp": opp["ypp"],
                    "ypr": own["ypr"],
                    "def_ypr": opp["ypr"],
                    "ypa": own["ypa"],
                    "def_ypa": opp["ypa"],
                    "turnovers": own["turnovers"],
                    "def_turnovers": opp["turnovers"],
                    "sacks_allowed": opp["sacks"],
                    "def_sacks": own["sacks"],
                    "tfl_allowed": opp["tfl"],
                    "def_tfl": own["tfl"],
                    "completion": own["completion"],
                    "def_completion": opp["completion"],
                }
    return found


def _season_frame(year: int) -> pd.DataFrame:
    print(f"snaps {year}", flush=True)
    frame = pd.DataFrame(_advanced_rows(year))
    box = _box(year)
    extra = [box.get((row.game_id, row.team), {}) for row in frame.itertuples(index=False)]
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(extra)], axis=1)


def refresh(now: datetime | None = None) -> None:
    """Bring schedules and snaps up to the current season."""
    end = season_year(now or datetime.now(timezone.utc))
    years = list(range(FIRST_SEASON, end + 1))
    SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)
    for year in years:
        path = SCHEDULE_DIR / f"cfb_schedules_{year}.parquet"
        if path.exists() and year != end:
            continue
        print(f"schedule {year}", flush=True)
        try:
            rows = _schedule_rows(year)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429 and path.exists():
                print(f"schedule {year} is rate limited, keeping the file on disk", flush=True)
                continue
            raise
        pd.DataFrame(rows).to_parquet(path, index=False)
    if SNAPS_PATH.exists():
        held = pd.read_parquet(SNAPS_PATH)
        held = held[held["season"] != end]
        try:
            fresh_year = _season_frame(end)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                print("snaps are rate limited, keeping the file on disk", flush=True)
                return
            raise
        fresh = pd.concat([held, fresh_year], ignore_index=True)
    else:
        fresh = pd.concat([_season_frame(year) for year in years], ignore_index=True)
    SNAPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_parquet(SNAPS_PATH, index=False)
    print(f"snaps {len(fresh)} -> {SNAPS_PATH}")
