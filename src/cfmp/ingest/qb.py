"""Quarterback and pass-rate from box scores.

Used only after a game, to update continuity. The prediction for that game
does not see that game's quarterback.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from cfmp.ingest.cfbd import game_players
from cfmp.ingest.results import FIRST_SEASON, season_year

QB_PATH = Path("data/qb.parquet")
_ATTEMPTS = re.compile(r"(\d+)\s*/\s*(\d+)")


def _stat(categories: list[dict[str, Any]], category: str, stat_name: str) -> dict[str, float]:
    found: dict[str, float] = {}
    for cat in categories:
        if str(cat.get("name", "")).lower() != category:
            continue
        for kind in cat.get("types") or []:
            if str(kind.get("name", "")).upper() != stat_name.upper():
                continue
            for athlete in kind.get("athletes") or []:
                name = athlete.get("name") or athlete.get("athlete")
                raw = str(athlete.get("stat", ""))
                match = _ATTEMPTS.search(raw)
                if match and stat_name.upper() == "C/ATT":
                    value = float(match.group(2))
                else:
                    try:
                        value = float(raw.split()[0])
                    except (TypeError, ValueError, IndexError):
                        continue
                if name:
                    found[str(name)] = found.get(str(name), 0.0) + value
    return found


def _team_row(team: dict[str, Any]) -> dict[str, Any] | None:
    categories = team.get("categories") or []
    attempts = _stat(categories, "passing", "C/ATT")
    rushes = _stat(categories, "rushing", "CAR")
    if not attempts and not rushes:
        return None
    starter = max(attempts, key=attempts.get) if attempts else None
    pass_att = sum(attempts.values())
    rush_att = sum(rushes.values())
    denom = pass_att + rush_att
    return {
        "qb": starter,
        "pass_rate": (pass_att / denom) if denom else None,
    }


def normalize_players(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for game in payload:
        home: dict[str, Any] = {}
        away: dict[str, Any] = {}
        for team in game.get("teams") or []:
            side = str(team.get("homeAway") or team.get("home_away") or "").lower()
            parsed = _team_row(team)
            if not parsed:
                continue
            if side == "home":
                home = parsed
            elif side == "away":
                away = parsed
        if not home and not away:
            continue
        rows.append(
            {
                "id": game.get("id"),
                "home_qb": home.get("qb"),
                "away_qb": away.get("qb"),
                "home_pass_rate": home.get("pass_rate"),
                "away_pass_rate": away.get("pass_rate"),
            }
        )
    return rows


def fetch_year(year: int) -> list[dict[str, Any]]:
    try:
        payload = game_players(year, season_type="both")
        return normalize_players(payload)
    except Exception as exc:
        print(f"Quarterback pull failed for {year}: {exc}")
        return []


def ensure_qb(years: list[int] | None = None, dest: Path = QB_PATH, refresh_year: int | None = None) -> Path | None:
    """Best-effort. A missing box score must not stop the rating."""
    if years is None:
        years = list(range(FIRST_SEASON, season_year() + 1))
    cache = Path("data/qb_cache.json")
    cached: dict[str, list] = {}
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
    frames: list[pd.DataFrame] = []
    try:
        for year in years:
            key = str(year)
            if key not in cached or year == refresh_year:
                cached[key] = fetch_year(year)
                print(f"Quarterbacks for {len(cached[key])} games in {year}")
            if cached[key]:
                frames.append(pd.DataFrame(cached[key]))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(cached), encoding="utf-8")
    except Exception as exc:
        print(f"Quarterback cache skipped: {exc}")
        return None
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return None
    df = df.drop_duplicates(subset=["id"], keep="last")
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    print(f"Wrote {len(df)} quarterback rows -> {dest}")
    return dest
