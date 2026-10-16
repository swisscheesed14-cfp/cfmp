"""Points to take off a team whose starting quarterback is not playing.

The season rating still includes his games. The dock stays until the
replacement has started three times, because those games are then in the
rating. A one-week absence does not change the rest of the year. A
season-ending absence does.
"""

from __future__ import annotations

import pandas as pd

from cfmp.ingest.cfbd import game_players
from cfmp.ingest.injuries import fetch_absences
from cfmp.ingest.qb import _team_row

SEASON_POINTS = 6.0
WEEK_POINTS = 4.0
DOUBTFUL_POINTS = 2.0
FADE_STARTS = 3


def quarterback_docks(schedule: pd.DataFrame, season: int) -> pd.DataFrame:
    """One row per team that is short a starter. `now` is this week, `rest` is the year."""
    names = _fbs_names(schedule, season)
    if not names:
        return _empty()
    absences = fetch_absences(names)
    if not absences:
        return _empty()
    starters, replaced = _starters(season)
    rows = []
    for absence in absences:
        team = absence["team"]
        last = absence["qb"].split()[-1].lower()
        starter = starters.get(team)
        if starter and starter.split()[-1].lower() != last:
            continue
        fade = max(0.0, (FADE_STARTS - replaced.get(team, 0)) / FADE_STARTS)
        if fade == 0:
            continue
        if absence["scope"] == "season":
            base = SEASON_POINTS
        elif absence.get("level") == "doubtful":
            base = DOUBTFUL_POINTS
        else:
            base = WEEK_POINTS
        points = round(base * fade, 2)
        rows.append(
            {
                "team": team,
                "qb": starter or absence["qb"],
                "scope": absence["scope"],
                "now": points,
                "rest": points if absence["scope"] == "season" else 0.0,
            }
        )
    if not rows:
        return _empty()
    return pd.DataFrame(rows).drop_duplicates("team", keep="first")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["team", "qb", "scope", "now", "rest"])


def _fbs_names(schedule: pd.DataFrame, season: int) -> list[str]:
    frame = schedule.loc[schedule["season"] == season]
    names: set[str] = set()
    for side in ("home", "away"):
        fbs = frame[f"{side}_division"].astype(str).str.lower().eq("fbs")
        names.update(frame.loc[fbs, f"{side}_team"].astype(str))
    return sorted(names)


def _starters(season: int) -> tuple[dict[str, str], dict[str, int]]:
    """Attempt leader for the year, and how many recent games someone else started."""
    try:
        payload = []
        for week in range(1, 16):
            payload.extend(game_players(season, week=week, season_type="regular"))
    except Exception as exc:
        print(f"Quarterback starts skipped: {exc}")
        return {}, {}
    games: list[tuple[str, int, str, str]] = []
    for game in payload:
        week = int(game.get("week") or 0)
        for team in game.get("teams") or []:
            parsed = _team_row(team)
            if not parsed or not parsed.get("qb"):
                continue
            school = str(team.get("school") or team.get("team") or "")
            if school:
                games.append((school, week, str(parsed["qb"]), str(game.get("id"))))
    if not games:
        return {}, {}
    frame = pd.DataFrame(games, columns=["team", "week", "qb", "id"]).drop_duplicates(["team", "id"])
    counts = frame.groupby(["team", "qb"]).size().reset_index(name="starts")
    leader = counts.sort_values(["starts", "qb"], ascending=[False, True]).groupby("team").first()
    starters = leader["qb"].to_dict()
    replaced: dict[str, int] = {}
    for team, games_played in frame.groupby("team"):
        ordered = games_played.sort_values("week")
        name = starters.get(str(team))
        streak = 0
        for qb in reversed(ordered["qb"].tolist()):
            if qb == name:
                break
            streak += 1
        replaced[str(team)] = streak
    return starters, replaced
