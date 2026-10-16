"""Site adjustments on top of a neutral-field rating.

Home field is a crowd edge, not a dome edge. Weather is applied by the caller.
Travel and rest are separate so a normal Saturday does not get double-counted
with home field: only a short week, a bye, or a genuinely long trip moves the number.
The league home number stays in the rating. Crowd, size, and kickoff time
only move it by about a point.
"""

from __future__ import annotations

import math

import pandas as pd

from cfmp.models.params import PowerParams

CROWD_CAP = 1.5


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


def timezone_index(lon: float | None) -> int | None:
    if lon is None or not math.isfinite(lon):
        return None
    if lon >= -85:
        return 0
    if lon >= -102:
        return 1
    if lon >= -115:
        return 2
    return 3


def _penalty(miles: float | None, zones: int | None, params: PowerParams) -> float:
    extra = 0.0
    if miles is not None and miles > params.travel_free_miles:
        extra += (miles - params.travel_free_miles) / 1000.0 * params.travel_per_1000
    if zones:
        extra += abs(zones) * params.tz_points
    return extra


def travel_adjustment(
    home_miles: float | None,
    away_miles: float | None,
    home_zones: int | None,
    away_zones: int | None,
    params: PowerParams,
) -> float:
    """Points added to the home spread. The team that traveled farther is penalized."""
    return _penalty(away_miles, away_zones, params) - _penalty(home_miles, home_zones, params)


def rest_score(days: float | None, params: PowerParams) -> float:
    if days is None or not math.isfinite(days):
        return 0.0
    if days <= 6:
        return -params.short_week_points
    if days >= 13:
        return params.bye_points
    return 0.0


def rest_adjustment(home_days: float | None, away_days: float | None, params: PowerParams) -> float:
    raw = rest_score(home_days, params) - rest_score(away_days, params)
    cap = params.rest_cap
    return max(-cap, min(cap, raw))


def qb_penalty(games_since_change: int | None, params: PowerParams) -> float:
    """Points to subtract from a team whose quarterback just changed.

    The first observed starter is not a change. The penalty fades across the
    next two starts, because those games are already moving the rating.
    """
    if games_since_change is None or games_since_change >= 3:
        return 0.0
    return params.qb_change_points * (3 - games_since_change) / 3.0


def qb_adjustment(home_since: int | None, away_since: int | None, params: PowerParams) -> float:
    return qb_penalty(away_since, params) - qb_penalty(home_since, params)


def kick_period(start: object) -> str:
    """Night, early, or the middle of the day, from a UTC kickoff."""
    text = str(start or "")
    if "T" not in text:
        return "day"
    try:
        hour = int(text.split("T", 1)[1][:2])
    except ValueError:
        return "day"
    if hour >= 23 or hour <= 3:
        return "night"
    if hour <= 17:
        return "early"
    return "day"


def crowd_points(fill: float | None, capacity: float | None, period: str) -> float:
    """Extra home points on top of the league number. Capped at a point and a half.

    How full the building usually is matters more than the number of seats.
    Night is a small bump. Noon is a small cut. Home wins are not an input.
    """
    extra = 0.0
    if fill is not None and math.isfinite(fill):
        extra += (fill - 0.80) * 2.0
    if capacity is not None and math.isfinite(capacity):
        if capacity >= 80000:
            extra += 0.40
        elif capacity >= 65000:
            extra += 0.25
    if period == "night":
        extra += 0.40
    elif period == "early":
        extra -= 0.25
    return max(-CROWD_CAP, min(CROWD_CAP, extra))


def home_edges(schedule: pd.DataFrame, season: int) -> pd.Series:
    """Extra home points for each game this season. Neutral sites stay at zero.

    Capacity is the most people that venue has held in this file. The fill
    rate is that team's own crowd, this year when they have played at home
    twice, otherwise last year too.
    """
    frame = schedule.copy()
    if "attendance" not in frame.columns or "venue_id" not in frame.columns:
        return pd.Series(dtype=float)
    frame["game_id"] = pd.to_numeric(frame["game_id"], errors="coerce")
    frame["attendance"] = pd.to_numeric(frame["attendance"], errors="coerce")
    frame["venue_id"] = pd.to_numeric(frame["venue_id"], errors="coerce")
    caps = frame.groupby("venue_id")["attendance"].max()
    history = frame[frame["season"].between(season - 1, season) & frame["attendance"].gt(0)]
    typical: dict[str, tuple[float, float]] = {}
    for team, games in history.groupby("home_team"):
        venue = games["venue_id"].mode()
        if venue.empty:
            continue
        capacity = caps.get(venue.iloc[0])
        if capacity is None or not math.isfinite(float(capacity)) or float(capacity) < 15000:
            continue
        this_year = games[games["season"] == season]
        sample = this_year if len(this_year) >= 2 else games
        typical[str(team)] = (float(sample["attendance"].median() / float(capacity)), float(capacity))
    extras = {}
    for row in frame[frame["season"] == season].itertuples(index=False):
        if bool(getattr(row, "neutral_site", False)):
            extras[row.game_id] = 0.0
            continue
        fill, capacity = typical.get(str(row.home_team), (None, None))
        extras[row.game_id] = crowd_points(fill, capacity, kick_period(getattr(row, "start_date", None)))
    return pd.Series(extras, dtype=float)
