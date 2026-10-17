"""Site adjustments on top of a neutral-field rating.

Home field comes from the last ten years at that stadium, after the teams
themselves are taken out. A packed night game does not add more on top.
Some buildings are just not hard to win in. Weather is applied by the caller.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from cfmp.models.params import PowerParams

CROWD_CAP = 1.5
VENUE_PRIOR = 25
HOME_FLOOR = 0.5
HOME_CAP = 3.5
MIN_VENUE_GAMES = 40
# The ten-year result still leads. Size and the crowd that actually shows up are the rest.
RESULT_WEIGHT = 0.40
CAPACITY_WEIGHT = 0.30
ATTENDANCE_WEIGHT = 0.30


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


def shrunk_venue(residual: float, games: int, league: float) -> float:
    """Ten-year venue edge, pulled toward the league and kept between 0.5 and 3.5."""
    pulled = league + games / (games + VENUE_PRIOR) * residual
    return max(HOME_FLOOR, min(HOME_CAP, pulled))


def _venue_home_field(schedule: pd.DataFrame, season: int) -> tuple[dict[int, float], float]:
    """Home points at each stadium from the previous ten seasons. 2020 is left out."""
    frame = schedule.copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    years = frame["season"].between(season - 10, season - 1) & frame["season"].ne(2020)
    played = frame["home_points"].notna() & frame["away_points"].notna()
    hist = frame[years & played & ~frame["neutral_site"].fillna(False).astype(bool)].copy()
    if "home_division" in hist.columns and "away_division" in hist.columns:
        fbs = hist["home_division"].astype(str).str.lower().eq("fbs") & hist["away_division"].astype(str).str.lower().eq("fbs")
        hist = hist[fbs]
    hist["venue_id"] = pd.to_numeric(hist["venue_id"], errors="coerce")
    hist = hist[hist["venue_id"].notna()]
    if hist.empty:
        return {}, 2.5
    league_values = []
    residuals: dict[int, list[float]] = {}
    for _, games in hist.groupby("season"):
        if len(games) < 50:
            continue
        teams = pd.Index(sorted(set(games["home_team"]) | set(games["away_team"])))
        index = {team: pos for pos, team in enumerate(teams)}
        design = np.zeros((len(games), len(teams)))
        margin = (games["home_points"] - games["away_points"]).to_numpy(dtype=float)
        for row, home, away in zip(design, games["home_team"], games["away_team"]):
            row[index[home]] = 1.0
            row[index[away]] = -1.0
        design = np.column_stack([design[:, :-1], np.ones(len(games))])
        coef, *_ = np.linalg.lstsq(design, margin, rcond=None)
        residual = margin - design @ coef
        league_values.append(float(coef[-1]))
        for venue, value in zip(games["venue_id"].astype(int), residual):
            residuals.setdefault(int(venue), []).append(float(value))
    league = float(np.mean(league_values)) if league_values else 2.5
    venues = {}
    for venue, values in residuals.items():
        games = len(values)
        venues[venue] = (league + games / (games + VENUE_PRIOR) * float(np.mean(values)), games)
    return venues, league


def _z(values: pd.Series) -> pd.Series:
    spread = float(values.std(ddof=0))
    if spread < 1e-8:
        return pd.Series(0.0, index=values.index)
    return (values - float(values.mean())) / spread


def _crowd(schedule: pd.DataFrame, season: int) -> pd.DataFrame:
    """Biggest crowd and the usual crowd at each stadium over the same ten years."""
    frame = schedule.copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["attendance"] = pd.to_numeric(frame["attendance"], errors="coerce")
    frame["venue_id"] = pd.to_numeric(frame["venue_id"], errors="coerce")
    years = frame["season"].between(season - 10, season - 1) & frame["season"].ne(2020)
    frame = frame[years & frame["attendance"].gt(0) & frame["venue_id"].notna()]
    if frame.empty:
        return pd.DataFrame(columns=["capacity", "attendance"])
    grouped = frame.groupby("venue_id")["attendance"]
    return pd.DataFrame({"capacity": grouped.max(), "attendance": grouped.median()})


def stadium_table(schedule: pd.DataFrame, season: int) -> pd.DataFrame:
    """Current buildings scored by the ten-year result, capacity, and the crowd that shows up."""
    venues, _league = _venue_home_field(schedule, season)
    crowd = _crowd(schedule, season)
    frame = schedule[pd.to_numeric(schedule["season"], errors="coerce") == season].copy()
    frame = frame[~frame["neutral_site"].fillna(False).astype(bool)]
    if "home_division" in frame.columns:
        frame = frame[frame["home_division"].astype(str).str.lower().eq("fbs")]
    frame["venue_id"] = pd.to_numeric(frame["venue_id"], errors="coerce")
    rows = []
    for venue_id, games in frame.groupby("venue_id"):
        found = venues.get(int(venue_id))
        if found is None or found[1] < MIN_VENUE_GAMES or int(venue_id) not in crowd.index:
            continue
        stadium = str(games["venue"].dropna().mode().iloc[0]) if games["venue"].notna().any() else str(int(venue_id))
        team = str(games["home_team"].dropna().mode().iloc[0])
        rows.append(
            {
                "venue_id": int(venue_id),
                "stadium": stadium,
                "team": team,
                "edge": float(found[0]),
                "capacity": float(crowd.at[int(venue_id), "capacity"]),
                "attendance": float(crowd.at[int(venue_id), "attendance"]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["venue_id", "stadium", "team", "score", "points"])
    out = pd.DataFrame(rows)
    out["score"] = RESULT_WEIGHT * _z(out["edge"]) + CAPACITY_WEIGHT * _z(out["capacity"]) + ATTENDANCE_WEIGHT * _z(out["attendance"])
    span = float(out["score"].max() - out["score"].min())
    if span < 1e-8:
        out["points"] = (HOME_FLOOR + HOME_CAP) / 2
    else:
        out["points"] = HOME_FLOOR + (out["score"] - float(out["score"].min())) / span * (HOME_CAP - HOME_FLOOR)
    return out


def home_edges(schedule: pd.DataFrame, season: int, league_hfa: float = 3.0) -> pd.Series:
    """Points added on top of the model's home field. Neutral sites stay at zero.

    A big full stadium can rank ahead of a smaller one with a lucky decade.
    The points stay between 0.5 and 3.5.
    """
    if "venue_id" not in schedule.columns:
        return pd.Series(dtype=float)
    table = stadium_table(schedule, season).set_index("venue_id")["points"]
    venues, league = _venue_home_field(schedule, season)
    frame = schedule[pd.to_numeric(schedule["season"], errors="coerce") == season]
    extras = {}
    for row in frame.itertuples(index=False):
        game_id = pd.to_numeric(getattr(row, "game_id", None), errors="coerce")
        if pd.isna(game_id):
            continue
        if bool(getattr(row, "neutral_site", False)):
            extras[int(game_id)] = 0.0
            continue
        venue = pd.to_numeric(getattr(row, "venue_id", None), errors="coerce")
        if pd.notna(venue) and int(venue) in table.index:
            total = float(table.at[int(venue)])
        else:
            found = venues.get(int(venue)) if pd.notna(venue) else None
            total = found[0] if found else league
            total = max(HOME_FLOOR, min(HOME_CAP, float(total)))
        extras[int(game_id)] = float(total) - float(league_hfa)
    return pd.Series(extras, dtype=float)


def toughest_places(schedule: pd.DataFrame, season: int) -> pd.DataFrame:
    """Current stadiums, hardest first. The point value is not in the file."""
    table = stadium_table(schedule, season)
    if table.empty:
        return pd.DataFrame(columns=["rank", "stadium", "team"])
    out = table.sort_values(["score", "stadium"], ascending=[False, True]).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    return out[["rank", "stadium", "team"]]
