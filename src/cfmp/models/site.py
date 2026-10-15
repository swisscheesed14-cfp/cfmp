"""Site adjustments on top of a neutral-field rating.

Home field is a crowd edge, not a dome edge. Weather is applied by the caller.
Travel and rest are separate so a normal Saturday does not get double-counted
with home field: only a short week, a bye, or a genuinely long trip moves the number.
"""

from __future__ import annotations

import math

from cfmp.models.params import PowerParams


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
