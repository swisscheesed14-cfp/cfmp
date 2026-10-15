# cfmp/models/elo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from cfmp.models.weather import adjust_spread, is_indoor, weather_severity


@dataclass
class EloConfig:
    k: float = 20.0
    home_field_adv: float = 55.0  # Elo points. Equal teams at home win ~58%.
    # 55 Elo is ~2.75 points. The old divisor of 3.5 turned that same edge
    # into a 16-point favorite while the win probability still said 58%.
    elo_per_point: float = 20.0


def _p_home(elo_home: float, elo_away: float, hfa: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(elo_home + hfa - elo_away) / 400.0))


def train_elo(
    results: pd.DataFrame,
    config: EloConfig | None = None,
) -> pd.DataFrame:
    """
    results columns: season, week, start_date, neutral_site, home, away, home_points, away_points
    """
    cfg = config or EloConfig()
    ratings: Dict[str, float] = {}
    order_cols = ["season", "week", "start_date"]
    results = results.sort_values(order_cols, kind="stable")

    def elo(team: str) -> float:
        return ratings.get(team, 1500.0)

    for _, r in results.iterrows():
        if pd.isna(r.home_points) or pd.isna(r.away_points):
            continue
        h, a = str(r.home), str(r.away)
        eh, ea = elo(h), elo(a)
        hfa = 0.0 if bool(r.neutral_site) else cfg.home_field_adv
        p = _p_home(eh, ea, hfa)
        outcome_home = 1.0 if r.home_points > r.away_points else (0.0 if r.home_points < r.away_points else 0.5)
        delta = cfg.k * (outcome_home - p)
        ratings[h] = eh + delta
        ratings[a] = ea - delta

    df = pd.DataFrame({"team": list(ratings.keys()), "elo": list(ratings.values())})
    return df.sort_values("elo", ascending=False, ignore_index=True)


def _spread(elo_home: float, elo_away: float, hfa: float, per_point: float) -> float:
    return (elo_home + hfa - elo_away) / per_point


def _severity_from_row(row: pd.Series) -> float:
    venue = row["venue"] if "venue" in row.index else None
    dome = row["indoor"] if "indoor" in row.index else None
    if is_indoor(venue, dome):
        return 0.0

    def cell(name: str):
        if name not in row.index:
            return None
        value = row[name]
        return None if pd.isna(value) else value

    severity = weather_severity(
        temp_f=cell("wx_temp_f"),
        wind_mph=cell("wx_wind_mph"),
        pop=cell("wx_pop"),
        forecast=cell("wx_short"),
    )
    # forecast_conf is how close we are to kickoff (0.5 to 0.95). A stale
    # forecast should not move the number as much as one from the day before.
    confidence = cell("forecast_conf")
    if confidence is not None:
        severity *= max(0.0, min(1.0, float(confidence)))
    return severity


def predict_week(
    slate: pd.DataFrame,
    ratings: pd.DataFrame,
    hfa: float = 55.0,
    elo_per_point: float = 20.0,
) -> pd.DataFrame:
    """Elo spread, then a weather compression. No simulations.

    `spread_elo` is the raw number. `spread_pred` is what to use: the same
    number on a mild day, a few points closer to zero in real weather.
    `win_prob_home` matches `spread_pred`, not the pre-weather spread.
    """
    rmap = dict(zip(ratings["team"], ratings["elo"]))

    def elo(team: str) -> float:
        return float(rmap.get(team, 1500.0))

    out = slate.copy()
    spreads: list[float] = []
    probs: list[float] = []
    adjusted: list[float] = []
    deltas: list[float] = []
    severities: list[float] = []

    for _, row in out.iterrows():
        neutral = False
        if "neutral_site" in row.index and not pd.isna(row["neutral_site"]):
            neutral = str(row["neutral_site"]).strip().lower() in {"1", "true", "yes", "y"}
        edge = 0.0 if neutral else hfa
        raw = _spread(elo(str(row["home"])), elo(str(row["away"])), edge, elo_per_point)
        severity = _severity_from_row(row)
        final, delta = adjust_spread(raw, severity)
        spreads.append(raw)
        adjusted.append(final)
        deltas.append(delta)
        severities.append(severity)
        probs.append(_p_home(final * elo_per_point, 0.0, 0.0))

    out["spread_elo"] = spreads
    out["weather_severity"] = severities
    out["weather_adj"] = deltas
    out["spread_pred"] = adjusted
    out["win_prob_home"] = probs
    return out
