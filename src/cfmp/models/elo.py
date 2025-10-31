# cfmp/models/elo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd


@dataclass
class EloConfig:
    k: float = 20.0
    home_field_adv: float = 55.0  # Elo points


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


def predict_week(
    slate: pd.DataFrame,
    ratings: pd.DataFrame,
    hfa: float = 55.0,
) -> pd.DataFrame:
    """Return slate with win_prob_home and spread_pred."""
    rmap = dict(zip(ratings["team"], ratings["elo"]))
    def Elo(t: str) -> float:
        return float(rmap.get(t, 1500.0))

    def prob_and_spread(home: str, away: str) -> tuple[float, float]:
        eh, ea = Elo(home), Elo(away)
        p = _p_home(eh, ea, hfa)
        spread = (eh + hfa - ea) / 3.5  # ~3.5 Elo per point
        return p, spread

    out = slate.copy()
    out["win_prob_home"], out["spread_pred"] = zip(
        *[
            prob_and_spread(str(r["home"]), str(r["away"]))
            for _, r in slate.iterrows()
        ]
    )
    return out
