"""Run/pass matchup tilt, then a simulated final margin.

The power rating already knows who scored. It does not know whether those
points came on the ground. This layer only moves a spread when one team's
run-pass mix hits the other team's stronger side. An average mix adds zero,
so a good defense is not counted twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np
import pandas as pd

from cfmp.models.power import final_ratings

# This season counts three times last year, which still counts more than the years before it.
SEASON_WEIGHT = {0: 3.0, 1: 1.2, 2: 0.55}
OLDER_WEIGHT = 0.25
MARGIN_SD = 14.5
SIMS = 5000


@dataclass
class Style:
    rush_off: float
    pass_off: float
    rush_def: float
    pass_def: float
    rush_rate: float
    plays: float
    rush_def_dev: float
    pass_def_dev: float
    rush_off_dev: float
    pass_off_dev: float


def _season_weight(season: int, current: int) -> float:
    return SEASON_WEIGHT.get(int(current) - int(season), OLDER_WEIGHT)


def attach_names(splits: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Name each offense row from the schedule. The splits file only carries ids."""
    names = schedule[["game_id", "home_team", "away_team"]].drop_duplicates("game_id")
    out = splits.merge(names, on="game_id", how="inner")
    home = out["is_home"].fillna(False).astype(bool)
    out["team"] = np.where(home, out["home_team"], out["away_team"])
    out["opp"] = np.where(home, out["away_team"], out["home_team"])
    return out


def build_styles(splits: pd.DataFrame, schedule: pd.DataFrame, current_season: int) -> dict[str, Style]:
    games = attach_names(splits, schedule)
    games = games.dropna(subset=["team", "opp", "epa_per_rush", "epa_per_pass", "rush_plays", "pass_plays"])
    games = games[(games["rush_plays"] + games["pass_plays"]) > 0].copy()
    games["w"] = games["season"].map(lambda year: _season_weight(int(year), current_season))

    # The other row in the same game is what this defense allowed.
    allowed = games[
        ["game_id", "team", "epa_per_rush", "epa_per_pass", "rush_plays", "pass_plays"]
    ].rename(
        columns={
            "team": "opp",
            "epa_per_rush": "rush_allowed",
            "epa_per_pass": "pass_allowed",
            "rush_plays": "rush_faced",
            "pass_plays": "pass_faced",
        }
    )
    games = games.merge(allowed, on=["game_id", "opp"], how="inner")
    games["rush_w"] = games["w"] * games["rush_plays"]
    games["pass_w"] = games["w"] * games["pass_plays"]
    games["rush_faced_w"] = games["w"] * games["rush_faced"]
    games["pass_faced_w"] = games["w"] * games["pass_faced"]

    raw_rows = []
    for team, frame in games.groupby("team"):
        raw_rows.append(
            {
                "team": team,
                "off_rush": _weighted(frame, "epa_per_rush", "rush_w"),
                "off_pass": _weighted(frame, "epa_per_pass", "pass_w"),
                "def_rush": _weighted(frame, "rush_allowed", "rush_faced_w"),
                "def_pass": _weighted(frame, "pass_allowed", "pass_faced_w"),
                "games": int(frame["game_id"].nunique()),
            }
        )
    raw = pd.DataFrame(raw_rows).set_index("team")
    league_def_rush = float(raw["def_rush"].mean())
    league_def_pass = float(raw["def_pass"].mean())
    league_off_rush = float(raw["off_rush"].mean())
    league_off_pass = float(raw["off_pass"].mean())

    adj = games.merge(
        raw[["def_rush", "def_pass"]].rename(columns={"def_rush": "opp_def_rush", "def_pass": "opp_def_pass"}),
        left_on="opp",
        right_index=True,
        how="inner",
    )
    adj = adj.merge(
        raw[["off_rush", "off_pass"]].rename(columns={"off_rush": "opp_off_rush", "off_pass": "opp_off_pass"}),
        left_on="opp",
        right_index=True,
        how="inner",
    )
    adj["off_rush_adj"] = adj["epa_per_rush"] - (adj["opp_def_rush"] - league_def_rush)
    adj["off_pass_adj"] = adj["epa_per_pass"] - (adj["opp_def_pass"] - league_def_pass)
    adj["def_rush_adj"] = adj["rush_allowed"] - (adj["opp_off_rush"] - league_off_rush)
    adj["def_pass_adj"] = adj["pass_allowed"] - (adj["opp_off_pass"] - league_off_pass)

    rush_rates = []
    styles: dict[str, Style] = {}
    built: list[tuple[str, float, float, float, float, float, float]] = []
    for team, frame in adj.groupby("team"):
        rushes = float(frame["rush_plays"].sum())
        passes = float(frame["pass_plays"].sum())
        rush_rate = rushes / max(rushes + passes, 1.0)
        plays = float(frame["plays"].mean()) if len(frame) else 65.0
        built.append(
            (
                team,
                _weighted(frame, "off_rush_adj", "rush_w"),
                _weighted(frame, "off_pass_adj", "pass_w"),
                _weighted(frame, "def_rush_adj", "rush_faced_w"),
                _weighted(frame, "def_pass_adj", "pass_faced_w"),
                rush_rate,
                plays,
            )
        )
        rush_rates.append(rush_rate)
    league_rush_rate = float(np.mean(rush_rates)) if rush_rates else 0.5

    for team, off_rush, off_pass, def_rush, def_pass, rush_rate, plays in built:
        off_overall = rush_rate * off_rush + (1.0 - rush_rate) * off_pass
        def_overall = league_rush_rate * def_rush + (1.0 - league_rush_rate) * def_pass
        styles[team] = Style(
            rush_off=off_rush,
            pass_off=off_pass,
            rush_def=def_rush,
            pass_def=def_pass,
            rush_rate=rush_rate,
            plays=plays,
            rush_off_dev=off_rush - off_overall,
            pass_off_dev=off_pass - off_overall,
            rush_def_dev=def_rush - def_overall,
            pass_def_dev=def_pass - def_overall,
        )
    return styles


def _weighted(frame: pd.DataFrame, value: str, weight: str) -> float:
    w = frame[weight].to_numpy(dtype=float)
    total = float(w.sum())
    if total <= 0:
        return 0.0
    return float((frame[value].to_numpy(dtype=float) * w).sum() / total)


def style_points(offense: Style, defense: Style) -> float:
    """Points the offense scores above its own overall level against this defense."""
    mix = offense.rush_rate
    epa = mix * (offense.rush_off_dev + defense.rush_def_dev) + (1.0 - mix) * (
        offense.pass_off_dev + defense.pass_def_dev
    )
    return epa * offense.plays


def matchup_spread(rating_a: float, rating_b: float, style_a: Style | None, style_b: Style | None) -> dict[str, float]:
    """Neutral-field margin for A. Positive means A is favored."""
    base = rating_a - rating_b
    tilt = 0.0
    if style_a is not None and style_b is not None:
        tilt = style_points(style_a, style_b) - style_points(style_b, style_a)
    return {"base_spread": base, "style_adj": tilt, "spread": base + tilt}


def simulate(spread: float, n: int = SIMS, sd: float = MARGIN_SD, seed: int = 7) -> dict[str, float]:
    draws = np.random.default_rng(seed).normal(spread, sd, size=n)
    return {
        "sims": float(n),
        "mean_margin": float(draws.mean()),
        "median_margin": float(np.median(draws)),
        "win_prob": float((draws > 0).mean()),
        "p10": float(np.percentile(draws, 10)),
        "p90": float(np.percentile(draws, 90)),
    }


def simulate_pair(
    team: str,
    opponent: str,
    ratings: pd.DataFrame,
    styles: dict[str, Style],
    n: int = SIMS,
) -> dict[str, float | str]:
    table = ratings.set_index("team")
    if team not in table.index or opponent not in table.index:
        missing = team if team not in table.index else opponent
        raise KeyError(missing)
    priced = matchup_spread(
        float(table.loc[team, "rating"]),
        float(table.loc[opponent, "rating"]),
        styles.get(team),
        styles.get(opponent),
    )
    return {"team": team, "opponent": opponent, **priced, **simulate(priced["spread"], n=n)}


def _win_prob(spread: float) -> float:
    return 0.5 * (1.0 + erf(spread / (MARGIN_SD * sqrt(2.0))))


def all_neutral(ratings: pd.DataFrame, styles: dict[str, Style]) -> pd.DataFrame:
    """Every FBS team against every other, on a neutral field."""
    fbs = ratings[ratings["classification"].astype(str).str.lower() == "fbs"]
    teams = fbs["team"].tolist()
    rate = dict(zip(fbs["team"], fbs["rating"].astype(float)))
    rows = []
    for team in teams:
        for opponent in teams:
            if team == opponent:
                continue
            priced = matchup_spread(rate[team], rate[opponent], styles.get(team), styles.get(opponent))
            rows.append({"team": team, "opponent": opponent, "win_prob": _win_prob(priced["spread"]), **priced})
    return pd.DataFrame(rows)


def ratings_from_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    games = pd.DataFrame(
        {
            "season": schedule["season"].astype(int),
            "week": schedule["week"].fillna(0).astype(int),
            "start_date": pd.to_datetime(schedule["start_date"], utc=True, errors="coerce"),
            "neutral_site": schedule["neutral_site"].fillna(False).astype(bool),
            "home": schedule["home_team"],
            "away": schedule["away_team"],
            "home_points": pd.to_numeric(schedule["home_points"], errors="coerce"),
            "away_points": pd.to_numeric(schedule["away_points"], errors="coerce"),
            "id": schedule["game_id"],
            "home_classification": schedule["home_division"].astype(str).str.lower(),
            "away_classification": schedule["away_division"].astype(str).str.lower(),
        }
    )
    played = games.dropna(subset=["home", "away", "home_points", "away_points", "start_date"])
    return final_ratings(played)
