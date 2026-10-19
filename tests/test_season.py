"""Current rankings stay on this season. Older seasons only set the weights."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cfmp.models.season import STATS, Weights, fit_weights, rate_season


def _game(season: int, week: int, game_id: int, home_line: float, away_line: float, margin: float) -> dict:
    row = {
        "season": season,
        "week": week,
        "game_id": game_id,
        "home_team": "Home",
        "away_team": "Away",
        "margin": margin,
        "neutral_site": True,
        "fbs": True,
        "prior_games_home": 4,
        "prior_games_away": 4,
    }
    for off, deff, _hurts in STATS:
        row[f"in_{off}_home"] = 0.0
        row[f"in_{off}_away"] = 0.0
        row[f"in_{deff}_home"] = 0.0
        row[f"in_{deff}_away"] = 0.0
    row["in_line_yards_home"] = home_line
    row["in_line_yards_away"] = away_line
    return row


def test_current_season_does_not_set_the_weights() -> None:
    rows = []
    for index in range(40):
        # History: more line yards, bigger home margin.
        rows.append(_game(2024, 5, index, 3.5, 2.2, 14))
        rows.append(_game(2024, 6, 100 + index, 2.2, 3.5, -14))
        # Current season says the opposite. It must not flip the weight.
        rows.append(_game(2026, 5, 200 + index, 3.5, 2.2, -20))
    weights = fit_weights(pd.DataFrame(rows), train_through=2025)
    assert weights.coef[list(weights.names).index("line_yards")] > 0


def test_average_team_is_the_middle_of_this_season() -> None:
    average = {"games": 6}
    good = {"games": 6}
    for off, deff, hurts in STATS:
        if hurts:
            average[off], average[deff] = 0.2, 0.2
            good[off], good[deff] = 0.1, 0.3
        elif off == "line_yards":
            average[off], average[deff] = 3.0, 3.0
            good[off], good[deff] = 3.8, 2.2
        else:
            average[off], average[deff] = 0.0, 0.0
            good[off], good[deff] = 1.0, -1.0
    profile = pd.DataFrame([average, good], index=["Average", "Good"])
    weights = Weights(
        names=tuple(off for off, _, _ in STATS),
        coef=np.ones(len(STATS)),
        intercept=0.0,
        hfa=2.5,
        residual_sd=14.0,
        train_games=10,
        mae=10.0,
    )
    rated = rate_season(profile, weights)
    assert rated.loc["Good", "rating"] > rated.loc["Average", "rating"]


def test_reports_share_one_ranking() -> None:
    profile = pd.DataFrame(
        [
            {"games": 6, "sos": 1.0},
            {"games": 6, "sos": 8.0},
            {"games": 2, "sos": 3.0},
        ],
        index=["Soft", "Hard", "Short"],
    )
    for off, deff, hurts in STATS:
        if hurts:
            profile[off] = [0.2, 0.1, 0.2]
            profile[deff] = [0.2, 0.3, 0.2]
        else:
            profile[off] = [0.0, 2.0, 0.5]
            profile[deff] = [0.0, -1.0, 0.0]
    weights = Weights(
        names=tuple(off for off, _, _ in STATS),
        coef=np.ones(len(STATS)),
        intercept=0.0,
        hfa=2.5,
        residual_sd=14.0,
        train_games=10,
        mae=10.0,
    )
    from cfmp.models.season import fpi_board, sos_board, top25

    full = fpi_board(profile, weights)
    assert list(top25(profile, weights, n=2)["team"]) == list(full["team"].head(2))
    sos = sos_board(profile, weights)
    assert sos.iloc[0]["team"] == "Hard"
    assert set(sos["team"]) == set(full["team"])


def test_next_slate_skips_played_games() -> None:
    from cfmp.models.season import next_slate

    schedule = pd.DataFrame(
        [
            {"season": 2026, "week": 6, "home_points": 21, "home_division": "fbs", "away_division": "fbs", "home_team": "A", "away_team": "B"},
            {"season": 2026, "week": 7, "home_points": 10, "home_division": "fbs", "away_division": "fbs", "home_team": "C", "away_team": "D"},
            {"season": 2026, "week": 7, "home_points": None, "home_division": "fbs", "away_division": "fbs", "home_team": "E", "away_team": "F"},
            {"season": 2026, "week": 8, "home_points": None, "home_division": "fbs", "away_division": "fbs", "home_team": "G", "away_team": "H"},
        ]
    )
    slate = next_slate(schedule, 2026, through_week=6)
    assert list(slate["home_team"]) == ["E"]


def test_a_win_counts_and_a_blowout_is_capped() -> None:
    from cfmp.models.season import recency_weight, result_points

    assert result_points(1) == 7
    assert result_points(40) == 20
    assert result_points(-28) == -20
    # A one-point home win stays a win after removing about 2.5 points of home field.
    assert result_points(1, site=2.5) > 0
    assert abs(recency_weight(2, 7) - 0.70) < 1e-9
