"""The results board ranks wins, not snap stats."""

from __future__ import annotations

import pandas as pd

from cfmp.models.resume import _value, resume_board


def _game(week, home, away, home_points, away_points, home_conf="SEC", away_conf="SEC"):
    return {
        "season": 2026,
        "week": week,
        "home_team": home,
        "away_team": away,
        "home_points": home_points,
        "away_points": away_points,
        "home_division": "fbs",
        "away_division": "fbs",
        "home_conference": home_conf,
        "away_conference": away_conf,
        "neutral_site": False,
    }


def test_a_blowout_is_not_worth_two_wins() -> None:
    close = _value(1, at_home=False, neutral=True)
    blowout = _value(40, at_home=False, neutral=True)
    assert close > 3
    assert blowout > close
    assert blowout - close < close


def test_a_better_win_ranks_higher() -> None:
    rows = []
    for week, margin in ((1, 7), (2, 7), (3, 7), (4, 7)):
        rows.append(_game(week, "Strong", "Good", 20 + margin, 20))
        rows.append(_game(week, "Soft", "Weak", 20 + margin, 20))
    rows.append(_game(5, "Good", "Weak", 24, 17))
    rows.append(_game(5, "Beater", "Soft", 21, 20))
    rows.append(_game(6, "Beater", "FillerA", 21, 14))
    rows.append(_game(6, "Beater", "FillerB", 21, 14))
    rows.append(_game(6, "Beater", "FillerC", 21, 14))
    board = resume_board(pd.DataFrame(rows), 2026, through_week=6).set_index("team")
    assert board.loc["Strong", "rank"] < board.loc["Soft", "rank"]
    assert board.loc["Strong", "best_win"] == "Good"
