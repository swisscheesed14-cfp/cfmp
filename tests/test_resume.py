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


def test_a_winner_is_ranked_ahead_of_the_team_it_beat() -> None:
    rows = []
    for week in range(1, 5):
        rows.append(_game(week, "Favorite", "Cupcake", 30, 7, "Big 12", "Big 12"))
        rows.append(_game(week, "Underdog", "Doormat", 21, 20, "Big 12", "Big 12"))
    rows.append(_game(5, "Underdog", "Favorite", 24, 21, "Big 12", "Big 12"))
    rows.append(_game(6, "Favorite", "Filler", 28, 14, "Big 12", "Big 12"))
    rows.append(_game(6, "Underdog", "FillerB", 17, 14, "Big 12", "Big 12"))
    board = resume_board(pd.DataFrame(rows), 2026, through_week=6).set_index("team")
    assert board.loc["Underdog", "rank"] < board.loc["Favorite", "rank"]


def test_the_same_number_of_losses_still_respects_the_winner() -> None:
    rows = []
    for week in range(1, 5):
        rows.append(_game(week, "Winner", "Solid", 24, 17))
        rows.append(_game(week, "Loser", "Solid", 24, 17))
    rows.append(_game(5, "Winner", "Loser", 21, 17))
    rows.append(_game(6, "Good", "Winner", 28, 14))
    rows.append(_game(6, "Good", "Loser", 28, 14))
    board = resume_board(pd.DataFrame(rows), 2026, through_week=6).set_index("team")
    assert board.loc["Winner", "rank"] < board.loc["Loser", "rank"]



def test_cupcake_wins_do_not_rank_with_a_real_win() -> None:
    rows = []
    for week in range(1, 6):
        rows.append(_game(week, "Cupcake", "Doormat", 30, 7, "American Athletic", "American Athletic"))
        rows.append(_game(week, "Quality", "Solid", 24, 17, "SEC", "SEC"))
    rows.append(_game(6, "Cupcake", "GoodLoss", 10, 17, "American Athletic", "SEC"))
    rows.append(_game(6, "Quality", "GoodLoss", 14, 21, "SEC", "SEC"))
    rows.append(_game(7, "Solid", "Doormat", 28, 14, "SEC", "American Athletic"))
    board = resume_board(pd.DataFrame(rows), 2026, through_week=7).set_index("team")
    assert board.loc["Quality", "rank"] < board.loc["Cupcake", "rank"]
