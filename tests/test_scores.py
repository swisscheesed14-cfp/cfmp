"""Final scores land on the card. A game still being played does not."""

import pandas as pd

from cfmp.cli.fpi_report import apply_final_scores, attach_board_ranks


def test_a_final_is_written_and_a_live_game_is_not() -> None:
    card = pd.DataFrame(
        [
            {"away": "Kentucky", "home": "Tennessee", "away_score": pd.NA, "home_score": pd.NA},
            {"away": "Ohio State", "home": "Indiana", "away_score": pd.NA, "home_score": pd.NA},
        ]
    )
    games = [
        {"awayTeam": "Kentucky", "homeTeam": "Tennessee", "awayPoints": 17, "homePoints": 31, "completed": True},
        {"awayTeam": "Ohio State", "homeTeam": "Indiana", "awayPoints": 10, "homePoints": 7, "completed": False},
    ]
    out = apply_final_scores(card, games)
    assert int(out.iloc[0]["away_score"]) == 17
    assert int(out.iloc[0]["home_score"]) == 31
    assert pd.isna(out.iloc[1]["away_score"])
    assert pd.isna(out.iloc[1]["home_score"])


def test_only_the_top_25_gets_a_rank() -> None:
    card = pd.DataFrame([{"away": "Kentucky", "home": "Tulane"}])
    rankings = pd.DataFrame([{"rank": 3, "team": "Kentucky"}, {"rank": 40, "team": "Tulane"}])
    out = attach_board_ranks(card, rankings)
    assert int(out.iloc[0]["away_rank"]) == 3
    assert pd.isna(out.iloc[0]["home_rank"])
