"""The same final score must not produce the same rating when the snaps disagree."""

from __future__ import annotations

import pandas as pd

from cfmp.models.params import PowerParams
from cfmp.models.power import final_ratings


def _game(team: str, ppa: float, line: float, stuff: float, when: str) -> dict:
    return {
        "season": 2024,
        "week": 1,
        "id": abs(hash(team)) % 10000,
        "start_date": when,
        "home": team,
        "away": "Cup",
        "home_points": 31,
        "away_points": 24,
        "neutral_site": True,
        "home_classification": "fbs",
        "away_classification": "fbs",
        "home_off_ppa": ppa,
        "away_off_ppa": 7.2,
        "home_rush_ppa": ppa * 0.4,
        "away_rush_ppa": 2.8,
        "home_pass_ppa": ppa * 0.6,
        "away_pass_ppa": 5.6,
        "home_line_yards": line,
        "away_line_yards": 2.9,
        "home_second_yards": 1.5 if line > 3 else 0.7,
        "away_second_yards": 1.08,
        "home_open_yards": 1.6 if line > 3 else 0.5,
        "away_open_yards": 1.3,
        "home_stuff_rate": stuff,
        "away_stuff_rate": 0.19,
    }


def test_snaps_beat_the_same_score() -> None:
    games = pd.DataFrame([
        _game("Snaps", 22.0, 4.1, 0.08, "2024-10-01T16:00:00Z"),
        _game("Empty", -8.0, 1.7, 0.36, "2024-10-01T20:00:00Z"),
    ])
    games["start_date"] = pd.to_datetime(games["start_date"], utc=True)
    ratings = final_ratings(games, PowerParams()).set_index("team")
    assert ratings.loc["Snaps", "rating"] > ratings.loc["Empty", "rating"]
    assert ratings.loc["Snaps", "rush_off"] > ratings.loc["Empty", "rush_off"]


if __name__ == "__main__":
    test_snaps_beat_the_same_score()
    print("process ok")
