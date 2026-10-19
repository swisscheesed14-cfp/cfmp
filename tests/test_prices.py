"""The weekly open is our Sunday DraftKings snapshot, not the book's first line."""

from pathlib import Path

import pandas as pd

from cfmp.cli.fpi_report import apply_open, games_of_the_week


def test_sunday_open_stays_put_when_the_line_moves(tmp_path: Path) -> None:
    first = pd.DataFrame([{"away": "Kansas", "home": "Kansas State", "draftkings": 7.5}])
    locked = apply_open(first, tmp_path, True)
    assert locked.dk_open.iloc[0] == 7.5
    locked.to_csv(tmp_path / "Week7Matchups.csv", index=False)
    moved = pd.DataFrame([{"away": "Kansas", "home": "Kansas State", "draftkings": 4.5}])
    later = apply_open(moved, tmp_path, False)
    assert later.dk_open.iloc[0] == 7.5
    assert later.draftkings.iloc[0] == 4.5


def test_top_25_games_come_before_the_big_edges() -> None:
    card = pd.DataFrame(
        [
            {"start_date": "2026-10-24T16:00:00.000Z", "away": "Pitt", "home": "Miami", "model": 4.9, "draftkings": 23.5, "edge": -18.6, "total": 45.5, "away_ml": 800, "home_ml": -1400, "forecast": ""},
            {"start_date": "2026-10-24T23:30:00.000Z", "away": "Ole Miss", "home": "Texas", "model": 4.0, "draftkings": 3.0, "edge": 1.0, "total": 53.5, "away_ml": 140, "home_ml": -166, "forecast": ""},
            {"start_date": "2026-10-24T19:00:00.000Z", "away": "Kansas", "home": "Baylor", "model": 3.0, "draftkings": 3.5, "edge": -0.5, "total": 57.0, "away_ml": 150, "home_ml": -180, "forecast": ""},
        ]
    )
    rankings = pd.DataFrame(
        [
            {"rank": 7, "team": "Ole Miss", "record": "5-1"},
            {"rank": 20, "team": "Texas", "record": "4-2"},
            {"rank": 2, "team": "Miami", "record": "6-0"},
            {"rank": 40, "team": "Pitt", "record": "4-2"},
            {"rank": 60, "team": "Kansas", "record": "3-3"},
            {"rank": 70, "team": "Baylor", "record": "3-3"},
        ]
    )
    games = games_of_the_week(card, rankings)
    assert list(games["group"]) == ["Top 25", "Edge"]
    assert games.iloc[0]["away"] == "Ole Miss"
    assert games.iloc[1]["side"] == "Pitt +23.5"
    assert "Kansas" not in set(games["away"])



def test_a_weekday_does_not_invent_the_open(tmp_path: Path) -> None:
    frame = pd.DataFrame([{"away": "Memphis", "home": "Tulane", "draftkings": -7.0}])
    out = apply_open(frame, tmp_path, False)
    assert pd.isna(out.dk_open.iloc[0])
    assert not (tmp_path / "open_locked.txt").exists()
