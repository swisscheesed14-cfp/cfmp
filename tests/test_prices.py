"""The weekly open is our Sunday DraftKings snapshot, not the book's first line."""

from pathlib import Path

import pandas as pd

from cfmp.cli.fpi_report import apply_open


def test_sunday_open_stays_put_when_the_line_moves(tmp_path: Path) -> None:
    first = pd.DataFrame([{"away": "Kansas", "home": "Kansas State", "draftkings": 7.5}])
    locked = apply_open(first, tmp_path, True)
    assert locked.dk_open.iloc[0] == 7.5
    locked.to_csv(tmp_path / "Week7Matchups.csv", index=False)
    moved = pd.DataFrame([{"away": "Kansas", "home": "Kansas State", "draftkings": 4.5}])
    later = apply_open(moved, tmp_path, False)
    assert later.dk_open.iloc[0] == 7.5
    assert later.draftkings.iloc[0] == 4.5


def test_a_weekday_does_not_invent_the_open(tmp_path: Path) -> None:
    frame = pd.DataFrame([{"away": "Memphis", "home": "Tulane", "draftkings": -7.0}])
    out = apply_open(frame, tmp_path, False)
    assert pd.isna(out.dk_open.iloc[0])
    assert not (tmp_path / "open_locked.txt").exists()
