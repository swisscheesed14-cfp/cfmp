"""Write the three weekly FPI reports once that week's snaps are in.

The reports are FPI Top 25, FPI for every team, and SOS for every team.
A week is ready when most of its scored FBS games have snap stats, which
is usually the Sunday or Monday after the games.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from cfmp.ingest.snaps import SCHEDULE_DIR, refresh
from cfmp.models.season import _current_means, fit_weights, fpi_board, game_table, ready_week, sos_board, top25

SEASON = 2026
SNAPS = Path("data/cfbd/game_model.parquet")
OUT = Path("data/reports")


def _schedule() -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in sorted(SCHEDULE_DIR.glob("cfb_schedules_*.parquet"))]
    return pd.concat(frames, ignore_index=True)


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def publish(force: bool = False) -> int | None:
    snaps = pd.read_parquet(SNAPS)
    schedule = _schedule()
    week = ready_week(snaps, schedule, SEASON)
    if week is None:
        print("No week has snaps yet.")
        return None
    stamp = OUT / "latest" / "week.txt"
    if not force and stamp.exists() and stamp.read_text(encoding="utf-8").strip() == str(week):
        print(f"Week {week} is already published.")
        return week
    games = game_table(schedule, snaps)
    weights = fit_weights(games, train_through=SEASON - 1)
    profile = _current_means(snaps, schedule, SEASON, weights, through_week=week)
    top = top25(profile, weights)[["rank", "team", "fpi", "games"]]
    full = fpi_board(profile, weights)[["rank", "team", "fpi", "games"]]
    schedules = sos_board(profile, weights)[["rank", "team", "sos", "games"]]
    folder = OUT / f"{SEASON}-w{week:02d}"
    _write(top, folder / "fpi_top25.csv")
    _write(full, folder / "fpi.csv")
    _write(schedules, folder / "sos.csv")
    for name, frame in (("fpi_top25.csv", top), ("fpi.csv", full), ("sos.csv", schedules)):
        _write(frame, OUT / "latest" / name)
    stamp.write_text(str(week), encoding="utf-8")
    print(f"FPI Top 25, FPI, and SOS through week {week} -> {folder}")
    print(top.round(2).to_string(index=False))
    print(f"FPI teams: {len(full)}. SOS teams: {len(schedules)}.")
    return week


def main() -> None:
    refresh()
    publish(force="--force" in sys.argv[1:])


if __name__ == "__main__":
    main()
