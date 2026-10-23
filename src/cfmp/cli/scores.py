"""Write final scores onto the current week's card. Rankings are not rebuilt.

Thursday through Sunday morning. A game that is still being played is left blank.
A team in the published Top 25 gets that rank. Everyone else stays blank.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from cfmp.cli.fpi_report import RESULT_COLUMNS, attach_board_ranks, apply_final_scores
from cfmp.ingest.cfbd import schedule

SEASON = 2026
LATEST = Path("data/reports/latest")


def current_card(folder: Path = LATEST) -> Path:
    cards = sorted(folder.glob("Week*Matchups.csv"), key=lambda path: int(re.search(r"\d+", path.name).group(0)))  # type: ignore[union-attr]
    if not cards:
        raise SystemExit(f"No matchup card in {folder}")
    return cards[-1]


def week_of(path: Path) -> int:
    match = re.search(r"Week(\d+)Matchups", path.name)
    if match is None:
        raise SystemExit(f"Cannot read a week from {path.name}")
    return int(match.group(1))


def stamp(card: pd.DataFrame, games: list[dict], rankings: pd.DataFrame | None) -> pd.DataFrame:
    out = apply_final_scores(card, games)
    out = attach_board_ranks(out, rankings)
    for column in RESULT_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
    return out


def main() -> None:
    path = current_card()
    week = week_of(path)
    card = pd.read_csv(path)
    games = schedule(year=SEASON, week=week, season_type="regular")
    board = LATEST / "Top 25.csv"
    rankings = pd.read_csv(board) if board.exists() else None
    stamped = stamp(card, games, rankings)
    finals = int(stamped["away_score"].notna().sum()) if "away_score" in stamped.columns else 0
    folder = Path(f"data/reports/{SEASON}-w{week:02d}")
    for dest in (path, folder / path.name):
        dest.parent.mkdir(parents=True, exist_ok=True)
        stamped.to_csv(dest, index=False)
    print(f"Week {week}: {finals} finals on {len(stamped)} games -> {path}")


if __name__ == "__main__":
    main()
