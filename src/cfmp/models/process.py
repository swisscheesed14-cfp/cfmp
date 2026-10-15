"""Turn per-game snap stats into the columns the rating learns from.

The final score is not one of them. Total PPA, rush PPA, pass PPA, line yards,
second-level yards, open-field yards, and stuff rate are averaged into one
point estimate. Havoc is not on the game feed, so it is not invented here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ADVANCED = Path("data/cfbd/game_advanced.parquet")
COLUMNS = (
    "off_ppa",
    "rush_ppa",
    "pass_ppa",
    "line_yards",
    "second_yards",
    "open_yards",
    "stuff_rate",
    "rush_plays",
    "pass_plays",
)


def attach_process(games: pd.DataFrame, path: Path | None = None) -> pd.DataFrame:
    source = path or ADVANCED
    if not source.exists() or "id" not in games.columns or "home" not in games.columns:
        return games
    advanced = pd.read_parquet(source, columns=["game_id", "team", *COLUMNS])
    out = games.copy()
    out["id"] = pd.to_numeric(out["id"], errors="coerce")
    for side in ("home", "away"):
        piece = advanced.rename(columns={"game_id": "id", "team": side, **{column: f"{side}_{column}" for column in COLUMNS}})
        piece["id"] = pd.to_numeric(piece["id"], errors="coerce")
        out = out.merge(piece, on=["id", side], how="left")
    return out
