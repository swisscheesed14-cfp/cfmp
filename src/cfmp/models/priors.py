"""Preseason priors from CollegeFootballData.

Talent is a point target, not a rating. Returning production decides how
much of last year's rating survives. Teams with neither keep the old
shrink toward their own division.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

POINTS_PER_SD = 4.0
PRIOR_CAP = 10.0
ROOT = Path("data/cfbd")


def keep_from_returning(pct: float) -> float:
    """0% back keeps a quarter of last year. A full roster keeps 85%."""
    pct = min(1.0, max(0.0, pct))
    return min(0.85, max(0.25, 0.25 + 0.60 * pct))


def _clamp01(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return min(1.0, max(0.0, float(value)))


@dataclass
class Priors:
    talent: dict[tuple[int, str], float]
    returning: dict[tuple[int, str], tuple[float | None, float | None]]

    def talent_points(self, season: int, team: str) -> float | None:
        return self.talent.get((season, team))

    def returning_split(self, season: int, team: str) -> tuple[float | None, float | None]:
        return self.returning.get((season, team), (None, None))


def _talent_points(frame: pd.DataFrame) -> dict[tuple[int, str], float]:
    out: dict[tuple[int, str], float] = {}
    for season, part in frame.groupby(frame["year"].astype(int)):
        # Older files mix in FCS and pull the average down. Use the top 130
        # when the file is that mixed, so a blue blood is not "average."
        base = part.nlargest(130, "talent") if len(part) > 160 else part
        sd = float(base["talent"].std(ddof=0))
        if sd <= 0:
            continue
        mean = float(base["talent"].mean())
        for row in part.itertuples(index=False):
            z = (float(row.talent) - mean) / sd
            out[(int(season), str(row.team))] = max(-PRIOR_CAP, min(PRIOR_CAP, z * POINTS_PER_SD))
    return out


def _returning_split(frame: pd.DataFrame) -> dict[tuple[int, str], tuple[float | None, float | None]]:
    out: dict[tuple[int, str], tuple[float | None, float | None]] = {}
    for row in frame.itertuples(index=False):
        overall = _clamp01(getattr(row, "percentPPA", None))
        passing = _clamp01(getattr(row, "percentPassingPPA", None))
        if overall is None and passing is None:
            continue
        defense = overall
        offense = overall if passing is None or overall is None else 0.5 * overall + 0.5 * passing
        out[(int(row.season), str(row.team))] = (offense, defense)
    return out


def load_priors(root: Path | None = None) -> Priors | None:
    folder = root or ROOT
    talent_path = folder / "talent.parquet"
    returning_path = folder / "returning.parquet"
    if not talent_path.exists() and not returning_path.exists():
        return None
    talent = _talent_points(pd.read_parquet(talent_path)) if talent_path.exists() else {}
    returning = _returning_split(pd.read_parquet(returning_path)) if returning_path.exists() else {}
    return Priors(talent=talent, returning=returning)
