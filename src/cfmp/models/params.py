"""Fitted knobs for the power rating. Defaults until a holdout backtest earns a change."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

PACKAGE_DEFAULT = Path(__file__).with_name("params_default.json")
DATA_OVERRIDE = Path("data/model_params.json")


@dataclass
class PowerParams:
    learning_rate: float = 0.08
    mov_cap: float = 28.0
    hfa: float = 2.75
    season_regress: float = 0.65
    points_per_team: float = 28.0
    short_week_points: float = 1.25
    bye_points: float = 0.75
    rest_cap: float = 2.0
    travel_per_1000: float = 0.35
    travel_free_miles: float = 500.0
    tz_points: float = 0.35
    qb_change_points: float = 2.5
    weather_points: float = 3.0
    # Points added to a team's own rating. FBS stays 0. These are priors;
    # cross-division games move the non-FBS numbers, FBS never moves.
    fcs_baseline: float = -20.0
    ii_baseline: float = -32.0
    iii_baseline: float = -42.0
    # How many games the prior is worth. Stops one blowout from moving a division.
    division_prior_games: float = 400.0

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_mapping(cls, raw: dict) -> "PowerParams":
        known = {field: raw[field] for field in cls.__dataclass_fields__ if field in raw}
        return cls(**known)


def load_params(path: Path | None = None) -> PowerParams:
    candidates = [path] if path else [DATA_OVERRIDE, PACKAGE_DEFAULT]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return PowerParams.from_mapping(json.loads(candidate.read_text(encoding="utf-8")))
    return PowerParams()
