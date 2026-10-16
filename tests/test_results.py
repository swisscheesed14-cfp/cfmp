"""The results file grades the card that was sent, not a rebuilt number."""

import pandas as pd

from cfmp.models.availability import blend_dock, prior_dock
from cfmp.models.results import grade_card, week_rates
from cfmp.models.season import cap_if_both_good, sample_trust
from cfmp.models.season import Weights
from cfmp.models.site import crowd_points, kick_period
import numpy as np


def test_the_worst_miss_is_first_and_the_rates_are_the_week() -> None:
    card = pd.DataFrame(
        [
            {"away": "Kansas", "home": "Kansas State", "model": 10.0, "draftkings": 3.0},
            {"away": "Nebraska", "home": "Oregon", "model": 1.0, "draftkings": 7.0},
        ]
    )
    played = pd.DataFrame(
        [
            {"away": "Kansas", "home": "Kansas State", "actual": 0.0},
            {"away": "Nebraska", "home": "Oregon", "actual": -10.0},
        ]
    )
    graded = grade_card(card, played)
    assert "closer_rate" not in graded.columns
    assert graded.iloc[0]["home"] == "Oregon"
    assert graded.iloc[0]["closer"] == "model"
    assert bool(graded.iloc[0]["covered"]) is True
    rates = week_rates(graded)
    assert rates.closer_rate.iloc[0] == 50.0
    assert rates.cover_rate.iloc[0] == 50.0


def test_a_short_sample_keeps_only_part_of_the_gap() -> None:
    assert round(sample_trust(4, 5), 3) == round(4 / 12, 3)
    # Both well above average, so 14 points becomes 10.
    assert cap_if_both_good(14, 18, 12) == 10
    assert cap_if_both_good(14, 18, 3) == 14
    assert crowd_points(1.0, 90000, "night") == 1.2
    assert crowd_points(1.3, 90000, "night") == 1.5
    assert crowd_points(0.4, 20000, "early") == -1.05
    assert kick_period("2026-10-18T03:00:00.000Z") == "night"
    assert kick_period("2026-10-17T16:00:00.000Z") == "early"


def test_two_backup_starts_only_move_part_of_the_way() -> None:
    weights = Weights(
        names=("pass_ppa",),
        coef=np.array([10.0]),
        intercept=0.0,
        hfa=2.5,
        residual_sd=14.0,
        train_games=10,
        mae=10.0,
    )
    # The season line is half the backup. Two starts only trust him one third of the way off the starter.
    dock = blend_dock({"rush_ppa": 1.0}, {"rush_ppa": -1.0}, 2, {"rush_ppa": 0.0}, weights)
    assert round(dock, 2) == -3.33
    profile = pd.DataFrame({"pass_ppa": [1.0, -1.0]}, index=["Star", "Manager"])
    assert prior_dock(profile, "Star", 4.5) > prior_dock(profile, "Manager", 4.5)
    assert 2.0 <= prior_dock(profile, "Manager", 4.5) <= 8.0
