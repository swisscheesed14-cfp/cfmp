"""A run-first offense should lose ground against a run-stuffing defense, and nowhere else."""

from __future__ import annotations

import pandas as pd

from cfmp.models.style import Style, matchup_spread, style_points


def _style(rush_rate: float, rush_def: float, pass_def: float) -> Style:
    off_rush, off_pass = 0.0, 0.0
    off_overall = rush_rate * off_rush + (1 - rush_rate) * off_pass
    def_overall = 0.5 * rush_def + 0.5 * pass_def
    return Style(
        rush_off=off_rush,
        pass_off=off_pass,
        rush_def=rush_def,
        pass_def=pass_def,
        rush_rate=rush_rate,
        plays=60,
        rush_off_dev=off_rush - off_overall,
        pass_off_dev=off_pass - off_overall,
        rush_def_dev=rush_def - def_overall,
        pass_def_dev=pass_def - def_overall,
    )


def test_run_defense_only_bites_a_running_team() -> None:
    # rush_def_dev is negative when the run defense is the stronger side.
    stopper = _style(0.5, rush_def=-0.20, pass_def=0.20)
    runner = _style(0.70, rush_def=0.0, pass_def=0.0)
    passer = _style(0.30, rush_def=0.0, pass_def=0.0)
    assert style_points(runner, stopper) < style_points(passer, stopper)
    runner_spread = matchup_spread(0, 0, stopper, runner)["style_adj"]
    passer_spread = matchup_spread(0, 0, stopper, passer)["style_adj"]
    # Stopper is team A. Against a runner, A's margin should be larger.
    assert runner_spread > passer_spread
    assert runner_spread > 0


if __name__ == "__main__":
    test_run_defense_only_bites_a_running_team()
    print("style ok")
