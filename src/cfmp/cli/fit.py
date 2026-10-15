"""Search rating knobs. Write them only if the held-out season agrees.

The closing line is the scoreboard, not a feature.
"""

from __future__ import annotations

from cfmp.ingest.dataset import RESULTS_PATH, load_games
from cfmp.models.params import PACKAGE_DEFAULT
from cfmp.models.power import select_params


def main() -> None:
    if not RESULTS_PATH.exists():
        raise SystemExit("No results yet. Run python -m cfmp.cli.train first.")
    games = load_games()
    seasons = sorted(int(year) for year in games["season"].dropna().unique())
    if len(seasons) < 2:
        raise SystemExit("Need at least two seasons to hold one out.")
    holdout = [seasons[-1]]
    fit = seasons[:-1]
    params, report = select_params(games, fit, holdout)
    print(f"Tried {report['tried']} settings on {fit[0]}-{fit[-1]}, held out {holdout[0]}.")
    print(f"Holdout with the candidate: {report['holdout']}")
    print(f"Holdout with the current defaults: {report['default_holdout']}")
    if report["accepted"]:
        params.to_json(PACKAGE_DEFAULT)
        print(f"Holdout improved. Wrote {PACKAGE_DEFAULT}")
    else:
        print("Holdout did not improve. Kept the current defaults.")


if __name__ == "__main__":
    main()
