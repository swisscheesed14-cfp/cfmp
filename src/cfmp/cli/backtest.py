"""Walk-forward grade against the final margin and the closing number."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cfmp.ingest.dataset import RESULTS_PATH, load_games
from cfmp.models.params import load_params
from cfmp.models.power import summarize, walk_forward
from cfmp.models.priors import load_priors
from cfmp.models.process import attach_process

SUMMARY = Path("data/backtest_summary.json")
DETAIL = Path("data/backtest_games.parquet")


def _print(summary: dict) -> None:
    print(f"Completed games: {summary['games']}")
    print(f"Games with a closing number: {summary['lined_games']}")
    if summary["mae_model"] is not None:
        print(f"Mean absolute error, model: {summary['mae_model']:.2f} points")
    if summary["mae_vegas"] is not None:
        print(f"Mean absolute error, close: {summary['mae_vegas']:.2f} points")
    if summary["mean_closer"] is not None:
        print(f"Model closer than the close by: {summary['mean_closer']:.2f} points (negative means the close won)")
    if summary["ats"] is not None:
        print(f"Against the close: {summary['ats']:.1%} on {summary['ats_n']} decided games")
    if summary["mean_clv"] is not None:
        print(f"Mean closing-line value vs the open: {summary['mean_clv']:.2f} points")


def main() -> None:
    if not RESULTS_PATH.exists():
        raise SystemExit("No results yet. Run python -m cfmp.cli.train first.")
    games = attach_process(load_games())
    graded = walk_forward(games, load_params(), load_priors())
    summary = summarize(graded)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    graded.to_parquet(DETAIL, index=False)
    _print(summary)
    runs = sorted(Path("runs").glob("*/predictions.csv"), key=lambda path: path.stat().st_mtime)
    if runs:
        target = runs[-1].parent / "backtest_summary.json"
        target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Copied summary -> {target}")
    print(f"Wrote {SUMMARY}")


if __name__ == "__main__":
    main()
