"""Write the weekly FPI reports, then price the next unplayed slate.

Rankings wait until that week's snaps are in. The price file is the next
week's unplayed FBS games, with the Vegas number beside the model when
the line is posted. Positive numbers favor the home team.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from cfmp.ingest.cfbd import lines as fetch_lines
from cfmp.ingest.lines import normalize_lines
from cfmp.ingest.snaps import SCHEDULE_DIR, refresh
from cfmp.models.season import (
    _current_means,
    fit_weights,
    fpi_board,
    game_table,
    next_slate,
    price_slate,
    ready_week,
    sos_board,
    top25,
)

SEASON = 2026
SNAPS = Path("data/cfbd/game_model.parquet")
OUT = Path("data/reports")


def _schedule() -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in sorted(SCHEDULE_DIR.glob("cfb_schedules_*.parquet"))]
    return pd.concat(frames, ignore_index=True)


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _market(season: int, week: int) -> pd.DataFrame:
    try:
        rows = normalize_lines(fetch_lines(season, week=week, season_type="regular"))
    except Exception as exc:
        print(f"Lines pull failed, prices will have no Vegas number: {exc}")
        return pd.DataFrame(columns=["id", "vegas_spread"])
    if not rows:
        return pd.DataFrame(columns=["id", "vegas_spread"])
    return pd.DataFrame(rows)


def price_next(schedule: pd.DataFrame, profile: pd.DataFrame, weights, through_week: int) -> int | None:
    """Price the next unplayed FBS slate. Positive model and Vegas favor the home team."""
    slate = next_slate(schedule, SEASON, through_week)
    if slate.empty:
        print("No unplayed slate to price.")
        return None
    week = int(slate["week"].iloc[0])
    priced = price_slate(slate, profile, weights)
    market = _market(SEASON, week)
    if not market.empty:
        priced["game_id"] = pd.to_numeric(priced["game_id"], errors="coerce")
        market["id"] = pd.to_numeric(market["id"], errors="coerce")
        priced = priced.merge(market[["id", "vegas_spread"]], left_on="game_id", right_on="id", how="left")
        priced = priced.drop(columns=["id"])
    else:
        priced["vegas_spread"] = pd.NA
    priced["edge"] = priced["model"] - priced["vegas_spread"]
    if "start_date" in slate.columns:
        kick = slate[["game_id", "start_date"]].drop_duplicates("game_id")
        kick["game_id"] = pd.to_numeric(kick["game_id"], errors="coerce")
        priced = priced.merge(kick, on="game_id", how="left")
        priced = priced.sort_values(["start_date", "home"])
    else:
        priced = priced.sort_values("home")
    columns = ["away", "home", "neutral", "model", "vegas_spread", "edge"]
    if "start_date" in priced.columns:
        columns = ["start_date", *columns]
    out = priced[columns].copy()
    for column in ("model", "vegas_spread", "edge"):
        out[column] = pd.to_numeric(out[column], errors="coerce").round(2)
    folder = OUT / f"{SEASON}-w{week:02d}"
    _write(out, folder / "prices.csv")
    _write(out, OUT / "latest" / "prices.csv")
    (OUT / "latest" / "prices_week.txt").write_text(str(week), encoding="utf-8")
    covered = int(out["vegas_spread"].notna().sum())
    print(f"Priced {len(out)} week {week} games ({covered} with a Vegas number) -> {folder / 'prices.csv'}")
    return week


def main() -> None:
    refresh()
    snaps = pd.read_parquet(SNAPS)
    schedule = _schedule()
    week = ready_week(snaps, schedule, SEASON)
    if week is None:
        print("No week has snaps yet.")
        return
    games = game_table(schedule, snaps)
    weights = fit_weights(games, train_through=SEASON - 1)
    profile = _current_means(snaps, schedule, SEASON, weights, through_week=week)
    force = "--force" in sys.argv[1:]
    stamp = OUT / "latest" / "week.txt"
    if force or not stamp.exists() or stamp.read_text(encoding="utf-8").strip() != str(week):
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
    else:
        print(f"Week {week} reports are already published.")
    price_next(schedule, profile, weights, week)


if __name__ == "__main__":
    main()
