"""Write the weekly FPI reports, then price the next unplayed slate.

Rankings wait until that week's snaps are in. The price file is the next
week's unplayed FBS games. Open is the DraftKings number from the Sunday
run, and later runs in the week move only the current DraftKings number.
Positive numbers favor the home team.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from cfmp.ingest.cfbd import lines as fetch_lines
from cfmp.ingest.lines import normalize_lines
from cfmp.ingest.snaps import SCHEDULE_DIR, refresh
from cfmp.models.season import (
    _current_means,
    cap_if_both_good,
    fit_weights,
    fpi_board,
    game_table,
    next_slate,
    price_slate,
    ready_week,
    sos_board,
    top25,
)
from cfmp.models.availability import quarterback_docks
from cfmp.models.playoff import playoff_odds
from cfmp.models.results import grade_card, week_rates
from cfmp.models.site import home_edges
from cfmp.models.weather import apply_kickoff_weather

SEASON = 2026
SNAPS = Path("data/cfbd/game_model.parquet")
OUT = Path("data/reports")


def _schedule() -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in sorted(SCHEDULE_DIR.glob("cfb_schedules_*.parquet"))]
    return pd.concat(frames, ignore_index=True)


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _open_day() -> bool:
    """Sunday locks the open. Monday does it only if Sunday never ran."""
    return datetime.now(ZoneInfo("America/Chicago")).weekday() in (6, 0)


def result_name(week: int) -> str:
    """The graded card. Week 6 is Week6Results.csv."""
    return f"Week{int(week)}Results.csv"


def rate_name(week: int) -> str:
    """The week's two rates. Week 6 is Week6Rates.csv."""
    return f"Week{int(week)}Rates.csv"


def _cap_model(priced: pd.DataFrame) -> pd.DataFrame:
    """Last step. Two good teams cannot finish more than 10 points apart."""
    if priced.empty or "home_fpi" not in priced.columns:
        return priced
    out = priced.copy()
    out["model"] = [
        cap_if_both_good(float(margin), float(home), float(away))
        for margin, home, away in zip(out["model"], out["home_fpi"], out["away_fpi"], strict=True)
    ]
    return out


def card_name(week: int) -> str:
    """The weekly card. Week 7 is Week7Matchups.csv."""
    return f"Week{int(week)}Matchups.csv"


def _prior_card(folder: Path) -> Path | None:
    cards = sorted(folder.glob("Week*Matchups.csv"))
    if cards:
        return cards[-1]
    legacy = folder / "prices.csv"
    return legacy if legacy.exists() else None


def apply_open(out: pd.DataFrame, folder: Path, lock_today: bool) -> pd.DataFrame:
    """Keep the Sunday DraftKings number. Do not use the book's original open.

    A line that was blank on Sunday is filled the first time it appears,
    then left alone while the current number keeps moving.
    """
    prior = _prior_card(folder)
    lock = folder / "open_locked.txt"
    if lock.exists() and prior is not None:
        saved = pd.read_csv(prior)
        if {"away", "home", "dk_open"} <= set(saved.columns):
            saved = saved[["away", "home", "dk_open"]].drop_duplicates(["away", "home"])
            out = out.drop(columns=["dk_open"], errors="ignore").merge(saved, on=["away", "home"], how="left")
    if "dk_open" not in out.columns:
        out["dk_open"] = pd.NA
    out["dk_open"] = pd.to_numeric(out["dk_open"], errors="coerce")
    if lock_today or lock.exists():
        missing = out["dk_open"].isna() & out["draftkings"].notna()
        out.loc[missing, "dk_open"] = out.loc[missing, "draftkings"]
        if out["dk_open"].notna().any():
            folder.mkdir(parents=True, exist_ok=True)
            lock.write_text("sunday", encoding="utf-8")
    return out


def _market(season: int, week: int) -> pd.DataFrame:
    try:
        rows = normalize_lines(fetch_lines(season, week=week, season_type="regular"))
    except Exception as exc:
        print(f"Lines pull failed, prices will have no Vegas number: {exc}")
        return pd.DataFrame(columns=["id", "vegas_spread", "vegas_open", "total", "total_open", "home_ml", "away_ml"])
    if not rows:
        return pd.DataFrame(columns=["id", "vegas_spread", "vegas_open", "total", "total_open", "home_ml", "away_ml"])
    return pd.DataFrame(rows)


def price_next(schedule: pd.DataFrame, profile: pd.DataFrame, weights, through_week: int, docks: pd.DataFrame | None = None) -> int | None:
    """Price the next unplayed FBS slate. Positive model and DraftKings favor the home team."""
    if docks is None:
        docks = pd.DataFrame(columns=["team", "now"])
    slate = next_slate(schedule, SEASON, through_week)
    if slate.empty:
        print("No unplayed slate to price.")
        return None
    week = int(slate["week"].iloc[0])
    priced = price_slate(slate, profile, weights)
    if not docks.empty:
        now = docks.set_index("team")["now"]
        priced["model"] = priced["model"] - priced["home"].map(now).fillna(0) + priced["away"].map(now).fillna(0)
    edges = home_edges(schedule, SEASON, weights.hfa)
    if not edges.empty:
        edges.index = pd.to_numeric(edges.index, errors="coerce")
        priced["model"] = priced["model"] + pd.to_numeric(priced["game_id"], errors="coerce").map(edges).fillna(0)
    market = _market(SEASON, week)
    if not market.empty:
        priced["game_id"] = pd.to_numeric(priced["game_id"], errors="coerce")
        market["id"] = pd.to_numeric(market["id"], errors="coerce")
        keep = ["id", "vegas_spread", "total", "home_ml", "away_ml"]
        priced = priced.merge(market[keep], left_on="game_id", right_on="id", how="left")
        priced = priced.drop(columns=["id"])
        priced = priced.rename(columns={"vegas_spread": "draftkings"})
    else:
        priced["draftkings"] = pd.NA
        priced["total"] = pd.NA
        priced["home_ml"] = pd.NA
        priced["away_ml"] = pd.NA
    priced = apply_kickoff_weather(priced, slate)
    priced = _cap_model(priced)
    priced["edge"] = priced["model"] - priced["draftkings"]
    if "start_date" in slate.columns:
        kick = slate[["game_id", "start_date"]].drop_duplicates("game_id")
        kick["game_id"] = pd.to_numeric(kick["game_id"], errors="coerce")
        priced = priced.merge(kick, on="game_id", how="left")
        priced = priced.sort_values(["start_date", "home"])
    else:
        priced = priced.sort_values("home")
    columns = ["away", "home", "dk_open", "model", "draftkings", "edge", "total", "away_ml", "home_ml", "weather", "forecast"]
    if "start_date" in priced.columns:
        columns = ["start_date", *columns]
    folder = OUT / f"{SEASON}-w{week:02d}"
    out = apply_open(priced, folder, _open_day())
    out = out[columns].copy()
    for column in ("dk_open", "model", "draftkings", "edge", "total", "weather"):
        out[column] = pd.to_numeric(out[column], errors="coerce").round(2)
    for column in ("home_ml", "away_ml"):
        out[column] = pd.to_numeric(out[column], errors="coerce").round(0)
    name = card_name(week)
    _write(out, folder / name)
    latest = OUT / "latest"
    for stale in latest.glob("Week*Matchups.csv"):
        if stale.name != name:
            stale.unlink()
    _write(out, latest / name)
    for stale in (folder / "prices.csv", latest / "prices.csv"):
        if stale.exists():
            stale.unlink()
    (OUT / "latest" / "prices_week.txt").write_text(str(week), encoding="utf-8")
    covered = int(out["draftkings"].notna().sum())
    print(f"Priced {len(out)} week {week} games ({covered} with a DraftKings number) -> {folder / name}")
    return week


def _with_quarterback(board: pd.DataFrame, dock: pd.Series) -> pd.DataFrame:
    out = board.copy()
    out["qb"] = out["team"].map(dock).fillna(0.0)
    out["fpi"] = (out["fpi"] - out["qb"]).round(3)
    out["qb"] = out["qb"].round(2)
    out = out.sort_values(["fpi", "team"], ascending=[False, True]).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out


def publish_boards(profile: pd.DataFrame, weights, week: int, dock: pd.Series) -> None:
    top = _with_quarterback(top25(profile, weights)[["rank", "team", "fpi", "games"]], dock)
    full = _with_quarterback(fpi_board(profile, weights)[["rank", "team", "fpi", "games"]], dock)
    schedules = sos_board(profile, weights)[["rank", "team", "sos", "games"]].copy()
    schedules["fpi"] = (
        fpi_board(profile, weights).set_index("team")["fpi"].reindex(schedules["team"]).to_numpy()
        - schedules["team"].map(dock).fillna(0.0).to_numpy()
    ).round(3)
    folder = OUT / f"{SEASON}-w{week:02d}"
    _write(top, folder / "fpi_top25.csv")
    _write(full, folder / "fpi.csv")
    _write(schedules, folder / "sos.csv")
    for name, frame in (("fpi_top25.csv", top), ("fpi.csv", full), ("sos.csv", schedules)):
        _write(frame, OUT / "latest" / name)
    print(f"FPI Top 25, FPI, and SOS through week {week} -> {folder}")


def publish_playoff(schedule: pd.DataFrame, profile: pd.DataFrame, weights, week: int, docks: pd.DataFrame, crowd: pd.Series | None = None) -> None:
    """Playoff odds from the games still left. Written on every run, not just Mondays."""
    ratings = fpi_board(profile, weights, min_games=1).set_index("team")["fpi"]
    rest = docks.set_index("team")["rest"] if not docks.empty else pd.Series(dtype=float)
    this_week = docks.set_index("team")["now"] - docks.set_index("team")["rest"] if not docks.empty else pd.Series(dtype=float)
    slate = next_slate(schedule, SEASON, week)
    dock_week = int(slate["week"].iloc[0]) if not slate.empty else None
    odds = playoff_odds(
        schedule,
        ratings,
        weights.hfa,
        weights.residual_sd,
        SEASON,
        season_dock=rest,
        week_dock=this_week,
        dock_week=dock_week,
        crowd=crowd,
    )
    folder = OUT / f"{SEASON}-w{week:02d}"
    _write(odds, folder / "playoff.csv")
    _write(odds, OUT / "latest" / "playoff.csv")
    print(f"Playoff odds through week {week} -> {folder / 'playoff.csv'}")


def publish_results(schedule: pd.DataFrame, snaps: pd.DataFrame, weights, week: int) -> None:
    """Grade the card that was saved before this week kicked off.

    The first week this ran had no saved card. That week is priced from the
    games already played, then graded, so the file still exists.
    """
    folder = OUT / f"{SEASON}-w{week:02d}"
    path = _prior_card(folder)
    if path is None:
        print(f"No saved card for week {week}. Pricing it from the prior week.")
        card = _card_from_prior_week(schedule, snaps, weights, week)
        if card.empty:
            print(f"Week {week} could not be rebuilt.")
            return
        _write(card, folder / card_name(week))
    else:
        card = pd.read_csv(path)
    info = schedule.loc[schedule["season"] == SEASON].copy()
    info = info[pd.to_numeric(info["week"], errors="coerce") == week]
    info = info[info["home_points"].notna() & info["away_points"].notna()]
    played = pd.DataFrame(
        {
            "away": info["away_team"].to_numpy(),
            "home": info["home_team"].to_numpy(),
            "actual": (info["home_points"] - info["away_points"]).to_numpy(),
        }
    )
    graded = grade_card(card, played)
    if graded.empty:
        print(f"Week {week} card has no final scores yet.")
        return
    rates = week_rates(graded)
    name = result_name(week)
    rates_file = rate_name(week)
    _write(graded, folder / name)
    _write(rates, folder / rates_file)
    latest = OUT / "latest"
    for stale in latest.glob("Week*Results.csv"):
        if stale.name != name:
            stale.unlink()
    for stale in latest.glob("Week*Rates.csv"):
        if stale.name != rates_file:
            stale.unlink()
    _write(graded, latest / name)
    _write(rates, latest / rates_file)
    print(
        f"Week {week} results -> {folder / name} "
        f"(closer {rates['closer_than_dk'].iloc[0]}%, covered {rates['cover_rate'].iloc[0]}%)"
    )


def _card_from_prior_week(schedule: pd.DataFrame, snaps: pd.DataFrame, weights, week: int) -> pd.DataFrame:
    """The card this week would have been, using nothing from the games themselves."""
    profile = _current_means(snaps, schedule, SEASON, weights, through_week=week - 1)
    info = schedule.loc[schedule["season"] == SEASON].copy()
    info = info[pd.to_numeric(info["week"], errors="coerce") == week]
    fbs = info["home_division"].astype(str).str.lower().eq("fbs") & info["away_division"].astype(str).str.lower().eq("fbs")
    slate = info[fbs].copy()
    if slate.empty:
        return pd.DataFrame()
    priced = price_slate(slate, profile, weights)
    if priced.empty:
        return priced
    edges = home_edges(schedule, SEASON, weights.hfa)
    if not edges.empty:
        edges.index = pd.to_numeric(edges.index, errors="coerce")
        priced["model"] = priced["model"] + pd.to_numeric(priced["game_id"], errors="coerce").map(edges).fillna(0)
    priced = _cap_model(priced)
    market = _market(SEASON, week)
    if not market.empty:
        priced["game_id"] = pd.to_numeric(priced["game_id"], errors="coerce")
        market["id"] = pd.to_numeric(market["id"], errors="coerce")
        priced = priced.merge(market[["id", "vegas_spread"]], left_on="game_id", right_on="id", how="left")
        priced = priced.drop(columns=["id"])
        priced = priced.rename(columns={"vegas_spread": "draftkings"})
    else:
        priced["draftkings"] = pd.NA
    priced["model"] = pd.to_numeric(priced["model"], errors="coerce").round(2)
    priced["draftkings"] = pd.to_numeric(priced["draftkings"], errors="coerce").round(2)
    return priced[["away", "home", "model", "draftkings"]]


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
    docks = quarterback_docks(schedule, SEASON, snaps, weights, profile)
    if docks.empty:
        print("No starting quarterback is ruled out.")
    else:
        print(docks.to_string(index=False))
    rest = docks.set_index("team")["rest"] if not docks.empty else pd.Series(dtype=float)
    crowd = home_edges(schedule, SEASON, weights.hfa)
    publish_boards(profile, weights, week, rest)
    (OUT / "latest" / "week.txt").write_text(str(week), encoding="utf-8")
    publish_results(schedule, snaps, weights, week)
    price_next(schedule, profile, weights, week, docks)
    publish_playoff(schedule, profile, weights, week, docks, crowd)


if __name__ == "__main__":
    main()
