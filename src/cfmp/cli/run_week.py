# src/cfmp/cli/run_week.py
from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Tuple

import pandas as pd

# project imports
from cfmp.ingest import cfbd                 # <- uses your existing wrapper (schedule endpoint)
from cfmp.weather.nws import hourly_forecast_many  # optional weather batching

# -----------------------------
# Constants / config
# -----------------------------
LOCAL_TZ = ZoneInfo("America/Chicago")
CACHE_FILE = ".cache/venues.parquet"
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def _to_int(x: Any) -> int | None:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None

def gval(g: Dict[str, Any], *keys: str) -> Any:
    """Get the first non-empty value among keys from dict g."""
    for k in keys:
        if k in g and g[k] not in (None, "", []):
            return g[k]
    return None

def pick_year_week() -> Tuple[int, int]:
    """
    Determine target year/week.
    Priority:
      1) CFB_YEAR / CFB_WEEK env
      2) Local ISO week as fallback
    """
    now = datetime.now(LOCAL_TZ)
    year = _to_int(os.getenv("CFB_YEAR")) or now.year
    week = _to_int(os.getenv("CFB_WEEK"))
    if week is None:
        week = now.isocalendar().week
    return year, week

def ensure_cached_venues() -> None:
    """
    If you have a dedicated tool that materializes venues into .cache/venues.parquet,
    you’ll see a log line in Actions like 'Saved 833 venues ...'. If it’s already there,
    we just reuse it.
    """
    Path(".cache").mkdir(exist_ok=True)
    # If your pipeline elsewhere writes to CACHE_FILE, do nothing here.

def load_cached_venues() -> Dict[int, Dict[str, Any]]:
    """Load venues parquet and index by id."""
    if not Path(CACHE_FILE).exists():
        return {}
    df = pd.read_parquet(CACHE_FILE)
    recs = df.to_dict("records")
    out = {}
    for r in recs:
        vid = _to_int(r.get("id")) or _to_int(r.get("venue_id"))
        if vid is None:
            continue
        out[vid] = r
    return out

def venue_from_record(g: Dict[str, Any], VENUES: Dict[int, Dict[str, Any]]) -> Tuple[str | None, int | None, float | None, float | None]:
    """
    Resolve venue fields from a schedule game record + cached venues.
    Returns (venue_name, venue_id, lat, lon)
    """
    vid = _to_int(gval(g, "venue_id", "venueId"))
    vname = gval(g, "venue", "venue_name")
    vlat = None
    vlon = None

    if vid and vid in VENUES:
        v = VENUES[vid]
        vname = vname or v.get("name")
        vlat = v.get("latitude")
        vlon = v.get("longitude")

    # If no vid but venue name exists in cache? We won’t try an O(N) match here.
    return vname, vid, vlat, vlon

def compute_forecast_conf(kickoff_iso: str | None, now: datetime) -> float:
    """
    Confidence bucket based on hours to kickoff:
      >7d        -> 0.50
      5–7d       -> 0.70
      3–5d       -> 0.85
      <=48 h     -> 0.95
    If kickoff unknown -> 0.50.
    """
    if not kickoff_iso:
        return 0.50
    try:
        dt = datetime.fromisoformat(kickoff_iso.replace("Z", "+00:00")).astimezone(LOCAL_TZ)
    except Exception:
        return 0.50

    delta_h = (dt - now).total_seconds() / 3600.0
    if delta_h <= 48:
        return 0.95
    if delta_h <= 5 * 24:
        return 0.85
    if delta_h <= 7 * 24:
        return 0.70
    return 0.50

def kickoff_iso_from_record(g: Dict[str, Any]) -> str | None:
    """
    Try typical keys CFBD returns for datetime.
    Prefer UTC-like forms; format to 3dp with Z for CSV stability.
    """
    raw = gval(g, "start_date", "startDate", "kickoff", "start_time_tbd")
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    except Exception:
        return None

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    year, week = pick_year_week()
    print(f"Running for {year}-W{week:02d}")

    ensure_cached_venues()
    VENUES = load_cached_venues()

    # Fetch week’s games from your cfbd wrapper
    games = cfbd.schedule(year=year, week=week)
    print(f"games ok: {len(games)}")

    # Build rows
    now_local = datetime.now(LOCAL_TZ)
    rows: List[Dict[str, Any]] = []
    coords_for_weather: List[Tuple[float, float]] = []
    coord_keys: List[str] = []  # map key to row index

    for g in games:
        # teams
        away = gval(g, "away", "away_team")
        home = gval(g, "home", "home_team")

        # kickoff ISO
        kickoff_iso = kickoff_iso_from_record(g)

        # venue details
        vname, vid, vlat, vlon = venue_from_record(g, VENUES)

        # confidence
        fconf = compute_forecast_conf(kickoff_iso, now_local)

        row = {
            "id": _to_int(gval(g, "id", "game_id", "gameId")),
            "year": year,
            "week": week,
            "kickoff": kickoff_iso,
            "away": away,
            "home": home,
            "venue": vname,
            "venue_id": vid,
            "venue_lat": vlat,
            "venue_lon": vlon,
            "forecast_sample": None,   # filled if weather on
            "forecast_conf": fconf,
        }
        idx = len(rows)
        rows.append(row)

        # schedule weather pulls only when we have coords
        if os.getenv("CFMP_WEATHER", "").lower() == "on" and vlat is not None and vlon is not None:
            coords_for_weather.append((float(vlat), float(vlon)))
            coord_keys.append(str(idx))

    # Weather batch (optional)
    if coords_for_weather:
        print(f"Fetching weather for {len(coords_for_weather)} venues…")
        # concurrency tuned to be gentle; adjust if desired
        fc_map = hourly_forecast_many(coords_for_weather, hours=6, concurrency=16)
        # fc_map is keyed by "lat,lon" or similar. We’ll stitch by index order we sent.
        for k, (lat, lon) in zip(coord_keys, coords_for_weather):
            key = f"{lat:.6f},{lon:.6f}"
            sample = fc_map.get(key)
            if sample:
                rows[int(k)]["forecast_sample"] = sample
        print("weather ok")

    # Write outputs
    outdir = RUNS_DIR / f"{year}-W{week:02d}"
    outdir.mkdir(parents=True, exist_ok=True)

    # JSON
    jpath = outdir / "slate_preview.json"
    with jpath.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # CSV
    cpath = outdir / "slate.csv"
    df = pd.DataFrame(rows)
    # Keep column order stable
    want_cols = [
        "id", "year", "week", "kickoff",
        "away", "home", "venue", "venue_id",
        "venue_lat", "venue_lon", "forecast_sample",
        "forecast_conf",
    ]
    for col in want_cols:
        if col not in df.columns:
            df[col] = None
    df = df[want_cols]
    df.to_csv(cpath, index=False)

    print(f"Wrote {outdir} in {year}-W{week:02d}")

if __name__ == "__main__":
    main()
