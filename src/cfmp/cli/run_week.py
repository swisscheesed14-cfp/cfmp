# src/cfmp/cli/run_week.py
import os
import json
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from cfmp.ingest import cfbd
from cfmp.weather.nws import hourly_forecast_many

LOCAL_TZ = ZoneInfo("America/Chicago")
CACHE_FILE = ".cache/venues.parquet"


# ---------- helpers ----------
def _to_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def pick_year_week():
    """Pick target year/week from env or 'current' ISO week as a fallback."""
    now = datetime.now(LOCAL_TZ)
    year = _to_int(os.getenv("CFB_YEAR")) or now.year
    week = _to_int(os.getenv("CFB_WEEK"))
    if week is None:
        # ISO week is good enough when running ad-hoc
        week = now.isocalendar().week
    return year, week


def gval(g: dict, *keys):
    """Get first present, non-empty value among keys from dict g."""
    for k in keys:
        if k in g and g[k] not in (None, ""):
            return g[k]
    return None


def load_cached_venues():
    if os.path.exists(CACHE_FILE):
        return pd.read_parquet(CACHE_FILE).to_dict("records")
    # first time: pull once and cache
    v = cfbd.venues()
    cols = ["id", "name", "latitude", "longitude"]
    df = pd.DataFrame(v)[cols]
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    df.to_parquet(CACHE_FILE, index=False)
    return df.to_dict("records")


# ---------- forecast confidence ----------
def conf_from_hours(hours_to_kick: float | None) -> float:
    """
    Map hours until kickoff to a [0,1] confidence weight for weather usefulness.
    Coarse bins, intentionally simple.
    """
    if hours_to_kick is None:
        return 0.5
    h = max(0.0, hours_to_kick)
    if h <= 48:
        return 0.95
    if h <= 72:
        return 0.90
    if h <= 120:
        return 0.85
    if h <= 168:
        return 0.70
    return 0.50


# ---------- main ----------
def main():
    year, week = pick_year_week()
    print(f"Fetching games {year} W{week:02d}…")

    games = cfbd.schedule(year=year, week=week)
    print(f"games ok: {len(games)}")

    venues = load_cached_venues()
    v_by_id = {v["id"]: v for v in venues}
    print(f"venues ok: {len(venues)}")

    # Prepare coordinates for weather calls
    coords = []
    rows = []
    for g in games:
        vid = gval(g, "venue_id")
        v = v_by_id.get(vid) if vid is not None else None
        lat = gval(v or {}, "latitude")
        lon = gval(v or {}, "longitude")

        # kickoff parsing
        ko = gval(g, "start_date") or gval(g, "kickoff") or gval(g, "start_time_tbd")
        kickoff_dt = None
        if isinstance(ko, str) and len(ko) >= 10:
            # CFBD returns ISO8601; make naive-UTC then convert to LOCAL_TZ for confidence calc
            try:
                kickoff_dt = datetime.fromisoformat(ko.replace("Z", "+00:00")).astimezone(LOCAL_TZ)
            except Exception:
                kickoff_dt = None

        if lat is not None and lon is not None:
            coords.append((lat, lon))

        rows.append(
            {
                "id": gval(g, "id"),
                "year": year,
                "week": week,
                "kickoff": kickoff_dt.isoformat() if kickoff_dt else None,
                "away": gval(g, "away_team"),
                "home": gval(g, "home_team"),
                "venue": gval(v or {}, "name"),
                "venue_id": vid,
                "venue_lat": lat,
                "venue_lon": lon,
                # filled after weather call:
                "forecast_sample": None,
                "forecast_conf": None,
            }
        )

    # Weather: sample next 6 hours around now, de-duplicated coords
    print("Loading weather…")
    uniq = sorted(set([c for c in coords if c[0] is not None and c[1] is not None]))
    fc_map = hourly_forecast_many(uniq, hours=6, concurrency=32)  # {(lat,lon): [...]}
    print("weather ok")

    now_local = datetime.now(LOCAL_TZ)

    # Attach one forecast sample and confidence per game
    for r in rows:
        lat, lon = r["venue_lat"], r["venue_lon"]
        if lat is not None and lon is not None:
            r["forecast_sample"] = fc_map.get((lat, lon))  # list[dict] or None

        # confidence from time-to-kickoff
        if r["kickoff"]:
            try:
                ko = datetime.fromisoformat(r["kickoff"])
                dt_hours = (ko - now_local).total_seconds() / 3600.0
            except Exception:
                dt_hours = None
        else:
            dt_hours = None
        r["forecast_conf"] = round(conf_from_hours(dt_hours), 2)

    # Write outputs
    run_dir = os.path.join("runs", f"{year}-W{week:02d}")
    os.makedirs(run_dir, exist_ok=True)

    # CSV (flat view)
    csv_cols = [
        "id",
        "year",
        "week",
        "kickoff",
        "away",
        "home",
        "venue",
        "venue_id",
        "venue_lat",
        "venue_lon",
        "forecast_conf",
    ]
    pd.DataFrame([{k: r.get(k) for k in csv_cols} for r in rows]).to_csv(
        os.path.join(run_dir, "slate.csv"), index=False
    )

    # JSON (rich preview including forecast sample)
    with open(os.path.join(run_dir, "slate_preview.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()
