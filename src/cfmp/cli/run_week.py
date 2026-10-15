# -*- coding: utf-8 -*-
"""
Produce the weekly slate CSV + JSON.
Hard requirements:
- CFB_YEAR and CFB_WEEK are optional. If either is missing, the CFBD calendar picks the week.
- Writes to runs/{year}-W{week}/ (postseason: {year}-P{week}/)
- Adds forecast_conf based on time-to-kickoff
- Uses venues from data/venues_min.parquet (fallback: .cache/venues.parquet)
- When weather is on, attaches the NWS hourly period that covers kickoff
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd  # for parquet cache

# project ingest
from cfmp.ingest import cfbd as cfbd_ingest
from cfmp.models.weather import is_indoor

# optional weather import (kept off by default in CI to save time)
try:
    from cfmp.weather.nws import coord_key, finite_coord, forecast_many, period_at
except Exception as exc:
    coord_key = finite_coord = forecast_many = period_at = None  # type: ignore
    print(f"Weather import failed, forecasts will be empty: {exc}")


LOCAL_TZ = ZoneInfo(os.getenv("TZ") or "America/Chicago")
VENUE_FILES = (Path("data/venues_min.parquet"), Path(".cache/venues.parquet"))


# ---------------------------
# helpers
# ---------------------------

def _to_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def pick_year_week() -> Tuple[int, int, str]:
    """Explicit CFB_YEAR and CFB_WEEK win. Otherwise use the CFBD calendar.

    ISO week is never a fallback. That is what previously produced week 44.
    """
    year = _to_int(os.getenv("CFB_YEAR"))
    week = _to_int(os.getenv("CFB_WEEK"))
    season_type = (os.getenv("CFB_SEASON_TYPE") or "").strip().lower() or None
    if year is not None and week is not None:
        return year, week, season_type or "regular"

    detected_year, detected_week, detected_type = cfbd_ingest.current_week(
        datetime.now(tz=LOCAL_TZ),
        year=year,
    )
    if week is None:
        week = detected_week
    if season_type is None:
        season_type = detected_type
    print(f"Calendar selected {detected_year} {season_type} week {week}")
    return detected_year, week, season_type


def gval(d: Optional[dict], *keys: str) -> Optional[Any]:
    if not d:
        return None
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def parse_kickoff(start: Optional[str]) -> Optional[datetime]:
    if not start:
        return None
    # CFBD uses '...Z'. Make it aware.
    s = start.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def forecast_confidence(now: datetime, kickoff: Optional[datetime]) -> float:
    """
    Simple confidence curve:
      <=48h: 0.95
      3–5d : 0.85
      5–7d : 0.70
      >7d  : 0.50
      unknown kickoff: 0.50
    """
    if not kickoff:
        return 0.50
    hours = (kickoff - now).total_seconds() / 3600.0
    if hours <= 48:
        return 0.95
    if hours <= 5 * 24:
        return 0.85
    if hours <= 7 * 24:
        return 0.70
    return 0.50


def weather_enabled() -> bool:
    flag = os.getenv("CFMP_WEATHER", "on").strip().lower()
    return flag not in {"off", "false", "0", "no"}


def load_cached_venues() -> tuple[Dict[int, Dict[str, Any]], Optional[Path]]:
    path = next((p for p in VENUE_FILES if p.exists()), None)
    if path is None:
        return {}, path
    df = pd.read_parquet(path)
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("id")
    if id_col is None:
        return {}, path
    out: Dict[int, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        try:
            vid = int(r[id_col])
        except Exception:
            continue
        out[vid] = {
            "name": r.get(cols["name"]) if "name" in cols else None,
            "latitude": r.get(cols["latitude"]) if "latitude" in cols else None,
            "longitude": r.get(cols["longitude"]) if "longitude" in cols else None,
            "dome": r.get(cols["dome"]) if "dome" in cols else None,
        }
    return out, path


# ---------------------------
# main
# ---------------------------

def main() -> None:
    year, week, season_type = pick_year_week()
    print(f"Running for {year} {season_type} week {week}")

    # venues cache
    venues, venue_path = load_cached_venues()
    if venues:
        print(f"Loaded {len(venues)} venues from {venue_path}")
    else:
        print("No venue file found (data/venues_min.parquet). Weather will have no coordinates.")

    # fetch games from our ingest (uses CFBD_KEY env internally)
    # NOTE: the correct function in this repo is `schedule`, not `games`.
    games: List[dict] = cfbd_ingest.schedule(year=year, week=week, season_type=season_type)
    print(f"Fetched {len(games)} games from CFBD")

    if not games:
        raise SystemExit(f"No CFBD games found for year={year}, week={week}.")

    # build rows
    now_local = datetime.now(tz=LOCAL_TZ)
    rows: List[Dict[str, Any]] = []

    # pre-collect coordinates for optional weather
    coords_for_weather: List[Tuple[float, float, Optional[datetime]]] = []

    for g in games:
        gid = gval(g, "id") or gval(g, "game_id")
        start = gval(g, "startDate", "start_date", "kickoff")
        kickoff_dt = parse_kickoff(start)
        kickoff_local = kickoff_dt.astimezone(LOCAL_TZ) if kickoff_dt else None

        vid = gval(g, "venue_id") or gval(g, "venueId")
        vid_int = None
        try:
            vid_int = int(vid) if vid is not None else None
        except Exception:
            vid_int = None

        v = venues.get(vid_int, {}) if vid_int is not None else {}

        lat = v.get("latitude")
        lon = v.get("longitude")
        point = finite_coord(lat, lon) if finite_coord is not None else None
        lat_f = point[0] if point else None
        lon_f = point[1] if point else None

        venue_name = v.get("name") or gval(g, "venue")
        indoor = is_indoor(venue_name, v.get("dome") if v else None)
        neutral = gval(g, "neutralSite", "neutral_site")

        row = {
            "id": gid,
            "year": year,
            "week": week,
            "season_type": season_type,
            "kickoff": kickoff_local.isoformat() if kickoff_local else None,
            "away": gval(g, "away_team", "awayTeam"),
            "home": gval(g, "home_team", "homeTeam"),
            "neutral_site": bool(neutral),
            "venue": venue_name,
            "indoor": indoor,
            "venue_id": vid_int,
            "venue_lat": lat_f,
            "venue_lon": lon_f,
            "forecast_sample": None,
            "wx_temp_f": None,
            "wx_wind": None,
            "wx_wind_mph": None,
            "wx_pop": None,
            "wx_short": None,
            "wx_at": None,
            "forecast_conf": forecast_confidence(now_local, kickoff_local),
        }
        rows.append(row)

        if weather_enabled() and point is not None and not indoor:
            coords_for_weather.append((lat_f, lon_f, kickoff_dt))

    # optional weather pull
    if weather_enabled() and coords_for_weather and forecast_many is not None and period_at is not None:
        try:
            series = forecast_many(coords_for_weather, concurrency=4)
            got = 0
            missed = 0
            for row in rows:
                point = finite_coord(row.get("venue_lat"), row.get("venue_lon"))
                if point is None:
                    continue
                periods = series.get(coord_key(*point))
                kick = parse_kickoff(row.get("kickoff"))
                snap = period_at(periods or [], kick)
                row["forecast_sample"] = snap
                if not snap:
                    missed += 1
                    continue
                got += 1
                row["wx_temp_f"] = snap.get("tempF")
                row["wx_wind"] = snap.get("wind")
                row["wx_wind_mph"] = snap.get("wind_mph")
                row["wx_pop"] = snap.get("pop")
                row["wx_short"] = snap.get("forecast")
                row["wx_at"] = snap.get("start")
            print(
                f"Weather: {got} kickoff forecasts, {missed} without coverage or outside the hourly window "
                f"({len(series)} unique grids)"
            )
        except Exception as e:
            print(f"Weather step skipped due to error: {e}")
    elif weather_enabled() and not coords_for_weather:
        print("Weather enabled, but no venue coordinates to query.")

    # output
    tag = f"W{week:02d}" if season_type == "regular" else f"P{week:02d}"
    out_dir = Path(f"runs/{year}-{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON preview
    with (out_dir / "slate_preview.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # CSV
    cols = [
        "id",
        "year",
        "week",
        "season_type",
        "kickoff",
        "away",
        "home",
        "neutral_site",
        "venue",
        "indoor",
        "venue_id",
        "venue_lat",
        "venue_lon",
        "forecast_sample",
        "wx_temp_f",
        "wx_wind",
        "wx_wind_mph",
        "wx_pop",
        "wx_short",
        "wx_at",
        "forecast_conf",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_dir / "slate.csv", index=False)
    print(f"Wrote {out_dir} with {len(rows)} rows")


if __name__ == "__main__":
    main()
