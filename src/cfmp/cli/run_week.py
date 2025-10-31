# -*- coding: utf-8 -*-
"""
Produce the weekly slate CSV + JSON.
Hard requirements:
- CFB_YEAR and CFB_WEEK must be set (we fail if not).
- Writes to runs/{year}-W{week}/slate.csv and slate_preview.json
- Adds forecast_conf based on time-to-kickoff
- Uses cached venues from .cache/venues.parquet if present
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd  # for parquet cache

# project ingest
from cfmp.ingest import cfbd as cfbd_ingest

# optional weather import (kept off by default in CI to save time)
try:
    from cfmp.weather.nws import hourly_forecast_many  # type: ignore
except Exception:
    hourly_forecast_many = None  # pragma: no cover


LOCAL_TZ = ZoneInfo(os.getenv("TZ") or "America/Chicago")
CACHE_FILE = Path(".cache/venues.parquet")


# ---------------------------
# helpers
# ---------------------------

def _to_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def pick_year_week() -> Tuple[int, int]:
    """Refuse to run unless both are supplied. Prevents silent drift to ISO week 44."""
    y = _to_int(os.getenv("CFB_YEAR"))
    w = _to_int(os.getenv("CFB_WEEK"))
    if y is None or w is None:
        raise SystemExit("CFB_YEAR and CFB_WEEK must be set (e.g., 2025 / 10)")
    return y, w


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


def load_cached_venues() -> Dict[int, Dict[str, Any]]:
    if not CACHE_FILE.exists():
        return {}
    df = pd.read_parquet(CACHE_FILE)
    need = {"id", "name", "latitude", "longitude"}
    avail = set(map(str, df.columns))
    # tolerate different casings
    cols = {c.lower(): c for c in df.columns}
    got = {k for k in ("id", "name", "latitude", "longitude") if k in cols}
    if not got:
        return {}
    id_col = cols.get("id")
    out: Dict[int, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        try:
            vid = int(r[id_col])
        except Exception:
            continue
        out[vid] = {
            "name": r.get(cols.get("name", ""), None),
            "latitude": r.get(cols.get("latitude", ""), None),
            "longitude": r.get(cols.get("longitude", ""), None),
        }
    return out


# ---------------------------
# main
# ---------------------------

def main() -> None:
    year, week = pick_year_week()
    print(f"Running for {year}-W{week:02d}")

    # venues cache
    venues = load_cached_venues()
    if venues:
        print(f"Loaded {len(venues)} venues from {CACHE_FILE}")

    # fetch games from our ingest (uses CFBD_KEY env internally)
    # NOTE: the correct function in this repo is `schedule`, not `games`.
    games: List[dict] = cfbd_ingest.schedule(year=year, week=week)
    print(f"Fetched {len(games)} games from CFBD")

    if not games:
        raise SystemExit(f"No CFBD games found for year={year}, week={week}.")

    # build rows
    now_local = datetime.now(tz=LOCAL_TZ)
    rows: List[Dict[str, Any]] = []

    # pre-collect coordinates for optional weather
    coords_for_weather: List[Tuple[float, float, int]] = []

    for g in games:
        gid = gval(g, "id") or gval(g, "game_id")
        start = gval(g, "startDate") or gval(g, "kickoff")
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
        try:
            lat_f = float(lat) if lat is not None else None
            lon_f = float(lon) if lon is not None else None
        except Exception:
            lat_f = lon_f = None

        row = {
            "id": gid,
            "year": year,
            "week": week,
            "kickoff": kickoff_local.isoformat() if kickoff_local else None,
            "away": gval(g, "away_team", "awayTeam"),
            "home": gval(g, "home_team", "homeTeam"),
            "venue": v.get("name"),
            "venue_id": vid_int,
            "venue_lat": lat_f,
            "venue_lon": lon_f,
            # filled later if weather is enabled
            "forecast_sample": None,
            "forecast_conf": forecast_confidence(now_local, kickoff_local),
        }
        rows.append(row)

        if os.getenv("CFMP_WEATHER", "").lower() in {"on", "true", "1"}:
            if lat_f is not None and lon_f is not None:
                coords_for_weather.append((lat_f, lon_f, gid if gid is not None else -1))

    # optional weather pull
    if coords_for_weather and hourly_forecast_many is not None:
        try:
            # throttle defaults are fine; map back into rows by id
            uniq = {(lat, lon) for (lat, lon, _) in coords_for_weather}
            fc_map = hourly_forecast_many(list(uniq), hours=6, concurrency=32)
            m = {(lat, lon): fc for (lat, lon), fc in fc_map.items()}
            for r in rows:
                lat = r.get("venue_lat")
                lon = r.get("venue_lon")
                if lat is None or lon is None:
                    continue
                r["forecast_sample"] = m.get((lat, lon))  # may be None
        except Exception as e:
            print(f"Weather step skipped due to error: {e}")

    # output
    out_dir = Path(f"runs/{year}-W{week:02d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON preview
    with (out_dir / "slate_preview.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # CSV
    cols = [
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
        "forecast_sample",
        "forecast_conf",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_dir / "slate.csv", index=False)
    print(f"Wrote {out_dir} with {len(rows)} rows")


if __name__ == "__main__":
    main()
