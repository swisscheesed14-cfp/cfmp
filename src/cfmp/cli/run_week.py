import os, json, time, pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from cfmp.ingest import cfbd
from cfmp.weather.nws import hourly_forecast_many

LOCAL_TZ = ZoneInfo("America/Chicago")
CACHE_FILE = ".cache/venues.parquet"


def _to_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None

def pick_year_week():
    now = datetime.now()
    year = _to_int(os.getenv("CFB_YEAR")) or now.year
    week = _to_int(os.getenv("CFB_WEEK"))
    if week is None:
        # fallback: ISO week number (good enough for automation)
        week = now.isocalendar().week
    return year, week


def gval(g: dict, *keys):
    for k in keys:
        if k in g and g[k] not in (None, ""):
            return g[k]
    return None

def load_cached_venues():
    if os.path.exists(CACHE_FILE):
        return pd.read_parquet(CACHE_FILE).to_dict("records")
    v = cfbd.venues()
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    pd.DataFrame(v).to_parquet(CACHE_FILE, index=False)
    return v

def main():
    year, week = pick_year_week()

    t0 = time.time()
    print(f"Fetching games {year} W{week}…", flush=True)
    games = cfbd.schedule(year=year, week=week, season_type="regular")
    print(f"games ok: {len(games)} in {time.time()-t0:.2f}s", flush=True)

    t1 = time.time()
    print("Loading venues…", flush=True)
    vlist = load_cached_venues()
    print(f"venues ok: {len(vlist)} in {time.time()-t1:.2f}s", flush=True)

    v_by_id = {v.get("id"): (v.get("latitude"), v.get("longitude")) for v in vlist}
    v_by_name = {(v.get("name") or "").strip().lower(): (v.get("latitude"), v.get("longitude")) for v in vlist}

    rows = []
    for g in games:
        home = gval(g, "home_team", "homeTeam")
        away = gval(g, "away_team", "awayTeam")
        kickoff = gval(g, "start_date", "startDate")
        venue_name = (gval(g, "venue") or "").strip()
        vid = gval(g, "venue_id", "venueId")

        lat, lon = (None, None)
        if vid in v_by_id:
            lat, lon = v_by_id[vid]
        elif venue_name:
            lat, lon = v_by_name.get(venue_name.lower(), (None, None))

        rows.append({
            "id": gval(g, "id", "gameId"),
            "year": year,
            "week": week,
            "kickoff": kickoff,
            "away": away,
            "home": home,
            "venue": venue_name,
            "venue_id": vid,
            "venue_lat": lat,
            "venue_lon": lon,
            "forecast_sample": None,  # filled below if weather on
        })

    # Batched async weather (fast)
    if os.getenv("CFMP_WEATHER", "on").lower() != "off":
        tW = time.time()
        coords = [(r["venue_lat"], r["venue_lon"]) for r in rows if r["venue_lat"] and r["venue_lon"]]
        fc_map = hourly_forecast_many(coords, hours=6, concurrency=32)
        for r in rows:
            if r["venue_lat"] and r["venue_lon"]:
                key = (round(r["venue_lat"], 4), round(r["venue_lon"], 4))
                r["forecast_sample"] = fc_map.get(key)
        print(f"weather ok in {time.time()-tW:.2f}s", flush=True)

    outdir = "runs/" + datetime.now(LOCAL_TZ).strftime("%Y-W%V")
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{outdir}/slate.csv", index=False)
    with open(f"{outdir}/slate_preview.json", "w") as f:
        json.dump(rows[:5], f, indent=2)
    print(f"Wrote {outdir} in {time.time()-t0:.2f}s", flush=True)

if __name__ == "__main__":
    raise SystemExit(main())
