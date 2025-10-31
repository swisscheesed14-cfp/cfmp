import os, json, time, pandas as pd
from datetime import datetime, timedelta
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


def gval(g: dict, *keys):
    for k in keys:
        if k in g and g[k] not in (None, ""):
            return g[k]
    return None


def load_cached_venues():
    if os.path.exists(CACHE_FILE):
        return pd.read_parquet(CACHE_FILE).to_dict("records")
    v = cfbd.venues()
    df = pd.DataFrame(v)[["id", "name", "latitude", "longitude"]]
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    df.to_parquet(CACHE_FILE, index=False)
    print(f"Saved {len(df)} venues to {CACHE_FILE}")
    return df.to_dict("records")


# ---------- new: robust year/week logic ----------
def pick_year_week():
    now_dt = datetime.now(LOCAL_TZ)
    today = now_dt.date()
    year = int(os.getenv("CFB_YEAR", now_dt.year))

    wk_env = os.getenv("CFB_WEEK")
    if wk_env:
        return year, int(wk_env)

    try:
        weeks = cfbd.calendar(year=year)
        for w in weeks:
            s = datetime.fromisoformat(w["startDate"]).date()
            e = datetime.fromisoformat(w["endDate"]).date()
            if s <= today <= e:
                return year, int(w["week"])
        if weeks:
            first = int(weeks[0]["week"])
            last = int(weeks[-1]["week"])
            if today < datetime.fromisoformat(weeks[0]["startDate"]).date():
                return year, first
            return year, last
    except Exception as e:
        print(f"Calendar lookup failed: {e}")

    # ultimate fallback
    return year, now_dt.isocalendar().week


# ---------- main ----------

def main():
    year, week = pick_year_week()
    print(f"Running for {year}-W{week}")

    venues = {v["id"]: v for v in load_cached_venues()}
    games = cfbd.schedule(year=year, week=week)
    rows = []

    for g in games:
        vid = gval(g, "venue_id")
        v = venues.get(vid, {})
        lat = gval(v, "latitude")
        lon = gval(v, "longitude")

        kickoff_raw = gval(g, "start_date")
        kickoff_dt = None
        if kickoff_raw:
            try:
                kickoff_dt = datetime.fromisoformat(kickoff_raw.replace("Z", "+00:00"))
            except Exception:
                pass

        rows.append(
    {
        "id": gval(g, "id"),
        "year": year,
        "week": week,
        "kickoff": kickoff_dt.isoformat() if kickoff_dt else None,
        "away": gval(g, "away_team", "awayTeam", "away"),
        "home": gval(g, "home_team", "homeTeam", "home"),
        # venue info (robust fallbacks)
        "venue": (
            VENUES.get(gval(g, "venue_id", "venueId"), {}).get("name")
            if gval(g, "venue_id", "venueId")
            else gval(g, "venue", "venue_name", "venueName", "stadium", "location")
        ),
        "venue_id": gval(g, "venue_id", "venueId"),
        "venue_lat": (
            VENUES.get(gval(g, "venue_id", "venueId"), {}).get("latitude")
            if gval(g, "venue_id", "venueId")
            else gval(g, "venue_lat", "venueLatitude", "venue_latitude", "latitude")
        ),
        "venue_lon": (
            VENUES.get(gval(g, "venue_id", "venueId"), {}).get("longitude")
            if gval(g, "venue_id", "venueId")
            else gval(g, "venue_lon", "venueLongitude", "venue_longitude", "longitude")
        ),
        # filled after weather call
        "forecast_sample": None,
        "forecast_conf": None,
    }
)

    df = pd.DataFrame(rows)

    # weather forecast block
    forecasts = hourly_forecast_many(
        df[["venue_id", "venue_lat", "venue_lon"]].dropna().to_dict("records")
    )
    fc_map = {f["venue_id"]: f for f in forecasts}

    # assign confidence and samples
    for i, row in df.iterrows():
        kickoff_str = row["kickoff"]
        conf = None
        if kickoff_str:
            try:
                kickoff = datetime.fromisoformat(kickoff_str)
                dt_to_kick = kickoff - datetime.now(LOCAL_TZ)
                days = dt_to_kick.total_seconds() / 86400
                if days >= 7:
                    conf = 0.5
                elif 5 <= days < 7:
                    conf = 0.7
                elif 2 <= days < 5:
                    conf = 0.85
                else:
                    conf = 0.95
            except Exception:
                conf = None
        df.at[i, "forecast_conf"] = conf

        fid = row["venue_id"]
        if fid in fc_map:
            df.at[i, "forecast_sample"] = json.dumps(fc_map[fid].get("hours", []))

    # write output files
    os.makedirs(f"runs/{year}-W{week:02d}", exist_ok=True)
    csv_path = f"runs/{year}-W{week:02d}/slate.csv"
    json_path = f"runs/{year}-W{week:02d}/slate.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print(f"Saved {len(df)} games to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
