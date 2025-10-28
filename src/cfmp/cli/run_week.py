import os, json, pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from cfmp.ingest import cfbd
from cfmp.weather.nws import hourly_forecast

LOCAL_TZ = ZoneInfo("America/Chicago")

def main():
    now_local = datetime.now(LOCAL_TZ)
    year = now_local.year
    week = int(os.getenv("CFB_WEEK", "0")) or now_local.isocalendar().week
    games = cfbd.schedule(year=year, week=week)

    rows = []
    for g in games:
        venue = g.get("venue") or {}
        lat, lon = venue.get("latitude"), venue.get("longitude")
        fc = None
        if lat and lon:
            try:
                fc = hourly_forecast(lat, lon)[:6]
            except Exception:
                fc = None
        rows.append({
            "id": g.get("id"),
            "kickoff": g.get("start_date"),
            "away": g.get("away_team"),
            "home": g.get("home_team"),
            "venue_lat": lat, "venue_lon": lon,
            "forecast_sample": fc,
        })

    outdir = "runs/" + now_local.strftime("%Y-W%V")
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{outdir}/slate.csv", index=False)
    with open(f"{outdir}/slate_preview.json", "w") as f:
        json.dump(rows[:5], f, indent=2)
    print(f"Wrote {outdir}")

if __name__ == "__main__":
    raise SystemExit(main())
