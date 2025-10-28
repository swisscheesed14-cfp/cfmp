import httpx, time
from typing import Any
from ..config import SETTINGS

def _ua():
    return {"User-Agent": SETTINGS.nws_user_agent, "Accept": "application/geo+json"}

def _get(url: str) -> dict:
    with httpx.Client(headers=_ua(), timeout=20.0) as c:
        r = c.get(url)
        r.raise_for_status()
        return r.json()

def hourly_forecast(lat: float, lon: float) -> list[dict[str, Any]]:
    pt = _get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}")
    grid = pt["properties"]["forecastHourly"]
    fc = _get(grid)
    periods = fc["properties"]["periods"]
    time.sleep(0.25)
    return periods
