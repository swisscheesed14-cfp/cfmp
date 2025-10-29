import asyncio, httpx
from typing import List, Tuple, Dict, Any
from ..config import SETTINGS

# ---- low-level helpers -------------------------------------------------

async def _get_json(client: httpx.AsyncClient, url: str, sem: asyncio.Semaphore, retries: int = 3):
    for i in range(retries):
        try:
            async with sem:
                r = await client.get(url, timeout=10.0)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == retries - 1:
                return None
            await asyncio.sleep(0.5 * (2 ** i))

async def _one_point(client: httpx.AsyncClient, sem: asyncio.Semaphore, lat: float, lon: float, hours: int):
    # round to reduce key explosions from tiny float diffs
    key = (round(lat, 4), round(lon, 4))

    meta = await _get_json(client, f"https://api.weather.gov/points/{key[0]},{key[1]}", sem)
    if not meta:
        return key, None

    url = (meta.get("properties") or {}).get("forecastHourly")
    if not url:
        return key, None

    data = await _get_json(client, url, sem)
    if not data:
        return key, None

    periods = ((data.get("properties") or {}).get("periods") or [])[:hours]
    out = [
        {
            "start": p.get("startTime"),
            "tempF": p.get("temperature"),
            "wind": p.get("windSpeed"),
            "forecast": p.get("shortForecast"),
        }
        for p in periods
    ]
    return key, out

async def _gather(coords: List[Tuple[float, float]], hours: int, concurrency: int) -> Dict[Tuple[float, float], Any]:
    sem = asyncio.Semaphore(concurrency)
    headers = {
        "User-Agent": SETTINGS.nws_user_agent or "cfmp/0.0.1 (contact@nowhere)",
        "Accept": "application/geo+json",
    }
    async with httpx.AsyncClient(headers=headers) as client:
        tasks = [_one_point(client, sem, lat, lon, hours) for (lat, lon) in coords]
        results = await asyncio.gather(*tasks)
    return {k: v for k, v in results}

# ---- public API --------------------------------------------------------

def hourly_forecast_many(coords: List[Tuple[float, float]], hours: int = 6, concurrency: int = 24) -> Dict[Tuple[float, float], Any]:
    # de-dup and drop Nones
    uniq = list({(round(lat, 4), round(lon, 4)) for lat, lon in coords if lat is not None and lon is not None})
    if not uniq:
        return {}
    return asyncio.run(_gather(uniq, hours, concurrency))

def hourly_forecast(lat: float, lon: float, hours: int = 6):
    return hourly_forecast_many([(lat, lon)], hours).get((round(lat, 4), round(lon, 4)))
