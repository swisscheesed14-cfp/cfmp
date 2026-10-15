"""Kickoff-aligned forecasts from api.weather.gov.

NWS covers the US and its territories. A point outside that area is a 404,
which we record as no forecast rather than a pipeline failure.

Coordinates are snapped to 2 decimal degrees (~1 km). That is finer than the
NWS forecast grid and lets venues that share a grid share one request pair.
"""

from __future__ import annotations

import asyncio
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from ..config import SETTINGS

POINTS_URL = "https://api.weather.gov/points/{lat},{lon}"
# The hourly product runs about 7 days. Pull it once, then slice at kickoff.
MAX_HOURS = 168
DEFAULT_CONCURRENCY = 4


def coord_key(lat: float, lon: float) -> tuple[float, float]:
    return (round(float(lat), 2), round(float(lon), 2))


def finite_coord(lat: Any, lon: Any) -> tuple[float, float] | None:
    try:
        la, lo = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(la) and math.isfinite(lo)):
        return None
    if not (-90.0 <= la <= 90.0 and -180.0 <= lo <= 180.0):
        return None
    return la, lo


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _pop(period: dict[str, Any]) -> float | None:
    pop = period.get("probabilityOfPrecipitation")
    val = pop.get("value") if isinstance(pop, dict) else None
    try:
        return None if val is None else float(val)
    except (TypeError, ValueError):
        return None


def _humidity(period: dict[str, Any]) -> float | None:
    hum = period.get("relativeHumidity")
    val = hum.get("value") if isinstance(hum, dict) else None
    try:
        return None if val is None else float(val)
    except (TypeError, ValueError):
        return None


def _wind_mph(text: str | None) -> float | None:
    if not text:
        return None
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(text))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _compact(period: dict[str, Any]) -> dict[str, Any]:
    return {
        "start": period.get("startTime"),
        "tempF": period.get("temperature"),
        "wind": period.get("windSpeed"),
        "wind_mph": _wind_mph(period.get("windSpeed")),
        "wind_dir": period.get("windDirection"),
        "pop": _pop(period),
        "humidity": _humidity(period),
        "forecast": period.get("shortForecast"),
    }


def period_at(periods: list[dict[str, Any]], kickoff: datetime | None) -> dict[str, Any] | None:
    """Hourly period that contains kickoff. None if kickoff is outside the product."""
    parsed: list[tuple[datetime, dict[str, Any]]] = []
    for period in periods:
        start = _parse_dt(period.get("startTime"))
        if start is not None:
            parsed.append((start, period))
    if not parsed:
        return None
    kickoff = _aware(kickoff)
    if kickoff is None:
        return _compact(parsed[0][1])
    for start, period in parsed:
        if start <= kickoff < start + timedelta(hours=1):
            return _compact(period)
    nearest_start, nearest = min(parsed, key=lambda item: abs((item[0] - kickoff).total_seconds()))
    if abs((nearest_start - kickoff).total_seconds()) > 3 * 3600:
        return None
    return _compact(nearest)


async def _get_json(
    client: httpx.AsyncClient,
    url: str,
    sem: asyncio.Semaphore,
    retries: int = 5,
) -> Any:
    for attempt in range(retries):
        try:
            async with sem:
                response = await client.get(url)
            # Outside the US grid. Not a retryable failure.
            if response.status_code == 404:
                return None
            if response.status_code in {429, 500, 502, 503, 504}:
                retry_after = response.headers.get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after else 0.8 * (2**attempt)
                except ValueError:
                    wait = 0.8 * (2**attempt)
                await asyncio.sleep(min(wait, 20.0))
                continue
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            if attempt == retries - 1:
                return None
            await asyncio.sleep(0.5 * (2**attempt))
    return None


async def _periods_for(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    lat: float,
    lon: float,
) -> tuple[tuple[float, float], list[dict[str, Any]] | None]:
    key = coord_key(lat, lon)
    meta = await _get_json(client, POINTS_URL.format(lat=key[0], lon=key[1]), sem)
    if not meta:
        return key, None
    url = (meta.get("properties") or {}).get("forecastHourly")
    if not url:
        return key, None
    data = await _get_json(client, url, sem)
    if not data:
        return key, None
    periods = ((data.get("properties") or {}).get("periods") or [])[:MAX_HOURS]
    return key, periods


async def _gather(
    coords: list[tuple[float, float]],
    concurrency: int,
) -> dict[tuple[float, float], list[dict[str, Any]] | None]:
    sem = asyncio.Semaphore(max(1, concurrency))
    headers = {
        "User-Agent": SETTINGS.nws_user_agent or "cfmp/0.0.1 (contact@nowhere)",
        "Accept": "application/geo+json",
    }
    timeout = httpx.Timeout(20.0, connect=5.0)
    limits = httpx.Limits(max_connections=concurrency + 2, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(
        headers=headers,
        timeout=timeout,
        limits=limits,
        follow_redirects=True,
    ) as client:
        results = await asyncio.gather(*[_periods_for(client, sem, lat, lon) for lat, lon in coords])
    return {key: periods for key, periods in results}


def _unique_coords(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    seen: dict[tuple[float, float], tuple[float, float]] = {}
    for lat, lon in coords:
        point = finite_coord(lat, lon)
        if point is None:
            continue
        key = coord_key(*point)
        seen.setdefault(key, point)
    return list(seen.values())


def hourly_forecast_many(
    coords: list[tuple[float, float]],
    hours: int = 6,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict[tuple[float, float], Any]:
    """Next `hours` of hourly forecast, keyed by coord snapped to 2 decimals."""
    uniq = _unique_coords(coords)
    if not uniq:
        return {}
    raw = asyncio.run(_gather(uniq, concurrency))
    out: dict[tuple[float, float], Any] = {}
    for key, periods in raw.items():
        if not periods:
            out[key] = None
            continue
        out[key] = [_compact(p) for p in periods[:hours]]
    return out


def hourly_forecast(lat: float, lon: float, hours: int = 6) -> Any:
    return hourly_forecast_many([(lat, lon)], hours).get(coord_key(lat, lon))


def forecast_many(
    points: list[tuple[float, float, datetime | None]],
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict[tuple[float, float], list[dict[str, Any]] | None]:
    """Full hourly series per snapped coordinate. Slice with period_at."""
    coords = [(lat, lon) for lat, lon, _kick in points]
    uniq = _unique_coords(coords)
    if not uniq:
        return {}
    return asyncio.run(_gather(uniq, concurrency))
