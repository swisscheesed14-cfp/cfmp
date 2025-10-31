import ossdfsdsdddd
import httpx, time
from typing import Any
from ..config import SETTINGS

BASE = "https://api.collegefootballdata.com"

def _client() -> httpx.Client:
    headers = {"Accept": "application/json"}
    if SETTINGS.cfbd_key:
        headers["Authorization"] = f"Bearer {SETTINGS.cfbd_key}"
    return httpx.Client(base_url=BASE, headers=headers, timeout=httpx.Timeout(10.0, connect=5.0))

def _get(path: str, params: dict[str, Any] | None = None) -> Any:
    for attempt in range(4):
        try:
            with _client() as c:
                r = c.get(path, params=params)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(0.5 * (2 ** attempt))
    raise RuntimeError("unreachable")

def schedule(year: int, week: int | None = None, season_type: str | None = None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"year": year}
    if week: params["week"] = week
    if season_type: params["seasonType"] = season_type
    return _get("/games", params)

def venues() -> list[dict[str, Any]]:
    return _get("/venues")

def calendar(year: int):
    """
    Return CFBD calendar rows for a season.
    Each row has: week, season, startDate, endDate, etc.
    """
    r = httpx.get(
        "https://api.collegefootballdata.com/calendar",
        params={"year": year},
        headers={"Authorization": f"Bearer {os.getenv('CFBD_KEY')}"},  # secret already set
        timeout=30,
    )
    r.raise_for_status()
    return r.json()