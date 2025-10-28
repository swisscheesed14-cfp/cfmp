import httpx
from typing import Any
from ..config import SETTINGS

BASE = "https://api.collegefootballdata.com"

def _client() -> httpx.Client:
    headers = {"Accept": "application/json"}
    if SETTINGS.cfbd_key:
        headers["Authorization"] = f"Bearer {SETTINGS.cfbd_key}"
    return httpx.Client(base_url=BASE, headers=headers, timeout=20.0)

def schedule(year: int, week: int | None = None) -> list[dict[str, Any]]:
    params = {"year": year}
    if week:
        params["week"] = week
    with _client() as c:
        r = c.get("/games", params=params)
        r.raise_for_status()
        return r.json()

def venues() -> list[dict[str, Any]]:
    with _client() as c:
        r = c.get("/venues")
        r.raise_for_status()
        return r.json()
