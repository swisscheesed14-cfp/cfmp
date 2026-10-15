import time
from datetime import datetime, timezone
from typing import Any

import httpx

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
        except Exception:
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

def lines(year: int, week: int | None = None, season_type: str | None = "both") -> list[dict[str, Any]]:
    params: dict[str, Any] = {"year": year}
    if week:
        params["week"] = week
    if season_type:
        params["seasonType"] = season_type
    return _get("/lines", params)

def game_players(year: int, week: int | None = None, season_type: str | None = "both") -> list[dict[str, Any]]:
    params: dict[str, Any] = {"year": year}
    if week:
        params["week"] = week
    if season_type:
        params["seasonType"] = season_type
    return _get("/games/players", params)

def _parse_cfbd_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def calendar(year: int):
    """CFBD calendar rows: week, seasonType, startDate, endDate."""
    return _get("/calendar", {"year": year})


def select_week(weeks: list[dict[str, Any]], now: datetime) -> tuple[int, str]:
    """Week that contains `now`, else the next one, else the last one.

    Spring and all-star rows are ignored. Regular season wins if it overlaps
    postseason. Dates are compared in UTC.
    """
    now_utc = now.astimezone(timezone.utc) if now.tzinfo else now.replace(tzinfo=timezone.utc)
    parsed: list[tuple[datetime, datetime, int, str]] = []
    for row in weeks:
        season_type = str(row.get("seasonType") or row.get("season_type") or "regular")
        if season_type not in {"regular", "postseason"}:
            continue
        start = _parse_cfbd_dt(row.get("startDate") or row.get("start_date"))
        end = _parse_cfbd_dt(row.get("endDate") or row.get("end_date"))
        week = row.get("week")
        if start is None or end is None or week is None:
            continue
        parsed.append((start, end, int(week), season_type))
    if not parsed:
        raise SystemExit("CFBD calendar had no usable weeks")
    parsed.sort()

    inside = [row for row in parsed if row[0] <= now_utc <= row[1]]
    regular = [row for row in inside if row[3] == "regular"]
    chosen = (regular or inside or [None])[0]
    if chosen is None:
        upcoming = [row for row in parsed if row[0] > now_utc]
        chosen = upcoming[0] if upcoming else parsed[-1]
    return chosen[2], chosen[3]


def current_week(now: datetime, year: int | None = None) -> tuple[int, int, str]:
    """Return (season year, week, season_type) for a timestamp.

    January and February still belong to the previous season.
    """
    season = year if year is not None else (now.year if now.month >= 3 else now.year - 1)
    week, season_type = select_week(calendar(season), now)
    return season, week, season_type
