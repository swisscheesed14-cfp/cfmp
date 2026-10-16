"""Weather moves a site spread only when the conditions are actually bad.

Mild conditions do nothing. A sunny 50s or 60s day, even with a 15 mph
breeze, is not an edge. Indoors, the forecast is ignored.

When we know which offense throws more, that side absorbs the penalty.
Otherwise the spread shrinks slightly toward a pick'em. Weather never
flips the favorite by itself. The weekly card uses the forecast at
kickoff, not the weather on the day the report runs.
"""

from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Names that do not contain "dome" but are roofed or retractable.
INDOOR_VENUES = {
    "alerus center",
    "allegiant stadium",
    "at&t stadium",
    "ford field",
    "holt arena",
    "lucas oil stadium",
    "mercedes-benz stadium",
    "nrg stadium",
    "sofi stadium",
    "state farm stadium",
    "tropicana field",
    "u.s. bank stadium",
}

_DOME_RE = re.compile(r"dome", re.IGNORECASE)
_WET_RE = re.compile(r"rain|snow|storm|sleet|drizzle|shower|thunder", re.IGNORECASE)
VENUES_PATH = Path("data/venues_min.parquet")

# At severity 1.0, shrink the spread by this fraction, and never by more than
# this many points. A hurricane is worth about a field goal, not a touchdown.
MAX_SHRINK_FRACTION = 0.15
MAX_SHRINK_POINTS = 3.0


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def is_indoor(venue: Any = None, dome: Any = None) -> bool:
    if isinstance(dome, bool) and dome:
        return True
    if isinstance(dome, (int, float)) and not isinstance(dome, bool) and dome == 1:
        return True
    if isinstance(dome, str) and dome.strip().lower() in {"1", "true", "yes", "y"}:
        return True
    name = str(venue or "").strip().lower()
    if not name or name == "nan":
        return False
    if name in INDOOR_VENUES:
        return True
    return _DOME_RE.search(name) is not None


def weather_severity(
    temp_f: Any = None,
    wind_mph: Any = None,
    pop: Any = None,
    forecast: Any = None,
) -> float:
    """0.0 means ignore the weather. 1.0 is the worst bucket we model.

    Thresholds are intentional:
    - wind does nothing through 15 mph, then ramps to full at 35 mph
    - precipitation does nothing unless the text says it and the chance is
      at least 50 percent
    - temperature does nothing at 45 F and above, then ramps toward 15 F
    """
    severity = 0.0

    wind = _num(wind_mph) or 0.0
    if wind > 15.0:
        severity += min(1.0, (wind - 15.0) / 20.0)

    chance = _num(pop) or 0.0
    text = "" if forecast is None else str(forecast)
    if text.lower() != "nan" and _WET_RE.search(text) and chance >= 50.0:
        severity += min(1.0, (chance - 50.0) / 50.0) * 0.75

    temp = _num(temp_f)
    if temp is not None and temp < 45.0:
        severity += min(1.0, (45.0 - temp) / 30.0) * 0.5

    return min(1.0, severity)


def adjust_spread(spread_home: float, severity: float) -> tuple[float, float]:
    """Return (adjusted home spread, points added to the home spread).

    Positive spread means the home team is favored. Bad weather pulls that
    number toward zero and never through it.
    """
    if severity <= 0.0 or spread_home == 0.0:
        return spread_home, 0.0
    shrink = min(abs(spread_home) * MAX_SHRINK_FRACTION * severity, MAX_SHRINK_POINTS * severity)
    delta = -shrink if spread_home > 0.0 else shrink
    return spread_home + delta, delta


def weather_site_adjustment(
    severity: float,
    spread_home: float,
    pass_home: float | None = None,
    pass_away: float | None = None,
    max_points: float = 3.0,
) -> float:
    """Points added to the home spread.

    If we know who throws more, the pass-heavier side is the one that pays.
    If we do not, fall back to a small shrink toward a pick'em.
    """
    if severity <= 0.0 or max_points <= 0.0:
        return 0.0
    if pass_home is not None and pass_away is not None:
        tilt = max(-1.0, min(1.0, (pass_away - pass_home) * 2.0))
        return tilt * severity * max_points
    _, delta = adjust_spread(spread_home, severity)
    return delta


def forecast_confidence(now: datetime, kickoff: datetime | None) -> float:
    """How much of the weather adjustment to keep as kickoff gets further away.

    <=48h is 0.95, through 5 days is 0.85, through 7 days is 0.70, and
    anything further out, or an unknown kickoff, is 0.50.
    """
    if kickoff is None:
        return 0.50
    hours = (kickoff - now).total_seconds() / 3600.0
    if hours <= 48:
        return 0.95
    if hours <= 5 * 24:
        return 0.85
    if hours <= 7 * 24:
        return 0.70
    return 0.50


def _kickoff(value: Any) -> datetime | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def forecast_label(snap: dict[str, Any] | None, indoor: bool) -> str:
    if indoor:
        return "dome"
    if not snap:
        return ""
    parts: list[str] = []
    temp = snap.get("tempF")
    if temp is not None:
        parts.append(f"{int(temp)} F")
    wind = snap.get("wind")
    if wind:
        parts.append(str(wind))
    text = snap.get("forecast")
    if text:
        parts.append(str(text))
    return ", ".join(parts)


def adjust_for_forecast(
    spread_home: float,
    indoor: bool,
    snap: dict[str, Any] | None,
    confidence: float,
) -> tuple[float, float]:
    """Return the weather-adjusted home spread and the points added to it."""
    if indoor or not snap:
        return spread_home, 0.0
    severity = weather_severity(
        temp_f=snap.get("tempF"),
        wind_mph=snap.get("wind_mph"),
        pop=snap.get("pop"),
        forecast=snap.get("forecast"),
    )
    severity *= max(0.0, min(1.0, confidence))
    delta = weather_site_adjustment(severity, spread_home)
    return spread_home + delta, delta


def _weather_on() -> bool:
    return os.getenv("CFMP_WEATHER", "on").strip().lower() not in {"off", "false", "0", "no"}


def apply_kickoff_weather(priced: pd.DataFrame, slate: pd.DataFrame, venues_path: Path = VENUES_PATH) -> pd.DataFrame:
    """Move each game's home spread by the forecast at that stadium's kickoff.

    A roofed stadium stays put. A miss from the weather service leaves the
    unadjusted number in place rather than failing the weekly card.
    """
    out = priced.copy()
    out["weather"] = 0.0
    out["forecast"] = ""
    if out.empty or not _weather_on():
        return out
    needed = [column for column in ("game_id", "venue_id", "venue", "start_date") if column in slate.columns]
    if "game_id" not in needed or "venue" not in needed:
        print("Weather skipped: the slate has no venue.")
        return out
    try:
        from cfmp.weather.nws import coord_key, finite_coord, forecast_many, period_at

        info = slate[needed].drop_duplicates("game_id")
        info["game_id"] = pd.to_numeric(info["game_id"], errors="coerce")
        out["game_id"] = pd.to_numeric(out["game_id"], errors="coerce")
        detail = out[["game_id"]].merge(info, on="game_id", how="left")
        if venues_path.exists() and "venue_id" in detail.columns:
            venues = pd.read_parquet(venues_path).rename(columns={"id": "venue_id", "latitude": "venue_lat", "longitude": "venue_lon"})
            detail["venue_id"] = pd.to_numeric(detail["venue_id"], errors="coerce")
            venues["venue_id"] = pd.to_numeric(venues["venue_id"], errors="coerce")
            venues = venues.drop_duplicates("venue_id")
            detail = detail.merge(venues[["venue_id", "venue_lat", "venue_lon"]], on="venue_id", how="left")
        else:
            detail["venue_lat"] = None
            detail["venue_lon"] = None

        now = datetime.now(timezone.utc)
        requests: list[tuple[float, float, datetime | None]] = []
        for row in detail.itertuples(index=False):
            if is_indoor(getattr(row, "venue", None)):
                continue
            point = finite_coord(getattr(row, "venue_lat", None), getattr(row, "venue_lon", None))
            if point is None:
                continue
            requests.append((point[0], point[1], _kickoff(getattr(row, "start_date", None))))
        series = forecast_many(requests) if requests else {}
        if len(detail) != len(out):
            print(f"Weather skipped: {len(detail)} venue rows for {len(out)} games")
            return out

        adjusted: list[float] = []
        deltas: list[float] = []
        labels: list[str] = []
        moved = domes = missed = 0
        for row, game in zip(detail.itertuples(index=False), out.itertuples(index=False)):
            spread = float(game.model)
            indoor = is_indoor(getattr(row, "venue", None))
            if indoor:
                adjusted.append(spread)
                deltas.append(0.0)
                labels.append("dome")
                domes += 1
                continue
            point = finite_coord(getattr(row, "venue_lat", None), getattr(row, "venue_lon", None))
            kick = _kickoff(getattr(row, "start_date", None))
            snap = None
            if point is not None:
                snap = period_at(series.get(coord_key(*point)) or [], kick)
            new_spread, delta = adjust_for_forecast(spread, False, snap, forecast_confidence(now, kick))
            adjusted.append(new_spread)
            deltas.append(delta)
            labels.append(forecast_label(snap, False))
            if snap is None:
                missed += 1
            elif delta != 0.0:
                moved += 1
        out["model"] = adjusted
        out["weather"] = deltas
        out["forecast"] = labels
        print(f"Weather: {moved} games moved, {domes} indoors, {missed} without a kickoff forecast")
    except Exception as exc:
        print(f"Weather step skipped: {exc}")
    return out
