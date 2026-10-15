"""Weather moves a site spread only when the conditions are actually bad.

Mild conditions do nothing. A sunny 50s or 60s day, even with a 15 mph
breeze, is not an edge. Indoors, the forecast is ignored.

When we know which offense throws more, that side absorbs the penalty.
Otherwise the spread shrinks slightly toward a pick'em. Weather never
flips the favorite by itself.
"""

from __future__ import annotations

import math
import re
from typing import Any

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
