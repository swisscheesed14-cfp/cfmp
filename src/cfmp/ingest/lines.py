"""Closing spreads from CFBD. The number is the exam, never a model input.

CFBD stores the home-team spread, negative when the home team is favored.
We store the home margin instead, so +7 means the same thing as our site spread.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cfmp.ingest.cfbd import lines as fetch_lines
from cfmp.ingest.results import FIRST_SEASON, season_year

LINES_PATH = Path("data/lines.parquet")


def _floats(values: list[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number:
            out.append(number)
    return out


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def normalize_lines(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for game in payload:
        books = game.get("lines") or []
        spreads = _floats([book.get("spread") for book in books])
        opens = _floats([book.get("spreadOpen") for book in books])
        close = _median(spreads)
        opened = _median(opens)
        if close is None and opened is None:
            continue
        rows.append(
            {
                "id": game.get("id"),
                # Positive means the home team is favored, matching site_spread.
                "vegas_spread": None if close is None else -close,
                "vegas_open": None if opened is None else -opened,
                "n_books": len(spreads),
            }
        )
    return rows


def fetch_year(year: int) -> list[dict[str, Any]]:
    return normalize_lines(fetch_lines(year, season_type="both"))


def ensure_lines(years: list[int] | None = None, dest: Path = LINES_PATH, refresh_year: int | None = None) -> Path:
    if years is None:
        years = list(range(FIRST_SEASON, season_year() + 1))
    frames: list[pd.DataFrame] = []
    cache = dest.with_suffix(".json")
    cached: dict[str, list] = {}
    if cache.exists():
        import json

        cached = json.loads(cache.read_text(encoding="utf-8"))
    for year in years:
        key = str(year)
        if key not in cached or year == refresh_year:
            cached[key] = fetch_year(year)
            print(f"Fetched lines for {len(cached[key])} games in {year}")
        frames.append(pd.DataFrame(cached[key]))
    import json

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(cached), encoding="utf-8")
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        df = df.drop_duplicates(subset=["id"], keep="last")
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    print(f"Wrote {len(df)} lines -> {dest}")
    return dest
