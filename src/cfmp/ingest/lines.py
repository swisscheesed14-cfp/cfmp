"""DraftKings spreads from CFBD. The number is the exam, never a model input.

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


def _one(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _draftkings(books: list[dict[str, Any]]) -> dict[str, Any] | None:
    for book in books:
        if str(book.get("provider") or "").lower() == "draftkings" and _one(book.get("spread")) is not None:
            return book
    return None


def normalize_lines(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per game. The spread is DraftKings, not a blend of books.

    CFBD's home spread is negative when the home team is favored. We flip it
    so a positive number means the home team is favored.
    """
    rows: list[dict[str, Any]] = []
    for game in payload:
        book = _draftkings(game.get("lines") or [])
        if book is None:
            continue
        close = _one(book.get("spread"))
        opened = _one(book.get("spreadOpen"))
        rows.append(
            {
                "id": game.get("id"),
                "book": "DraftKings",
                "vegas_spread": None if close is None else -close,
                "vegas_open": None if opened is None else -opened,
                "total": _one(book.get("overUnder")),
                "total_open": _one(book.get("overUnderOpen")),
                "home_ml": _one(book.get("homeMoneyline")),
                "away_ml": _one(book.get("awayMoneyline")),
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
