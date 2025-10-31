# src/cfmp/ingest/results.py
from __future__ import annotations

import os
import sys
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional

import cfbd  # CollegeFootballData SDK

from cfmp.config import SETTINGS


# ---- CFBD client -------------------------------------------------------------

def get_cfbd_client() -> cfbd.GamesApi:
    """
    Return an authenticated GamesApi client. Uses SETTINGS.cfbd_key from .env.
    We set `access_token` so the SDK sends `Authorization: Bearer <key>`.
    """
    cfg = cfbd.Configuration()
    cfg.access_token = SETTINGS.cfbd_key  # <-- correct way to send Bearer token
    return cfbd.GamesApi(cfbd.ApiClient(cfg))


# ---- Helpers -----------------------------------------------------------------

def to_iso(v: Any) -> Any:
    """Convert datetime/date to ISO string, leave everything else as-is."""
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return v


def normalize(obj: Any) -> Any:
    """
    Recursively convert dataclasses and objects to JSON-serializable dicts.
    - datetimes -> ISO strings
    - cfbd models -> dict via __dict__ filtering private attrs and callables
    - lists/tuples/dicts handled recursively
    """
    if obj is None:
        return None
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (list, tuple)):
        return [normalize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: normalize(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return normalize(asdict(obj))
    # cfbd model objects are plain objects; filter __dict__
    if hasattr(obj, "__dict__"):
        d = {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        return normalize(d)
    return obj


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(normalize(r), separators=(",", ":")) + "\n")


# ---- Ingest logic ------------------------------------------------------------

def fetch_year(y: int) -> List[Dict[str, Any]]:
    api = get_cfbd_client()
    # Pull both regular + postseason
    games = api.get_games(year=y, season_type="both")  # returns list[Game]
    rows: List[Dict[str, Any]] = []

    for g in games:
        # Keep completed games; keep scheduled too if you want a full ledger.
        # Here we write everything and rely on None for points if not played.
        rows.append(
            {
                "season": y,
                "week": getattr(g, "week", None),
                "id": getattr(g, "id", None),
                "start_date": getattr(g, "start_date", None) or getattr(g, "start_time", None),
                "home_team": getattr(g, "home_team", None),
                "away_team": getattr(g, "away_team", None),
                "home_points": getattr(g, "home_points", None),
                "away_points": getattr(g, "away_points", None),
                "neutral_site": getattr(g, "neutral_site", None),
                "conference_game": getattr(g, "conference_game", None),
                "venue": getattr(g, "venue", None),
                "venue_id": getattr(g, "venue_id", None),
                "home_conference": getattr(g, "home_conference", None),
                "away_conference": getattr(g, "away_conference", None),
                # Keep original object flattened for anything else you might need
                # without risking non-serializable types.
            }
        )
    return rows


def write_year(y: int, rows: List[Dict[str, Any]]) -> None:
    out = f"data/results/{y}.jsonl"
    save_jsonl(out, rows)


# ---- CLI ---------------------------------------------------------------------

def main(argv: List[str]) -> None:
    # Default: current and previous year if none provided
    if len(argv) > 1:
        years = [int(a) for a in argv[1:]]
    else:
        now = datetime.now()
        years = [now.year - 1, now.year]

    print(f"Fetching results for years: {years}")
    for y in years:
        rows = fetch_year(y)
        write_year(y, rows)
        print(f"Wrote {len(rows)} rows to data/results/{y}.jsonl")


if __name__ == "__main__":
    try:
        main(sys.argv)
    except KeyboardInterrupt:
        sys.exit(130)
