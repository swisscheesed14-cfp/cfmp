"""Who is not playing. The season stats do not know that until the backup has started.

Covers has the weekly injury table. ESPN's team news catches a starter ruled
out before that table does. A missing page returns nothing and the rating
stands.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

COVERS = "https://www.covers.com/sport/football/ncaaf/injuries"
ESPN_NEWS = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/news"
ESPN_TEAMS = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams"
HEADERS = {"User-Agent": "cfmp/0.0.1"}
SEASON_WORDS = ("remainder of the season", "rest of the season", "rest of the year", "out for the season", "for the season")


def fetch_absences(team_names: list[str]) -> list[dict[str, str]]:
    """Quarterbacks ruled out. `scope` is 'season' or 'week'."""
    found: list[dict[str, str]] = []
    try:
        html = httpx.get(COVERS, headers={"User-Agent": "Mozilla/5.0"}, timeout=30, follow_redirects=True).text
        found.extend(parse_covers(html, team_names))
    except Exception as exc:
        print(f"Injury table skipped: {exc}")
    try:
        found.extend(_espn_news(team_names))
    except Exception as exc:
        print(f"Injury news skipped: {exc}")
    return _prefer_season(found)


def parse_covers(html: str, team_names: list[str]) -> list[dict[str, str]]:
    parts = re.split(r'<a id="([^"]+)"></a>', html)
    rows: list[dict[str, str]] = []
    for index in range(1, len(parts), 2):
        team = _match_team(parts[index], team_names)
        if team is None:
            continue
        body = parts[index + 1]
        for match in re.finditer(
            r"player-link'>\s*[A-Z]\.\s+([^<]+?)</span>\s*</td>\s*<td>([^<]+)</td>\s*<td><b>([^<]+)</b>",
            body,
            re.S,
        ):
            if match.group(2).strip() != "QB":
                continue
            status = match.group(3).strip()
            decided = _scope(status, body[match.end(): match.end() + 1600])
            if decided is None:
                continue
            scope, level = decided
            rows.append({"team": team, "qb": re.sub(r"\s+", " ", match.group(1)).strip(), "scope": scope, "level": level})
    return rows


def parse_headline(headline: str, description: str) -> tuple[str, str, str] | None:
    """Name, scope, and level when a headline rules the quarterback out."""
    text = f"{headline} {description}"
    match = re.search(r"quarterbacks?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s+(?:is\s+)?out\b", text)
    if not match:
        return None
    scope = "season" if any(phrase in text.lower() for phrase in SEASON_WORDS) else "week"
    return match.group(1), scope, "out"


def _scope(status: str, note_html: str) -> tuple[str, str] | None:
    note = re.sub(r"<[^>]+>", " ", note_html).lower()
    label = status.lower()
    season = any(phrase in note or phrase in label for phrase in SEASON_WORDS) or "shut down" in note
    if season and label.startswith(("out", "doubtful")):
        return "season", "out"
    if label.startswith("out"):
        return "week", "out"
    if label.startswith("doubtful"):
        return "week", "doubtful"
    return None


def _espn_news(team_names: list[str]) -> list[dict[str, str]]:
    teams = httpx.get(ESPN_TEAMS, params={"limit": 900}, headers=HEADERS, timeout=30).json()
    ids: dict[str, str] = {}
    for item in teams["sports"][0]["leagues"][0]["teams"]:
        info = item["team"]
        name = _match_team(info.get("displayName", ""), team_names)
        if name and name not in ids:
            ids[name] = str(info["id"])
    cutoff = datetime.now(timezone.utc) - timedelta(days=8)

    def one(pair: tuple[str, str]) -> dict[str, str] | None:
        team, team_id = pair
        try:
            payload = httpx.get(ESPN_NEWS, params={"team": team_id, "limit": 6}, headers=HEADERS, timeout=15).json()
        except Exception:
            return None
        for article in payload.get("articles") or []:
            published = _when(article.get("published"))
            if published is not None and published < cutoff:
                continue
            parsed = parse_headline(str(article.get("headline") or ""), str(article.get("description") or ""))
            if not parsed:
                continue
            qb, scope, level = parsed
            headline = str(article.get("headline") or "")
            prefix = headline.split("quarterback", 1)[0].strip()
            if _match_team(prefix, team_names) != team:
                continue
            return {"team": team, "qb": qb, "scope": scope, "level": level}
        return None

    rows: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for row in pool.map(one, ids.items()):
            if row:
                rows.append(row)
    return rows


def _when(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _prefer_season(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["team"], row["qb"].split()[-1].lower())
        current = best.get(key)
        if current is None or (row["scope"] == "season" and current["scope"] != "season"):
            best[key] = row
    return list(best.values())


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _match_team(label: str, team_names: list[str]) -> str | None:
    key = _norm(label)
    if not key:
        return None
    alias = {
        "pitt": "Pittsburgh",
        "pittpanthers": "Pittsburgh",
        "olemiss": "Ole Miss",
        "olemissrebels": "Ole Miss",
        "southernmiss": "Southern Miss",
        "southernmississippigoldeneagles": "Southern Miss",
    }
    if key in alias and alias[key] in team_names:
        return alias[key]
    exact = [name for name in team_names if _norm(name) == key]
    if exact:
        return exact[0]
    prefixes = [name for name in team_names if key.startswith(_norm(name))]
    if not prefixes:
        return None
    return max(prefixes, key=lambda name: len(_norm(name)))
