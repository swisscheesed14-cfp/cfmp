"""What a team is without the quarterback who built the season line.

His games come out of the forecast. A backup with a couple of starts is
used in their place, pulled hard toward the starter so one bad game is
not the offense. A backup who has not started gets a drop measured from
the quarterback changes we can see, larger when the starter was the
offense, and capped. A one-week absence does not change the rankings.
An absence that is already a streak, or is for the season, does.
"""

from __future__ import annotations

import pandas as pd

from cfmp.ingest.cfbd import game_players
from cfmp.ingest.injuries import fetch_absences
from cfmp.ingest.qb import _team_row
from cfmp.models.season import STATS

BACKUP_PRIOR = 4
DROP_FLOOR = 2.0
DROP_CAP = 8.0
DEFAULT_DROP = 4.5
MIN_CHANGES = 5


def quarterback_docks(
    schedule: pd.DataFrame,
    season: int,
    snaps: pd.DataFrame | None = None,
    weights=None,
    profile: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Points to take off the forward rating. `rest` is the rest of the year."""
    names = _fbs_names(schedule, season)
    if not names:
        return _empty()
    absences = fetch_absences(names)
    if not absences:
        return _empty()
    starts = _start_frame(season)
    if starts.empty:
        return _empty()
    base = _measured_drop(schedule, starts)
    groups = _offense_groups(snaps, starts, season) if snaps is not None else {}
    rows = []
    for absence in absences:
        team = absence["team"]
        last = absence["qb"].split()[-1].lower()
        group = groups.get(team)
        starter = None if group is None else group["starter"]
        if starter and str(starter).split()[-1].lower() != last:
            continue
        if starter is None:
            known = _season_starter(starts, team)
            if known and known.split()[-1].lower() != last:
                continue
            starter = known or absence["qb"]
        missed = 0 if group is None else group["missed"]
        lasting = absence["scope"] == "season" or (absence.get("level") == "out" and missed > 0)
        sample = 0 if group is None else group["backup_games"]
        if sample >= 2 and weights is not None and group is not None:
            points = blend_dock(group["starter_off"], group["backup_off"], sample, group["season_off"], weights)
        else:
            points = prior_dock(profile, team, base)
        if absence.get("level") == "doubtful" and absence["scope"] != "season":
            points = round(points * 0.5, 2)
            lasting = False
        points = round(points, 2)
        rows.append(
            {
                "team": team,
                "qb": starter,
                "scope": "season" if lasting else "week",
                "now": points,
                "rest": points if lasting else 0.0,
            }
        )
    if not rows:
        return _empty()
    return pd.DataFrame(rows).drop_duplicates("team", keep="first")


def blend_dock(starter_off: dict, backup_off: dict, backup_games: int, season_off: dict, weights) -> float:
    """Points the season line is high by, once the starter's games are set aside."""
    trust = backup_games / (backup_games + BACKUP_PRIOR)
    delta = {}
    for off, _, _ in STATS:
        starter = _num(starter_off.get(off))
        backup = _num(backup_off.get(off))
        season = _num(season_off.get(off))
        if starter is None or backup is None or season is None:
            continue
        forward = (1.0 - trust) * starter + trust * backup
        delta[off] = forward - season
    change = _offense_points(delta, weights)
    return max(-DROP_CAP, min(DROP_CAP, -change))


def prior_dock(profile: pd.DataFrame | None, team: str, base: float) -> float:
    """Capped drop for a backup with no real sample. Bigger if the starter was the offense."""
    scale = 1.0
    if profile is not None and "pass_ppa" in profile.columns and team in profile.index:
        passing = pd.to_numeric(profile["pass_ppa"], errors="coerce")
        center = float(passing.mean())
        spread = float(passing.std())
        value = _num(profile.at[team, "pass_ppa"])
        if value is not None and spread and spread == spread and spread > 1e-6:
            distance = max(-1.5, min(1.5, (value - center) / spread))
            scale = 1.0 + 0.35 * distance
    return max(DROP_FLOOR, min(DROP_CAP, base * scale))


def _offense_points(delta: dict, weights) -> float:
    total = 0.0
    for coef, (off, _, hurts) in zip(weights.coef, STATS):
        if off not in delta:
            continue
        sign = -1.0 if hurts else 1.0
        total += float(coef) * sign * float(delta[off])
    return total


def _offense_groups(snaps: pd.DataFrame | None, starts: pd.DataFrame, season: int) -> dict:
    if snaps is None or snaps.empty:
        return {}
    offense = [off for off, _, _ in STATS if off in snaps.columns]
    if not offense:
        return {}
    frame = snaps.loc[snaps["season"] == season, ["game_id", "team", "week", *offense]].copy()
    frame["game_id"] = pd.to_numeric(frame["game_id"], errors="coerce")
    labels = starts.loc[starts["season"] == season, ["game_id", "team", "qb"]].copy()
    labels["game_id"] = pd.to_numeric(labels["game_id"], errors="coerce")
    frame = frame.merge(labels, on=["game_id", "team"], how="inner")
    if frame.empty:
        return {}
    found = {}
    for team, games in frame.groupby("team"):
        counts = games.groupby("qb").size().sort_values(ascending=False)
        starter = str(counts.index[0])
        starter_games = games[games["qb"] == starter]
        backup_games = games[games["qb"] != starter]
        ordered = games.sort_values("week")
        missed = 0
        for name in reversed(ordered["qb"].tolist()):
            if name == starter:
                break
            missed += 1
        found[str(team)] = {
            "starter": starter,
            "missed": missed,
            "backup_games": int(len(backup_games)),
            "starter_off": starter_games[offense].mean().to_dict(),
            "backup_off": backup_games[offense].mean().to_dict() if len(backup_games) else {},
            "season_off": games[offense].mean().to_dict(),
        }
    return found


def _measured_drop(schedule: pd.DataFrame, starts: pd.DataFrame) -> float:
    """Median points a team lost when someone else started. Falls back to 4.5."""
    info = schedule[["game_id", "home_team", "away_team", "home_points", "away_points"]].copy()
    info["game_id"] = pd.to_numeric(info["game_id"], errors="coerce")
    info = info.dropna(subset=["home_points", "away_points"])
    labels = starts.copy()
    labels["game_id"] = pd.to_numeric(labels["game_id"], errors="coerce")
    home = labels.merge(info, left_on=["game_id", "team"], right_on=["game_id", "home_team"], how="inner")
    home["margin"] = home["home_points"] - home["away_points"]
    away = labels.merge(info, left_on=["game_id", "team"], right_on=["game_id", "away_team"], how="inner")
    away["margin"] = away["away_points"] - away["home_points"]
    played = pd.concat([home[["season", "team", "qb", "margin"]], away[["season", "team", "qb", "margin"]]], ignore_index=True)
    drops = []
    for (_, _), games in played.groupby(["season", "team"]):
        counts = games.groupby("qb").size()
        if counts.empty:
            continue
        starter = counts.idxmax()
        own = games[games["qb"] == starter]
        other = games[games["qb"] != starter]
        if len(own) < 2 or len(other) < 2:
            continue
        drops.append(float(own["margin"].mean() - other["margin"].mean()))
    if len(drops) < MIN_CHANGES:
        print(f"Quarterback drop is the prior {DEFAULT_DROP:.1f}. Only {len(drops)} changes had a real sample.")
        return DEFAULT_DROP
    base = float(pd.Series(drops).median())
    base = max(DROP_FLOOR, min(6.0, base))
    print(f"Quarterback drop measured at {base:.2f} from {len(drops)} changes.")
    return base


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["team", "qb", "scope", "now", "rest"])


def _fbs_names(schedule: pd.DataFrame, season: int) -> list[str]:
    frame = schedule.loc[schedule["season"] == season]
    names: set[str] = set()
    for side in ("home", "away"):
        fbs = frame[f"{side}_division"].astype(str).str.lower().eq("fbs")
        names.update(frame.loc[fbs, f"{side}_team"].astype(str))
    return sorted(names)


def _season_starter(starts: pd.DataFrame, team: str) -> str | None:
    games = starts[starts["team"] == team]
    if games.empty:
        return None
    return str(games.groupby("qb").size().sort_values(ascending=False).index[0])


def _start_frame(season: int) -> pd.DataFrame:
    try:
        payload = []
        for week in range(1, 16):
            payload.extend(game_players(season, week=week, season_type="regular"))
    except Exception as exc:
        print(f"Quarterback starts skipped: {exc}")
        return pd.DataFrame(columns=["season", "week", "game_id", "team", "qb"])
    rows = []
    for game in payload:
        week = int(game.get("week") or 0)
        for team in game.get("teams") or []:
            parsed = _team_row(team)
            if not parsed or not parsed.get("qb"):
                continue
            school = str(team.get("school") or team.get("team") or "")
            if school:
                rows.append((season, week, game.get("id"), school, str(parsed["qb"])))
    if not rows:
        return pd.DataFrame(columns=["season", "week", "game_id", "team", "qb"])
    frame = pd.DataFrame(rows, columns=["season", "week", "game_id", "team", "qb"])
    return frame.drop_duplicates(["game_id", "team"])


def _num(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number
