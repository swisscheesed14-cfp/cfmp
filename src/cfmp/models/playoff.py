"""Chance each team makes the 2026 College Football Playoff.

The field is 12 teams. The ACC, Big Ten, Big 12, and SEC champions are in
even with a bad record. One spot goes to the highest-rated Group of Six
team, champion or not. Notre Dame is in when the final rating puts them
in the top 12, and that spot comes out of the at-large pool. Everyone
else fills what is left, in order of the rating after the simulated games.

The rating starts at the current FPI. Wins and losses move it, and a win
over a better team moves it more. Conference championship games are not
on the schedule yet, so each simulation crowns a champion from the top
two conference records. Pac-12 games after week 12 are flex games and do
not count in that league's standings. Army-Navy is week 15, after the
title games, so it counts for the record and not for a championship berth.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

POWER = ("ACC", "Big Ten", "Big 12", "SEC")
GROUP = ("American Athletic", "Conference USA", "Mid-American", "Mountain West", "Pac-12", "Sun Belt")
FCS_FPI = -18.0
WIN_BASE = 1.6
LOSS_BASE = 2.2
SIMS = 4000


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def win_chance(spread: float, residual_sd: float) -> float:
    if residual_sd <= 0:
        return 1.0 if spread > 0 else 0.0
    return float(min(0.99, max(0.01, _phi(spread / residual_sd))))


def _win_bump(opponent: float) -> float:
    return WIN_BASE + 0.06 * opponent


def _loss_bump(opponent: float) -> float:
    return -(LOSS_BASE - 0.04 * opponent)


def select_field(
    resume: dict[str, float],
    champs: dict[str, str],
    conference: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    """The 12-team field and why each team is in it."""
    ranked = sorted(resume, key=lambda team: (resume[team], team), reverse=True)
    rank = {team: index + 1 for index, team in enumerate(ranked)}
    field: list[str] = []
    reason: dict[str, str] = {}

    def add(team: str | None, why: str) -> None:
        if not team or team in field:
            return
        field.append(team)
        reason[team] = why

    for name in POWER:
        add(champs.get(name), "champ")
    group = [team for team, conf in conference.items() if conf in GROUP and team in resume]
    if group:
        add(max(group, key=lambda team: (resume[team], team)), "group")
    if rank.get("Notre Dame", 99) <= 12:
        add("Notre Dame", "notre dame")
    for team in ranked:
        if len(field) >= 12:
            break
        add(team, "at-large")
    return field[:12], reason


def _fbs_mask(frame: pd.DataFrame, side: str) -> pd.Series:
    return frame[f"{side}_division"].astype(str).str.lower().eq("fbs")


def playoff_odds(
    schedule: pd.DataFrame,
    ratings: pd.Series,
    hfa: float,
    residual_sd: float,
    season: int = 2026,
    n: int = SIMS,
    seed: int = 7,
    season_dock: pd.Series | None = None,
    week_dock: pd.Series | None = None,
    dock_week: int | None = None,
    crowd: pd.Series | None = None,
) -> pd.DataFrame:
    """Share of simulations in which each team makes the 12-team field."""
    games = schedule.loc[schedule["season"] == season].copy()
    home_fbs = _fbs_mask(games, "home")
    away_fbs = _fbs_mask(games, "away")
    games = games[home_fbs | away_fbs]
    names = sorted(set(games.loc[home_fbs, "home_team"]) | set(games.loc[away_fbs, "away_team"]))
    index = {name: pos for pos, name in enumerate(names)}
    teams = len(names)
    rating = np.array([float(ratings.get(name, 0.0)) for name in names])
    if season_dock is not None:
        rating = rating - np.array([float(season_dock.get(name, 0.0)) for name in names])
    week_penalty = np.zeros(teams) if week_dock is None else np.array([float(week_dock.get(name, 0.0)) for name in names])
    crowd_map = {}
    if crowd is not None:
        for key, value in crowd.items():
            try:
                crowd_map[int(key)] = float(value)
            except (TypeError, ValueError):
                continue
    conference = {}
    for row in games.itertuples(index=False):
        if str(row.home_division).lower() == "fbs":
            conference[row.home_team] = row.home_conference
        if str(row.away_division).lower() == "fbs":
            conference[row.away_team] = row.away_conference

    wins = np.zeros(teams)
    losses = np.zeros(teams)
    conf_wins = np.zeros(teams)
    conf_played = np.zeros(teams)
    resume = rating.copy()
    played_beat = np.zeros((teams, teams), dtype=bool)

    def standing_game(row) -> bool:
        if not bool(row.conference_game):
            return False
        if row.home_conference != row.away_conference:
            return False
        if int(row.week) >= 15:
            return False
        if row.home_conference == "Pac-12" and int(row.week) > 12:
            return False
        return str(row.home_division).lower() == "fbs" and str(row.away_division).lower() == "fbs"

    def apply_result(home: int, away: int, home_won: bool, conference_game: bool) -> None:
        winner, loser = (home, away) if home_won else (away, home)
        wins[winner] += 1
        losses[loser] += 1
        resume[winner] += _win_bump(rating[loser])
        resume[loser] += _loss_bump(rating[winner])
        if conference_game:
            conf_wins[winner] += 1
            conf_played[winner] += 1
            conf_played[loser] += 1
            played_beat[winner, loser] = True

    remaining = []
    late = []
    for row in games.itertuples(index=False):
        home_is = str(row.home_division).lower() == "fbs"
        away_is = str(row.away_division).lower() == "fbs"
        home = index[row.home_team] if home_is else None
        away = index[row.away_team] if away_is else None
        counts = standing_game(row)
        if pd.notna(row.home_points) and pd.notna(row.away_points):
            if home is None or away is None:
                side = home if home is not None else away
                fbs_won = (row.home_points > row.away_points) if home is not None else (row.away_points > row.home_points)
                if side is not None and row.home_points != row.away_points:
                    if fbs_won:
                        wins[side] += 1
                        resume[side] += _win_bump(FCS_FPI)
                    else:
                        losses[side] += 1
                        resume[side] += _loss_bump(FCS_FPI)
                continue
            if row.home_points == row.away_points:
                continue
            apply_result(home, away, row.home_points > row.away_points, counts)
            continue
        spread = _spread(home, away, rating, hfa, bool(row.neutral_site))
        if not bool(row.neutral_site):
            try:
                spread += crowd_map.get(int(row.game_id), 0.0)
            except (TypeError, ValueError):
                pass
        if dock_week is not None and int(row.week) == dock_week:
            spread += _side(week_penalty, away) - _side(week_penalty, home)
        item = (home, away, win_chance(spread, residual_sd), counts, int(row.week))
        if int(row.week) >= 15:
            late.append(item)
        else:
            remaining.append(item)

    members = {name: [index[team] for team, conf in conference.items() if conf == name] for name in (*POWER, *GROUP)}
    playoff = np.zeros(teams)
    auto = np.zeros(teams)
    at_large = np.zeros(teams)
    champ_hits = np.zeros(teams)
    sim_wins = np.zeros(teams)
    rng = np.random.default_rng(seed)
    base_wins, base_losses = wins.copy(), losses.copy()
    base_conf_wins, base_conf_played = conf_wins.copy(), conf_played.copy()
    base_resume = resume.copy()

    for _ in range(n):
        wins[:] = base_wins
        losses[:] = base_losses
        conf_wins[:] = base_conf_wins
        conf_played[:] = base_conf_played
        resume[:] = base_resume
        beat = played_beat.copy()
        for home, away, chance, counts, _week in remaining:
            _play(home, away, chance, counts, rating, wins, losses, conf_wins, conf_played, resume, beat, rng)
        champs = {}
        for name, sides in members.items():
            if len(sides) < 2:
                continue
            order = _order(sides, conf_wins, conf_played, rating, beat)
            host, visitor = order[0], order[1]
            neutral = name != "Pac-12"
            chance = win_chance(_spread(host, visitor, rating, hfa, neutral), residual_sd)
            home_won = _play(host, visitor, chance, False, rating, wins, losses, conf_wins, conf_played, resume, beat, rng)
            champs[name] = names[host if home_won else visitor]
            champ_hits[host if home_won else visitor] += 1
        for home, away, chance, counts, _week in late:
            _play(home, away, chance, False, rating, wins, losses, conf_wins, conf_played, resume, beat, rng)
        scored = {names[pos]: float(resume[pos]) for pos in range(teams)}
        field, reasons = select_field(scored, champs, conference)
        sim_wins += wins
        for team in field:
            pos = index[team]
            playoff[pos] += 1
            if reasons[team] == "at-large":
                at_large[pos] += 1
            else:
                auto[pos] += 1

    out = pd.DataFrame(
        {
            "team": names,
            "conference": [conference[name] for name in names],
            "playoff": (100.0 * playoff / n).round(1),
            "auto": (100.0 * auto / n).round(1),
            "at_large": (100.0 * at_large / n).round(1),
            "champ": (100.0 * champ_hits / n).round(1),
            "wins": (sim_wins / n).round(2),
        }
    )
    out = out.sort_values(["playoff", "wins", "team"], ascending=[False, False, True]).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    return out


def _side(penalty: np.ndarray, side: int | None) -> float:
    if side is None:
        return 0.0
    return float(penalty[side])


def _spread(home: int | None, away: int | None, rating: np.ndarray, hfa: float, neutral: bool) -> float:
    home_rating = FCS_FPI if home is None else float(rating[home])
    away_rating = FCS_FPI if away is None else float(rating[away])
    return home_rating - away_rating + (0.0 if neutral else hfa)


def _play(
    home: int | None,
    away: int | None,
    chance: float,
    conference_game: bool,
    rating: np.ndarray,
    wins: np.ndarray,
    losses: np.ndarray,
    conf_wins: np.ndarray,
    conf_played: np.ndarray,
    resume: np.ndarray,
    beat: np.ndarray,
    rng: np.random.Generator,
) -> bool:
    home_won = bool(rng.random() < chance)
    if home is None or away is None:
        side = home if home is not None else away
        if side is None:
            return home_won
        if home_won if home is not None else not home_won:
            wins[side] += 1
            resume[side] += _win_bump(FCS_FPI)
        else:
            losses[side] += 1
            resume[side] += _loss_bump(FCS_FPI)
        return home_won
    winner, loser = (home, away) if home_won else (away, home)
    wins[winner] += 1
    losses[loser] += 1
    resume[winner] += _win_bump(float(rating[loser]))
    resume[loser] += _loss_bump(float(rating[winner]))
    if conference_game:
        conf_wins[winner] += 1
        conf_played[winner] += 1
        conf_played[loser] += 1
        beat[winner, loser] = True
    return home_won


def _order(sides: list[int], conf_wins: np.ndarray, conf_played: np.ndarray, rating: np.ndarray, beat: np.ndarray) -> list[int]:
    pct = {side: conf_wins[side] / conf_played[side] if conf_played[side] else 0.0 for side in sides}

    def key(side: int, tied: list[int]) -> tuple[float, int, float]:
        h2h = 0
        for other in tied:
            if other == side or abs(pct[side] - pct[other]) > 1e-9:
                continue
            if beat[side, other]:
                h2h += 1
            elif beat[other, side]:
                h2h -= 1
        return (pct[side], h2h, float(rating[side]))

    order = sorted(sides, key=lambda side: key(side, []), reverse=True)
    tied = [side for side in sides if any(abs(pct[side] - pct[top]) < 1e-9 for top in order[:2])]
    if len(tied) < 2:
        return order
    return sorted(sides, key=lambda side: key(side, tied), reverse=True)
