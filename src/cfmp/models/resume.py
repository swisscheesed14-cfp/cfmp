"""Results top 25. Wins and losses, not the snap rating.

A win is worth more than the margin. Beating a good team is worth more than
beating a bad one. A road win and a home loss count a little extra.
Strength of wins is who you beat. Strength of losses is who you lost to.
A loss to a good team is not the same as a loss to a bad one, and one loss
does not keep a team ranked when the wins are not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WIN_POINTS = 3.0
MARGIN_CAP = 10.0
MARGIN_POINT = 0.08
ROAD_POINTS = 0.75
HOME_FIELD = 2.5
CONFERENCE_PULL = 0.0
# Schedule is reported and used to order teams that have the same number of losses.
# It is not added a second time on top of the result.
SOS_WEIGHT = 0.0
# The first loss starts at this cost. Each loss after that starts higher.
LOSS_COST = 1.25
# How much steeper the next loss is. 1.0 means the second loss starts at twice the first.
LOSS_ACCELERATION = 1.0
# Subtracted from that loss for each point of opponent quality. A bad loss costs more.
LOSS_QUALITY = 0.15
# Same number of losses still counts, unless the winner's losses were clearly worse.
LOSS_COMPARE = 1.0
# A win pulls you just ahead of that opponent when your record is at least as good.
HEAD_TO_HEAD_CUSHION = 0.001
RECENCY_PER_WEEK = 0.04
RECENCY_FLOOR = 0.75
ITERATIONS = 8
MIN_GAMES = 4


def _value(margin: float, at_home: bool, neutral: bool) -> float:
    """The result first. The margin is a tiebreak, and it stops at 10 points."""
    if not np.isfinite(margin) or margin == 0:
        return 0.0
    site = 0.0 if neutral else (HOME_FIELD if at_home else -HOME_FIELD)
    bonus = WIN_POINTS if margin > 0 else -WIN_POINTS
    scored = bonus + float(np.clip(margin - site, -MARGIN_CAP, MARGIN_CAP)) * MARGIN_POINT
    if not neutral and margin > 0 and not at_home:
        scored += ROAD_POINTS
    if not neutral and margin < 0 and at_home:
        scored -= ROAD_POINTS
    return scored


def _recency(week: float, through: float) -> float:
    age = max(0.0, float(through) - float(week))
    return max(RECENCY_FLOOR, 1.0 - RECENCY_PER_WEEK * age)


def _games(schedule: pd.DataFrame, season: int, through_week: int) -> pd.DataFrame:
    frame = schedule.loc[schedule["season"] == season].copy()
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame = frame[frame["week"] <= through_week]
    frame = frame[frame["home_points"].notna() & frame["away_points"].notna()]
    rows = []
    for game in frame.itertuples(index=False):
        margin = float(game.home_points) - float(game.away_points)
        neutral = bool(game.neutral_site)
        home_fbs = str(game.home_division).lower() == "fbs"
        away_fbs = str(game.away_division).lower() == "fbs"
        if not home_fbs and not away_fbs:
            continue
        rows.append(
            {
                "week": int(game.week),
                "team": game.home_team,
                "opponent": game.away_team,
                "conference": game.home_conference,
                "fbs": home_fbs,
                "opponent_fbs": away_fbs,
                "margin": margin,
                "value": _value(margin, True, neutral),
                "weight": _recency(game.week, through_week),
            }
        )
        rows.append(
            {
                "week": int(game.week),
                "team": game.away_team,
                "opponent": game.home_team,
                "conference": game.away_conference,
                "fbs": away_fbs,
                "opponent_fbs": home_fbs,
                "margin": -margin,
                "value": _value(-margin, False, neutral),
                "weight": _recency(game.week, through_week),
            }
        )
    played = pd.DataFrame(rows)
    if played.empty:
        return played
    return played[played["fbs"]].copy()


def _head_to_head(played: pd.DataFrame, score: pd.Series, loss_strength: pd.Series) -> pd.Series:
    """Put a winner just ahead of the team it beat.

    More losses does not jump fewer losses. The same number of losses does,
    unless the winner's losses were clearly worse. That is what keeps a loss
    to Tulsa from copying Oregon's rank.
    """
    wins = played[played["opponent_fbs"] & (played["margin"] > 0)]
    decided = played[played["margin"].ne(0)]
    losses = decided.groupby("team")["margin"].apply(lambda margins: int((margins < 0).sum()))
    adjusted = score.copy()
    for game in wins.itertuples(index=False):
        winner, loser = game.team, game.opponent
        if winner not in score.index or loser not in score.index:
            continue
        winner_losses = int(losses.get(winner, 0))
        loser_losses = int(losses.get(loser, 0))
        if winner_losses > loser_losses:
            continue
        if winner_losses == loser_losses:
            winner_quality = float(loss_strength.get(winner, np.nan))
            loser_quality = float(loss_strength.get(loser, np.nan))
            if np.isfinite(winner_quality) and np.isfinite(loser_quality) and winner_quality + LOSS_COMPARE < loser_quality:
                continue
        if float(score[winner]) > float(score[loser]):
            continue
        adjusted[winner] = max(float(adjusted[winner]), float(score[loser]) + HEAD_TO_HEAD_CUSHION)
    return adjusted


def _loss_penalty(opponents: list[float]) -> float:
    """Each loss is priced from who it was against.

    The best loss fills the cheapest slot. The next loss starts higher, then
    the opponent's rating is taken off that slot. A loss never becomes a bonus.
    """
    if not opponents:
        return 0.0
    ordered = sorted((float(rating) if np.isfinite(rating) else 0.0 for rating in opponents), reverse=True)
    total = 0.0
    for index, rating in enumerate(ordered):
        slot = LOSS_COST * (1.0 + LOSS_ACCELERATION * index)
        total += max(0.0, slot - LOSS_QUALITY * rating)
    return total


def _record_score(win_strength: float, opponents: list[float]) -> float:
    """Who you beat, minus what each loss cost."""
    wins = 0.0 if not np.isfinite(win_strength) else float(win_strength)
    return wins - _loss_penalty(opponents)


def resume_board(schedule: pd.DataFrame, season: int, through_week: int) -> pd.DataFrame:
    """Every FBS team, best results first. Score is points versus an average résumé."""
    played = _games(schedule, season, through_week)
    empty = pd.DataFrame(columns=["rank", "team", "record", "score", "conference", "conf", "sos", "win_sos", "loss_sos", "best_win", "games"])
    if played.empty:
        return empty
    teams = pd.Index(sorted(played["team"].unique()))
    conference = played.groupby("team")["conference"].agg(lambda values: values.dropna().iloc[-1] if values.dropna().size else "")
    rating = pd.Series(0.0, index=teams)
    for _ in range(ITERATIONS):
        fcs_score = float(rating.quantile(0.10) - 1.5)
        opponent = played["opponent"].map(rating)
        opponent = opponent.where(played["opponent_fbs"], fcs_score)
        shown = played["value"] + opponent.fillna(fcs_score)
        weighted = shown * played["weight"]
        raw = weighted.groupby(played["team"]).sum() / played["weight"].groupby(played["team"]).sum()
        raw = raw.reindex(teams).fillna(0.0)
        raw = raw - raw.mean()
        table = pd.DataFrame({"rating": raw, "conference": conference.reindex(teams)})
        league = table.groupby("conference")["rating"].transform("mean")
        independent = table["conference"].isin(["FBS Independents", ""])
        rating = ((1.0 - CONFERENCE_PULL) * raw + CONFERENCE_PULL * league).where(~independent, raw)
        rating = rating.reindex(teams).fillna(0.0)
        rating = rating - rating.mean()
    wins = played["margin"] > 0
    losses = played["margin"] < 0
    record = pd.DataFrame(
        {
            "wins": wins.groupby(played["team"]).sum(),
            "losses": losses.groupby(played["team"]).sum(),
        }
    ).reindex(teams).fillna(0).astype(int)
    fbs_played = played[played["opponent_fbs"]].copy()
    fbs_played["opponent_rating"] = fbs_played["opponent"].map(rating)
    sos = fbs_played.groupby("team")["opponent_rating"].mean()
    won_fbs = fbs_played[fbs_played["margin"] > 0]
    lost_fbs = fbs_played[fbs_played["margin"] < 0]
    win_sos = won_fbs.groupby("team")["opponent_rating"].mean()
    loss_sos = lost_fbs.groupby("team")["opponent_rating"].mean()
    no_win = float(rating.quantile(0.10) - 1.5)
    lost = played[played["margin"] < 0].copy()
    lost["opponent_rating"] = lost["opponent"].map(rating)
    lost["opponent_rating"] = lost["opponent_rating"].where(lost["opponent_fbs"], no_win)
    loss_lists = lost.groupby("team")["opponent_rating"].apply(list)
    score = pd.Series(
        {
            team: _record_score(float(win_sos.get(team, no_win)), list(loss_lists.get(team, [])))
            for team in teams
        }
    )
    score = _head_to_head(played, score, loss_sos)
    conf_score = score.groupby(conference.reindex(teams).fillna("").to_numpy()).transform("mean")
    conf_score.index = teams
    won = played[wins & played["opponent_fbs"]].copy()
    won["opp_score"] = won["opponent"].map(score)
    best = won.sort_values("opp_score", ascending=False).groupby("team").first()
    board = pd.DataFrame(index=teams)
    board["score"] = score
    board["wins"] = record["wins"]
    board["losses"] = record["losses"]
    board["conference"] = conference.reindex(teams).fillna("")
    board["conf"] = conf_score
    board["sos"] = sos
    board["win_sos"] = win_sos
    board["loss_sos"] = loss_sos
    board["best_win"] = best["opponent"] if not best.empty else pd.Series(dtype=object)
    board["games"] = played.groupby("team").size()
    board = board[board["games"] >= MIN_GAMES].sort_values(["score", "win_sos"], ascending=[False, False])
    board["record"] = board["wins"].astype(str) + "-" + board["losses"].astype(str)
    board = board.reset_index(names="team")
    board.insert(0, "rank", range(1, len(board) + 1))
    board["score"] = board["score"].round(2)
    board["conf"] = board["conf"].round(2)
    board["sos"] = board["sos"].round(2)
    board["win_sos"] = board["win_sos"].round(2)
    board["loss_sos"] = board["loss_sos"].round(2)
    return board[["rank", "team", "record", "score", "conference", "conf", "sos", "win_sos", "loss_sos", "best_win", "games"]]
