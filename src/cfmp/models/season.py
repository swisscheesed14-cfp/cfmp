"""This year's ratings, with weights learned from older seasons.

2014-2025 is only used to answer two questions: how many points each snap
stat has been worth, and how much a team bounces around its own average.
The 2026 top 25 uses 2026 games only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Offense column, defense column, True when a bigger defensive number helps the defense.
# Stuff, takeaways, sacks, and tackles for loss work that way. Everything else is
# a rate or a yardage where a bigger defensive number means you allowed more.
STATS = (
    ("rush_ppa", "def_rush_ppa", False),
    ("pass_ppa", "def_pass_ppa", False),
    ("line_yards", "def_line", False),
    ("second_yards", "def_second", False),
    ("open_yards", "def_open", False),
    ("stuff_rate", "def_stuff", True),
    ("off_success", "def_success", False),
    ("explosiveness", "def_explosiveness", False),
    ("power_success", "def_power", False),
    ("standard_success", "def_standard_success", False),
    ("passing_success", "def_passing_success", False),
    ("third_down", "def_third", False),
    ("fourth_down", "def_fourth", False),
    ("ypp", "def_ypp", False),
    ("ypr", "def_ypr", False),
    ("ypa", "def_ypa", False),
    ("completion", "def_completion", False),
    ("turnovers", "def_turnovers", True),
    ("sacks_allowed", "def_sacks", True),
    ("tfl_allowed", "def_tfl", True),
)
MIN_PRIOR = 3
# How many games the roster is worth before this year's box score takes over.
# Two is what 2024 and 2025 games supported. More than that made the old seasons worse.
ROSTER_GAMES = 2.0
# Points per 100 of roster talent, and per 1.0 of returning production. Fit on 2015-2023.
TALENT_POINTS = 4.15
RETURNING_POINTS = 7.62
TALENT_PATH = Path("data/cfbd/talent.parquet")
RETURNING_PATH = Path("data/cfbd/returning.parquet")
MIN_CURRENT = 4
SHRINK_GAMES = 3
SIMS = 4000
# A game against a bottom-tier opponent counts this fraction of a game against a top one.
SOS_FLOOR = 0.35
# The result is 30% of the rating. Snaps stay 70%. A win is worth this many
# points even when the snaps were a tie, and the margin adds no more than the cap.
RESULT_WEIGHT = 0.30
WIN_POINTS = 6.0
MARGIN_CAP = 14.0
# A Week 2 game is worth about 70% of a Week 7 game. Older than that keeps fading
# until the floor.
RECENCY_PER_WEEK = 0.06
RECENCY_FLOOR = 0.55
# The rating gap is not fully trusted until the shorter sample is long.
TRUST_PRIOR = 8
# Two teams this far above average do not get a blowout number.
GOOD_TEAM = 7.0
GOOD_CAP = 10.0


@dataclass
class Weights:
    names: tuple[str, ...]
    coef: np.ndarray
    intercept: float
    hfa: float
    residual_sd: float
    train_games: int
    mae: float

    def as_dict(self) -> dict[str, float]:
        out = {name: float(value) for name, value in zip(self.names, self.coef)}
        out["intercept"] = self.intercept
        out["hfa"] = self.hfa
        out["residual_sd"] = self.residual_sd
        return out


def _edge(home_off: float, home_def: float, away_off: float, away_def: float, hurts_offense: bool) -> float:
    """What the home offense does to that defense, minus the reverse.

    A leaky defense is added to the opposing offense, not subtracted.
    Higher defensive numbers are yards or rates allowed, except stuff,
    sacks, and takeaways, where a higher number helps the defense.
    """
    if hurts_offense:
        return (home_def - away_off) - (away_def - home_off)
    return (home_off + away_def) - (away_off + home_def)


def entering_profile(team_games: pd.DataFrame) -> pd.DataFrame:
    """What a team had done earlier in that same season, before this game."""
    frame = team_games.sort_values(["team", "season", "week", "game_id"]).copy()
    columns = [name for pair in STATS for name in pair[:2]]
    grouped = frame.groupby(["team", "season"], sort=False)
    seen = grouped.cumcount()
    for column in columns:
        prior = grouped[column].cumsum() - frame[column]
        frame[f"in_{column}"] = prior / seen.where(seen > 0)
    frame["prior_games"] = seen
    return frame


def _ridge(features: np.ndarray, target: np.ndarray, penalty: float) -> tuple[np.ndarray, float]:
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standard = (features - center) / scale
    gram = standard.T @ standard + penalty * np.eye(standard.shape[1])
    beta = np.linalg.solve(gram, standard.T @ target)
    coef = beta / scale
    intercept = float(target.mean() - np.dot(center, coef))
    return coef, intercept


def game_table(schedule: pd.DataFrame, advanced: pd.DataFrame) -> pd.DataFrame:
    """One row per game, with each team's pregame profile from that season only."""
    columns = ["game_id", "team", "season", "week", *[name for pair in STATS for name in pair[:2]]]
    snaps = advanced[columns].copy()
    for made, allowed in (("third_down", "sacks_allowed"), ("third_down", "def_sacks"), ("third_down", "tfl_allowed"), ("third_down", "def_tfl")):
        if made in snaps.columns and allowed in snaps.columns:
            snaps.loc[snaps[made].notna() & snaps[allowed].isna(), allowed] = 0.0
    for column in ("open_yards", "def_open"):
        if column in snaps.columns:
            snaps[column] = snaps[column].clip(upper=snaps[column].quantile(0.99))
    snaps = entering_profile(snaps)
    info = schedule[
        ["game_id", "home_team", "away_team", "home_points", "away_points", "neutral_site", "home_division", "away_division"]
    ].copy()
    info["game_id"] = pd.to_numeric(info["game_id"], errors="coerce")
    snaps["game_id"] = pd.to_numeric(snaps["game_id"], errors="coerce")
    home = snaps.merge(info, on="game_id", how="inner")
    home = home[home["team"] == home["home_team"]]
    away = snaps.merge(info[["game_id", "away_team"]], on="game_id", how="inner")
    away = away[away["team"] == away["away_team"]]
    keep_away = ["game_id", "prior_games", *[f"in_{name}" for pair in STATS for name in pair[:2]]]
    games = home.merge(away[keep_away], on="game_id", how="inner", suffixes=("_home", "_away"))
    games["margin"] = games["home_points"] - games["away_points"]
    games["fbs"] = (
        games["home_division"].astype(str).str.lower().eq("fbs")
        & games["away_division"].astype(str).str.lower().eq("fbs")
    )
    return games.dropna(subset=["margin"])


def _design(games: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = []
    columns = []
    for off, deff, hurts in STATS:
        home_off = pd.to_numeric(games[f"in_{off}_home"], errors="coerce")
        home_def = pd.to_numeric(games[f"in_{deff}_home"], errors="coerce")
        away_off = pd.to_numeric(games[f"in_{off}_away"], errors="coerce")
        away_def = pd.to_numeric(games[f"in_{deff}_away"], errors="coerce")
        if hurts:
            value = (home_def - away_off) - (away_def - home_off)
        else:
            value = (home_off + away_def) - (away_off + home_def)
        filled = value.fillna(value.median())
        if not np.isfinite(filled.to_numpy(dtype=float)).any():
            filled = pd.Series(0.0, index=value.index)
        columns.append(filled.to_numpy(dtype=float))
        names.append(off)
    neutral = games["neutral_site"].fillna(False).astype(bool).to_numpy()
    columns.append((~neutral).astype(float))
    names.append("hfa")
    return np.column_stack(columns), games["margin"].to_numpy(dtype=float), names


def fit_weights(games: pd.DataFrame, train_through: int = 2025) -> Weights:
    """Learn point values from completed seasons. The current season is not in the fit."""
    usable = games[
        games["fbs"]
        & games["season"].between(2014, train_through)
        & (games["prior_games_home"] >= MIN_PRIOR)
        & (games["prior_games_away"] >= MIN_PRIOR)
    ].dropna(subset=["margin"])
    early = usable[usable["season"] <= train_through - 2]
    late = usable[usable["season"] > train_through - 2]
    features, target, names = _design(usable)
    best_penalty, best_mae = 30.0, None
    if len(early) > 500 and len(late) > 200:
        early_x, early_y, _ = _design(early)
        late_x, late_y, _ = _design(late)
        for penalty in (3.0, 10.0, 30.0, 100.0, 300.0):
            coef, intercept = _ridge(early_x, early_y, penalty)
            mae = float(np.abs(late_y - (intercept + late_x @ coef)).mean())
            if best_mae is None or mae < best_mae:
                best_penalty, best_mae = penalty, mae
    coef, intercept = _ridge(features, target, best_penalty)
    predicted = intercept + features @ coef
    hfa = float(coef[-1])
    return Weights(
        names=tuple(names[:-1]),
        coef=coef[:-1],
        intercept=intercept,
        hfa=hfa,
        residual_sd=float(np.std(target - predicted)),
        train_games=int(len(usable)),
        mae=float(np.abs(target - predicted).mean()),
    )


def recency_weight(week: float, through_week: float) -> float:
    """Newer games count more. Five weeks back is about 70% of this week."""
    age = max(0.0, float(through_week) - float(week))
    return max(RECENCY_FLOOR, 1.0 - RECENCY_PER_WEEK * age)


def result_points(margin: float, site: float = 0.0) -> float:
    """Win or loss first, then the margin, capped.

    `site` is the home-field points already inside the score, subtracted so a
    home win is not bigger just because it was at home. The sign stays with
    the actual result, so a one-point home win is still a win.
    """
    if not np.isfinite(margin) or margin == 0:
        return 0.0
    bonus = WIN_POINTS if margin > 0 else -WIN_POINTS
    return bonus + float(np.clip(margin - site, -MARGIN_CAP, MARGIN_CAP))


def _weighted_mean(values: pd.Series, weights: pd.Series, teams: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"v": values.to_numpy(), "w": weights.to_numpy(), "team": teams.to_numpy()})
    frame = frame[np.isfinite(frame["v"]) & np.isfinite(frame["w"])]
    if frame.empty:
        return pd.Series(dtype=float)
    frame["vw"] = frame["v"] * frame["w"]
    grouped = frame.groupby("team")
    return grouped["vw"].sum() / grouped["w"].sum()


def _score_means(offense: pd.DataFrame, defense: pd.DataFrame, weights: Weights | None) -> pd.Series:
    """Point rating from current offense and defense means. Missing weights start everyone at zero."""
    if weights is None:
        return pd.Series(0.0, index=offense.index)
    league_off = offense.mean()
    league_def = defense.mean()
    rating = np.full(len(offense), weights.intercept)
    for coef, (off, deff, hurts) in zip(weights.coef, STATS):
        if hurts:
            # Home pressure against the away offense, versus the reverse, against an average team.
            team_net = defense[deff].fillna(league_def[deff]) - offense[off].fillna(league_off[off])
            league_net = league_def[deff] - league_off[off]
            rating = rating + coef * (team_net.to_numpy() - league_net)
        else:
            offense_above = offense[off].fillna(league_off[off]) - league_off[off]
            defense_above = league_def[deff] - defense[deff].fillna(league_def[deff])
            rating = rating + coef * (offense_above.to_numpy() + defense_above.to_numpy())
    return pd.Series(rating, index=offense.index)


def _current_means(
    advanced: pd.DataFrame,
    schedule: pd.DataFrame,
    season: int,
    weights: Weights | None = None,
    through_week: int | None = None,
) -> pd.DataFrame:
    """2026 means, opponent-adjusted and weighted by our own schedule strength.

    A game against a weak opponent counts less than a game against a strong one.
    Opponent strength is this model's rating, recomputed until it stops moving.
    It is not an outside poll.
    """
    info = schedule.loc[
        schedule["season"] == season,
        ["game_id", "home_team", "away_team", "home_division", "away_division", "home_points", "away_points", "neutral_site"],
    ]
    snaps = advanced.loc[advanced["season"] == season].merge(info, on="game_id", how="inner")
    snaps = snaps[(snaps["team"] == snaps["home_team"]) | (snaps["team"] == snaps["away_team"])]
    snaps["opponent"] = np.where(snaps["team"] == snaps["home_team"], snaps["away_team"], snaps["home_team"])
    snaps["at_home"] = snaps["team"] == snaps["home_team"]
    snaps["is_fbs"] = np.where(
        snaps["team"] == snaps["home_team"],
        snaps["home_division"].astype(str).str.lower().eq("fbs"),
        snaps["away_division"].astype(str).str.lower().eq("fbs"),
    )
    fbs_names = set(snaps.loc[snaps["is_fbs"], "team"])
    snaps = snaps[snaps["team"].isin(fbs_names) & snaps["opponent"].isin(fbs_names)].copy()
    if through_week is not None and "week" in snaps.columns:
        snaps = snaps[pd.to_numeric(snaps["week"], errors="coerce") <= through_week]
    current = float(snaps["week"].max()) if "week" in snaps.columns and len(snaps) else 1.0
    if through_week is not None:
        current = float(through_week)
    snaps["recency"] = snaps["week"].map(lambda week: recency_weight(week, current))
    hfa = 0.0 if weights is None else float(weights.hfa)
    neutral = snaps["neutral_site"].fillna(False).astype(bool)
    site = np.where(neutral, 0.0, np.where(snaps["at_home"], hfa, -hfa))
    margin = np.where(snaps["at_home"], snaps["home_points"] - snaps["away_points"], snaps["away_points"] - snaps["home_points"])
    snaps["margin"] = pd.to_numeric(margin, errors="coerce")
    snaps["result_raw"] = [
        result_points(float(scored), float(field)) if np.isfinite(scored) else np.nan
        for scored, field in zip(snaps["margin"], site)
    ]
    snaps["result_weight"] = np.where(np.isfinite(snaps["result_raw"]), snaps["recency"], 0.0)
    off_cols = [off for off, _, _ in STATS]
    def_cols = [deff for _, deff, _ in STATS]
    offense = snaps.groupby("team")[off_cols].mean()
    defense = snaps.groupby("team")[def_cols].mean()
    games = snaps.groupby("team").size().rename("games")
    teams = offense.index
    rating = pd.Series(0.0, index=teams)
    for _ in range(6):
        league_off = offense.mean()
        league_def = defense.mean()
        percentile = rating.rank(pct=True).fillna(0.5)
        game_weight = (SOS_FLOOR + (1.0 - SOS_FLOOR) * snaps["opponent"].map(percentile).fillna(0.5)) * snaps["recency"]
        new_off = {}
        for column, (_, def_column, _) in zip(off_cols, STATS):
            opp_def = snaps["opponent"].map(defense[def_column])
            piece = snaps[column] - (opp_def - league_def[def_column])
            new_off[column] = _weighted_mean(piece, game_weight, snaps["team"])
        new_def = {}
        for column, (off_column, _, _) in zip(def_cols, STATS):
            opp_off = snaps["opponent"].map(offense[off_column])
            piece = snaps[column] - (opp_off - league_off[off_column])
            new_def[column] = _weighted_mean(piece, game_weight, snaps["team"])
        offense = pd.DataFrame(new_off).reindex(teams)
        defense = pd.DataFrame(new_def).reindex(teams)
        snap_rating = _score_means(offense, defense, weights).reindex(teams).fillna(0.0)
        demonstrated = snaps["result_raw"] + snaps["opponent"].map(rating).fillna(0.0)
        result_rating = _weighted_mean(demonstrated, snaps["result_weight"], snaps["team"]).reindex(teams).fillna(0.0)
        rating = (1.0 - RESULT_WEIGHT) * snap_rating + RESULT_WEIGHT * result_rating
    league_off = offense.mean()
    league_def = defense.mean()
    shrink = games / (games + SHRINK_GAMES)
    for column in off_cols:
        offense[column] = shrink * offense[column] + (1.0 - shrink) * league_off[column]
    for column in def_cols:
        defense[column] = shrink * defense[column] + (1.0 - shrink) * league_def[column]
    snap_rating = _score_means(offense, defense, weights).reindex(teams).fillna(0.0)
    result_rating = (shrink * result_rating).reindex(teams).fillna(0.0)
    rating = (1.0 - RESULT_WEIGHT) * snap_rating + RESULT_WEIGHT * result_rating
    sos = snaps.groupby("team")["opponent"].apply(lambda opponents: float(rating.reindex(opponents).mean()))
    profile = pd.concat([games, offense, defense], axis=1)
    profile["result_rating"] = result_rating
    profile["sos"] = sos
    profile["fbs"] = True
    profile["season"] = season
    return profile


def blend_rating(box: float, roster: float, games: float) -> float:
    """This year's results, pulled toward the roster while the sample is short."""
    weight = float(games) / (float(games) + ROSTER_GAMES)
    return weight * float(box) + (1.0 - weight) * float(roster)


def _roster_points(teams: pd.Index, season: int) -> pd.Series:
    """Talent plus returning production, in points versus an average roster."""
    points = pd.Series(0.0, index=teams)
    if TALENT_PATH.exists():
        talent = pd.read_parquet(TALENT_PATH)
        talent = talent[talent["year"].astype(int) == season].drop_duplicates("team").set_index("team")["talent"]
        known = pd.to_numeric(talent.reindex(teams), errors="coerce").dropna()
        if len(known) > 10:
            points = points.add((known - known.mean()) / 100.0 * TALENT_POINTS, fill_value=0.0)
    if RETURNING_PATH.exists():
        returning = pd.read_parquet(RETURNING_PATH)
        returning = returning[returning["season"].astype(int) == season].drop_duplicates("team").set_index("team")["percentPPA"]
        known = pd.to_numeric(returning.reindex(teams), errors="coerce").dropna()
        if len(known) > 10:
            points = points.add((known - known.mean()) * RETURNING_POINTS, fill_value=0.0)
    return points.reindex(teams).fillna(0.0)


def rate_season(profile: pd.DataFrame, weights: Weights, min_games: int = MIN_CURRENT) -> pd.DataFrame:
    """Neutral-field margin versus an average team from this season."""
    board = profile[profile["games"] >= min_games].copy()
    league_off = profile[[off for off, _, _ in STATS]].mean()
    league_def = profile[[deff for _, deff, _ in STATS]].mean()
    rating = np.full(len(board), weights.intercept)
    for coef, (off, deff, hurts) in zip(weights.coef, STATS):
        if hurts:
            team_net = board[deff] - board[off]
            league_net = league_def[deff] - league_off[off]
            rating = rating + coef * (team_net - league_net)
        else:
            offense_above = board[off] - league_off[off]
            defense_above = league_def[deff] - board[deff]
            rating = rating + coef * (offense_above + defense_above)
    if "result_rating" in board.columns:
        result = pd.to_numeric(board["result_rating"], errors="coerce").fillna(pd.Series(rating, index=board.index))
        rating = (1.0 - RESULT_WEIGHT) * rating + RESULT_WEIGHT * result.to_numpy()
    board["rating"] = rating
    if "season" in board.columns and len(board):
        roster = _roster_points(board.index, int(board["season"].iloc[0]))
        board["rating"] = [
            blend_rating(box, prior, games)
            for box, prior, games in zip(board["rating"], roster.to_numpy(), board["games"].to_numpy())
        ]
    board = board.sort_values("rating", ascending=False)
    return board


def fpi_board(profile: pd.DataFrame, weights: Weights, min_games: int = 1) -> pd.DataFrame:
    """Every team, best FPI first. FPI is the neutral-field margin versus an average team."""
    board = rate_season(profile, weights, min_games=min_games).reset_index()
    if "team" not in board.columns:
        board = board.rename(columns={"index": "team"})
    board = board.rename(columns={"rating": "fpi"})
    board.insert(0, "rank", range(1, len(board) + 1))
    columns = ["rank", "team", "fpi", "games"]
    if "sos" in board.columns:
        columns.insert(3, "sos")
    return board[columns]


def sos_board(profile: pd.DataFrame, weights: Weights, min_games: int = 1) -> pd.DataFrame:
    """Every team, hardest schedule first. Rank 1 played the strongest opponents."""
    board = fpi_board(profile, weights, min_games=min_games)
    board = board.sort_values(["sos", "fpi"], ascending=[False, False]).reset_index(drop=True)
    board["rank"] = range(1, len(board) + 1)
    return board[["rank", "team", "sos", "fpi", "games"]]


def top25(profile: pd.DataFrame, weights: Weights, n: int = 25) -> pd.DataFrame:
    """FPI Top 25. The first 25 rows of the full FPI list."""
    return fpi_board(profile, weights).head(n)


def ready_week(advanced: pd.DataFrame, schedule: pd.DataFrame, season: int, coverage: float = 0.8) -> int | None:
    """Latest week that is mostly played and mostly in the snap file.

    Five Thursday games do not make a week. The report waits until the slate
    is in, which is usually the Sunday or Monday after Saturday.
    """
    info = schedule.loc[schedule["season"] == season, ["game_id", "week", "home_points", "home_division", "away_division"]].copy()
    fbs = info["home_division"].astype(str).str.lower().eq("fbs") & info["away_division"].astype(str).str.lower().eq("fbs")
    played = info[fbs].copy()
    if played.empty:
        return None
    played["played"] = played["home_points"].notna()
    snapped = set(pd.to_numeric(advanced.loc[advanced["season"] == season, "game_id"], errors="coerce"))
    played["in_snaps"] = pd.to_numeric(played["game_id"], errors="coerce").isin(snapped)
    by_week = played.groupby("week").agg(scheduled=("game_id", "size"), played=("played", "sum"), snapped=("in_snaps", "sum"))
    complete = (by_week["played"] / by_week["scheduled"] >= coverage) & (by_week["snapped"] / by_week["played"].clip(lower=1) >= coverage)
    ready = by_week[complete]
    if ready.empty:
        return None
    return int(ready.index.max())


def next_slate(schedule: pd.DataFrame, season: int, through_week: int) -> pd.DataFrame:
    """Unplayed FBS games in the first week after the last finished week."""
    info = schedule.loc[schedule["season"] == season].copy()
    fbs = info["home_division"].astype(str).str.lower().eq("fbs") & info["away_division"].astype(str).str.lower().eq("fbs")
    left = info[fbs & info["home_points"].isna() & (pd.to_numeric(info["week"], errors="coerce") > through_week)]
    if left.empty:
        return left
    week = int(left["week"].min())
    return left[left["week"] == week].copy()


def sample_trust(home_games: float, away_games: float) -> float:
    """Share of the rating gap to keep. The shorter sample sets it."""
    games = min(float(home_games), float(away_games))
    if games <= 0:
        return 0.0
    return games / (games + TRUST_PRIOR)


def cap_if_both_good(spread: float, home_fpi: float, away_fpi: float) -> float:
    """A good team against a good team stops at 10 points."""
    if home_fpi > GOOD_TEAM and away_fpi > GOOD_TEAM:
        return max(-GOOD_CAP, min(GOOD_CAP, spread))
    return spread


def price_slate(slate: pd.DataFrame, profile: pd.DataFrame, weights: Weights) -> pd.DataFrame:
    """Home spread for each game. Positive means the home team is favored.

    The gap is the full matchup. Home field is added on top. Two good teams
    are capped later, after the crowd and the weather.
    """
    ratings = rate_season(profile, weights, min_games=1)
    rows = []
    for game in slate.itertuples(index=False):
        home_name = game.home_team
        away_name = game.away_team
        if home_name not in profile.index or away_name not in profile.index:
            continue
        if home_name not in ratings.index or away_name not in ratings.index:
            continue
        neutral = bool(getattr(game, "neutral_site", False))
        home_fpi = float(ratings.at[home_name, "rating"])
        away_fpi = float(ratings.at[away_name, "rating"])
        hfa = 0.0 if neutral else weights.hfa
        rows.append(
            {
                "game_id": game.game_id,
                "week": int(game.week),
                "away": away_name,
                "home": home_name,
                "neutral": neutral,
                "home_fpi": home_fpi,
                "away_fpi": away_fpi,
                "model": hfa + (home_fpi - away_fpi),
            }
        )
    return pd.DataFrame(rows)


def _team_edge(home: pd.Series, away: pd.Series, weights: Weights, hfa: float) -> float:
    total = weights.intercept + hfa
    for coef, (off, deff, hurts) in zip(weights.coef, STATS):
        total += coef * _edge(float(home[off]), float(home[deff]), float(away[off]), float(away[deff]), hurts)
    return float(total)


def simulate_game(
    home: pd.Series,
    away: pd.Series,
    weights: Weights,
    neutral: bool = False,
    n: int = SIMS,
    seed: int = 7,
) -> dict[str, float]:
    """Sample the margin around the weighted snaps. The mean is the spread."""
    hfa = 0.0 if neutral else weights.hfa
    center = _team_edge(home, away, weights, hfa)
    draws = np.random.default_rng(seed).normal(center, weights.residual_sd, size=n)
    return {
        "spread": float(draws.mean()),
        "win_prob_home": float((draws > 0).mean()),
        "p10": float(np.percentile(draws, 10)),
        "p90": float(np.percentile(draws, 90)),
    }


def price_played(games: pd.DataFrame, weights: Weights) -> pd.DataFrame:
    """Price each 2026 game from the profiles entering that week, not from the result."""
    rows = []
    usable = games[
        games["fbs"]
        & (games["season"] == 2026)
        & (games["prior_games_home"] >= MIN_PRIOR)
        & (games["prior_games_away"] >= MIN_PRIOR)
    ]
    for game in usable.itertuples(index=False):
        # The entering columns are already on the row. Build two tiny profiles.
        home = {}
        away = {}
        for off, deff, _ in STATS:
            home[off] = getattr(game, f"in_{off}_home")
            home[deff] = getattr(game, f"in_{deff}_home")
            away[off] = getattr(game, f"in_{off}_away")
            away[deff] = getattr(game, f"in_{deff}_away")
        if any(value != value for value in (*home.values(), *away.values())):
            continue
        priced = simulate_game(pd.Series(home), pd.Series(away), weights, neutral=bool(game.neutral_site), n=800, seed=int(game.game_id) % 10000)
        rows.append(
            {
                "game_id": game.game_id,
                "week": game.week,
                "home": game.home_team,
                "away": game.away_team,
                "model": priced["spread"],
                "actual": game.margin,
                "neutral_site": bool(game.neutral_site),
            }
        )
    return pd.DataFrame(rows)
