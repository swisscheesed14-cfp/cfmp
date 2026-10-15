"""Neutral-field power ratings, then a site spread.

Each team has an offense and a defense, both in points versus average.
The neutral spread is the difference of those ratings. That is the Pate-style
number: who would be favored on a neutral field, and by how many points.

The site spread adds home field, travel, rest, quarterback continuity, and
weather. The closing line is never an input. Walk-forward scoring predicts
each game before that game is allowed to move the rating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

from cfmp.models.params import PowerParams, load_params
from cfmp.models.priors import Priors, keep_from_returning
from cfmp.models.site import qb_adjustment, rest_adjustment, timezone_index, travel_adjustment, haversine_miles
from cfmp.models.weather import is_indoor, weather_severity, weather_site_adjustment


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _cell(row: Any, name: str) -> Any:
    try:
        if name not in row.index:
            return None
    except AttributeError:
        return row.get(name)
    value = row[name]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


@dataclass
class Team:
    off: float = 0.0
    def_: float = 0.0
    lat: float | None = None
    lon: float | None = None
    nloc: int = 0
    qb: str | None = None
    since: int | None = None
    pass_rate: float | None = None
    last_start: pd.Timestamp | None = None
    classification: str | None = None
    rush_off: float = 0.0
    pass_off: float = 0.0
    rush_def: float = 0.0
    pass_def: float = 0.0
    snap_games: int = 0

    @property
    def rating(self) -> float:
        return self.off + self.def_


@dataclass
class Book:
    teams: dict[str, Team] = field(default_factory=dict)
    season: int | None = None
    baselines: dict[str, float] = field(default_factory=dict)
    baseline_seen: dict[str, int] = field(default_factory=dict)

    def team(self, name: str) -> Team:
        if name not in self.teams:
            self.teams[name] = Team()
        return self.teams[name]

    def baseline(self, classification: str | None) -> float:
        key = classification or "fbs"
        if key == "fbs":
            return 0.0
        return float(self.baselines.get(key, 0.0))

    def nudge_baseline(self, classification: str, residual: float, prior_games: float) -> None:
        """Move a non-FBS division. Positive residual means that division was better than expected."""
        if classification == "fbs":
            return
        seen = self.baseline_seen.get(classification, 0) + 1
        self.baseline_seen[classification] = seen
        weight = 1.0 / (prior_games + seen)
        updated = self.baseline(classification) + weight * residual
        self.baselines[classification] = min(0.0, max(-60.0, updated))

    def regress(self, keep: float, season: int | None = None, priors: Priors | None = None) -> None:
        """Shrink last year toward the preseason prior for the season being entered."""
        for name, team in self.teams.items():
            base = self.baseline(team.classification or "fbs")
            target = base if priors is None or season is None else priors.talent_points(season, name)
            if target is None:
                target = base
            side = (target - base) / 2.0
            off_pct, def_pct = (None, None) if priors is None or season is None else priors.returning_split(season, name)
            off_keep = keep if off_pct is None else keep_from_returning(off_pct)
            def_keep = keep if def_pct is None else keep_from_returning(def_pct)
            team.off = off_keep * team.off + (1.0 - off_keep) * side
            team.def_ = def_keep * team.def_ + (1.0 - def_keep) * side
            team.rush_off = off_keep * team.rush_off + (1.0 - off_keep) * side
            team.pass_off = off_keep * team.pass_off + (1.0 - off_keep) * side
            team.rush_def = def_keep * team.rush_def + (1.0 - def_keep) * side
            team.pass_def = def_keep * team.pass_def + (1.0 - def_keep) * side


def new_book(params: PowerParams) -> Book:
    return Book(
        baselines={
            "fbs": 0.0,
            "fcs": params.fcs_baseline,
            "ii": params.ii_baseline,
            "iii": params.iii_baseline,
        }
    )


def _days_between(later: Any, earlier: Any) -> float | None:
    if later is None or earlier is None:
        return None
    end = pd.Timestamp(later)
    start = pd.Timestamp(earlier)
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    return (end - start).total_seconds() / 86400.0


def _miles(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> float | None:
    a, b, c, d = _num(lat1), _num(lon1), _num(lat2), _num(lon2)
    if None in (a, b, c, d):
        return None
    return haversine_miles(a, b, c, d)


def _zones(origin_lon: Any, venue_lon: Any) -> int | None:
    origin = timezone_index(_num(origin_lon))
    venue = timezone_index(_num(venue_lon))
    if origin is None or venue is None:
        return None
    return venue - origin


def project_game(book: Book, row: Any, params: PowerParams) -> dict[str, Any]:
    """Price a game from the rating as it stands. Does not update anything."""
    home = book.team(str(_cell(row, "home")))
    away = book.team(str(_cell(row, "away")))
    neutral = _truthy(_cell(row, "neutral_site"))
    indoor = is_indoor(_cell(row, "venue"), _cell(row, "indoor") or _cell(row, "dome"))
    hfa = 0.0 if neutral else params.hfa
    home_abs = home.rating + book.baseline(home.classification or "fbs")
    away_abs = away.rating + book.baseline(away.classification or "fbs")
    fit = 0.0
    if home.snap_games and away.snap_games:
        fit = _fit_points(home, away) - _fit_points(away, home)
    neutral_spread = home_abs - away_abs + fit

    home_miles = away_miles = None
    home_zones = away_zones = None
    venue_lat, venue_lon = _cell(row, "venue_lat"), _cell(row, "venue_lon")
    if neutral:
        home_miles = _miles(home.lat, home.lon, venue_lat, venue_lon)
        away_miles = _miles(away.lat, away.lon, venue_lat, venue_lon)
        home_zones = _zones(home.lon, venue_lon)
        away_zones = _zones(away.lon, venue_lon)
    else:
        away_miles = _miles(away.lat, away.lon, venue_lat, venue_lon)
        away_zones = _zones(away.lon, venue_lon)
    travel = travel_adjustment(home_miles, away_miles, home_zones, away_zones, params)

    kick = _cell(row, "start_date") or _cell(row, "kickoff")
    rest = rest_adjustment(
        _days_between(kick, home.last_start),
        _days_between(kick, away.last_start),
        params,
    )
    qb = qb_adjustment(home.since, away.since, params)

    confidence = _num(_cell(row, "forecast_conf"))
    severity = 0.0 if indoor else weather_severity(
        temp_f=_cell(row, "wx_temp_f"),
        wind_mph=_cell(row, "wx_wind_mph"),
        pop=_cell(row, "wx_pop"),
        forecast=_cell(row, "wx_short"),
    )
    if confidence is not None:
        severity *= max(0.0, min(1.0, confidence))
    pre_weather = neutral_spread + hfa + travel + rest + qb
    weather = weather_site_adjustment(
        severity,
        pre_weather,
        None if indoor else home.pass_rate,
        None if indoor else away.pass_rate,
        params.weather_points,
    )
    site = pre_weather + weather
    # 20 rating points was the old Elo-per-point scale. Ratings are already points,
    # so one site-spread point is one point on this curve.
    win_prob = 1.0 / (1.0 + 10 ** (-site / 20.0))
    return {
        "rating_home": home_abs,
        "rating_away": away_abs,
        "off_home": home.off,
        "def_home": home.def_,
        "off_away": away.off,
        "def_away": away.def_,
        "neutral_spread": neutral_spread,
        "hfa_points": hfa,
        "travel_adj": travel,
        "rest_adj": rest,
        "qb_adj": qb,
        "weather_severity": severity,
        "weather_adj": weather,
        "site_spread": site,
        "win_prob_home": win_prob,
        "pass_rate_home": home.pass_rate,
        "pass_rate_away": away.pass_rate,
    }


# Snap stats are turned into points before they enter the rating. Each raw
# number is scaled so one standard deviation is about one standard deviation
# of total PPA (15.4 points), then the pieces are averaged. The score is not
# a piece. Means and spreads are from the 2014-2026 game feed.
_OFF_MEAN = 7.23
_RUSH_MEAN = 2.79
_PASS_MEAN = 5.63
_LINE_MEAN = 2.885
_SECOND_MEAN = 1.078
_OPEN_MEAN = 1.297
_STUFF_MEAN = 0.194
_LINE_SCALE = 15.4 / 0.756
_SECOND_SCALE = 15.4 / 0.388
_OPEN_SCALE = 15.4 / 1.157
_STUFF_SCALE = 15.4 / 0.089


def _pieces(row: Any, side: str, names: tuple[str, ...]) -> list[float]:
    values = []
    for name in names:
        number = _num(_cell(row, f"{side}_{name}"))
        if number is None:
            continue
        if name == "off_ppa":
            values.append(number - _OFF_MEAN)
        elif name == "rush_ppa":
            values.append(number - _RUSH_MEAN)
        elif name == "pass_ppa":
            values.append(number - _PASS_MEAN)
        elif name == "line_yards":
            values.append((number - _LINE_MEAN) * _LINE_SCALE)
        elif name == "second_yards":
            values.append((number - _SECOND_MEAN) * _SECOND_SCALE)
        elif name == "open_yards":
            values.append((number - _OPEN_MEAN) * _OPEN_SCALE)
        elif name == "stuff_rate":
            values.append((_STUFF_MEAN - number) * _STUFF_SCALE)
    return values


def _snap_points(row: Any, side: str) -> float | None:
    """Points the offense created, from the snaps. None when the feed is missing."""
    values = _pieces(
        row,
        side,
        ("off_ppa", "rush_ppa", "pass_ppa", "line_yards", "second_yards", "open_yards", "stuff_rate"),
    )
    if not values or _num(_cell(row, f"{side}_off_ppa")) is None:
        return None
    return sum(values) / len(values)


def _phase_points(row: Any, side: str, phase: str) -> float | None:
    if phase == "rush":
        values = _pieces(row, side, ("rush_ppa", "line_yards", "second_yards", "open_yards", "stuff_rate"))
    else:
        values = _pieces(row, side, ("pass_ppa",))
    if not values:
        return None
    return sum(values) / len(values)


def _fit_points(offense: Team, defense: Team) -> float:
    """How much this offense gains because its run/pass mix hits this defense."""
    rate = 0.5 if offense.pass_rate is None else offense.pass_rate
    rush_rate = 1.0 - rate
    offense_extra = rush_rate * (offense.rush_off - offense.off) + rate * (offense.pass_off - offense.off)
    defense_extra = rush_rate * (defense.rush_def - defense.def_) + rate * (defense.pass_def - defense.def_)
    return offense_extra - defense_extra


def _update(book: Book, row: Any, params: PowerParams) -> None:
    home_pts = _num(_cell(row, "home_points"))
    away_pts = _num(_cell(row, "away_points"))
    if home_pts is None or away_pts is None:
        return
    home = book.team(str(_cell(row, "home")))
    away = book.team(str(_cell(row, "away")))
    _assign_division(home, _cell(row, "home_classification"), book)
    _assign_division(away, _cell(row, "away_classification"), book)
    neutral = _truthy(_cell(row, "neutral_site"))
    hfa = 0.0 if neutral else params.hfa
    div_h = book.baseline(home.classification or "fbs")
    div_a = book.baseline(away.classification or "fbs")
    base = params.points_per_team
    home_snaps = _snap_points(row, "home")
    away_snaps = _snap_points(row, "away")
    home_obs = home_pts if home_snaps is None else base + home_snaps
    away_obs = away_pts if away_snaps is None else base + away_snaps
    exp_home = base + home.off - away.def_ + hfa / 2.0 + div_h / 2.0 - div_a / 2.0
    exp_away = base + away.off - home.def_ - hfa / 2.0 + div_a / 2.0 - div_h / 2.0
    err_home = home_obs - exp_home
    err_away = away_obs - exp_away
    margin = home_obs - away_obs
    expected_margin = (home.rating + div_h) - (away.rating + div_a) + hfa
    home_div = home.classification or "fbs"
    away_div = away.classification or "fbs"
    if home_div != away_div:
        resid = margin - expected_margin
        movers = [division for division in (home_div, away_div) if division != "fbs"]
        share = 1.0 / len(movers) if movers else 0.0
        if home_div != "fbs":
            book.nudge_baseline(home_div, resid * share, params.division_prior_games)
        if away_div != "fbs":
            book.nudge_baseline(away_div, -resid * share, params.division_prior_games)
    if abs(margin) > params.mov_cap and margin != 0:
        scale = params.mov_cap / abs(margin)
        err_home *= scale
        err_away *= scale
    step = params.learning_rate
    home.off += step * err_home
    away.def_ -= step * err_home
    away.off += step * err_away
    home.def_ -= step * err_away
    _update_phases(home, away, row, step, params.mov_cap)

    if not neutral:
        lat, lon = _num(_cell(row, "venue_lat")), _num(_cell(row, "venue_lon"))
        if lat is not None and lon is not None:
            home.nloc += 1
            home.lat = lat if home.lat is None else home.lat + (lat - home.lat) / home.nloc
            home.lon = lon if home.lon is None else home.lon + (lon - home.lon) / home.nloc

    kick = _cell(row, "start_date")
    if kick is not None:
        stamp = pd.Timestamp(kick)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        home.last_start = stamp
        away.last_start = stamp

    _note_qb(home, _cell(row, "home_qb"), _num(_cell(row, "home_pass_rate")))
    _note_qb(away, _cell(row, "away_qb"), _num(_cell(row, "away_pass_rate")))


def _update_phases(home: Team, away: Team, row: Any, step: float, cap: float) -> None:
    used = False
    for phase, off_name, def_name in (("rush", "rush_off", "rush_def"), ("pass", "pass_off", "pass_def")):
        home_obs = _phase_points(row, "home", phase)
        away_obs = _phase_points(row, "away", phase)
        if home_obs is None or away_obs is None:
            continue
        used = True
        for offense, defense, observed in ((home, away, home_obs), (away, home, away_obs)):
            expected = getattr(offense, off_name) - getattr(defense, def_name)
            error = observed - expected
            if abs(error) > cap:
                error = cap if error > 0 else -cap
            setattr(offense, off_name, getattr(offense, off_name) + step * error)
            setattr(defense, def_name, getattr(defense, def_name) - step * error)
    if not used:
        return
    home.snap_games += 1
    away.snap_games += 1
    for team, side in ((home, "home"), (away, "away")):
        rushes = _num(_cell(row, f"{side}_rush_plays"))
        passes = _num(_cell(row, f"{side}_pass_plays"))
        if not rushes or not passes:
            continue
        rate = passes / (rushes + passes)
        team.pass_rate = rate if team.pass_rate is None else 0.75 * team.pass_rate + 0.25 * rate


def _note_qb(team: Team, qb: Any, pass_rate: float | None) -> None:
    if pass_rate is not None:
        team.pass_rate = pass_rate if team.pass_rate is None else 0.75 * team.pass_rate + 0.25 * pass_rate
    if qb is None or (isinstance(qb, float) and math.isnan(qb)):
        return
    name = str(qb)
    if name.lower() in {"", "nan", "none"}:
        return
    if team.qb is None:
        team.qb = name
        team.since = None
        return
    if name != team.qb:
        team.qb = name
        team.since = 0
        return
    team.since = 99 if team.since is None else team.since + 1


def walk_forward(
    games: pd.DataFrame,
    params: PowerParams | None = None,
    priors: Priors | None = None,
) -> pd.DataFrame:
    """Predict every game from prior games only, then learn from the result."""
    params = params or load_params()
    book = new_book(params)
    rows: list[dict[str, Any]] = []
    ordered = games.sort_values(["start_date", "season", "week", "id"], kind="stable")
    for row in ordered.itertuples(index=False):
        record = row._asdict() if hasattr(row, "_asdict") else row._asdict()
        # itertuples renamed def and other fields; use a Series-like via _asdict
        view = pd.Series(record)
        season = _num(view.get("season"))
        season_i = int(season) if season is not None else None
        if season_i is not None and book.season is not None and season_i != book.season:
            book.regress(params.season_regress, season_i, priors)
        if season_i is not None:
            book.season = season_i
        prepare_divisions(book, view)
        projected = project_game(book, view, params)
        actual = None
        home_pts, away_pts = _num(view.get("home_points")), _num(view.get("away_points"))
        if home_pts is not None and away_pts is not None:
            actual = home_pts - away_pts
        vegas = _num(view.get("vegas_spread"))
        opened = _num(view.get("vegas_open"))
        edge = None if vegas is None else projected["site_spread"] - vegas
        closer = None
        ats = None
        clv = None
        if actual is not None and vegas is not None:
            closer = abs(vegas - actual) - abs(projected["site_spread"] - actual)
            if edge and actual != vegas:
                took_home = edge > 0
                covered = actual > vegas if took_home else actual < vegas
                ats = 1.0 if covered else 0.0
        if opened is not None and vegas is not None:
            side = projected["site_spread"] - opened
            if side:
                took_home = side > 0
                clv = (vegas - opened) if took_home else (opened - vegas)
        rows.append(
            {
                "id": view.get("id"),
                "season": view.get("season"),
                "week": view.get("week"),
                "home": view.get("home"),
                "away": view.get("away"),
                "neutral_spread": projected["neutral_spread"],
                "site_spread": projected["site_spread"],
                "hfa_points": projected["hfa_points"],
                "weather_adj": projected["weather_adj"],
                "rest_adj": projected["rest_adj"],
                "travel_adj": projected["travel_adj"],
                "qb_adj": projected["qb_adj"],
                "actual_margin": actual,
                "vegas_spread": vegas,
                "vegas_open": opened,
                "edge": edge,
                "model_closer": closer,
                "ats": ats,
                "clv": clv,
            }
        )
        _update(book, view, params)
    return pd.DataFrame(rows)


def final_ratings(
    games: pd.DataFrame,
    params: PowerParams | None = None,
    priors: Priors | None = None,
) -> pd.DataFrame:
    """Ratings after every completed game. This is what next week's slate uses."""
    params = params or load_params()
    book = new_book(params)
    ordered = games.sort_values(["start_date", "season", "week", "id"], kind="stable")
    for row in ordered.itertuples(index=False):
        view = pd.Series(row._asdict())
        season = _num(view.get("season"))
        season_i = int(season) if season is not None else None
        if season_i is not None and book.season is not None and season_i != book.season:
            book.regress(params.season_regress, season_i, priors)
        if season_i is not None:
            book.season = season_i
        _update(book, view, params)
    rows = []
    for name, team in book.teams.items():
        base_line = book.baseline(team.classification or "fbs")
        rows.append(
            {
                "team": name,
                "off": team.off,
                "defense": team.def_,
                "edge": team.rating,
                "division_baseline": base_line,
                "rating": team.rating + base_line,
                "rush_off": team.rush_off,
                "pass_off": team.pass_off,
                "rush_def": team.rush_def,
                "pass_def": team.pass_def,
                "home_lat": team.lat,
                "home_lon": team.lon,
                "qb": team.qb,
                "games_since_qb_change": team.since,
                "pass_rate": team.pass_rate,
                "classification": team.classification,
                "last_start": None if team.last_start is None else team.last_start.isoformat(),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("rating", ascending=False, ignore_index=True)


def _division(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    aliases = {
        "fbs": "fbs",
        "fcs": "fcs",
        "ii": "ii",
        "d2": "ii",
        "division ii": "ii",
        "iii": "iii",
        "d3": "iii",
        "ii/iii": "iii",
        "division iii": "iii",
    }
    return aliases.get(text)


def _assign_division(team: Team, raw: Any, book: Book) -> None:
    """Rating is edge over the team's own division. Moving up rebases that edge.

    A team that was +12 against FCS, with FCS at -20, is -8 against FBS.
    The absolute number does not change. The label does.
    """
    new = _division(raw)
    if new is None:
        return
    if team.classification is None:
        team.classification = new
        return
    if new == team.classification:
        return
    shift = (book.baseline(team.classification) - book.baseline(new)) / 2.0
    team.off += shift
    team.def_ += shift
    team.classification = new


def prepare_divisions(book: Book, row: Any) -> None:
    home = _cell(row, "home")
    away = _cell(row, "away")
    if home is not None:
        _assign_division(book.team(str(home)), _cell(row, "home_classification"), book)
    if away is not None:
        _assign_division(book.team(str(away)), _cell(row, "away_classification"), book)


def top25(ratings: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """Neutral-field top 25. FBS only when division is known.

    Rank is the power rating, not wins. A 9-0 team can sit behind a 7-2 team.
    """
    df = ratings.copy()
    if "classification" in df.columns and df["classification"].notna().any():
        fbs = df[df["classification"].astype(str).str.lower() == "fbs"]
        if not fbs.empty:
            df = fbs
    df = df.sort_values(["rating", "team"], ascending=[False, True]).head(n).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    keep = ["rank", "team", "rating", "off", "defense"]
    out = df[[column for column in keep if column in df.columns]].copy()
    for column in ("rating", "off", "defense"):
        if column in out.columns:
            out[column] = out[column].astype(float).round(2)
    return out


def ratings_book(ratings: pd.DataFrame) -> Book:
    book = Book(baselines={"fbs": 0.0})
    if "classification" in ratings.columns and "division_baseline" in ratings.columns:
        for classification, part in ratings.groupby(ratings["classification"].fillna("fbs")):
            key = _division(classification) or "fbs"
            values = pd.to_numeric(part["division_baseline"], errors="coerce").dropna()
            if key != "fbs" and not values.empty:
                book.baselines[key] = float(values.iloc[-1])
    for _, row in ratings.iterrows():
        team = book.team(str(row["team"]))
        team.off = float(row["off"])
        team.def_ = float(row["defense"])
        team.classification = _division(row.get("classification"))
        team.lat = _num(row.get("home_lat"))
        team.lon = _num(row.get("home_lon"))
        qb = row.get("qb")
        team.qb = None if qb is None or (isinstance(qb, float) and math.isnan(qb)) else str(qb)
        since = row.get("games_since_qb_change")
        team.since = None if since is None or (isinstance(since, float) and math.isnan(since)) else int(since)
        team.pass_rate = _num(row.get("pass_rate"))
        last = row.get("last_start")
        if last is not None and not (isinstance(last, float) and math.isnan(last)):
            stamp = pd.Timestamp(last)
            team.last_start = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp
    return book


def predict_slate(slate: pd.DataFrame, ratings: pd.DataFrame, params: PowerParams | None = None) -> pd.DataFrame:
    params = params or load_params()
    book = ratings_book(ratings)
    # Rest uses the last completed game stored on the rating, not the slate.
    projected = [project_game(book, row, params) for _, row in slate.iterrows()]
    out = slate.copy()
    extra = pd.DataFrame(projected)
    for column in extra.columns:
        out[column] = extra[column].to_numpy()
    if "vegas_spread" in out.columns:
        out["edge"] = out["site_spread"] - pd.to_numeric(out["vegas_spread"], errors="coerce")
    return out


def summarize(graded: pd.DataFrame) -> dict[str, Any]:
    played = graded.dropna(subset=["actual_margin"])
    lined = played.dropna(subset=["vegas_spread"])
    decided = lined.dropna(subset=["ats"])
    summary: dict[str, Any] = {
        "games": int(len(played)),
        "lined_games": int(len(lined)),
        "mae_model": None if played.empty else float((played["site_spread"] - played["actual_margin"]).abs().mean()),
        "mae_vegas": None if lined.empty else float((lined["vegas_spread"] - lined["actual_margin"]).abs().mean()),
        "mean_closer": None if lined.empty else float(lined["model_closer"].mean()),
        "ats": None if decided.empty else float(decided["ats"].mean()),
        "ats_n": int(len(decided)),
        "mean_clv": None if lined["clv"].dropna().empty else float(lined["clv"].dropna().mean()),
    }
    return summary


def _objective(summary: dict[str, Any]) -> float:
    if summary["mean_closer"] is not None:
        return float(summary["mean_closer"])
    if summary["mae_model"] is None:
        return -1e9
    return -float(summary["mae_model"])


def select_params(
    games: pd.DataFrame,
    fit_seasons: list[int],
    holdout_seasons: list[int],
) -> tuple[PowerParams, dict[str, Any]]:
    """Grid the rating knobs. Keep a winner only when the holdout also improves.

    The market number is not a feature in any of these fits.
    """
    base = load_params()
    best = base
    best_score = -1e9
    tried = 0
    base_graded = walk_forward(games, base)
    best_graded = base_graded
    base_hold = summarize(base_graded[base_graded["season"].isin(holdout_seasons)])
    for rate in (0.05, 0.08, 0.12):
        for cap in (21.0, 28.0, 35.0):
            for hfa in (2.0, 2.75, 3.5):
                for keep in (0.55, 0.70):
                    tried += 1
                    candidate = replace(
                        base,
                        learning_rate=rate,
                        mov_cap=cap,
                        hfa=hfa,
                        season_regress=keep,
                    )
                    graded = walk_forward(games, candidate)
                    fit_score = _objective(summarize(graded[graded["season"].isin(fit_seasons)]))
                    if fit_score > best_score:
                        best_score = fit_score
                        best = candidate
                        best_graded = graded
    hold = summarize(best_graded[best_graded["season"].isin(holdout_seasons)])
    improved = True
    if base_hold.get("mean_closer") is not None and hold.get("mean_closer") is not None:
        improved = hold["mean_closer"] >= base_hold["mean_closer"]
    elif base_hold.get("mae_model") is not None and hold.get("mae_model") is not None:
        improved = hold["mae_model"] <= base_hold["mae_model"]
    report = {
        "tried": tried,
        "fit_seasons": fit_seasons,
        "holdout_seasons": holdout_seasons,
        "fit_objective": best_score,
        "holdout": hold,
        "default_holdout": base_hold,
        "accepted": improved,
        "params": best if improved else base,
    }
    return report["params"], report
