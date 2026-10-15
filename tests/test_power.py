"""Behavior checks for the neutral rating and the site line. No network."""

from __future__ import annotations

import pandas as pd

from cfmp.ingest.lines import normalize_lines
from cfmp.ingest.qb import normalize_players
from cfmp.models.params import PowerParams
from cfmp.models.power import final_ratings, predict_slate, select_params, summarize, top25, walk_forward


def _games() -> pd.DataFrame:
    # Two seasons. Week 1 is a blowout, week 2 is the same matchup.
    # A market number is attached so we can prove the rating ignores it.
    rows = [
        dict(season=2024, week=1, start_date="2024-09-01T16:00:00Z", id=1,
             home="Alpha", away="Beta", home_points=49, away_points=7, neutral_site=False,
             venue="Alpha Stadium", venue_lat=33.2, venue_lon=-87.5,
             vegas_spread=3.0, vegas_open=1.0, home_qb="A1", away_qb="B1",
             home_pass_rate=0.45, away_pass_rate=0.70),
        dict(season=2024, week=2, start_date="2024-09-08T16:00:00Z", id=2,
             home="Beta", away="Alpha", home_points=17, away_points=24, neutral_site=False,
             venue="Beta Stadium", venue_lat=42.3, venue_lon=-83.7,
             vegas_spread=-3.0, vegas_open=-2.0, home_qb="B1", away_qb="A1",
             home_pass_rate=0.70, away_pass_rate=0.45),
        dict(season=2024, week=3, start_date="2024-09-14T16:00:00Z", id=3,
             home="Alpha", away="Beta", home_points=21, away_points=17, neutral_site=True,
             venue="Dome", indoor=True, venue_lat=33.7, venue_lon=-84.4,
             vegas_spread=7.0, vegas_open=6.0, home_qb="A1", away_qb="B2",
             home_pass_rate=0.45, away_pass_rate=0.70,
             wx_temp_f=20, wx_wind_mph=30, wx_pop=90, wx_short="Snow", forecast_conf=0.95),
        dict(season=2025, week=1, start_date="2025-09-06T16:00:00Z", id=4,
             home="Alpha", away="Beta", home_points=28, away_points=14, neutral_site=False,
             venue="Alpha Stadium", venue_lat=33.2, venue_lon=-87.5,
             vegas_spread=6.0, vegas_open=4.0, home_qb="A1", away_qb="B2",
             home_pass_rate=0.45, away_pass_rate=0.70),
    ]
    df = pd.DataFrame(rows)
    df["start_date"] = pd.to_datetime(df["start_date"], utc=True)
    return df


def test_margin_beats_a_sneaky_win() -> None:
    params = PowerParams(learning_rate=0.15, mov_cap=28, hfa=2.5, season_regress=1.0)
    graded = walk_forward(_games(), params)
    # Before any games, neutral spread is zero. Home field is the whole line.
    first = graded.iloc[0]
    assert abs(first.neutral_spread) < 1e-9
    assert abs(first.hfa_points - 2.5) < 1e-9
    # After Alpha blows out Beta, Alpha is the better neutral team.
    second = graded.iloc[1]
    assert second.neutral_spread < 0  # Beta is home, so home rating minus away (Alpha) is negative
    # The market column does not change the update: drop it and ratings match.
    bare = _games().drop(columns=["vegas_spread", "vegas_open"])
    a = final_ratings(_games(), params).set_index("team").rating
    b = final_ratings(bare, params).set_index("team").rating
    assert abs(a["Alpha"] - b["Alpha"]) < 1e-9


def test_site_line_respects_weather_and_roof() -> None:
    params = PowerParams()
    ratings = final_ratings(_games().iloc[:1], params)
    mild = pd.DataFrame([{
        "home": "Alpha", "away": "Beta", "neutral_site": False, "venue": "Bryant-Denny Stadium",
        "indoor": False, "kickoff": "2024-09-08T20:00:00Z",
        "wx_temp_f": 68, "wx_wind_mph": 15, "wx_pop": 1, "wx_short": "Sunny", "forecast_conf": 0.95,
        "venue_lat": 33.2, "venue_lon": -87.5,
    }])
    storm = mild.copy()
    storm["wx_temp_f"] = 22
    storm["wx_wind_mph"] = 32
    storm["wx_pop"] = 90
    storm["wx_short"] = "Snow"
    roof = storm.copy()
    roof["venue"] = "Mercedes-Benz Stadium"
    roof["indoor"] = True
    mild_px = predict_slate(mild, ratings, params).iloc[0]
    storm_px = predict_slate(storm, ratings, params).iloc[0]
    roof_px = predict_slate(roof, ratings, params).iloc[0]
    assert mild_px.weather_adj == 0
    # Alpha's first game pass rate is 0.45, Beta 0.70, so the home passer is less pass-heavy.
    # A storm should help Alpha (the home team) relative to no weather.
    assert storm_px.weather_adj > 0
    assert roof_px.weather_adj == 0
    assert roof_px.hfa_points == params.hfa
    assert abs(mild_px.neutral_spread - (mild_px.rating_home - mild_px.rating_away)) < 1e-9


def test_walk_forward_does_not_peek() -> None:
    params = PowerParams(learning_rate=0.2, hfa=0, season_regress=1, mov_cap=28)
    one = _games().iloc[:1].copy()
    one["neutral_site"] = True
    one["hfa"] = 0
    before = walk_forward(one, params).iloc[0]
    assert abs(before.neutral_spread) < 1e-9
    # The blowout is visible to the next game only.
    nxt = walk_forward(_games().iloc[:2], params).iloc[1]
    assert nxt.neutral_spread != 0


def test_lines_and_quarterbacks_parse() -> None:
    lines = normalize_lines([{
        "id": 9,
        "lines": [
            {"provider": "DraftKings", "spread": -7.0, "spreadOpen": -3.5, "overUnder": 55.5, "overUnderOpen": 53.5, "homeMoneyline": -270, "awayMoneyline": 220},
            {"provider": "FanDuel", "spread": -6.0, "spreadOpen": -4.5},
        ],
    }])
    assert lines[0]["book"] == "DraftKings"
    assert lines[0]["vegas_spread"] == 7.0
    assert lines[0]["vegas_open"] == 3.5
    assert lines[0]["total"] == 55.5
    assert lines[0]["home_ml"] == -270
    assert normalize_lines([{"id": 10, "lines": [{"provider": "FanDuel", "spread": -6.0}]}]) == []
    players = normalize_players([{
        "id": 9,
        "teams": [
            {"homeAway": "home", "categories": [
                {"name": "passing", "types": [{"name": "C/ATT", "athletes": [{"name": "Milroe", "stat": "14/22"}]}]},
                {"name": "rushing", "types": [{"name": "CAR", "athletes": [{"name": "Haynes", "stat": "18"}]}]},
            ]},
            {"homeAway": "away", "categories": [
                {"name": "passing", "types": [{"name": "C/ATT", "athletes": [{"name": "Other", "stat": "20/40"}]}]},
                {"name": "rushing", "types": [{"name": "CAR", "athletes": [{"name": "Back", "stat": "10"}]}]},
            ]},
        ],
    }])
    assert players[0]["home_qb"] == "Milroe"
    assert abs(players[0]["home_pass_rate"] - 22 / 40) < 1e-9


def test_fit_keeps_defaults_when_holdout_gets_worse(monkeypatch=None) -> None:
    games = _games()
    params, report = select_params(games, [2024], [2025])
    assert "accepted" in report
    assert report["holdout"]["games"] == 1
    # The candidate is returned only when accepted; otherwise the defaults.
    if not report["accepted"]:
        assert params.hfa == PowerParams().hfa


def test_top25_is_fbs_power_not_wins() -> None:
    ratings = pd.DataFrame([
        {"team": "Undefeated", "rating": 4.0, "off": 2.0, "defense": 2.0, "classification": "fbs"},
        {"team": "Better", "rating": 12.0, "off": 6.0, "defense": 6.0, "classification": "fbs"},
        {"team": "Small School", "rating": 30.0, "off": 15.0, "defense": 15.0, "classification": "fcs"},
    ])
    board = top25(ratings)
    assert list(board.team) == ["Better", "Undefeated"]
    assert list(board["rank"]) == [1, 2]
    assert "Small School" not in set(board.team)


def test_promotion_does_not_keep_the_fcs_number() -> None:
    rows = []
    game_id = 1
    for season in range(2014, 2018):
        for week in range(1, 11):
            rows.append({
                "season": season, "week": week, "id": game_id,
                "start_date": f"{season}-10-{week:02d}T16:00:00Z",
                "home": "Dynasty", "away": "Cup",
                "home_points": 45, "away_points": 17, "neutral_site": True,
                "home_classification": "fcs", "away_classification": "fcs",
            })
            game_id += 1
            rows.append({
                "season": season, "week": week, "id": game_id,
                "start_date": f"{season}-10-{week:02d}T20:00:00Z",
                "home": "Solid", "away": "Even",
                "home_points": 24, "away_points": 24, "neutral_site": True,
                "home_classification": "fbs", "away_classification": "fbs",
            })
            game_id += 1
    history = pd.DataFrame(rows)
    history["start_date"] = pd.to_datetime(history["start_date"], utc=True)
    before = final_ratings(history, PowerParams()).set_index("team")
    assert before.loc["Dynasty", "classification"] == "fcs"
    assert before.loc["Dynasty", "edge"] > 5
    assert before.loc["Dynasty", "rating"] < before.loc["Solid", "rating"]

    moved = pd.concat([history, pd.DataFrame([{
        "season": 2018, "week": 1, "id": 999,
        "start_date": pd.Timestamp("2018-09-01T16:00:00Z"),
        "home": "Solid", "away": "Dynasty",
        "home_points": 24, "away_points": 17, "neutral_site": True,
        "home_classification": "fbs", "away_classification": "fbs",
    }])], ignore_index=True)
    after = final_ratings(moved, PowerParams()).set_index("team")
    assert after.loc["Dynasty", "classification"] == "fbs"
    assert after.loc["Dynasty", "rating"] < 5
    assert after.loc["Dynasty", "rating"] < after.loc["Solid", "rating"]


def test_summary_metrics() -> None:
    graded = walk_forward(_games(), PowerParams())
    summary = summarize(graded)
    assert summary["games"] == 4
    assert summary["lined_games"] == 4
    assert summary["mae_model"] is not None
    assert summary["mae_vegas"] is not None


if __name__ == "__main__":
    test_margin_beats_a_sneaky_win()
    test_site_line_respects_weather_and_roof()
    test_walk_forward_does_not_peek()
    test_lines_and_quarterbacks_parse()
    test_fit_keeps_defaults_when_holdout_gets_worse()
    test_top25_is_fbs_power_not_wins()
    test_promotion_does_not_keep_the_fcs_number()
    test_summary_metrics()
    print("all power checks passed")
