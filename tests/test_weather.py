"""Kickoff weather moves the weekly home spread. A roof does not."""

from cfmp.models.weather import adjust_for_forecast


def test_a_dome_ignores_the_forecast() -> None:
    storm = {"tempF": 35, "wind_mph": 40, "pop": 100, "forecast": "Thunderstorm"}
    spread, delta = adjust_for_forecast(14.0, True, storm, 1.0)
    assert spread == 14.0
    assert delta == 0.0


def test_a_gale_shrinks_the_favorite_and_does_not_flip_it() -> None:
    gale = {"tempF": 70, "wind_mph": 35, "pop": 0, "forecast": "Windy"}
    spread, delta = adjust_for_forecast(20.0, False, gale, 1.0)
    assert delta == -3.0
    assert spread == 17.0


def test_a_mild_day_is_not_an_edge() -> None:
    mild = {"tempF": 62, "wind_mph": 12, "pop": 10, "forecast": "Partly Cloudy"}
    spread, delta = adjust_for_forecast(7.5, False, mild, 1.0)
    assert delta == 0.0
    assert spread == 7.5


def test_no_forecast_leaves_the_number_alone() -> None:
    spread, delta = adjust_for_forecast(-6.0, False, None, 0.7)
    assert spread == -6.0
    assert delta == 0.0
