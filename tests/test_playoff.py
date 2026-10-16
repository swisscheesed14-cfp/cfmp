"""The 2026 field is 12 teams, with the bids this season actually uses."""

from cfmp.models.playoff import select_field, win_chance


CONFERENCE = {
    "Duke": "ACC",
    "Ohio State": "Big Ten",
    "Texas Tech": "Big 12",
    "Alabama": "SEC",
    "Memphis": "American Athletic",
    "Tulane": "American Athletic",
    "Notre Dame": "FBS Independents",
    "UConn": "FBS Independents",
}


def _pool() -> dict[str, float]:
    resume = {name: 20.0 - index for index, name in enumerate(
        [
            "Alabama",
            "Ohio State",
            "Texas Tech",
            "Georgia",
            "Oregon",
            "Texas",
            "Indiana",
            "Ole Miss",
            "Miami",
            "Tennessee",
            "LSU",
            "Michigan",
            "Penn State",
            "Oklahoma",
        ]
    )}
    resume["Notre Dame"] = 8.0
    resume["Duke"] = -4.0
    resume["Memphis"] = 3.0
    resume["Tulane"] = 1.0
    resume["UConn"] = -8.0
    return resume


def test_a_bad_power_champion_still_gets_in() -> None:
    resume = _pool()
    conference = {**CONFERENCE, **{name: "SEC" for name in resume if name not in CONFERENCE}}
    conference["Ohio State"] = "Big Ten"
    conference["Texas Tech"] = "Big 12"
    conference["Duke"] = "ACC"
    field, reason = select_field(resume, {"ACC": "Duke", "Big Ten": "Ohio State", "Big 12": "Texas Tech", "SEC": "Alabama"}, conference)
    assert len(field) == 12
    assert "Duke" in field
    assert reason["Duke"] == "champ"


def test_the_group_bid_follows_the_rating_not_the_trophy() -> None:
    resume = _pool()
    conference = {name: "SEC" for name in resume}
    conference.update(CONFERENCE)
    field, reason = select_field(
        resume,
        {"ACC": "Duke", "Big Ten": "Ohio State", "Big 12": "Texas Tech", "SEC": "Alabama", "American Athletic": "Tulane"},
        conference,
    )
    assert "Memphis" in field
    assert reason["Memphis"] == "group"
    assert "Tulane" not in field


def test_notre_dame_in_the_top_12_takes_a_spot() -> None:
    resume = _pool()
    resume["Notre Dame"] = resume["Michigan"] + 0.1
    ordered = sorted(resume, key=resume.get, reverse=True)
    assert ordered.index("Notre Dame") < 12
    conference = {name: "SEC" for name in resume}
    conference.update(CONFERENCE)
    field, reason = select_field(resume, {"ACC": "Duke", "Big Ten": "Ohio State", "Big 12": "Texas Tech", "SEC": "Alabama"}, conference)
    assert "Notre Dame" in field
    assert reason["Notre Dame"] == "notre dame"
    assert len(field) == 12


def test_notre_dame_outside_the_top_12_is_not_protected() -> None:
    resume = _pool()
    resume["Notre Dame"] = -1.0
    ordered = sorted(resume, key=resume.get, reverse=True)
    assert ordered.index("Notre Dame") >= 12
    conference = {name: "SEC" for name in resume}
    conference.update(CONFERENCE)
    field, _reason = select_field(resume, {"ACC": "Duke", "Big Ten": "Ohio State", "Big 12": "Texas Tech", "SEC": "Alabama"}, conference)
    assert "Notre Dame" not in field


def test_a_pickem_is_a_coin_flip() -> None:
    assert abs(win_chance(0.0, 17.0) - 0.5) < 1e-9
    assert win_chance(20.0, 17.0) > 0.85
