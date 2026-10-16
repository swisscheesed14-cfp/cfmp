"""A ruled-out starter has to be read off the report, not out of last week's box score."""

from cfmp.ingest.injuries import parse_covers, parse_headline

HTML = """
<a id="UCLA"></a>
<span class='player-link'>N.                                        Iamaleava</span>
</td>
<td>QB</td>
<td><b>Out - Knee</b><br>(Wed, Oct 14)</td>
<td>out for Saturday</td>
<a id="Pittsburgh"></a>
<span class='player-link'>M. Heintschel</span></td><td>QB</td><td><b>Out - Knee</b>
<td>shut down for the remainder of the season</td>
<a id="Backup U"></a>
<span class='player-link'>J. Someone</span></td><td>WR</td><td><b>Out - Ankle</b>
"""


def test_a_saturday_absence_is_only_this_week() -> None:
    rows = parse_covers(HTML, ["UCLA", "Pittsburgh"])
    ucla = next(row for row in rows if row["team"] == "UCLA")
    assert ucla["qb"] == "Iamaleava"
    assert ucla["scope"] == "week"
    assert ucla["level"] == "out"


def test_a_shutdown_is_the_rest_of_the_year() -> None:
    rows = parse_covers(HTML, ["UCLA", "Pittsburgh"])
    pitt = next(row for row in rows if row["team"] == "Pittsburgh")
    assert pitt["scope"] == "season"


def test_a_headline_names_the_quarterback() -> None:
    parsed = parse_headline(
        "UCLA quarterback Nico Iamaleava out Saturday vs. Wisconsin",
        "UCLA quarterback Nico Iamaleava is out for Saturday's matchup against Wisconsin.",
    )
    assert parsed == ("Nico Iamaleava", "week", "out")
    season = parse_headline(
        "Pitt quarterback Mason Heintschel out for the season",
        "He is out for the season with a torn ACL.",
    )
    assert season is not None and season[1] == "season"
    assert parse_headline("Oregon DT won't face discipline", "He tried to punch the football.") is None
