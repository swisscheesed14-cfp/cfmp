"""A thin roster shrinks harder, and talent is the place it shrinks toward."""

from __future__ import annotations

from cfmp.models.params import PowerParams
from cfmp.models.power import new_book
from cfmp.models.priors import Priors, keep_from_returning


def test_keep_bounds() -> None:
    assert keep_from_returning(0.0) == 0.25
    assert abs(keep_from_returning(1.0) - 0.85) < 1e-9
    assert keep_from_returning(0.0) < keep_from_returning(1.0)


def test_low_returning_falls_farther_than_a_full_roster() -> None:
    params = PowerParams()
    book = new_book(params)
    for name in ("Kept", "Gutted"):
        team = book.team(name)
        team.off = 6.0
        team.def_ = 6.0
        team.classification = "fbs"
    priors = Priors(
        talent={(2025, "Kept"): 4.0, (2025, "Gutted"): 4.0},
        returning={(2025, "Kept"): (1.0, 1.0), (2025, "Gutted"): (0.0, 0.0)},
    )
    book.regress(params.season_regress, 2025, priors)
    assert book.team("Kept").rating > book.team("Gutted").rating
    # No roster info shrinks toward zero, which is the old rule.
    plain = new_book(params)
    team = plain.team("Plain")
    team.off = 6.0
    team.def_ = 6.0
    team.classification = "fbs"
    plain.regress(params.season_regress, 2025, None)
    assert abs(plain.team("Plain").rating - 12.0 * params.season_regress) < 1e-9


if __name__ == "__main__":
    test_keep_bounds()
    test_low_returning_falls_farther_than_a_full_roster()
    print("priors ok")
