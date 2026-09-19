import pytest
from cpmpy import Model, intvar

from socialcop_cpmpy.social import (
    envy_free_constraints,
    minimum_utility,
    nash_welfare,
    total_utility,
)


def test_social_expressions_have_expected_values() -> None:
    utilities = intvar(0, 10, shape=3, name="utility")
    model = Model(utilities == [2, 4, 3])

    assert model.solve()
    assert total_utility(utilities).value() == 9
    assert minimum_utility(utilities).value() == 2
    assert nash_welfare(utilities).value() == 24


@pytest.mark.parametrize(
    "function", [minimum_utility, total_utility, nash_welfare]
)
def test_social_expressions_reject_empty_utilities(function) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        function([])


def test_envy_free_constraints_accept_non_envious_shares() -> None:
    shares = [[10, 2], [3, 8]]
    assert Model(envy_free_constraints(shares)).solve()


def test_envy_free_constraints_reject_envy() -> None:
    shares = [[1, 2], [3, 8]]
    assert not Model(envy_free_constraints(shares)).solve()


@pytest.mark.parametrize("shares", [[], [[1, 2]], [[1], [2]]])
def test_envy_free_constraints_require_square_matrix(shares) -> None:
    with pytest.raises(ValueError, match="non-empty square matrix"):
        envy_free_constraints(shares)
