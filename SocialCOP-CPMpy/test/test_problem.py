import pytest
from cpmpy import Model, intvar

from socialcop_cpmpy import SocialProblem, envy_free


def test_problem_requires_utilities() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SocialProblem(Model(), [])


@pytest.mark.parametrize("shares", [[[1, 2]], [[1], [2]]])
def test_problem_requires_square_shares(shares) -> None:
    with pytest.raises(ValueError, match="square"):
        SocialProblem(Model(), [1, 2], shares)


def test_envy_free_requires_shares() -> None:
    utility = intvar(0, 1, name="utility")
    problem = SocialProblem(Model(utility == 1), [utility])
    with pytest.raises(ValueError, match="requires a shares matrix"):
        envy_free(problem)
