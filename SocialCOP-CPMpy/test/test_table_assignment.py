import pytest

from models.table_assignment import build_table_assignment
from socialcop_cpmpy import envy_free, nash, rawls, solve, utilitarian


def test_utilitarian_table_assignment() -> None:
    assignment = build_table_assignment([[10, 1], [9, 8]], [1, 1])
    assert utilitarian(assignment.problem)
    assert [utility.value() for utility in assignment.problem.utilities] == [10, 8]


def test_rawls_table_assignment() -> None:
    assignment = build_table_assignment([[10, 5], [4, 3]], [1, 1])
    assert rawls(assignment.problem)
    assert [utility.value() for utility in assignment.problem.utilities] == [5, 4]


def test_nash_table_assignment() -> None:
    assignment = build_table_assignment([[10, 5], [4, 3]], [1, 1])
    assert nash(assignment.problem)
    assert [utility.value() for utility in assignment.problem.utilities] == [10, 3]


def test_envy_free_table_assignment() -> None:
    assignment = build_table_assignment([[10, 1], [1, 10]], [1, 1])
    assert envy_free(assignment.problem)
    assert [utility.value() for utility in assignment.problem.utilities] == [10, 10]


def test_plain_feasibility_solver() -> None:
    assignment = build_table_assignment([[2, 1], [1, 2]], [1, 1])
    assert solve(assignment.problem)
    assert assignment.assigned.value() is not None


def test_insufficient_capacity_is_infeasible() -> None:
    assignment = build_table_assignment([[2], [1]], [1])
    assert not solve(assignment.problem)


@pytest.mark.parametrize(
    ("preferences", "capacities", "message"),
    [
        ([], [1], "must not be empty"),
        ([[1]], [], "must not be empty"),
        ([[1, 2], [3]], [1, 1], "one value per table"),
    ],
)
def test_invalid_table_assignment_input(preferences, capacities, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_table_assignment(preferences, capacities)
