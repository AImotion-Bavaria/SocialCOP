"""Social-choice expressions and constraints."""

from functools import reduce
from operator import mul
from typing import Any, Sequence

from cpmpy import min as cp_min


def minimum_utility(utilities: Sequence[Any]) -> Any:
    """Return the utility of the worst-off agent."""
    if len(utilities) == 0:
        raise ValueError("utilities must not be empty")
    return cp_min(list(utilities))


def total_utility(utilities: Sequence[Any]) -> Any:
    """Return utilitarian social welfare."""
    if len(utilities) == 0:
        raise ValueError("utilities must not be empty")
    return sum(utilities)


def nash_welfare(utilities: Sequence[Any]) -> Any:
    """Return the Nash product; callers must provide non-negative utilities."""
    if len(utilities) == 0:
        raise ValueError("utilities must not be empty")
    return reduce(mul, utilities)


def envy_free_constraints(shares: Sequence[Sequence[Any]]) -> list[Any]:
    """Ensure every agent values its own allocation at least as highly as others'.

    `shares[i][j]` is agent i's valuation of agent j's allocated share.
    """
    size = len(shares)
    if size == 0 or any(len(row) != size for row in shares):
        raise ValueError("shares must be a non-empty square matrix")
    return [shares[i][i] >= shares[i][j] for i in range(size) for j in range(size) if i != j]
