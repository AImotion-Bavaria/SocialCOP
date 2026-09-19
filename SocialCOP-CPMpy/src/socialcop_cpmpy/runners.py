"""Solver-independent entry points for CPMpy SocialCOP problems."""

from typing import Any

from cpmpy import SolverLookup

from .problem import SocialProblem
from .social import envy_free_constraints, minimum_utility, nash_welfare, total_utility


def solve(problem: SocialProblem, solver: str = "ortools", time_limit: float | None = None) -> bool:
    backend = SolverLookup.get(solver, problem.model)
    return bool(backend.solve(time_limit=time_limit))


def utilitarian(problem: SocialProblem, solver: str = "ortools", time_limit: float | None = None) -> bool:
    problem.model.maximize(total_utility(problem.utilities))
    return solve(problem, solver, time_limit)


def rawls(problem: SocialProblem, solver: str = "ortools", time_limit: float | None = None) -> bool:
    problem.model.maximize(minimum_utility(problem.utilities))
    return solve(problem, solver, time_limit)


def nash(problem: SocialProblem, solver: str = "ortools", time_limit: float | None = None) -> bool:
    problem.model += [utility >= 0 for utility in problem.utilities]
    problem.model.maximize(nash_welfare(problem.utilities))
    return solve(problem, solver, time_limit)


def envy_free(problem: SocialProblem, solver: str = "ortools", time_limit: float | None = None) -> bool:
    if problem.shares is None:
        raise ValueError("envy-free solving requires a shares matrix")
    problem.model += envy_free_constraints(problem.shares)
    return solve(problem, solver, time_limit)
