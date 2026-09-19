"""Small, native CPMpy table-assignment model."""

from dataclasses import dataclass
from typing import Sequence

from cpmpy import Model, boolvar

from socialcop_cpmpy import SocialProblem


@dataclass
class TableAssignment:
    problem: SocialProblem
    assigned: object


def build_table_assignment(
    preferences: Sequence[Sequence[int]], table_capacities: Sequence[int]
) -> TableAssignment:
    """Assign every agent to one table and expose each agent's preference utility."""
    if not preferences or not table_capacities:
        raise ValueError("preferences and table_capacities must not be empty")
    table_count = len(table_capacities)
    if any(len(row) != table_count for row in preferences):
        raise ValueError("each preference row must contain one value per table")

    agent_count = len(preferences)
    assigned = boolvar(shape=(agent_count, table_count), name="assigned")
    constraints = [sum(assigned[a, :]) == 1 for a in range(agent_count)]
    constraints += [sum(assigned[:, t]) <= table_capacities[t] for t in range(table_count)]
    utilities = [sum(preferences[a][t] * assigned[a, t] for t in range(table_count)) for a in range(agent_count)]

    # An agent evaluates another agent's share as the table occupied by that agent.
    shares = [
        [sum(preferences[i][t] * assigned[j, t] for t in range(table_count)) for j in range(agent_count)]
        for i in range(agent_count)
    ]
    return TableAssignment(SocialProblem(Model(constraints), utilities, shares), assigned)
