"""Common model interface used by the CPMpy SocialCOP runners."""

from dataclasses import dataclass
from typing import Any, Sequence

from cpmpy import Model


@dataclass
class SocialProblem:
    """A CPMpy model plus the expressions needed by social-choice objectives."""

    model: Model
    utilities: Sequence[Any]
    shares: Sequence[Sequence[Any]] | None = None

    def __post_init__(self) -> None:
        if len(self.utilities) == 0:
            raise ValueError("utilities must contain at least one expression")
        if self.shares is not None:
            size = len(self.utilities)
            if len(self.shares) != size or any(len(row) != size for row in self.shares):
                raise ValueError("shares must be a square agents-by-agents matrix")
