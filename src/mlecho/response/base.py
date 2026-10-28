"""The response interface: every policy (rules, PPO) takes a State and gives an Action.

Keeping one small interface means the rule table and the PPO agent can be
swapped without changing anything around them (proposal 3.4.4).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class Action(str, Enum):
    """What the guard can do after a window (proposal: the fixed action set)."""

    CONTINUE = "continue"  # keep training as normal
    QUARANTINE = "quarantine"  # from now on, leave suspicious images out of training
    ROLLBACK = "rollback"  # go back to the last clean checkpoint, then quarantine
    RETRAIN = "retrain"  # go back to the oldest clean checkpoint, then quarantine
    ESCALATE = "escalate"  # ask a human; training continues unchanged


@dataclass(frozen=True)
class State:
    """What a policy sees after each window. Only dataset-independent numbers."""

    attack_type: str  # the classifier's answer for this window
    confidence: float  # the classifier's probability for that answer
    probabilities: tuple[float, ...]  # every type's probability (ATTACK_TYPES order)
    streak: int  # windows in a row with this same answer (non-clean types only)
    acted: bool  # a quarantine / rollback / retrain is already in force for this incident
    escalated: bool  # a human was already asked about this incident
    loss_change: float  # mean training loss of this window minus the previous one


class ResponsePolicy(Protocol):
    """Anything with a `decide` method is a policy."""

    def decide(self, state: State) -> Action: ...
