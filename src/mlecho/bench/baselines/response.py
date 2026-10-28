"""Response baselines to compare MLEcho's rules with (step 5).

- NoResponse: the guard watches but never acts (the control).
- QuarantineEverything: same detector and the same two-windows-in-a-row rule
  as MLEcho's rules, but it never rolls back, never escalates and treats every
  alarm the same way. It shows what the attack classifier adds.
"""

from dataclasses import dataclass

from mlecho.response.base import Action, State


@dataclass(frozen=True)
class QuarantineEverything:
    persistence: int = 2

    def decide(self, state: State) -> Action:
        if state.attack_type == "clean" or state.streak < self.persistence or state.acted:
            return Action.CONTINUE
        return Action.QUARANTINE


class NoResponse:
    def decide(self, state: State) -> Action:
        return Action.CONTINUE
