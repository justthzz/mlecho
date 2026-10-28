"""A simple response baseline: quarantine whatever is flagged, without asking which attack.

Same detector and the same two-windows-in-a-row rule as MLEcho's rules, but
it never rolls back, never escalates and treats every alarm the same way. It
shows what the attack classifier adds (step 5 ablation).
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
