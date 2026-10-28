"""The rule-based policy: a fixed table from attack type to action (proposal 3.4.4).

The table and the two settings were fixed before the step 5 evaluation, from
the step 4 results:
- backdoor -> rollback: a learned trigger doesn't fade by itself, so undo the
  training since it arrived;
- accumulative -> retrain: a slow drip may have started long before it was
  noticed, so go back further (the oldest clean checkpoint);
- label-flip -> quarantine: wrong labels are unlearned by further clean
  training, so stopping new poison is enough;
- harmless shift, or low confidence -> escalate: never delete data on a guess.
"""

from dataclasses import dataclass

from mlecho.response.base import Action, State

RULES = {
    "backdoor": Action.ROLLBACK,
    "accumulative": Action.RETRAIN,
    "label_flip": Action.QUARANTINE,
    "benign_shift": Action.ESCALATE,
}


@dataclass(frozen=True)
class RulePolicy:
    """`persistence`: windows in a row needed before acting (cuts false alarms:
    3% of clean windows are misread, so two in a row are rare).
    `min_confidence`: below it, escalate instead (step 4: mean confidence 0.78)."""

    persistence: int = 2
    min_confidence: float = 0.6

    def decide(self, state: State) -> Action:
        if state.attack_type == "clean" or state.streak < self.persistence:
            return Action.CONTINUE
        action = RULES[state.attack_type]
        if state.confidence < self.min_confidence or action == Action.ESCALATE:
            return Action.CONTINUE if state.escalated else Action.ESCALATE
        return Action.CONTINUE if state.acted else action
