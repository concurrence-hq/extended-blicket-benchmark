"""5-object blicket detector environment.

Extends the blicket task from 3 objects (A, B, C) to 5 objects (A, B, C, D, E).
Reuses CausalRule, CausalRuleType, and Observation from blicket_env but defines
its own Action5 enum and BlicketEnv5 class. The original blicket_env.py is not
modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .blicket_env import CausalRule, CausalRuleType, Observation


class Action5(Enum):
    PLACE_A = "place_a"
    PLACE_B = "place_b"
    PLACE_C = "place_c"
    PLACE_D = "place_d"
    PLACE_E = "place_e"
    REMOVE_A = "remove_a"
    REMOVE_B = "remove_b"
    REMOVE_C = "remove_c"
    REMOVE_D = "remove_d"
    REMOVE_E = "remove_e"
    CHECK = "check"


OBJECT_NAMES_5 = ["A", "B", "C", "D", "E"]

PLACE_ACTIONS_5 = {
    Action5.PLACE_A: "A",
    Action5.PLACE_B: "B",
    Action5.PLACE_C: "C",
    Action5.PLACE_D: "D",
    Action5.PLACE_E: "E",
}

REMOVE_ACTIONS_5 = {
    Action5.REMOVE_A: "A",
    Action5.REMOVE_B: "B",
    Action5.REMOVE_C: "C",
    Action5.REMOVE_D: "D",
    Action5.REMOVE_E: "E",
}


@dataclass
class EnvState5:
    """Observable state of the 5-object blicket detector."""

    objects_on_detector: set[str] = field(default_factory=set)
    detector_active: bool = False
    step_count: int = 0
    action_history: list[Action5] = field(default_factory=list)
    activation_history: list[bool] = field(default_factory=list)


class BlicketEnv5:
    """5-object blicket detector environment.

    Five objects (A, B, C, D, E) can be placed on or removed from a detector.
    The detector activates according to a hidden causal rule. The agent's
    goal is to determine which objects are "blickets" and the rule type.
    """

    def __init__(self, causal_rule: CausalRule, max_steps: int = 75) -> None:
        self.causal_rule = causal_rule
        self.max_steps = max_steps
        self.state = EnvState5()
        self._done = False
        self._answer_correct: bool | None = None

    def reset(self) -> EnvState5:
        """Reset the environment to initial state."""
        self.state = EnvState5()
        self._done = False
        self._answer_correct = None
        return self.state

    @property
    def done(self) -> bool:
        return self._done

    @property
    def answer_correct(self) -> bool | None:
        return self._answer_correct

    def get_valid_actions(self) -> list[Action5]:
        """Return actions valid in the current state."""
        if self._done:
            return []

        valid = []
        for action, obj in PLACE_ACTIONS_5.items():
            if obj not in self.state.objects_on_detector:
                valid.append(action)
        for action, obj in REMOVE_ACTIONS_5.items():
            if obj in self.state.objects_on_detector:
                valid.append(action)
        valid.append(Action5.CHECK)
        return valid

    def step(self, action: Action5) -> Observation:
        """Execute an action and return the observation."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        previous_active = self.state.detector_active
        action_valid = True
        message = ""

        if action in PLACE_ACTIONS_5:
            obj = PLACE_ACTIONS_5[action]
            if obj in self.state.objects_on_detector:
                action_valid = False
                message = f"Object {obj} is already on the detector."
            else:
                self.state.objects_on_detector.add(obj)
                message = f"You placed object {obj} on the detector."

        elif action in REMOVE_ACTIONS_5:
            obj = REMOVE_ACTIONS_5[action]
            if obj not in self.state.objects_on_detector:
                action_valid = False
                message = f"Object {obj} is not on the detector."
            else:
                self.state.objects_on_detector.discard(obj)
                message = f"You removed object {obj} from the detector."

        elif action == Action5.CHECK:
            message = "You decided to check your answer."
            self._done = True

        # Update detector state
        new_active = self._compute_activation()
        self.state.detector_active = new_active
        detector_changed = new_active != previous_active

        if action != Action5.CHECK:
            if new_active and detector_changed:
                message += " The detector is now ACTIVE (glowing)."
            elif not new_active and detector_changed:
                message += " The detector is now INACTIVE (not glowing)."
            elif new_active:
                message += " The detector remains ACTIVE (glowing)."
            else:
                message += " The detector remains INACTIVE (not glowing)."

        self.state.step_count += 1
        self.state.action_history.append(action)
        self.state.activation_history.append(new_active)

        if self.state.step_count >= self.max_steps and not self._done:
            self._done = True
            message += " Maximum steps reached."

        return Observation(
            action_taken=action,
            action_valid=action_valid,
            objects_on_detector=set(self.state.objects_on_detector),
            detector_active=new_active,
            detector_changed=detector_changed,
            message=message,
        )

    def _compute_activation(self) -> bool:
        """Compute whether the detector should be active."""
        return self.causal_rule.is_satisfied(self.state.objects_on_detector)

    def check_answer(
        self,
        rule_type: CausalRuleType,
        required_objects: set[str],
        guessed_order: list[str] | None = None,
    ) -> dict[str, Any]:
        """Validate the agent's proposed rule against the true rule."""
        type_correct = rule_type == self.causal_rule.rule_type
        objects_correct = required_objects == self.causal_rule.required_objects
        fully_correct = type_correct and objects_correct

        self._answer_correct = fully_correct

        return {
            "correct": fully_correct,
            "rule_type_correct": type_correct,
            "objects_correct": objects_correct,
            "true_rule_type": self.causal_rule.rule_type.value,
            "true_required_objects": sorted(self.causal_rule.required_objects),
            "guessed_rule_type": rule_type.value,
            "guessed_required_objects": sorted(required_objects),
        }

    def get_info(self) -> dict[str, Any]:
        """Return environment info for logging."""
        return {
            "rule_type": self.causal_rule.rule_type.value,
            "required_objects": sorted(self.causal_rule.required_objects),
            "max_steps": self.max_steps,
            "step_count": self.state.step_count,
            "done": self._done,
            "objects_on_detector": sorted(self.state.objects_on_detector),
            "detector_active": self.state.detector_active,
            "num_objects": 5,
        }
