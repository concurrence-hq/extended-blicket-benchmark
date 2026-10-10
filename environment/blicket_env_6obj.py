"""6-object blicket detector environment.

Extends the blicket task from 5 objects (A-E) to 6 objects (A-F).
Reuses CausalRule, CausalRuleType, and Observation from blicket_env but defines
its own Action6 enum and BlicketEnv6 class. The original blicket_env.py and
blicket_env_5obj.py are not modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .blicket_env import CausalRule, CausalRuleType, Observation


class Action6(Enum):
    PLACE_A = "place_a"
    PLACE_B = "place_b"
    PLACE_C = "place_c"
    PLACE_D = "place_d"
    PLACE_E = "place_e"
    PLACE_F = "place_f"
    REMOVE_A = "remove_a"
    REMOVE_B = "remove_b"
    REMOVE_C = "remove_c"
    REMOVE_D = "remove_d"
    REMOVE_E = "remove_e"
    REMOVE_F = "remove_f"
    CHECK = "check"


OBJECT_NAMES_6 = ["A", "B", "C", "D", "E", "F"]

PLACE_ACTIONS_6 = {
    Action6.PLACE_A: "A",
    Action6.PLACE_B: "B",
    Action6.PLACE_C: "C",
    Action6.PLACE_D: "D",
    Action6.PLACE_E: "E",
    Action6.PLACE_F: "F",
}

REMOVE_ACTIONS_6 = {
    Action6.REMOVE_A: "A",
    Action6.REMOVE_B: "B",
    Action6.REMOVE_C: "C",
    Action6.REMOVE_D: "D",
    Action6.REMOVE_E: "E",
    Action6.REMOVE_F: "F",
}


@dataclass
class EnvState6:
    """Observable state of the 6-object blicket detector."""

    objects_on_detector: set[str] = field(default_factory=set)
    detector_active: bool = False
    step_count: int = 0
    action_history: list[Action6] = field(default_factory=list)
    activation_history: list[bool] = field(default_factory=list)


class BlicketEnv6:
    """6-object blicket detector environment.

    Six objects (A, B, C, D, E, F) can be placed on or removed from a detector.
    The detector activates according to a hidden causal rule. The agent's
    goal is to determine which objects are "blickets" and the rule type.
    """

    def __init__(self, causal_rule: CausalRule, max_steps: int = 75) -> None:
        self.causal_rule = causal_rule
        self.max_steps = max_steps
        self.state = EnvState6()
        self._done = False
        self._answer_correct: bool | None = None

    def reset(self) -> EnvState6:
        """Reset the environment to initial state."""
        self.state = EnvState6()
        self._done = False
        self._answer_correct = None
        return self.state

    @property
    def done(self) -> bool:
        return self._done

    @property
    def answer_correct(self) -> bool | None:
        return self._answer_correct

    def get_valid_actions(self) -> list[Action6]:
        """Return actions valid in the current state."""
        if self._done:
            return []

        valid = []
        for action, obj in PLACE_ACTIONS_6.items():
            if obj not in self.state.objects_on_detector:
                valid.append(action)
        for action, obj in REMOVE_ACTIONS_6.items():
            if obj in self.state.objects_on_detector:
                valid.append(action)
        valid.append(Action6.CHECK)
        return valid

    def step(self, action: Action6) -> Observation:
        """Execute an action and return the observation."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        previous_active = self.state.detector_active
        action_valid = True
        message = ""

        if action in PLACE_ACTIONS_6:
            obj = PLACE_ACTIONS_6[action]
            if obj in self.state.objects_on_detector:
                action_valid = False
                message = f"Object {obj} is already on the detector."
            else:
                self.state.objects_on_detector.add(obj)
                message = f"You placed object {obj} on the detector."

        elif action in REMOVE_ACTIONS_6:
            obj = REMOVE_ACTIONS_6[action]
            if obj not in self.state.objects_on_detector:
                action_valid = False
                message = f"Object {obj} is not on the detector."
            else:
                self.state.objects_on_detector.discard(obj)
                message = f"You removed object {obj} from the detector."

        elif action == Action6.CHECK:
            message = "You decided to check your answer."
            self._done = True

        # Update detector state
        new_active = self._compute_activation()
        self.state.detector_active = new_active
        detector_changed = new_active != previous_active

        if action != Action6.CHECK:
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
            "num_objects": 6,
        }
