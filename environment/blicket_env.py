"""Core blicket detector environment.

Implements the causal learning task from Kosoy et al. (2022): agents interact
with a "blicket detector" by placing/removing objects and observing activation
patterns to infer the underlying causal rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CausalRuleType(Enum):
    CONJUNCTIVE = "conjunctive"
    DISJUNCTIVE = "disjunctive"
    ORDER_SENSITIVE = "order_sensitive"
    STOCHASTIC = "stochastic"
    ORDER_SENSITIVE_STOCHASTIC = "order_sensitive_stochastic"


class Action(Enum):
    PLACE_A = "place_a"
    PLACE_B = "place_b"
    PLACE_C = "place_c"
    REMOVE_A = "remove_a"
    REMOVE_B = "remove_b"
    REMOVE_C = "remove_c"
    CHECK = "check"


OBJECT_NAMES = ["A", "B", "C"]

# Maps place/remove actions to the object they affect
PLACE_ACTIONS = {Action.PLACE_A: "A", Action.PLACE_B: "B", Action.PLACE_C: "C"}
REMOVE_ACTIONS = {Action.REMOVE_A: "A", Action.REMOVE_B: "B", Action.REMOVE_C: "C"}


@dataclass
class CausalRule:
    """Defines the causal rule governing the blicket detector.

    Args:
        rule_type: Conjunctive (all required) or disjunctive (any required).
        required_objects: Set of object names that are "blickets".
    """

    rule_type: CausalRuleType
    required_objects: set[str]

    def is_satisfied(self, objects_on_detector: set[str]) -> bool:
        """Check if the current object configuration satisfies the causal rule."""
        if not self.required_objects:
            return False

        present = self.required_objects & objects_on_detector

        if self.rule_type == CausalRuleType.CONJUNCTIVE:
            return present == self.required_objects
        elif self.rule_type == CausalRuleType.ORDER_SENSITIVE:
            # Object presence check only — order is validated by ExtendedBlicketEnv5
            return present == self.required_objects
        elif self.rule_type == CausalRuleType.STOCHASTIC:
            # Object presence check only — stochasticity validated by extended env
            return present == self.required_objects
        elif self.rule_type == CausalRuleType.ORDER_SENSITIVE_STOCHASTIC:
            # Object presence check only — order + stochasticity validated by extended env
            return present == self.required_objects
        else:  # disjunctive
            return len(present) > 0


@dataclass
class EnvState:
    """Observable state of the blicket detector."""

    objects_on_detector: set[str] = field(default_factory=set)
    detector_active: bool = False
    step_count: int = 0
    action_history: list[Action] = field(default_factory=list)
    activation_history: list[bool] = field(default_factory=list)


@dataclass
class Observation:
    """Result of taking an action in the environment."""

    action_taken: Action
    action_valid: bool
    objects_on_detector: set[str]
    detector_active: bool
    detector_changed: bool
    message: str


class BlicketEnv:
    """Blicket detector environment.

    Three objects (A, B, C) can be placed on or removed from a detector.
    The detector activates according to a hidden causal rule. The agent's
    goal is to determine which objects are "blickets" and the rule type.

    Actions:
        place_a, place_b, place_c: Place an object on the detector
        remove_a, remove_b, remove_c: Remove an object from the detector
        check: Submit a guess about the causal rule (ends the episode)
    """

    def __init__(self, causal_rule: CausalRule, max_steps: int = 50) -> None:
        self.causal_rule = causal_rule
        self.max_steps = max_steps
        self.state = EnvState()
        self._done = False
        self._answer_correct: bool | None = None

    def reset(self) -> EnvState:
        """Reset the environment to initial state."""
        self.state = EnvState()
        self._done = False
        self._answer_correct = None
        return self.state

    @property
    def done(self) -> bool:
        return self._done

    @property
    def answer_correct(self) -> bool | None:
        return self._answer_correct

    def get_valid_actions(self) -> list[Action]:
        """Return actions valid in the current state."""
        if self._done:
            return []

        valid = []
        for action, obj in PLACE_ACTIONS.items():
            if obj not in self.state.objects_on_detector:
                valid.append(action)
        for action, obj in REMOVE_ACTIONS.items():
            if obj in self.state.objects_on_detector:
                valid.append(action)
        valid.append(Action.CHECK)
        return valid

    def step(self, action: Action) -> Observation:
        """Execute an action and return the observation.

        Returns:
            Observation with action result and detector state.

        Raises:
            RuntimeError: If the episode is already done.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        previous_active = self.state.detector_active
        action_valid = True
        message = ""

        if action in PLACE_ACTIONS:
            obj = PLACE_ACTIONS[action]
            if obj in self.state.objects_on_detector:
                action_valid = False
                message = f"Object {obj} is already on the detector."
            else:
                self.state.objects_on_detector.add(obj)
                message = f"You placed object {obj} on the detector."

        elif action in REMOVE_ACTIONS:
            obj = REMOVE_ACTIONS[action]
            if obj not in self.state.objects_on_detector:
                action_valid = False
                message = f"Object {obj} is not on the detector."
            else:
                self.state.objects_on_detector.discard(obj)
                message = f"You removed object {obj} from the detector."

        elif action == Action.CHECK:
            message = "You decided to check your answer."
            self._done = True

        # Update detector state
        new_active = self._compute_activation()
        self.state.detector_active = new_active
        detector_changed = new_active != previous_active

        if action != Action.CHECK:
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
        """Validate the agent's proposed rule against the true rule.

        Args:
            rule_type: Agent's guess for the rule type.
            required_objects: Agent's guess for which objects are blickets.
            guessed_order: Agent's guess for required placement order (ignored in base).

        Returns:
            Dict with correctness details.
        """
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
        }
