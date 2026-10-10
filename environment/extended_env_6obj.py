"""Extended 6-object blicket detector environment with non-standard causal conditions.

Extends BlicketEnv6 with order-sensitivity, stochasticity, and hidden moderators.
Reuses condition types from extended_env but operates on 6-object action space.
"""

from __future__ import annotations

import numpy as np

from typing import Any

from .blicket_env import CausalRule, CausalRuleType, Observation
from .blicket_env_6obj import (
    Action6,
    BlicketEnv6,
    PLACE_ACTIONS_6,
    REMOVE_ACTIONS_6,
)
from .extended_env import (
    ExtendedCondition,
    HiddenModeratorConfig,
    OrderConfig,
    StochasticConfig,
)


class ExtendedBlicketEnv6(BlicketEnv6):
    """6-object blicket environment with extended causal conditions.

    Supports three extensions beyond standard conjunctive/disjunctive:
    - Order-sensitive: objects must be placed in a specific sequence
    - Stochastic: detector activates probabilistically when rule is satisfied
    - Hidden moderator: causal rule switches after N successful activations
    """

    def __init__(
        self,
        causal_rule: CausalRule,
        condition: ExtendedCondition,
        max_steps: int = 75,
        order_config: OrderConfig | None = None,
        stochastic_config: StochasticConfig | None = None,
        moderator_config: HiddenModeratorConfig | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(causal_rule=causal_rule, max_steps=max_steps)
        self.condition = condition
        self.order_config = order_config
        self.stochastic_config = stochastic_config
        self.moderator_config = moderator_config
        self._rng = np.random.default_rng(rng_seed)

        # Order-sensitive state
        self._placement_order: list[str] = []

        # Hidden moderator state
        self._successful_activations: int = 0
        self._rule_switched: bool = False
        self._original_rule = causal_rule

        self._validate_config()

    def _validate_config(self) -> None:
        if self.condition == ExtendedCondition.ORDER_SENSITIVE:
            if not self.order_config:
                raise ValueError("OrderConfig required for ORDER_SENSITIVE condition")
        elif self.condition == ExtendedCondition.STOCHASTIC:
            if not self.stochastic_config:
                raise ValueError(
                    "StochasticConfig required for STOCHASTIC condition"
                )
        elif self.condition == ExtendedCondition.ORDER_STOCHASTIC:
            if not self.order_config:
                raise ValueError("OrderConfig required for ORDER_STOCHASTIC condition")
            if not self.stochastic_config:
                raise ValueError("StochasticConfig required for ORDER_STOCHASTIC condition")
        elif self.condition == ExtendedCondition.HIDDEN_MODERATOR:
            if not self.moderator_config:
                raise ValueError(
                    "HiddenModeratorConfig required for HIDDEN_MODERATOR condition"
                )
            if not self.moderator_config.alternate_rule:
                raise ValueError(
                    "HiddenModeratorConfig.alternate_rule must be set"
                )

    def reset(self):
        state = super().reset()
        self._placement_order = []
        self._successful_activations = 0
        self._rule_switched = False
        self.causal_rule = self._original_rule
        return state

    def step(self, action: Action6):
        # Track placement order before calling super
        if action in PLACE_ACTIONS_6:
            obj = PLACE_ACTIONS_6[action]
            if obj not in self.state.objects_on_detector:
                self._placement_order.append(obj)
        elif action in REMOVE_ACTIONS_6:
            obj = REMOVE_ACTIONS_6[action]
            if obj in self.state.objects_on_detector:
                self._placement_order = [o for o in self._placement_order if o != obj]

        obs = super().step(action)

        # For hidden moderator: count successful activations and potentially switch
        if self.condition == ExtendedCondition.HIDDEN_MODERATOR:
            if obs.detector_active and obs.detector_changed:
                self._successful_activations += 1
                if (
                    not self._rule_switched
                    and self._successful_activations
                    >= self.moderator_config.switch_after_n
                ):
                    self._rule_switched = True
                    self.causal_rule = self.moderator_config.alternate_rule
                    # Recompute activation under new rule
                    new_active = self._compute_activation()
                    if new_active != obs.detector_active:
                        self.state.detector_active = new_active
                        obs = Observation(
                            action_taken=obs.action_taken,
                            action_valid=obs.action_valid,
                            objects_on_detector=obs.objects_on_detector,
                            detector_active=new_active,
                            detector_changed=True,
                            message=obs.message,
                        )

        return obs

    def _compute_activation(self) -> bool:
        """Compute activation with extended condition logic."""
        base_satisfied = self.causal_rule.is_satisfied(
            self.state.objects_on_detector
        )

        if self.condition == ExtendedCondition.ORDER_STOCHASTIC:
            if not base_satisfied:
                return False
            if not self._check_order():
                return False
            return self._rng.random() < self.stochastic_config.activation_probability

        elif self.condition == ExtendedCondition.ORDER_SENSITIVE:
            if not base_satisfied:
                return False
            return self._check_order()

        elif self.condition == ExtendedCondition.STOCHASTIC:
            if not base_satisfied:
                return False
            return self._rng.random() < self.stochastic_config.activation_probability

        elif self.condition == ExtendedCondition.HIDDEN_MODERATOR:
            return base_satisfied

        return base_satisfied

    def _check_order(self) -> bool:
        """Check if objects were placed in the required order."""
        required = self.order_config.required_order
        placed_required = [o for o in self._placement_order if o in required]
        return placed_required == required

    def check_answer(
        self,
        rule_type: CausalRuleType,
        required_objects: set[str],
        guessed_order: list[str] | None = None,
    ) -> dict[str, Any]:
        """Validate answer, including order for ORDER_SENSITIVE conditions."""
        result = super().check_answer(rule_type, required_objects, guessed_order)

        if self.condition == ExtendedCondition.ORDER_SENSITIVE and self.order_config:
            true_order = self.order_config.required_order
            order_correct = guessed_order == true_order if guessed_order else False
            result["order_correct"] = order_correct
            result["true_order"] = true_order
            result["guessed_order"] = guessed_order
            # Override: must get order right too
            result["correct"] = result["correct"] and order_correct

        if self.condition == ExtendedCondition.ORDER_STOCHASTIC and self.order_config:
            true_order = self.order_config.required_order
            order_correct = guessed_order == true_order if guessed_order else False
            result["order_correct"] = order_correct
            result["true_order"] = true_order
            result["guessed_order"] = guessed_order
            result["correct"] = result["correct"] and order_correct

        return result

    def get_info(self):
        info = super().get_info()
        info["condition"] = self.condition.value
        info["placement_order"] = list(self._placement_order)

        if self.condition == ExtendedCondition.ORDER_SENSITIVE and self.order_config:
            info["required_order"] = self.order_config.required_order

        if self.condition == ExtendedCondition.ORDER_STOCHASTIC:
            if self.order_config:
                info["required_order"] = self.order_config.required_order
            if self.stochastic_config:
                info["activation_probability"] = self.stochastic_config.activation_probability

        if self.condition == ExtendedCondition.HIDDEN_MODERATOR:
            info["successful_activations"] = self._successful_activations
            info["rule_switched"] = self._rule_switched
            if self._rule_switched:
                info["current_rule_type"] = self.causal_rule.rule_type.value
                info["current_required_objects"] = sorted(
                    self.causal_rule.required_objects
                )

        return info
