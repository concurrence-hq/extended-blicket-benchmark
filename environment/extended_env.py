"""Extended blicket detector environment with non-standard causal conditions.

Extends the base environment with three conditions that challenge standard
causal reasoning: order-sensitivity, stochasticity, and hidden moderators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .blicket_env import BlicketEnv, CausalRule, Action, Observation, PLACE_ACTIONS, REMOVE_ACTIONS


class ExtendedCondition(Enum):
    ORDER_SENSITIVE = "order_sensitive"
    STOCHASTIC = "stochastic"
    HIDDEN_MODERATOR = "hidden_moderator"
    ORDER_STOCHASTIC = "order_stochastic"


@dataclass
class OrderConfig:
    """Configuration for order-sensitive condition."""

    required_order: list[str]  # e.g. ["A", "B"] means A must be placed before B


@dataclass
class StochasticConfig:
    """Configuration for stochastic condition."""

    activation_probability: float = 0.7  # P(activate | rule satisfied)


@dataclass
class HiddenModeratorConfig:
    """Configuration for hidden moderator condition."""

    switch_after_n: int = 3  # Switch rule after N successful activations
    alternate_rule: CausalRule | None = None  # Rule to switch to


class ExtendedBlicketEnv(BlicketEnv):
    """Blicket environment with extended causal conditions.

    Supports three extensions beyond standard conjunctive/disjunctive:
    - Order-sensitive: objects must be placed in a specific sequence
    - Stochastic: detector activates probabilistically when rule is satisfied
    - Hidden moderator: causal rule switches after N successful activations
    """

    def __init__(
        self,
        causal_rule: CausalRule,
        condition: ExtendedCondition,
        max_steps: int = 50,
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

        # Stochastic state (no extra state needed beyond RNG)

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

    def step(self, action: Action):
        # Track placement order before calling super
        if action in PLACE_ACTIONS:
            obj = PLACE_ACTIONS[action]
            if obj not in self.state.objects_on_detector:
                self._placement_order.append(obj)
        elif action in REMOVE_ACTIONS:
            obj = REMOVE_ACTIONS[action]
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
        # Extract the subsequence of required objects from placement order
        placed_required = [o for o in self._placement_order if o in required]
        return placed_required == required

    def get_info(self):
        info = super().get_info()
        info["condition"] = self.condition.value
        info["placement_order"] = list(self._placement_order)

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
