"""Dynamic Behaviors v5 for Run 11e — Lean DB Configuration.

Only 3 behaviors active (down from 5 in 11c):
  1. order_hypothesis — inherited from 11c unchanged (10c revision, threshold 7.0)
  2. stochasticity_hypothesis — inherited from 11c unchanged (threshold 6.0)
  3. exploration_stagnation — modified: no DIMENSION_DISCOVERY state injection,
     revised notification that prompts strategy change without referencing a
     specific state

Disabled behaviors (not included in returned list):
  - moderator_detection (rule_change_hypothesis) — fired on weak evidence in 11c,
    disrupted exploration
  - dimension_discovery (exploration_stagnation's state injection) — redundant with
    ORS subset testing + ORDER_TESTING from lean DB

The trigger wording, thresholds, and injected states for order_hypothesis and
stochasticity_hypothesis are identical to 11c per the reproducibility constraint.
"""

from __future__ import annotations

from .dynamic_behaviors import (
    DynamicBehavior,
    TriggerCondition,
)
from .dynamic_behaviors_11c import create_blicket_behaviors_11c


def create_blicket_behaviors_11e() -> list[DynamicBehavior]:
    """Create lean 3-behavior config for Run 11e.

    Returns only order_hypothesis, stochasticity_hypothesis, and a modified
    exploration_stagnation. moderator_detection and dimension_discovery are
    excluded entirely.
    """
    base_behaviors = create_blicket_behaviors_11c()

    lean = []
    for behavior in base_behaviors:
        if behavior.name == "order_hypothesis":
            lean.append(behavior)
        elif behavior.name == "stochasticity_hypothesis":
            lean.append(behavior)
        elif behavior.name == "exploration_stagnation":
            lean.append(_create_revised_exploration_stagnation())
        # Skip rule_change_hypothesis (moderator_detection) — disabled for 11e

    return lean


def _create_revised_exploration_stagnation() -> DynamicBehavior:
    """Exploration stagnation with no DIMENSION_DISCOVERY injection.

    Fires on the same stagnation trigger as the base behavior, but instead of
    injecting a new CG state, it only delivers a notification message prompting
    the agent to change strategy.

    Key differences from base exploration_stagnation:
    - modifications=[] (no DIMENSION_DISCOVERY state added to CG)
    - Revised notification message that prompts strategy change without
      referencing a specific state name
    """
    return DynamicBehavior(
        name="exploration_stagnation",
        description=(
            "Fires when the agent has explored for several steps without "
            "ever activating the detector, suggesting standard exploration "
            "strategies are failing. Run 11e revision: no state injection, "
            "notification-only to prompt strategy change."
        ),
        trigger=TriggerCondition(
            evaluation_prompt=(
                "The agent has been exploring the detector for several steps "
                "but has NEVER achieved activation. Are they stuck in an "
                "ineffective exploration pattern? Consider whether they need "
                "to think about alternative causal mechanisms beyond simple "
                "conjunctive/disjunctive rules."
            ),
            min_steps=6,
            stagnation_mode=True,
            stagnation_threshold=6,
            fire_threshold=6.0,
        ),
        modifications=[],
        notification_message=(
            "[SYSTEM] Your observations don't fit standard causal rules. "
            "Consider changing your exploration strategy: try different "
            "object subsets, vary placement order, or test whether the same "
            "configuration gives different results on repeated trials."
        ),
    )
