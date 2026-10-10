"""Dynamic Behaviors v4 for Run 11c.

Lowers the stochasticity_hypothesis fire_threshold from 7.0 → 6.0 so the
behavior fires more readily.  All other behaviors are inherited unchanged
from 11b.

Change F: Lower threshold makes stochasticity detection more sensitive,
complementing Change E (v2 CG with OBJECT_RELEVANCE_SCREENING).
"""

from __future__ import annotations

from .context_graph import CGState, CGStateType, CGTransition
from .dynamic_behaviors import (
    CGModification,
    DynamicBehavior,
    TriggerCondition,
)
from .dynamic_behaviors_11b import create_blicket_behaviors_11b


def create_blicket_behaviors_11c() -> list[DynamicBehavior]:
    """Create behaviors with lowered stochasticity threshold for Run 11c.

    Returns the standard 4 behaviors (with 11b's revised stochasticity_hypothesis),
    but with stochasticity_hypothesis fire_threshold lowered from 7.0 to 6.0.
    """
    base_behaviors = create_blicket_behaviors_11b()

    revised = []
    for behavior in base_behaviors:
        if behavior.name == "stochasticity_hypothesis":
            revised.append(_create_revised_stochasticity_hypothesis())
        else:
            revised.append(behavior)

    return revised


def _create_revised_stochasticity_hypothesis() -> DynamicBehavior:
    """Stochasticity_hypothesis with lowered fire_threshold (6.0 vs 7.0).

    Identical to 11b revision except fire_threshold=6.0.
    """
    return DynamicBehavior(
        name="stochasticity_hypothesis",
        description=(
            "Fires when the same configuration gives inconsistent results. "
            "Run 11c revision: fire_threshold lowered from 7.0 to 6.0 for "
            "earlier detection of probabilistic activation."
        ),
        trigger=TriggerCondition(
            evaluation_prompt=(
                "Has the agent tried the EXACT same configuration of objects "
                "(same objects on detector) MORE THAN ONCE and gotten DIFFERENT "
                "results? This would suggest probabilistic/stochastic activation "
                "rather than a deterministic rule."
            ),
            min_steps=8,
            fire_threshold=6.0,
        ),
        modifications=[
            CGModification(
                action="add_state",
                state=CGState(
                    name="RELIABILITY_TESTING",
                    state_type=CGStateType.ACTION,
                    objective=(
                        "Test whether activation is probabilistic by repeating "
                        "the SAME configuration multiple times. Pick your best "
                        "candidate configuration and test it at least 5 times "
                        "to estimate the activation probability."
                    ),
                    guidelines=[
                        "Pick the configuration most likely to activate based on "
                        "your evidence so far (your best candidate)",
                        "Clear the detector completely before each repetition",
                        "Place the EXACT same objects in the EXACT same order each time",
                        "Repeat at least 5 times — more is better for probability estimates",
                        "Count: how many activations out of how many trials?",
                        "Calculate the activation probability explicitly: "
                        "P(activate) = activations / trials",
                        "If P = 0 after 5+ trials, this configuration is likely wrong — "
                        "try a different candidate",
                    ],
                    transitions=[
                        CGTransition(
                            target="VERIFICATION",
                            condition=(
                                "After 5+ repetitions with a clear probability estimate — "
                                "submit your answer incorporating the stochastic element"
                            ),
                        ),
                        CGTransition(
                            target="HYPOTHESIS_EVALUATION",
                            condition=(
                                "If repeated testing reveals the rule is actually "
                                "deterministic, or if you need to reconsider which "
                                "configuration to test"
                            ),
                        ),
                    ],
                ),
                connect_from=[
                    "DIMENSION_DISCOVERY",
                    "COMBINATION_TESTING",
                    "VERIFICATION",
                    "HYPOTHESIS_EVALUATION",
                    "ORDER_TESTING",
                ],
            ),
        ],
        notification_message=(
            "[SYSTEM] Inconsistent results detected — the same configuration "
            "produced different outcomes. This suggests PROBABILISTIC activation. "
            "Transition to RELIABILITY_TESTING to repeat your best candidate "
            "configuration 5+ times and estimate the activation probability."
        ),
    )
