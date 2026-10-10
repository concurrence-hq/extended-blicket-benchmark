"""Dynamic Behaviors v3 for Run 11b.

Overrides the stochasticity_hypothesis DB with a revised version that:
1. Expands connect_from to all 5 states the agent might occupy when it fires
2. Adds VERIFICATION exit transition from RELIABILITY_TESTING
3. Improves guidelines for explicit probability calculation

Root cause fix: The base stochasticity_hypothesis uses
connect_from=["DIMENSION_DISCOVERY", "COMBINATION_TESTING"]. When it fires
at step ~22, the agent is often in ORDER_TESTING or VERIFICATION — no
transition edge exists to RELIABILITY_TESTING. This revision adds all 5
reachable states.

All other behaviors are inherited unchanged from 10c.
"""

from __future__ import annotations

from .context_graph import CGState, CGStateType, CGTransition
from .dynamic_behaviors import (
    CGModification,
    DynamicBehavior,
    TriggerCondition,
)
from .dynamic_behaviors_10c import create_blicket_behaviors_10c


def create_blicket_behaviors_11b() -> list[DynamicBehavior]:
    """Create behaviors with revised stochasticity_hypothesis for Run 11b.

    Returns the standard 4 behaviors (with 10c's revised order_hypothesis),
    but with stochasticity_hypothesis modified for reachability.
    """
    base_behaviors = create_blicket_behaviors_10c()

    revised = []
    for behavior in base_behaviors:
        if behavior.name == "stochasticity_hypothesis":
            revised.append(_create_revised_stochasticity_hypothesis())
        else:
            revised.append(behavior)

    return revised


def _create_revised_stochasticity_hypothesis() -> DynamicBehavior:
    """Revised stochasticity_hypothesis with full reachability.

    Key changes from base:
    - connect_from expanded to all 5 states agent might occupy
    - RELIABILITY_TESTING gets two exit transitions (VERIFICATION + HYPOTHESIS_EVALUATION)
    - Guidelines are more directive about repeat-testing and probability calculation
    - Notification text explicitly directs transition to RELIABILITY_TESTING
    """
    return DynamicBehavior(
        name="stochasticity_hypothesis",
        description=(
            "Fires when the same configuration gives inconsistent results. "
            "Run 11b revision: RELIABILITY_TESTING reachable from all 5 states, "
            "two exit transitions, and improved repeat-testing guidelines."
        ),
        trigger=TriggerCondition(
            evaluation_prompt=(
                "Has the agent tried the EXACT same configuration of objects "
                "(same objects on detector) MORE THAN ONCE and gotten DIFFERENT "
                "results? This would suggest probabilistic/stochastic activation "
                "rather than a deterministic rule."
            ),
            min_steps=8,
            fire_threshold=7.0,
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
