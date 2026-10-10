"""Dynamic Behaviors v2 for Run 10c.

Overrides the order_hypothesis DB with revised ORDER_TESTING guidelines
that implement systematic permutation elimination. All other behaviors
are inherited unchanged from the base definitions.
"""

from __future__ import annotations

from .context_graph import CGState, CGStateType, CGTransition
from .dynamic_behaviors import (
    CGModification,
    DynamicBehavior,
    TriggerCondition,
    create_blicket_behaviors,
)


def create_blicket_behaviors_10c() -> list[DynamicBehavior]:
    """Create behaviors with revised ORDER_TESTING for Run 10c.

    Returns the standard 4 behaviors, but with the order_hypothesis DB
    modified to use systematic elimination guidelines.
    """
    base_behaviors = create_blicket_behaviors()

    revised = []
    for behavior in base_behaviors:
        if behavior.name == "order_hypothesis":
            revised.append(_create_revised_order_hypothesis())
        else:
            revised.append(behavior)

    return revised


def _create_revised_order_hypothesis() -> DynamicBehavior:
    """Revised order_hypothesis DB with systematic elimination protocol."""
    return DynamicBehavior(
        name="order_hypothesis",
        description=(
            "Fires when evidence suggests something beyond object identity "
            "matters for activation — particularly placement order or sequence. "
            "Run 10c revision: adds systematic permutation elimination protocol."
        ),
        trigger=TriggerCondition(
            evaluation_prompt=(
                "Has the agent observed inconsistent or surprising results that "
                "could be explained by the ORDER or SEQUENCE of object placement "
                "mattering? For example: the same set of objects producing different "
                "results, or configurations that 'should' work based on object "
                "identity alone failing to activate."
            ),
            min_steps=8,
            fire_threshold=7.0,
        ),
        modifications=[
            CGModification(
                action="add_state",
                state=CGState(
                    name="ORDER_TESTING",
                    state_type=CGStateType.ACTION,
                    objective=(
                        "Determine the exact placement order required for activation. "
                        "This is a systematic elimination task: there are a finite number "
                        "of possible orderings, and you must test them one by one until "
                        "you find the one that activates the detector."
                    ),
                    guidelines=[
                        "STEP 1: List all objects you believe are relevant based on "
                        "the evidence so far (e.g., from environmental observations "
                        "and your own testing)",
                        "STEP 2: Calculate how many possible orderings exist. "
                        "For 3 objects there are exactly 6 orderings. "
                        "For 4 objects there are exactly 24 orderings. "
                        "Write out ALL possible orderings explicitly",
                        "STEP 3: For each test — clear the detector completely, then "
                        "place objects ONE AT A TIME in a specific sequence, then check. "
                        "Record the result: 'Tested [X, Y, Z] → INACTIVE' or '→ ACTIVE'",
                        "STEP 4: After each failed test, cross that ordering off your list. "
                        "Choose the NEXT UNTESTED ordering from your list. Never repeat "
                        "an ordering you have already tested",
                        "STEP 5: Continue until you find the activating order or exhaust "
                        "all possibilities. If all orderings fail, your object set may "
                        "be wrong — go back and reconsider which objects are relevant",
                        "IMPORTANT: Do NOT guess based on alphabetical order, intuition, "
                        "or pattern-matching. The correct order could be ANY permutation. "
                        "Only an ordering you have TESTED and seen ACTIVATE is confirmed",
                        "IMPORTANT: Work through orderings methodically. A good strategy "
                        "is to fix the first object and cycle through all arrangements of "
                        "the remaining objects before moving to the next first object",
                    ],
                    transitions=[
                        CGTransition(
                            target="VERIFICATION",
                            condition="ONLY after you have tested an ordering that "
                            "produced ACTIVE — submit that specific ordering",
                        ),
                        CGTransition(
                            target="COMBINATION_TESTING",
                            condition="If ALL orderings of your current object set fail, "
                            "your object set is wrong — go back and revise which objects "
                            "are in the rule",
                        ),
                    ],
                ),
                connect_from=["DIMENSION_DISCOVERY", "COMBINATION_TESTING"],
            ),
        ],
        notification_message=(
            "[SYSTEM] Evidence suggests placement order may matter. "
            "A new ORDER_TESTING state is available for systematic order testing. "
            "There are a finite number of possible orderings — test them "
            "methodically to find the correct sequence."
        ),
    )
