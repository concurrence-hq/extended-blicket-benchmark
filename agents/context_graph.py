"""Context Graph data structure for the compositional agent architecture.

A directed graph of states with typed nodes (ACTION, DECISION, REFLECTION),
each carrying objectives, guidelines, available tools, and transitions.
The CG is injected into the agent's system prompt each turn.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger(__name__)


class CGStateType(Enum):
    ACTION = "action"
    DECISION = "decision"
    REFLECTION = "reflection"


class CGTransition(BaseModel):
    """A transition edge from the parent state to a target state."""

    target: str
    condition: str  # Human-readable condition description


class CGState(BaseModel):
    """A single node in the Context Graph."""

    name: str
    state_type: CGStateType
    objective: str
    guidelines: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    transitions: list[CGTransition] = Field(default_factory=list)


class ModificationLogEntry(BaseModel):
    """Records a runtime modification to the Context Graph."""

    timestamp: str
    step_number: int
    action: str  # "add_state", "add_transition", "modify_guidelines"
    details: dict[str, Any] = Field(default_factory=dict)


class ContextGraph(BaseModel):
    """Context Graph state machine.

    Manages a set of named states with transitions. The current state
    determines the objectives and guidelines injected into the LLM prompt.
    """

    states: dict[str, CGState] = Field(default_factory=dict)
    current_state: str = ""
    modification_log: list[ModificationLogEntry] = Field(default_factory=list)
    steps_in_current_state: int = 0
    initial_exploration_max_steps: int | None = None
    object_screening_max_steps: int | None = None
    _backstop_fired: bool = PrivateAttr(default=False)
    _screening_backstop_fired: bool = PrivateAttr(default=False)

    def transition_to(self, target_state: str) -> bool:
        """Attempt to transition to a target state.

        Returns True if transition was valid and executed, False otherwise.
        """
        if target_state not in self.states:
            return False

        current = self.states.get(self.current_state)
        if current is None:
            return False

        # Check if transition is allowed
        valid_targets = {t.target for t in current.transitions}
        if target_state not in valid_targets:
            return False

        self.current_state = target_state
        self.steps_in_current_state = 0
        return True

    def advance_step(self) -> tuple[str, str] | None:
        """Increment step counter and fire backstop if threshold reached.

        Returns ``(from_state, target_state)`` when a backstop fires, or
        ``None`` otherwise.

        Two backstops are supported:

        1. **INITIAL_EXPLORATION** backstop — fires after
           ``initial_exploration_max_steps`` steps.  In v2/v3 graphs (which
           contain OBJECT_RELEVANCE_SCREENING) the target is
           OBJECT_RELEVANCE_SCREENING; in v1 graphs the target is
           COMBINATION_TESTING.
        2. **OBJECT_RELEVANCE_SCREENING** backstop (v2/v3) — fires after
           ``object_screening_max_steps`` steps, transitioning to
           COMBINATION_TESTING.

        Each backstop fires at most once per episode.
        """
        self.steps_in_current_state += 1

        # Backstop 1: INITIAL_EXPLORATION →
        #   OBJECT_RELEVANCE_SCREENING (v2) or COMBINATION_TESTING (v1)
        if (
            self.initial_exploration_max_steps is not None
            and self.current_state == "INITIAL_EXPLORATION"
            and self.steps_in_current_state >= self.initial_exploration_max_steps
            and not self._backstop_fired
        ):
            target = (
                "OBJECT_RELEVANCE_SCREENING"
                if "OBJECT_RELEVANCE_SCREENING" in self.states
                else "COMBINATION_TESTING"
            )
            if target not in self.states:
                return None
            self._backstop_fired = True
            self.current_state = target
            self.steps_in_current_state = 0
            logger.info(
                f"Backstop fired: INITIAL_EXPLORATION -> {target} "
                f"after {self.initial_exploration_max_steps} steps"
            )
            return ("INITIAL_EXPLORATION", target)

        # Backstop 2: OBJECT_RELEVANCE_SCREENING → COMBINATION_TESTING (v2 only)
        if (
            self.object_screening_max_steps is not None
            and self.current_state == "OBJECT_RELEVANCE_SCREENING"
            and self.steps_in_current_state >= self.object_screening_max_steps
            and not self._screening_backstop_fired
        ):
            target = "COMBINATION_TESTING"
            if target not in self.states:
                return None
            self._screening_backstop_fired = True
            self.current_state = target
            self.steps_in_current_state = 0
            logger.info(
                f"Backstop fired: OBJECT_RELEVANCE_SCREENING -> {target} "
                f"after {self.object_screening_max_steps} steps"
            )
            return ("OBJECT_RELEVANCE_SCREENING", target)

        return None

    def add_state(
        self, state: CGState, connect_from: list[str] | None = None, step_number: int = 0
    ) -> None:
        """Add a new state to the graph, optionally connecting from existing states."""
        self.states[state.name] = state

        if connect_from:
            for source_name in connect_from:
                if source_name in self.states:
                    source = self.states[source_name]
                    # Avoid duplicate transitions
                    existing_targets = {t.target for t in source.transitions}
                    if state.name not in existing_targets:
                        source.transitions.append(
                            CGTransition(
                                target=state.name,
                                condition=f"Dynamic: transition to {state.name}",
                            )
                        )

        self.modification_log.append(
            ModificationLogEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                step_number=step_number,
                action="add_state",
                details={"state_name": state.name, "connected_from": connect_from or []},
            )
        )

    def modify_guidelines(
        self, state_name: str, add_guidelines: list[str], step_number: int = 0
    ) -> bool:
        """Add guidelines to an existing state."""
        if state_name not in self.states:
            return False

        self.states[state_name].guidelines.extend(add_guidelines)
        self.modification_log.append(
            ModificationLogEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                step_number=step_number,
                action="modify_guidelines",
                details={"state_name": state_name, "added": add_guidelines},
            )
        )
        return True

    def expose_tools(
        self, state_name: str, tools: list[str], step_number: int = 0
    ) -> bool:
        """Expose additional tools in a state."""
        if state_name not in self.states:
            return False

        self.states[state_name].tools.extend(tools)
        self.modification_log.append(
            ModificationLogEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                step_number=step_number,
                action="expose_tools",
                details={"state_name": state_name, "tools": tools},
            )
        )
        return True

    def render_for_prompt(self) -> str:
        """Render the CG for injection into the system prompt."""
        lines = []
        lines.append("=== Context Graph ===")

        current = self.states.get(self.current_state)
        if not current:
            lines.append("(No active state)")
            return "\n".join(lines)

        # Current state details
        lines.append(f"Current State: {current.name} [{current.state_type.value}]")
        lines.append(f"Objective: {current.objective}")

        if current.guidelines:
            lines.append("Guidelines:")
            for g in current.guidelines:
                lines.append(f"  - {g}")

        if current.tools:
            lines.append(f"Available tools: {', '.join(current.tools)}")

        # Available transitions
        if current.transitions:
            lines.append("Transitions:")
            for t in current.transitions:
                lines.append(f"  -> {t.target}: {t.condition}")

        # Graph overview (condensed)
        lines.append("")
        lines.append("Graph overview:")
        for name, state in self.states.items():
            marker = " <-- YOU ARE HERE" if name == self.current_state else ""
            targets = ", ".join(t.target for t in state.transitions)
            lines.append(f"  [{state.state_type.value}] {name} -> {targets}{marker}")

        # Recent modifications
        if self.modification_log:
            recent = self.modification_log[-3:]  # Last 3 modifications
            lines.append("")
            lines.append("Recent CG modifications:")
            for entry in recent:
                lines.append(f"  Step {entry.step_number}: {entry.action} - {entry.details}")

        return "\n".join(lines)


def create_blicket_cg(
    num_objects: int = 3,
    cg_version: str = "v1",
    object_list: list[str] | None = None,
) -> ContextGraph:
    """Create the Context Graph for the blicket task.

    Args:
        num_objects: Number of objects in the environment (3 or 5).
        cg_version: CG version string:
            - ``"v1"`` (default, 4-state)
            - ``"v2"`` (5-state with OBJECT_RELEVANCE_SCREENING)
            - ``"v3"`` (5-state like v2 but with revised ORS guidelines)
            - ``"v4"`` (5-state with subset-testing ORS for Run 11e)
        object_list: Labels for objects (e.g. ``["A","B","C","D","E"]``).
            Used in v2/v3 guidelines.  Defaults to first *num_objects* letters.

    Raises:
        ValueError: If *cg_version* is not a recognized version.
    """
    valid_versions = ("v1", "v2", "v3", "v4")
    if cg_version not in valid_versions:
        raise ValueError(
            f"Invalid cg_version: {cg_version!r}. "
            f"Must be one of {valid_versions}."
        )

    if object_list is None:
        object_list = [chr(ord("A") + i) for i in range(num_objects)]

    if cg_version == "v1":
        return _create_blicket_cg_v1(num_objects, object_list)
    elif cg_version == "v2":
        return _create_blicket_cg_v2(num_objects, object_list)
    elif cg_version == "v3":
        return _create_blicket_cg_v3(num_objects, object_list)
    else:
        return _create_blicket_cg_v4(num_objects, object_list)


def _create_blicket_cg_v1(num_objects: int, object_list: list[str]) -> ContextGraph:
    """Create the 4-state v1 Context Graph.

    States:
        INITIAL_EXPLORATION -> COMBINATION_TESTING -> HYPOTHESIS_EVALUATION -> VERIFICATION
    """
    exploration_guidelines = [
        "Observe detector responses as you place and remove objects",
        "Pay attention to what changes when each object is added or removed",
        "Notice any patterns in activation timing or sequence",
    ]
    if num_objects >= 5:
        exploration_guidelines.append(
            "With more objects, prioritize efficient exploration — "
            "focus on narrowing down which objects matter"
        )

    states = {
        "INITIAL_EXPLORATION": CGState(
            name="INITIAL_EXPLORATION",
            state_type=CGStateType.ACTION,
            objective="Understand how individual objects affect the detector.",
            guidelines=exploration_guidelines,
            transitions=[
                CGTransition(
                    target="COMBINATION_TESTING",
                    condition=f"After testing all {num_objects} individual objects",
                ),
            ],
        ),
        "COMBINATION_TESTING": CGState(
            name="COMBINATION_TESTING",
            state_type=CGStateType.ACTION,
            objective="Discover how object combinations and arrangements affect activation.",
            guidelines=[
                "Test different groupings of objects on the detector",
                "Vary not just which objects are present, but how you arrange and sequence them",
                "Compare results across different configurations",
                "If a configuration that should work doesn't, consider what else might matter",
            ],
            transitions=[
                CGTransition(
                    target="HYPOTHESIS_EVALUATION",
                    condition="After testing key combinations",
                ),
                CGTransition(
                    target="INITIAL_EXPLORATION",
                    condition="If results are confusing, go back to singles",
                ),
            ],
        ),
        "HYPOTHESIS_EVALUATION": CGState(
            name="HYPOTHESIS_EVALUATION",
            state_type=CGStateType.REFLECTION,
            objective="Evaluate your evidence to determine the causal rule.",
            guidelines=[
                "What rule type best explains ALL your observations?",
                "Which specific objects are blickets?",
                "Are there observations that don't fit your current hypothesis?",
                "If observations are inconsistent, consider whether factors beyond object identity might matter",
            ],
            transitions=[
                CGTransition(
                    target="VERIFICATION",
                    condition="When you have a clear hypothesis",
                ),
                CGTransition(
                    target="COMBINATION_TESTING",
                    condition="If hypothesis is unclear, test more combinations",
                ),
            ],
        ),
        "VERIFICATION": CGState(
            name="VERIFICATION",
            state_type=CGStateType.DECISION,
            objective="Confirm your hypothesis with targeted tests before submitting.",
            guidelines=[
                "Design a test that would distinguish your hypothesis from the next most likely alternative",
                "If the test confirms, submit your answer",
                "If the test disconfirms, reconsider your hypothesis",
            ],
            transitions=[
                CGTransition(
                    target="HYPOTHESIS_EVALUATION",
                    condition="If verification fails",
                ),
            ],
        ),
    }

    return ContextGraph(
        states=states,
        current_state="INITIAL_EXPLORATION",
    )


def _create_blicket_cg_v2(num_objects: int, object_list: list[str]) -> ContextGraph:
    """Create the 5-state v2 Context Graph.

    Inserts OBJECT_RELEVANCE_SCREENING between INITIAL_EXPLORATION and
    COMBINATION_TESTING.

    States:
        INITIAL_EXPLORATION -> OBJECT_RELEVANCE_SCREENING ->
        COMBINATION_TESTING -> HYPOTHESIS_EVALUATION -> VERIFICATION
    """
    labels_str = ", ".join(object_list)

    # v2 INITIAL_EXPLORATION: focused on individual object testing
    exploration_guidelines = [
        f"Test each object individually: place {labels_str} on the detector one at a time",
        "Record whether each individual object activates the detector",
        "Remove the object before testing the next one to get clean observations",
        "Pay attention to any inconsistencies when testing the same object multiple times",
    ]

    # v2 OBJECT_RELEVANCE_SCREENING: narrow down which objects matter
    screening_guidelines = [
        "Test pairs and triples of objects to find which combinations activate the detector",
        "When you find an activation, try removing one object at a time to see which are essential",
        "Identify distractor objects that don't contribute to activation",
        "Build a shortlist of objects that appear to be involved in the rule",
    ]

    states = {
        "INITIAL_EXPLORATION": CGState(
            name="INITIAL_EXPLORATION",
            state_type=CGStateType.ACTION,
            objective="Observe how individual objects affect the detector.",
            guidelines=exploration_guidelines,
            transitions=[
                CGTransition(
                    target="OBJECT_RELEVANCE_SCREENING",
                    condition=f"After testing all {num_objects} individual objects",
                ),
            ],
        ),
        "OBJECT_RELEVANCE_SCREENING": CGState(
            name="OBJECT_RELEVANCE_SCREENING",
            state_type=CGStateType.ACTION,
            objective="Narrow down which objects are involved in the rule.",
            guidelines=screening_guidelines,
            transitions=[
                CGTransition(
                    target="COMBINATION_TESTING",
                    condition="After identifying which objects appear relevant",
                ),
                CGTransition(
                    target="INITIAL_EXPLORATION",
                    condition="If results are confusing, go back to individual tests",
                ),
            ],
        ),
        "COMBINATION_TESTING": CGState(
            name="COMBINATION_TESTING",
            state_type=CGStateType.ACTION,
            objective="Discover how object combinations and arrangements affect activation.",
            guidelines=[
                "Test different groupings of objects on the detector",
                "Vary not just which objects are present, but how you arrange and sequence them",
                "Compare results across different configurations",
                "If a configuration that should work doesn't, consider what else might matter",
            ],
            transitions=[
                CGTransition(
                    target="HYPOTHESIS_EVALUATION",
                    condition="After testing key combinations",
                ),
                CGTransition(
                    target="OBJECT_RELEVANCE_SCREENING",
                    condition="If results are confusing, go back to screening",
                ),
            ],
        ),
        "HYPOTHESIS_EVALUATION": CGState(
            name="HYPOTHESIS_EVALUATION",
            state_type=CGStateType.REFLECTION,
            objective="Evaluate your evidence to determine the causal rule.",
            guidelines=[
                "What rule type best explains ALL your observations?",
                "Which specific objects are blickets?",
                "Are there observations that don't fit your current hypothesis?",
                "If observations are inconsistent, consider whether factors beyond object identity might matter",
            ],
            transitions=[
                CGTransition(
                    target="VERIFICATION",
                    condition="When you have a clear hypothesis",
                ),
                CGTransition(
                    target="COMBINATION_TESTING",
                    condition="If hypothesis is unclear, test more combinations",
                ),
            ],
        ),
        "VERIFICATION": CGState(
            name="VERIFICATION",
            state_type=CGStateType.DECISION,
            objective="Confirm your hypothesis with targeted tests before submitting.",
            guidelines=[
                "Design a test that would distinguish your hypothesis from the next most likely alternative",
                "If the test confirms, submit your answer",
                "If the test disconfirms, reconsider your hypothesis",
            ],
            transitions=[
                CGTransition(
                    target="HYPOTHESIS_EVALUATION",
                    condition="If verification fails",
                ),
            ],
        ),
    }

    return ContextGraph(
        states=states,
        current_state="INITIAL_EXPLORATION",
    )


# ---------------------------------------------------------------------------
# v3 variants
# ---------------------------------------------------------------------------


def _create_blicket_cg_v3(num_objects: int, object_list: list[str]) -> ContextGraph:
    """Create the 5-state v3 Context Graph.

    Same topology as v2 (IE → ORS → CT → HE → V) but with revised ORS
    guidelines: explicit elimination language, scorecard framing, and
    individual-first protocol.
    """
    labels_str = ", ".join(object_list)

    exploration_guidelines = [
        f"Test each object individually: place {labels_str} on the detector one at a time",
        "Record whether each individual object activates the detector",
        "Remove the object before testing the next one to get clean observations",
        "Pay attention to any inconsistencies when testing the same object multiple times",
    ]

    screening_guidelines = [
        f"Test each object individually first — place {labels_str} on the detector one at a time and record the result",
        "Track a scorecard: for each object note ACTIVATES / DOES NOT ACTIVATE / INCONCLUSIVE",
        "Test pairs that mix suspected relevant and irrelevant objects to confirm or ELIMINATE candidates",
        "Once you have eliminated at least 1 object OR found 2+ activations, move on to COMBINATION_TESTING",
        "Do NOT test the same pair or group twice in a row — each test should give you new information",
    ]

    states = {
        "INITIAL_EXPLORATION": CGState(
            name="INITIAL_EXPLORATION",
            state_type=CGStateType.ACTION,
            objective="Observe how individual objects affect the detector.",
            guidelines=exploration_guidelines,
            transitions=[
                CGTransition(
                    target="OBJECT_RELEVANCE_SCREENING",
                    condition=f"After testing all {num_objects} individual objects",
                ),
            ],
        ),
        "OBJECT_RELEVANCE_SCREENING": CGState(
            name="OBJECT_RELEVANCE_SCREENING",
            state_type=CGStateType.ACTION,
            objective="Narrow down which objects are involved in the rule by systematic elimination.",
            guidelines=screening_guidelines,
            transitions=[
                CGTransition(
                    target="COMBINATION_TESTING",
                    condition="After eliminating at least 1 object or finding 2+ activations",
                ),
                CGTransition(
                    target="INITIAL_EXPLORATION",
                    condition="If results are confusing, go back to individual tests",
                ),
            ],
        ),
        "COMBINATION_TESTING": CGState(
            name="COMBINATION_TESTING",
            state_type=CGStateType.ACTION,
            objective="Discover how object combinations and arrangements affect activation.",
            guidelines=[
                "Test different groupings of objects on the detector",
                "Vary not just which objects are present, but how you arrange and sequence them",
                "Compare results across different configurations",
                "If a configuration that should work doesn't, consider what else might matter",
            ],
            transitions=[
                CGTransition(
                    target="HYPOTHESIS_EVALUATION",
                    condition="After testing key combinations",
                ),
                CGTransition(
                    target="OBJECT_RELEVANCE_SCREENING",
                    condition="If results are confusing, go back to screening",
                ),
            ],
        ),
        "HYPOTHESIS_EVALUATION": CGState(
            name="HYPOTHESIS_EVALUATION",
            state_type=CGStateType.REFLECTION,
            objective="Evaluate your evidence to determine the causal rule.",
            guidelines=[
                "What rule type best explains ALL your observations?",
                "Which specific objects are blickets?",
                "Are there observations that don't fit your current hypothesis?",
                "If observations are inconsistent, consider whether factors beyond object identity might matter",
            ],
            transitions=[
                CGTransition(
                    target="VERIFICATION",
                    condition="When you have a clear hypothesis",
                ),
                CGTransition(
                    target="COMBINATION_TESTING",
                    condition="If hypothesis is unclear, test more combinations",
                ),
            ],
        ),
        "VERIFICATION": CGState(
            name="VERIFICATION",
            state_type=CGStateType.DECISION,
            objective="Confirm your hypothesis with targeted tests before submitting.",
            guidelines=[
                "Design a test that would distinguish your hypothesis from the next most likely alternative",
                "If the test confirms, submit your answer",
                "If the test disconfirms, reconsider your hypothesis",
            ],
            transitions=[
                CGTransition(
                    target="HYPOTHESIS_EVALUATION",
                    condition="If verification fails",
                ),
            ],
        ),
    }

    return ContextGraph(
        states=states,
        current_state="INITIAL_EXPLORATION",
    )


# ---------------------------------------------------------------------------
# v4 — subset-testing ORS (Run 11e)
# ---------------------------------------------------------------------------


def _create_blicket_cg_v4(num_objects: int, object_list: list[str]) -> ContextGraph:
    """Create the 5-state v4 Context Graph.

    Same topology as v2/v3 (IE -> ORS -> CT -> HE -> V) but with redesigned ORS
    guidelines: subset testing (2-3 objects), order variation, per-object
    scorecard, and small-subset enforcement.

    Key changes from v3:
    - Subset-first, not individual-first
    - Order variation within subsets
    - Scorecard tracks per-object participation in activating vs non-activating subsets
    - Explicit "note the order" instruction on activation
    - No elimination language
    - Small-subset enforcement ("Do NOT place all objects at once")
    """
    labels_str = ", ".join(object_list)

    exploration_guidelines = [
        f"Test each object individually: place {labels_str} on the detector one at a time",
        "Record whether each individual object activates the detector",
        "Remove the object before testing the next one to get clean observations",
        "Pay attention to any inconsistencies when testing the same object multiple times",
    ]

    screening_guidelines = [
        "Test subsets of 2\u20133 objects at a time. Place them on the detector "
        "in varied orders and observe whether the detector activates.",
        "Track a scorecard: for each object, count how many times it appeared "
        "in a subset that activated vs. subsets that did not activate.",
        "Try at least 3 different subsets before moving on. Vary which objects "
        "you include and the order you place them.",
        "If you observe any activation, note exactly which objects were present "
        "and in what order they were placed. This is critical information.",
        "If after testing several subsets you have zero activations, note which "
        "objects appeared most frequently in your tests \u2014 you may need to try "
        "different orders or combinations in the next phase.",
        "Do NOT place all objects at once. Keep subsets small (2\u20133) so you can "
        "identify which objects are contributing to activations.",
    ]

    states = {
        "INITIAL_EXPLORATION": CGState(
            name="INITIAL_EXPLORATION",
            state_type=CGStateType.ACTION,
            objective="Observe how individual objects affect the detector.",
            guidelines=exploration_guidelines,
            transitions=[
                CGTransition(
                    target="OBJECT_RELEVANCE_SCREENING",
                    condition=f"After testing all {num_objects} individual objects",
                ),
            ],
        ),
        "OBJECT_RELEVANCE_SCREENING": CGState(
            name="OBJECT_RELEVANCE_SCREENING",
            state_type=CGStateType.ACTION,
            objective="Test small subsets of objects to find which ones participate in activations.",
            guidelines=screening_guidelines,
            transitions=[
                CGTransition(
                    target="COMBINATION_TESTING",
                    condition=(
                        "After observing at least 1 activation, "
                        "or after testing 3+ subsets with zero activations"
                    ),
                ),
                CGTransition(
                    target="INITIAL_EXPLORATION",
                    condition="If results are confusing, go back to individual tests",
                ),
            ],
        ),
        "COMBINATION_TESTING": CGState(
            name="COMBINATION_TESTING",
            state_type=CGStateType.ACTION,
            objective="Discover how object combinations and arrangements affect activation.",
            guidelines=[
                "Test different groupings of objects on the detector",
                "Vary not just which objects are present, but how you arrange and sequence them",
                "Compare results across different configurations",
                "If a configuration that should work doesn't, consider what else might matter",
            ],
            transitions=[
                CGTransition(
                    target="HYPOTHESIS_EVALUATION",
                    condition="After testing key combinations",
                ),
                CGTransition(
                    target="OBJECT_RELEVANCE_SCREENING",
                    condition="If results are confusing, go back to screening",
                ),
            ],
        ),
        "HYPOTHESIS_EVALUATION": CGState(
            name="HYPOTHESIS_EVALUATION",
            state_type=CGStateType.REFLECTION,
            objective="Evaluate your evidence to determine the causal rule.",
            guidelines=[
                "What rule type best explains ALL your observations?",
                "Which specific objects are blickets?",
                "Are there observations that don't fit your current hypothesis?",
                "If observations are inconsistent, consider whether factors beyond object identity might matter",
            ],
            transitions=[
                CGTransition(
                    target="VERIFICATION",
                    condition="When you have a clear hypothesis",
                ),
                CGTransition(
                    target="COMBINATION_TESTING",
                    condition="If hypothesis is unclear, test more combinations",
                ),
            ],
        ),
        "VERIFICATION": CGState(
            name="VERIFICATION",
            state_type=CGStateType.DECISION,
            objective="Confirm your hypothesis with targeted tests before submitting.",
            guidelines=[
                "Design a test that would distinguish your hypothesis from the next most likely alternative",
                "If the test confirms, submit your answer",
                "If the test disconfirms, reconsider your hypothesis",
            ],
            transitions=[
                CGTransition(
                    target="HYPOTHESIS_EVALUATION",
                    condition="If verification fails",
                ),
            ],
        ),
    }

    return ContextGraph(
        states=states,
        current_state="INITIAL_EXPLORATION",
    )
