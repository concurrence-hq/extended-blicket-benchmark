"""Dynamic Behaviors engine for the compositional agent architecture.

Evaluates trigger conditions against conversation history and applies
modifications to the Context Graph at runtime. This is the mechanism
that enables "coordinate system revision" — the CG expands to accommodate
new causal dimensions that the agent discovers.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from .context_graph import CGState, CGStateType, CGTransition, ContextGraph

logger = logging.getLogger(__name__)


class TriggerCondition(BaseModel):
    """Defines when a Dynamic Behavior should fire."""

    evaluation_prompt: str  # Prompt for LLM-based trigger evaluation
    min_steps: int = 5  # Don't evaluate before this many steps
    requires_hypothesis_failure: bool = False  # Only fire after unexpected result
    stagnation_mode: bool = False  # Fire on zero-activation stagnation
    stagnation_threshold: int = 6  # Steps of zero activation before eligible
    fire_threshold: float = 6.0  # 0-10 scale; score must meet or exceed to fire


class CGModification(BaseModel):
    """A single modification to apply to the Context Graph."""

    action: str  # "add_state", "modify_guidelines", "expose_tools"
    state: CGState | None = None  # For add_state
    connect_from: list[str] = Field(default_factory=list)  # For add_state
    target_state: str = ""  # For modify_guidelines / expose_tools
    guidelines: list[str] = Field(default_factory=list)  # For modify_guidelines
    tools: list[str] = Field(default_factory=list)  # For expose_tools


class DynamicBehavior(BaseModel):
    """A complete behavior: trigger condition + modifications to apply."""

    name: str
    description: str
    trigger: TriggerCondition
    modifications: list[CGModification]
    notification_message: str  # Injected into conversation when behavior fires


class DynamicBehaviorEngine:
    """Evaluates triggers and applies CG modifications.

    Each behavior fires at most once per episode (one-shot for pilot simplicity).
    Trigger evaluation uses Haiku for cost savings.

    Run 3 changes:
    - _evaluate_trigger returns float|None (raw score) instead of bool
    - evaluate_and_apply uses two-phase competitive inhibition:
      Phase 1: Stagnation (independent) — fires on its own
      Phase 2: Diagnostic competition — winner-take-most with 1.0 proximity window
    - Score logging for analysis
    """

    def __init__(
        self,
        behaviors: list[DynamicBehavior],
        llm_client: Any,  # anthropic.AsyncAnthropic
        trigger_model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self.behaviors = behaviors
        self.llm_client = llm_client
        self.trigger_model = trigger_model
        self._fired: set[str] = set()
        self._trigger_token_usage: int = 0
        self._score_log: list[dict[str, Any]] = []
        self._trigger_response_models: list[str] = []

    def reset(self) -> None:
        """Reset fired state for a new episode."""
        self._fired = set()
        self._trigger_token_usage = 0
        self._score_log = []
        self._trigger_response_models = []

    @property
    def trigger_tokens_used(self) -> int:
        return self._trigger_token_usage

    @property
    def fired_behaviors(self) -> set[str]:
        return set(self._fired)

    @property
    def score_log(self) -> list[dict[str, Any]]:
        return list(self._score_log)

    async def evaluate_and_apply(
        self,
        context_graph: ContextGraph,
        conversation_history: list[dict[str, str]],
        step_number: int,
        had_unexpected_result: bool = False,
        activation_history: list[bool] | None = None,
        exploration_summary: str = "",
    ) -> list[str]:
        """Evaluate trigger conditions using two-phase competitive inhibition.

        Phase 1 — Stagnation (independent): DB1 (stagnation_mode=True) evaluated
        and fired independently. Stagnation is a meta-intervention, not a diagnostic.

        Phase 2 — Competitive diagnostic selection: DB2-4 (stagnation_mode=False)
        evaluated, scored, and the top scorer fires if it meets threshold. A runner-up
        within 1.0 point of the top score also fires if it meets its threshold.

        Returns:
            List of notification messages for behaviors that fired.
        """
        notifications = []

        # Partition behaviors into stagnation vs diagnostic
        stagnation_behaviors = []
        diagnostic_behaviors = []
        for behavior in self.behaviors:
            if behavior.trigger.stagnation_mode:
                stagnation_behaviors.append(behavior)
            else:
                diagnostic_behaviors.append(behavior)

        # Phase 1: Stagnation (independent)
        for behavior in stagnation_behaviors:
            if behavior.name in self._fired:
                continue
            if not self._pre_screen(
                behavior, step_number, had_unexpected_result, activation_history
            ):
                continue

            score = await self._evaluate_trigger(
                behavior, conversation_history, context_graph, exploration_summary
            )
            fired = score is not None and score >= behavior.trigger.fire_threshold
            self._score_log.append({
                "behavior": behavior.name,
                "step": step_number,
                "score": score,
                "fired": fired,
                "phase": "stagnation",
            })

            if fired:
                self._fired.add(behavior.name)
                self._apply_modifications(behavior, context_graph, step_number)
                notifications.append(behavior.notification_message)
                logger.info(f"DB fired (stagnation): {behavior.name} at step {step_number}, score={score}")

        # Phase 2: Competitive diagnostic selection
        candidates: list[tuple[DynamicBehavior, float]] = []
        for behavior in diagnostic_behaviors:
            if behavior.name in self._fired:
                continue
            if not self._pre_screen(
                behavior, step_number, had_unexpected_result, activation_history
            ):
                continue

            score = await self._evaluate_trigger(
                behavior, conversation_history, context_graph, exploration_summary
            )
            if score is not None:
                candidates.append((behavior, score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Determine which fire: top scorer + any runner-up within 1.0 point
        fired_diagnostics: list[tuple[DynamicBehavior, float]] = []
        if candidates:
            top_behavior, top_score = candidates[0]
            if top_score >= top_behavior.trigger.fire_threshold:
                fired_diagnostics.append((top_behavior, top_score))
                # Check runner-ups within 1.0 point proximity
                for behavior, score in candidates[1:]:
                    if (
                        top_score - score <= 1.0
                        and score >= behavior.trigger.fire_threshold
                    ):
                        fired_diagnostics.append((behavior, score))

        # Log all diagnostic candidates
        for behavior, score in candidates:
            fired = any(b.name == behavior.name for b, _ in fired_diagnostics)
            self._score_log.append({
                "behavior": behavior.name,
                "step": step_number,
                "score": score,
                "fired": fired,
                "phase": "diagnostic",
            })

        # Apply fired diagnostics
        for behavior, score in fired_diagnostics:
            self._fired.add(behavior.name)
            self._apply_modifications(behavior, context_graph, step_number)
            notifications.append(behavior.notification_message)
            logger.info(f"DB fired (diagnostic): {behavior.name} at step {step_number}, score={score}")

        return notifications

    def _pre_screen(
        self,
        behavior: DynamicBehavior,
        step_number: int,
        had_unexpected_result: bool,
        activation_history: list[bool] | None = None,
    ) -> bool:
        """Quick pre-screening before expensive LLM evaluation."""
        if step_number < behavior.trigger.min_steps:
            return False
        if behavior.trigger.requires_hypothesis_failure and not had_unexpected_result:
            return False
        if behavior.trigger.stagnation_mode:
            if activation_history and any(activation_history):
                return False  # Had activation, stagnation N/A
            if step_number < behavior.trigger.stagnation_threshold:
                return False
        return True

    async def _evaluate_trigger(
        self,
        behavior: DynamicBehavior,
        conversation_history: list[dict[str, str]],
        context_graph: ContextGraph | None = None,
        exploration_summary: str = "",
    ) -> float | None:
        """Use LLM to evaluate trigger condition strength (graded 0-10).

        Returns:
            Raw score (0-10) on success, None on error or parse failure.
            Threshold comparison is handled by the caller.
        """
        # Build condensed context: summary + last 5 messages
        recent = conversation_history[-5:]
        history_text = "\n".join(
            f"{m['role']}: {m['content'][:300]}" for m in recent
        )

        cg_context = ""
        if context_graph:
            cg_context = f"\nCurrent CG state: {context_graph.current_state}\n"

        summary_context = ""
        if exploration_summary:
            summary_context = f"\nExploration summary: {exploration_summary}\n"

        prompt = (
            f"You are evaluating whether a specific condition has been met in an "
            f"agent's exploration of a causal learning task.\n\n"
            f"Condition to evaluate:\n{behavior.trigger.evaluation_prompt}\n\n"
            f"{cg_context}{summary_context}"
            f"Recent conversation:\n{history_text}\n\n"
            f"Rate how strongly this condition is met on a scale of 0-10, "
            f"where 0 = definitely not met and 10 = clearly met.\n"
            f"Respond with just the number."
        )

        try:
            response = await self.llm_client.messages.create(
                model=self.trigger_model,
                max_tokens=10,
                temperature=1.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            self._trigger_token_usage += (
                response.usage.input_tokens + response.usage.output_tokens
            )
            self._trigger_response_models.append(
                getattr(response, "model", "unknown")
            )
            # Parse score from response
            match = re.search(r"(\d+(?:\.\d+)?)", text)
            if match:
                return float(match.group(1))
            return None
        except Exception as e:
            logger.warning(f"Trigger evaluation failed for {behavior.name}: {e}")
            return None

    def _apply_modifications(
        self,
        behavior: DynamicBehavior,
        context_graph: ContextGraph,
        step_number: int,
    ) -> None:
        """Apply all modifications from a fired behavior."""
        for mod in behavior.modifications:
            if mod.action == "add_state" and mod.state:
                context_graph.add_state(
                    mod.state,
                    connect_from=mod.connect_from,
                    step_number=step_number,
                )
            elif mod.action == "modify_guidelines":
                context_graph.modify_guidelines(
                    mod.target_state, mod.guidelines, step_number
                )
            elif mod.action == "expose_tools":
                context_graph.expose_tools(
                    mod.target_state, mod.tools, step_number
                )


def create_blicket_behaviors() -> list[DynamicBehavior]:
    """Create the 4 Dynamic Behaviors for the blicket pilot.

    These behaviors expand the CG when the agent encounters evidence
    that the standard conjunctive/disjunctive framework is insufficient.

    Run 2 changes from Run 1:
    - DB1: Stagnation trigger (fires on zero-activation stagnation, not reactive-only)
    - DB2-4: Graded triggers with calibrated thresholds
    - DB4: Removed requires_hypothesis_failure gate

    Run 3 changes from Run 2:
    - DB2-4: fire_threshold raised from 6.0 to 7.0
    - Competitive inhibition: diagnostic triggers compete, winner-take-most
    """
    return [
        # DB1: Exploration Stagnation
        DynamicBehavior(
            name="exploration_stagnation",
            description=(
                "Fires when the agent has explored for several steps without "
                "ever activating the detector, suggesting standard exploration "
                "strategies are failing."
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
            modifications=[
                CGModification(
                    action="add_state",
                    state=CGState(
                        name="DIMENSION_DISCOVERY",
                        state_type=CGStateType.REFLECTION,
                        objective=(
                            "The standard conjunctive/disjunctive framework doesn't explain "
                            "your observations. Consider alternative causal dimensions: "
                            "order of placement, probabilistic activation, or changing rules."
                        ),
                        guidelines=[
                            "Review ALL observations for patterns beyond which objects are present",
                            "Consider: does the ORDER of placement matter?",
                            "Consider: is activation PROBABILISTIC (same setup, different results)?",
                            "Consider: has the RULE CHANGED over time?",
                            "Form a new hypothesis about the causal mechanism",
                        ],
                        transitions=[
                            CGTransition(
                                target="COMBINATION_TESTING",
                                condition="Test new hypothesis with targeted experiments",
                            ),
                            CGTransition(
                                target="VERIFICATION",
                                condition="If new hypothesis is clear enough to verify",
                            ),
                        ],
                    ),
                    connect_from=["HYPOTHESIS_EVALUATION", "COMBINATION_TESTING"],
                ),
            ],
            notification_message=(
                "[SYSTEM] Your observations don't fit standard causal rules. "
                "A new exploration state DIMENSION_DISCOVERY is now available. "
                "Consider transitioning there to explore alternative causal dimensions."
            ),
        ),
        # DB2: Order Hypothesis
        DynamicBehavior(
            name="order_hypothesis",
            description=(
                "Fires when evidence suggests something beyond object identity "
                "matters for activation — particularly placement order or sequence."
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
                            "Test whether the order of object placement affects activation. "
                            "Systematically vary placement order while keeping the same objects."
                        ),
                        guidelines=[
                            "Clear detector completely before each test",
                            "Place the SAME objects in DIFFERENT orders",
                            "Record: which order activated, which didn't",
                            "Try at least 3 different orderings",
                        ],
                        transitions=[
                            CGTransition(
                                target="HYPOTHESIS_EVALUATION",
                                condition="After testing multiple orderings",
                            ),
                            CGTransition(
                                target="VERIFICATION",
                                condition="If order pattern is clear",
                            ),
                        ],
                    ),
                    connect_from=["DIMENSION_DISCOVERY", "COMBINATION_TESTING"],
                ),
            ],
            notification_message=(
                "[SYSTEM] Evidence suggests placement order may matter. "
                "A new ORDER_TESTING state is available for systematic order experiments."
            ),
        ),
        # DB3: Stochasticity Hypothesis
        DynamicBehavior(
            name="stochasticity_hypothesis",
            description="Fires when the same configuration gives inconsistent results.",
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
                            "the same configuration multiple times."
                        ),
                        guidelines=[
                            "Pick one configuration that activated before",
                            "Clear and re-place the same objects 5+ times",
                            "Count activations vs non-activations",
                            "Estimate the activation probability",
                        ],
                        transitions=[
                            CGTransition(
                                target="HYPOTHESIS_EVALUATION",
                                condition="After 5+ repetitions",
                            ),
                        ],
                    ),
                    connect_from=["DIMENSION_DISCOVERY", "COMBINATION_TESTING"],
                ),
            ],
            notification_message=(
                "[SYSTEM] Inconsistent results detected for same configurations. "
                "A new RELIABILITY_TESTING state is available to measure activation probability."
            ),
        ),
        # DB4: Hidden Moderator Hypothesis
        DynamicBehavior(
            name="rule_change_hypothesis",
            description="Fires when a previously working rule suddenly stops working.",
            trigger=TriggerCondition(
                evaluation_prompt=(
                    "Has the agent found a rule that WORKED CONSISTENTLY for several "
                    "trials but then STOPPED working? The same objects that reliably "
                    "activated the detector no longer do so, suggesting the underlying "
                    "rule has changed."
                ),
                min_steps=10,
                fire_threshold=7.0,
            ),
            modifications=[
                CGModification(
                    action="add_state",
                    state=CGState(
                        name="MODERATOR_DETECTION",
                        state_type=CGStateType.REFLECTION,
                        objective=(
                            "The causal rule appears to have changed. Investigate what "
                            "triggers rule changes and identify the new rule."
                        ),
                        guidelines=[
                            "Count how many successful activations occurred before the change",
                            "Re-test all objects individually under the new regime",
                            "Re-test combinations to identify the new rule",
                            "Consider: does the rule cycle or switch permanently?",
                        ],
                        transitions=[
                            CGTransition(
                                target="COMBINATION_TESTING",
                                condition="Start fresh exploration under new rule",
                            ),
                            CGTransition(
                                target="VERIFICATION",
                                condition="If new rule pattern is clear",
                            ),
                        ],
                    ),
                    connect_from=["DIMENSION_DISCOVERY", "HYPOTHESIS_EVALUATION"],
                ),
            ],
            notification_message=(
                "[SYSTEM] The causal rule appears to have changed. "
                "A new MODERATOR_DETECTION state is available to investigate rule shifts."
            ),
        ),
    ]
