"""Shared episode runner for executing agent-environment interaction loops.

Handles the core loop: render state -> get action -> step -> log trace.
Crash-resilient: saves per-episode JSON traces.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from environment.blicket_env import Action, BlicketEnv, CausalRuleType, Observation
from agents.base_llm_agent import BaseLLMAgent
from agents.full_agent import FullAgent
from experiments.config import (
    GradualFeedbackConfig,
    PartialFeedbackConfig,
    SONNET_INPUT_COST_PER_TOKEN,
    SONNET_OUTPUT_COST_PER_TOKEN,
    HAIKU_INPUT_COST_PER_TOKEN,
)

logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    """Result of a single episode."""

    condition: str
    agent_type: str
    episode_number: int
    success: bool  # Did the agent answer correctly?
    answer_submitted: bool  # Did the agent submit an answer at all?
    steps_taken: int
    rule_type_correct: bool = False
    objects_correct: bool = False
    true_rule_type: str = ""
    true_objects: list[str] = field(default_factory=list)
    guessed_rule_type: str = ""
    guessed_objects: list[str] = field(default_factory=list)
    order_correct: bool | None = None
    true_order: list[str] | None = None
    guessed_order: list[str] | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    trigger_tokens: int = 0
    parse_failures: int = 0
    wall_time_seconds: float = 0.0
    trace_path: str = ""
    error: str | None = None
    partial_feedback_step: int | None = None
    partial_feedback_message: str | None = None
    gradual_feedback_events: list[dict] = field(default_factory=list)
    gradual_feedback_confirmed_objects: list[str] = field(default_factory=list)


@dataclass
class StepTrace:
    """Trace data for a single step within an episode."""

    step: int
    action: str
    action_valid: bool
    objects_on_detector: list[str]
    detector_active: bool
    detector_changed: bool
    observation_text: str
    cg_state: str = ""
    db_notifications: list[str] = field(default_factory=list)
    partial_feedback_delivered: bool = False
    gradual_feedback_object: str | None = None


def _check_partial_feedback_trigger(
    step_count: int,
    unique_combinations: set[frozenset[str]],
    any_activation: bool,
    already_delivered: bool,
    config: "PartialFeedbackConfig",
) -> bool:
    """Check whether partial feedback should be delivered this step."""
    if not config.enabled:
        return False
    if already_delivered:
        return False
    if any_activation:
        return False
    if step_count < config.min_steps:
        return False
    if len(unique_combinations) < config.min_unique_combos:
        return False
    return True


def _render_partial_feedback_message(correct_objects: list[str]) -> str:
    """Render a partial insight message revealing object identities (not order)."""
    sorted_objects = sorted(correct_objects)
    obj_str = ", ".join(sorted_objects)
    return (
        f"PARTIAL INSIGHT: Based on your exploration, the detector appears to "
        f"respond to objects {obj_str}. The specific activation condition "
        f"involving these objects is still unknown."
    )


def _check_gradual_feedback_checkpoint(
    step_count: int,
    tested_objects_ordered: list[str],
    confirmed_objects: list[str],
    any_activation: bool,
    config: GradualFeedbackConfig,
) -> str | None:
    """Check whether a gradual feedback checkpoint should reveal an object.

    Returns the object name to confirm, or None.
    """
    if not config.enabled:
        return None
    if step_count == 0 or step_count % config.checkpoint_interval != 0:
        return None
    if any_activation:
        return None
    eligible = (
        set(tested_objects_ordered) & set(config.correct_objects) - set(confirmed_objects)
    )
    if not eligible:
        return None
    # Select first eligible from tested_objects_ordered (first-tested priority)
    for obj in tested_objects_ordered:
        if obj in eligible:
            return obj
    return None  # pragma: no cover


def _render_gradual_feedback_message(object_name: str) -> str:
    """Render a gradual feedback observation confirming a single object."""
    return (
        f"OBSERVATION: Based on your exploration patterns, the detector appears "
        f"to have some sensitivity to object {object_name}. The nature of this "
        f"sensitivity is unclear — further investigation is needed."
    )


async def run_episode(
    env: BlicketEnv,
    agent: BaseLLMAgent,
    condition: str,
    episode_number: int,
    trace_dir: Path,
    max_cost_per_episode: float = 2.0,
    partial_feedback_config: PartialFeedbackConfig | None = None,
    gradual_feedback_config: GradualFeedbackConfig | None = None,
) -> EpisodeResult:
    """Run a single episode of agent-environment interaction.

    Args:
        env: The blicket environment (already configured with causal rule).
        agent: The agent to run.
        condition: Name of the experimental condition.
        episode_number: Episode index for this condition/agent.
        trace_dir: Directory to save the episode trace.
        max_cost_per_episode: Cost ceiling in USD.

    Returns:
        EpisodeResult with outcomes and metrics.
    """
    start_time = time.time()
    text_interface = agent.text_interface

    env.reset()
    agent.reset()

    step_traces: list[StepTrace] = []
    parse_failures = 0
    had_unexpected_result = False

    # Partial feedback tracking
    pf_config = partial_feedback_config or PartialFeedbackConfig()
    pf_unique_combos: set[frozenset[str]] = set()
    pf_any_activation = False
    pf_delivered = False
    pf_step: int | None = None
    pf_message: str | None = None

    # Gradual feedback tracking
    gf_config = gradual_feedback_config or GradualFeedbackConfig()
    if pf_config.enabled and gf_config.enabled:
        raise ValueError(
            "partial_feedback_config and gradual_feedback_config are mutually exclusive"
        )
    gf_tested_objects_ordered: list[str] = []
    gf_confirmed_objects: list[str] = []
    gf_events: list[dict] = []
    gf_any_activation: bool = False

    try:
        while not env.done:
            step_num = env.state.step_count

            # Get action from agent
            action = await agent.get_action(env)

            if action is None:
                parse_failures += 1
                logger.warning(
                    f"[{condition}/{agent.__class__.__name__}/ep{episode_number}] "
                    f"Parse failure at step {step_num}, ending episode"
                )
                break

            # Execute action
            obs = env.step(action)

            # Track partial feedback state
            if obs.objects_on_detector:
                pf_unique_combos.add(frozenset(obs.objects_on_detector))
            if obs.detector_active:
                pf_any_activation = True

            # Track tested objects for gradual feedback (first-tested order)
            for obj in obs.objects_on_detector:
                if obj not in gf_tested_objects_ordered:
                    gf_tested_objects_ordered.append(obj)
            if obs.detector_active:
                gf_any_activation = True

            # Feed activation observation to FullAgent
            if isinstance(agent, FullAgent):
                agent.observe_activation(obs.detector_active)

            # Detect unexpected results for DB triggering
            # An "unexpected" result is when the detector changes state in a way
            # that contradicts simple hypotheses (approximation for pilot)
            if obs.detector_changed:
                had_unexpected_result = True
                if isinstance(agent, FullAgent):
                    agent.mark_unexpected_result()
            else:
                had_unexpected_result = False

            # Add observation to agent's conversation
            obs_text = text_interface.render_observation(obs)
            agent.add_observation(obs_text)

            # Check and deliver partial feedback
            pf_delivered_this_step = False
            if _check_partial_feedback_trigger(
                step_count=env.state.step_count,
                unique_combinations=pf_unique_combos,
                any_activation=pf_any_activation,
                already_delivered=pf_delivered,
                config=pf_config,
            ):
                pf_message = _render_partial_feedback_message(pf_config.correct_objects)
                pf_step = env.state.step_count
                pf_delivered = True
                pf_delivered_this_step = True
                agent.add_observation(pf_message)

            # Check and deliver gradual feedback
            gf_object_this_step: str | None = None
            if gf_config.enabled:
                revealed = _check_gradual_feedback_checkpoint(
                    step_count=env.state.step_count,
                    tested_objects_ordered=gf_tested_objects_ordered,
                    confirmed_objects=gf_confirmed_objects,
                    any_activation=gf_any_activation,
                    config=gf_config,
                )
                if revealed is not None:
                    gf_msg = _render_gradual_feedback_message(revealed)
                    gf_confirmed_objects.append(revealed)
                    gf_events.append({
                        "step": env.state.step_count,
                        "object": revealed,
                        "message": gf_msg,
                    })
                    gf_object_this_step = revealed
                    agent.add_observation(gf_msg)

            # Record step trace
            cg_state = ""
            if hasattr(agent, "context_graph"):
                cg_state = agent.context_graph.current_state

            step_db_notifications: list[str] = []
            if hasattr(agent, "get_last_step_notifications"):
                step_db_notifications = agent.get_last_step_notifications()

            step_traces.append(StepTrace(
                step=step_num,
                action=action.value,
                action_valid=obs.action_valid,
                objects_on_detector=sorted(obs.objects_on_detector),
                detector_active=obs.detector_active,
                detector_changed=obs.detector_changed,
                observation_text=obs_text,
                cg_state=cg_state,
                db_notifications=step_db_notifications,
                partial_feedback_delivered=pf_delivered_this_step,
                gradual_feedback_object=gf_object_this_step,
            ))

            # Check per-episode cost cap
            current_cost = (
                agent.token_usage.input_tokens * SONNET_INPUT_COST_PER_TOKEN
                + agent.token_usage.output_tokens * SONNET_OUTPUT_COST_PER_TOKEN
                + getattr(getattr(agent, "db_engine", None), "trigger_tokens_used", 0) * HAIKU_INPUT_COST_PER_TOKEN
            )
            if current_cost > max_cost_per_episode:
                logger.warning(
                    f"[{condition}/ep{episode_number}] Episode cost ${current_cost:.2f} "
                    f"exceeds cap ${max_cost_per_episode:.2f}. Terminating (timeout)."
                )
                break

            # If agent chose CHECK, get their answer
            if action.value == "check":
                break

        # Get answer if episode ended with CHECK
        answer_submitted = False
        answer_result = None

        if env.done and env.state.action_history and env.state.action_history[-1].value == "check":
            rule_type, blickets, guessed_order = await agent.get_answer()
            if rule_type is not None and blickets is not None:
                answer_submitted = True
                answer_result = env.check_answer(rule_type, blickets, guessed_order)

        # Build result
        wall_time = time.time() - start_time

        result = EpisodeResult(
            condition=condition,
            agent_type=agent.get_trace_data()["agent_type"],
            episode_number=episode_number,
            success=answer_result["correct"] if answer_result else False,
            answer_submitted=answer_submitted,
            steps_taken=env.state.step_count,
            rule_type_correct=answer_result.get("rule_type_correct", False) if answer_result else False,
            objects_correct=answer_result.get("objects_correct", False) if answer_result else False,
            true_rule_type=env.causal_rule.rule_type.value,
            true_objects=sorted(env.causal_rule.required_objects),
            guessed_rule_type=answer_result.get("guessed_rule_type", "") if answer_result else "",
            guessed_objects=answer_result.get("guessed_required_objects", []) if answer_result else [],
            order_correct=answer_result.get("order_correct") if answer_result else None,
            true_order=answer_result.get("true_order") if answer_result else None,
            guessed_order=answer_result.get("guessed_order") if answer_result else None,
            total_input_tokens=agent.token_usage.input_tokens,
            total_output_tokens=agent.token_usage.output_tokens,
            trigger_tokens=getattr(getattr(agent, "db_engine", None), "trigger_tokens_used", 0),
            parse_failures=parse_failures,
            wall_time_seconds=wall_time,
            partial_feedback_step=pf_step,
            partial_feedback_message=pf_message,
            gradual_feedback_events=gf_events,
            gradual_feedback_confirmed_objects=list(gf_confirmed_objects),
        )

    except Exception as e:
        wall_time = time.time() - start_time
        logger.error(f"Episode crashed: {e}", exc_info=True)
        result = EpisodeResult(
            condition=condition,
            agent_type=agent.get_trace_data()["agent_type"],
            episode_number=episode_number,
            success=False,
            answer_submitted=False,
            steps_taken=env.state.step_count,
            wall_time_seconds=wall_time,
            error=str(e),
            partial_feedback_step=pf_step,
            partial_feedback_message=pf_message,
            gradual_feedback_events=gf_events,
            gradual_feedback_confirmed_objects=list(gf_confirmed_objects),
        )

    # Save trace
    trace_path = _save_trace(
        trace_dir=trace_dir,
        condition=condition,
        agent_type=result.agent_type,
        episode_number=episode_number,
        result=result,
        step_traces=step_traces,
        agent_trace=agent.get_trace_data(),
        env_info=env.get_info(),
    )
    result.trace_path = str(trace_path)

    return result


def _save_trace(
    trace_dir: Path,
    condition: str,
    agent_type: str,
    episode_number: int,
    result: EpisodeResult,
    step_traces: list[StepTrace],
    agent_trace: dict[str, Any],
    env_info: dict[str, Any],
) -> Path:
    """Save episode trace as JSON. Crash-resilient: one file per episode."""
    trace_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{condition}_{agent_type}_ep{episode_number:03d}.json"
    trace_path = trace_dir / filename

    trace_data = {
        "result": {
            "condition": result.condition,
            "agent_type": result.agent_type,
            "episode_number": result.episode_number,
            "success": result.success,
            "answer_submitted": result.answer_submitted,
            "steps_taken": result.steps_taken,
            "rule_type_correct": result.rule_type_correct,
            "objects_correct": result.objects_correct,
            "true_rule_type": result.true_rule_type,
            "true_objects": result.true_objects,
            "guessed_rule_type": result.guessed_rule_type,
            "guessed_objects": result.guessed_objects,
            "order_correct": result.order_correct,
            "true_order": result.true_order,
            "guessed_order": result.guessed_order,
            "total_input_tokens": result.total_input_tokens,
            "total_output_tokens": result.total_output_tokens,
            "trigger_tokens": result.trigger_tokens,
            "parse_failures": result.parse_failures,
            "wall_time_seconds": result.wall_time_seconds,
            "error": result.error,
            "partial_feedback_step": result.partial_feedback_step,
            "partial_feedback_message": result.partial_feedback_message,
            "gradual_feedback_events": result.gradual_feedback_events,
            "gradual_feedback_confirmed_objects": result.gradual_feedback_confirmed_objects,
        },
        "steps": [
            {
                "step": s.step,
                "action": s.action,
                "action_valid": s.action_valid,
                "objects_on_detector": s.objects_on_detector,
                "detector_active": s.detector_active,
                "detector_changed": s.detector_changed,
                "observation_text": s.observation_text,
                "cg_state": s.cg_state,
                "db_notifications": s.db_notifications,
                "partial_feedback_delivered": s.partial_feedback_delivered,
                "gradual_feedback_object": s.gradual_feedback_object,
            }
            for s in step_traces
        ],
        "agent_trace": agent_trace,
        "env_info": env_info,
    }

    with open(trace_path, "w") as f:
        json.dump(trace_data, f, indent=2, default=str)

    logger.info(f"Trace saved: {trace_path}")
    return trace_path
