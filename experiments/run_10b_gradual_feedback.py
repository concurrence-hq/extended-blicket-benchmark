"""Run 10b: Gradual Periodic Feedback Order-Sensitive Pilot.

Tests whether gradual object revelation (one correct object per 5-step checkpoint)
combined with both 3-object and 4-object ordering conditions produces architectural
separation. Feedback is delivered as periodic environmental observations, not action-
triggered — the environment confirms previously-tested correct objects one at a time.

4 cells × 15 episodes = 60 episodes total.

Cell A: base_llm    × 3-obj order (A→D→C)   × 15 eps × 60 steps + gradual feedback
Cell B: full_cg_db  × 3-obj order (A→D→C)   × 15 eps × 60 steps + gradual feedback
Cell C: base_llm    × 4-obj order (A→B→E→C)  × 15 eps × 60 steps + gradual feedback
Cell D: full_cg_db  × 4-obj order (A→B→E→C)  × 15 eps × 60 steps + gradual feedback
"""

from __future__ import annotations

RUN_NAME = "run10b"

import asyncio
import json
import logging
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.blicket_env import CausalRule, CausalRuleType
from environment.extended_env import ExtendedCondition, OrderConfig
from environment.extended_env_5obj import ExtendedBlicketEnv5
from environment.text_interface import TextInterface5
from agents.base_llm_agent import BaseLLMAgent
from agents.full_agent import FullAgent
from experiments.config import (
    AgentType,
    CostTracker,
    ExperimentConfig,
    GradualFeedbackConfig,
)
from experiments.episode_runner import EpisodeResult, run_episode

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

NUM_OBJECTS = 5
BASE_SEED = 1051
BUDGET_CAP = 100.0
MAX_STEPS = 60
EPISODES_PER_CELL = 15

# 3-object condition (same rule as Runs 09-10)
REQUIRED_OBJECTS_3 = {"A", "D", "C"}
REQUIRED_ORDER_3 = ["A", "D", "C"]

# 4-object condition (new: non-alphabetical ordering)
REQUIRED_OBJECTS_4 = {"A", "B", "E", "C"}
REQUIRED_ORDER_4 = ["A", "B", "E", "C"]

# Cell definitions: (label, agent_type, n_obj, required_objects, required_order)
CELLS = [
    ("A", AgentType.BASE_LLM, 3, REQUIRED_OBJECTS_3, REQUIRED_ORDER_3),
    ("B", AgentType.FULL_CG_DB, 3, REQUIRED_OBJECTS_3, REQUIRED_ORDER_3),
    ("C", AgentType.BASE_LLM, 4, REQUIRED_OBJECTS_4, REQUIRED_ORDER_4),
    ("D", AgentType.FULL_CG_DB, 4, REQUIRED_OBJECTS_4, REQUIRED_ORDER_4),
]


def create_agent(
    agent_type: AgentType, client: anthropic.AsyncAnthropic, config: ExperimentConfig
) -> BaseLLMAgent:
    ti = TextInterface5(include_order_sensitive=True)
    if agent_type == AgentType.BASE_LLM:
        return BaseLLMAgent(llm_client=client, model=config.action_model, text_interface=ti)
    elif agent_type == AgentType.FULL_CG_DB:
        return FullAgent(
            llm_client=client,
            model=config.action_model,
            trigger_model=config.trigger_model,
            text_interface=ti,
            num_objects=NUM_OBJECTS,
        )
    raise ValueError(f"Unknown agent type: {agent_type}")


def create_env(
    required_objects: set[str],
    required_order: list[str],
    seed: int,
) -> ExtendedBlicketEnv5:
    rule = CausalRule(
        rule_type=CausalRuleType.ORDER_SENSITIVE,
        required_objects=required_objects,
    )
    return ExtendedBlicketEnv5(
        causal_rule=rule,
        condition=ExtendedCondition.ORDER_SENSITIVE,
        max_steps=MAX_STEPS,
        order_config=OrderConfig(required_order=required_order),
        rng_seed=seed,
    )


async def run_gradual_feedback_experiments(
    config: ExperimentConfig | None = None,
) -> list[EpisodeResult]:
    """Run all Run 10b gradual feedback experiments."""
    if config is None:
        config = ExperimentConfig(
            episodes_per_condition=EPISODES_PER_CELL,
            max_steps_per_episode=MAX_STEPS,
            trace_dir=Path(f"data/traces_{RUN_NAME}"),
            results_dir=Path(f"data/results_{RUN_NAME}"),
            base_seed=BASE_SEED,
            max_total_cost=BUDGET_CAP,
        )

    client = anthropic.AsyncAnthropic()
    cost_tracker = CostTracker(max_total=config.max_total_cost)
    results: list[EpisodeResult] = []

    total_episodes = len(CELLS) * EPISODES_PER_CELL
    completed = 0

    for label, agent_type, n_obj, req_objects, req_order in CELLS:
        condition_name = f"{n_obj}obj_order_gf_{label}"

        gf_config = GradualFeedbackConfig(
            enabled=True,
            checkpoint_interval=5,
            correct_objects=sorted(req_objects),
        )

        for ep in range(EPISODES_PER_CELL):
            if not cost_tracker.check_budget():
                logger.warning(
                    f"Budget exhausted (${cost_tracker.total_cost:.2f}/{BUDGET_CAP}). Stopping."
                )
                break

            logger.info(
                f"[{completed + 1}/{total_episodes}] "
                f"Cell {label}: {agent_type.value} / {n_obj}-obj order+gf / ep{ep}"
            )

            seed = config.get_episode_seed(condition_name, agent_type.value, ep)
            env = create_env(req_objects, req_order, seed)
            agent = create_agent(agent_type, client, config)

            result = await run_episode(
                env=env,
                agent=agent,
                condition=condition_name,
                episode_number=ep,
                trace_dir=config.trace_dir,
                max_cost_per_episode=config.max_cost_per_episode,
                gradual_feedback_config=gf_config,
            )

            episode_cost = cost_tracker.add_episode_cost(
                condition=condition_name,
                agent=agent_type.value,
                episode=ep,
                input_tokens=result.total_input_tokens,
                output_tokens=result.total_output_tokens,
                trigger_tokens=result.trigger_tokens,
            )

            results.append(result)
            completed += 1

            order_str = ""
            if result.order_correct is not None:
                order_str = f" | Order: {'CORRECT' if result.order_correct else 'WRONG'}"

            gf_str = f" | GF: {len(result.gradual_feedback_events)} objects revealed"

            logger.info(
                f"  Result: {'CORRECT' if result.success else 'WRONG'}{order_str}{gf_str} | "
                f"Steps: {result.steps_taken} | Cost: ${episode_cost:.3f} | "
                f"Total: ${cost_tracker.total_cost:.2f}"
            )

    # Save summary
    summary_path = config.results_dir / "gradual_feedback_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": RUN_NAME,
        "description": (
            "Gradual periodic feedback order-sensitive pilot: reveal one correct "
            "object per 5-step checkpoint; 3-obj and 4-obj conditions"
        ),
        "num_objects": NUM_OBJECTS,
        "base_seed": BASE_SEED,
        "gradual_feedback_config": {
            "enabled": True,
            "checkpoint_interval": 5,
        },
        "cells": [
            {
                "label": label,
                "agent_type": agent_type.value,
                "n_order_objects": n_obj,
                "required_objects": sorted(req_objects),
                "required_order": req_order,
                "max_steps": MAX_STEPS,
            }
            for label, agent_type, n_obj, req_objects, req_order in CELLS
        ],
        "cost_summary": cost_tracker.get_summary(),
        "results": [
            {
                "condition": r.condition,
                "agent_type": r.agent_type,
                "episode": r.episode_number,
                "success": r.success,
                "steps": r.steps_taken,
                "answer_submitted": r.answer_submitted,
                "rule_type_correct": r.rule_type_correct,
                "objects_correct": r.objects_correct,
                "order_correct": r.order_correct,
                "true_order": r.true_order,
                "guessed_order": r.guessed_order,
                "guessed_rule_type": r.guessed_rule_type,
                "guessed_objects": r.guessed_objects,
                "gradual_feedback_events": r.gradual_feedback_events,
                "gradual_feedback_confirmed_objects": r.gradual_feedback_confirmed_objects,
            }
            for r in results
        ],
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    logger.info(f"\n{'='*70}")
    logger.info("Run 10b: Gradual Periodic Feedback Order-Sensitive Results")
    logger.info(f"{'='*70}")
    for label, agent_type, n_obj, req_objects, req_order in CELLS:
        condition_name = f"{n_obj}obj_order_gf_{label}"
        cell_results = [
            r for r in results
            if r.condition == condition_name and r.agent_type == agent_type.value
        ]
        if cell_results:
            correct = sum(1 for r in cell_results if r.success)
            total = len(cell_results)
            obj_correct = sum(1 for r in cell_results if r.objects_correct)
            order_correct = sum(1 for r in cell_results if r.order_correct)
            avg_gf = sum(len(r.gradual_feedback_events) for r in cell_results) / total
            logger.info(
                f"  Cell {label} ({agent_type.value}/{n_obj}-obj): "
                f"{correct}/{total} ({100*correct/total:.1f}%) | "
                f"Objects: {obj_correct}/{total} | Order: {order_correct}/{total} | "
                f"Avg GF events: {avg_gf:.1f}"
            )
    logger.info(f"Total cost: ${cost_tracker.total_cost:.2f}")
    logger.info(f"Summary saved: {summary_path}")

    return results


if __name__ == "__main__":
    asyncio.run(run_gradual_feedback_experiments())
