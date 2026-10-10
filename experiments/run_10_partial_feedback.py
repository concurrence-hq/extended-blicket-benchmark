"""Run 10: Partial Feedback Order-Sensitive Pilot.

Tests whether revealing correct object identities (but NOT order) via partial
feedback helps agents solve order-sensitive tasks. Feedback is triggered once
per episode after sufficient exploration with no detector activations.

4 cells × 15 episodes = 60 episodes total.

Cell A: base_llm    × 3-obj order (A→D→C) × 15 eps × 20 steps + partial feedback
Cell B: full_cg_db  × 3-obj order (A→D→C) × 15 eps × 20 steps + partial feedback
Cell C: base_llm    × 3-obj order (A→D→C) × 15 eps × 40 steps + partial feedback
Cell D: full_cg_db  × 3-obj order (A→D→C) × 15 eps × 40 steps + partial feedback
"""

from __future__ import annotations

RUN_NAME = "run10"

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
    PartialFeedbackConfig,
)
from experiments.episode_runner import EpisodeResult, run_episode

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

NUM_OBJECTS = 5
BASE_SEED = 1001
BUDGET_CAP = 85.0

REQUIRED_OBJECTS = {"A", "D", "C"}
REQUIRED_ORDER = ["A", "D", "C"]

PF_CONFIG = PartialFeedbackConfig(
    enabled=True,
    min_steps=5,
    min_unique_combos=3,
    correct_objects=["A", "C", "D"],
)

# Cell definitions: (label, agent_type, max_steps)
CELLS = [
    ("A", AgentType.BASE_LLM, 20),
    ("B", AgentType.FULL_CG_DB, 20),
    ("C", AgentType.BASE_LLM, 40),
    ("D", AgentType.FULL_CG_DB, 40),
]
EPISODES_PER_CELL = 15


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
    max_steps: int,
    seed: int,
) -> ExtendedBlicketEnv5:
    rule = CausalRule(
        rule_type=CausalRuleType.ORDER_SENSITIVE,
        required_objects=REQUIRED_OBJECTS,
    )
    return ExtendedBlicketEnv5(
        causal_rule=rule,
        condition=ExtendedCondition.ORDER_SENSITIVE,
        max_steps=max_steps,
        order_config=OrderConfig(required_order=REQUIRED_ORDER),
        rng_seed=seed,
    )


async def run_partial_feedback_experiments(
    config: ExperimentConfig | None = None,
) -> list[EpisodeResult]:
    """Run all Run 10 partial feedback experiments."""
    if config is None:
        config = ExperimentConfig(
            episodes_per_condition=EPISODES_PER_CELL,
            max_steps_per_episode=40,
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

    for label, agent_type, max_steps in CELLS:
        condition_name = f"3obj_order_pf_{label}"

        for ep in range(EPISODES_PER_CELL):
            if not cost_tracker.check_budget():
                logger.warning(
                    f"Budget exhausted (${cost_tracker.total_cost:.2f}/{BUDGET_CAP}). Stopping."
                )
                break

            logger.info(
                f"[{completed + 1}/{total_episodes}] "
                f"Cell {label}: {agent_type.value} / 3-obj order+pf / ep{ep}"
            )

            seed = config.get_episode_seed(condition_name, agent_type.value, ep)
            env = create_env(max_steps, seed)
            agent = create_agent(agent_type, client, config)

            result = await run_episode(
                env=env,
                agent=agent,
                condition=condition_name,
                episode_number=ep,
                trace_dir=config.trace_dir,
                max_cost_per_episode=config.max_cost_per_episode,
                partial_feedback_config=PF_CONFIG,
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

            pf_str = ""
            if result.partial_feedback_step is not None:
                pf_str = f" | PF@step{result.partial_feedback_step}"

            logger.info(
                f"  Result: {'CORRECT' if result.success else 'WRONG'}{order_str}{pf_str} | "
                f"Steps: {result.steps_taken} | Cost: ${episode_cost:.3f} | "
                f"Total: ${cost_tracker.total_cost:.2f}"
            )

    # Save summary
    summary_path = config.results_dir / "partial_feedback_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": RUN_NAME,
        "description": "Partial feedback order-sensitive pilot: reveal correct objects (not order) after exploration",
        "num_objects": NUM_OBJECTS,
        "base_seed": BASE_SEED,
        "partial_feedback_config": {
            "enabled": PF_CONFIG.enabled,
            "min_steps": PF_CONFIG.min_steps,
            "min_unique_combos": PF_CONFIG.min_unique_combos,
            "correct_objects": PF_CONFIG.correct_objects,
        },
        "cells": [
            {
                "label": label,
                "agent_type": agent_type.value,
                "n_order_objects": len(REQUIRED_ORDER),
                "required_order": REQUIRED_ORDER,
                "max_steps": max_steps,
            }
            for label, agent_type, max_steps in CELLS
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
                "partial_feedback_step": r.partial_feedback_step,
                "partial_feedback_message": r.partial_feedback_message,
            }
            for r in results
        ],
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    logger.info(f"\n{'='*60}")
    logger.info("Run 10: Partial Feedback Order-Sensitive Results")
    logger.info(f"{'='*60}")
    for label, agent_type, max_steps in CELLS:
        condition_name = f"3obj_order_pf_{label}"
        cell_results = [
            r for r in results
            if r.condition == condition_name and r.agent_type == agent_type.value
        ]
        if cell_results:
            correct = sum(1 for r in cell_results if r.success)
            total = len(cell_results)
            obj_correct = sum(1 for r in cell_results if r.objects_correct)
            order_correct = sum(1 for r in cell_results if r.order_correct)
            pf_count = sum(1 for r in cell_results if r.partial_feedback_step is not None)
            logger.info(
                f"  Cell {label} ({agent_type.value}/{max_steps}steps): "
                f"{correct}/{total} ({100*correct/total:.1f}%) | "
                f"Objects: {obj_correct}/{total} | Order: {order_correct}/{total} | "
                f"PF delivered: {pf_count}/{total}"
            )
    logger.info(f"Total cost: ${cost_tracker.total_cost:.2f}")
    logger.info(f"Summary saved: {summary_path}")

    return results


if __name__ == "__main__":
    asyncio.run(run_partial_feedback_experiments())
