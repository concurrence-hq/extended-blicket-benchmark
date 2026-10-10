"""Run 11a: ORDER_STOCHASTIC Combined Condition Pilot.

Purpose:
  Test whether agents can identify a combined order-sensitive AND stochastic
  rule (A→B, p=0.7). The rule requires both correct placement order AND
  produces probabilistic activation — agents must detect both properties.

Cells (single session):
  A: base_llm   × 15 episodes (seeds 1101-1115)
  B: full_cg_db × 15 episodes (seeds 1101-1115)

Total: 30 episodes.

Usage:
  python experiments/run_11a_order_stochastic_pilot.py
"""

from __future__ import annotations

RUN_NAME = "run11a"

import asyncio
import json
import logging
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.blicket_env import CausalRule, CausalRuleType
from environment.extended_env import ExtendedCondition, OrderConfig, StochasticConfig
from environment.extended_env_5obj import ExtendedBlicketEnv5
from environment.text_interface import TextInterface5
from agents.base_llm_agent import BaseLLMAgent
from agents.full_agent import FullAgent
from agents.dynamic_behaviors_10c import create_blicket_behaviors_10c
from experiments.config import (
    AgentType,
    CostTracker,
    ExperimentConfig,
)
from experiments.episode_runner import EpisodeResult, run_episode

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Constants ---
NUM_OBJECTS = 5
REQUIRED_OBJECTS = {"A", "B"}
REQUIRED_ORDER = ["A", "B"]
ACTIVATION_PROBABILITY = 0.7
MAX_STEPS = 75
EPISODES_PER_CELL = 15
BASE_SEED = 1101
BUDGET_CAP = 75.0
MAX_COST_PER_EPISODE = 3.0


def create_agent(
    agent_type: AgentType,
    client: anthropic.AsyncAnthropic,
    config: ExperimentConfig,
    text_interface: TextInterface5,
) -> BaseLLMAgent:
    """Create an agent of the specified type."""
    if agent_type == AgentType.BASE_LLM:
        return BaseLLMAgent(
            llm_client=client,
            model=config.action_model,
            text_interface=text_interface,
        )
    elif agent_type == AgentType.FULL_CG_DB:
        return FullAgent(
            llm_client=client,
            model=config.action_model,
            trigger_model=config.trigger_model,
            text_interface=text_interface,
            num_objects=NUM_OBJECTS,
            behaviors=create_blicket_behaviors_10c(),
        )
    raise ValueError(f"Unknown agent type: {agent_type}")


def create_env(seed: int) -> ExtendedBlicketEnv5:
    """Create a 5-object order-stochastic environment."""
    rule = CausalRule(
        rule_type=CausalRuleType.ORDER_SENSITIVE_STOCHASTIC,
        required_objects=REQUIRED_OBJECTS,
    )
    return ExtendedBlicketEnv5(
        causal_rule=rule,
        condition=ExtendedCondition.ORDER_STOCHASTIC,
        max_steps=MAX_STEPS,
        order_config=OrderConfig(required_order=REQUIRED_ORDER),
        stochastic_config=StochasticConfig(activation_probability=ACTIVATION_PROBABILITY),
        rng_seed=seed,
    )


async def run_cell(
    cell_label: str,
    agent_type: AgentType,
    config: ExperimentConfig,
    client: anthropic.AsyncAnthropic,
    cost_tracker: CostTracker,
    trace_dir: Path,
    completed_ref: list[int],
    total_episodes: int,
) -> list[EpisodeResult]:
    """Run a single cell of episodes."""
    results: list[EpisodeResult] = []
    condition_name = f"order_stochastic_{cell_label}"

    for ep in range(EPISODES_PER_CELL):
        if not cost_tracker.check_budget():
            logger.warning(
                f"Budget exhausted (${cost_tracker.total_cost:.2f}/{BUDGET_CAP}). Stopping."
            )
            break

        seed = BASE_SEED + ep

        logger.info(
            f"[{completed_ref[0] + 1}/{total_episodes}] "
            f"Cell {cell_label}: {agent_type.value} / seed={seed} / ep{ep}"
        )

        env = create_env(seed)
        ti = TextInterface5(include_order_stochastic=True)
        agent = create_agent(agent_type, client, config, ti)

        result = await run_episode(
            env=env,
            agent=agent,
            condition=condition_name,
            episode_number=ep,
            trace_dir=trace_dir,
            max_cost_per_episode=MAX_COST_PER_EPISODE,
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
        completed_ref[0] += 1

        order_str = ""
        if result.order_correct is not None:
            order_str = f" | Order: {'CORRECT' if result.order_correct else 'WRONG'}"

        logger.info(
            f"  Result: {'CORRECT' if result.success else 'WRONG'}{order_str} | "
            f"Steps: {result.steps_taken} | Cost: ${episode_cost:.3f} | "
            f"Total: ${cost_tracker.total_cost:.2f}"
        )

    return results


async def run_order_stochastic_pilot() -> list[EpisodeResult]:
    """Run all Run 11a cells in a single session."""

    config = ExperimentConfig(
        episodes_per_condition=EPISODES_PER_CELL,
        max_steps_per_episode=MAX_STEPS,
        trace_dir=Path(f"data/traces_{RUN_NAME}"),
        results_dir=Path(f"data/results_{RUN_NAME}"),
        base_seed=BASE_SEED,
        max_total_cost=BUDGET_CAP,
        max_cost_per_episode=MAX_COST_PER_EPISODE,
    )

    client = anthropic.AsyncAnthropic()
    cost_tracker = CostTracker(max_total=config.max_total_cost)
    all_results: list[EpisodeResult] = []

    total_episodes = EPISODES_PER_CELL * 2  # A + B = 30
    completed_ref = [0]

    logger.info(f"Run 11a: ORDER_STOCHASTIC Combined Condition Pilot — {total_episodes} episodes")
    logger.info(f"  Rule: A→B, p={ACTIVATION_PROBABILITY}")
    logger.info(f"  Objects: {NUM_OBJECTS}, Required: {sorted(REQUIRED_OBJECTS)}, Order: {REQUIRED_ORDER}")
    logger.info(f"  Max steps: {MAX_STEPS}, Budget cap: ${BUDGET_CAP}")
    logger.info(f"  Seeds: {BASE_SEED}-{BASE_SEED + EPISODES_PER_CELL - 1}")

    cells = [
        {"label": "A", "agent_type": AgentType.BASE_LLM, "trace_subdir": "base_llm"},
        {"label": "B", "agent_type": AgentType.FULL_CG_DB, "trace_subdir": "full_cg_db"},
    ]

    for cell in cells:
        label = cell["label"]
        agent_type = cell["agent_type"]

        logger.info(f"\n{'='*70}")
        logger.info(f"CELL {label}: {agent_type.value} × {EPISODES_PER_CELL} episodes")
        logger.info(f"{'='*70}")

        trace_dir = Path(f"data/traces_{RUN_NAME}/{cell['trace_subdir']}")

        cell_results = await run_cell(
            cell_label=label,
            agent_type=agent_type,
            config=config,
            client=client,
            cost_tracker=cost_tracker,
            trace_dir=trace_dir,
            completed_ref=completed_ref,
            total_episodes=total_episodes,
        )
        all_results.extend(cell_results)

        if not cost_tracker.check_budget():
            logger.warning(f"Budget exhausted after Cell {label}. Stopping.")
            break

    # === Save summary ===
    summary_path = config.results_dir / "order_stochastic_pilot.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": RUN_NAME,
        "description": (
            "Run 11a: ORDER_STOCHASTIC Combined Condition Pilot. "
            "Tests whether agents can identify a combined order-sensitive AND stochastic "
            "rule (A→B, p=0.7). 2 cells: A (base_llm), B (full_cg_db). "
            f"{EPISODES_PER_CELL} episodes per cell, {total_episodes} total."
        ),
        "environment": {
            "num_objects": NUM_OBJECTS,
            "required_objects": sorted(REQUIRED_OBJECTS),
            "required_order": REQUIRED_ORDER,
            "activation_probability": ACTIVATION_PROBABILITY,
            "max_steps": MAX_STEPS,
            "condition": "order_stochastic",
            "rule_type": "order_sensitive_stochastic",
        },
        "instrumentation": {
            "temperature": 1.0,
            "action_model": config.action_model,
            "trigger_model": config.trigger_model,
        },
        "cells": [
            {
                "label": "A",
                "agent_type": AgentType.BASE_LLM.value,
                "episodes": EPISODES_PER_CELL,
                "seeds": f"{BASE_SEED}-{BASE_SEED + EPISODES_PER_CELL - 1}",
                "behavior_version": "N/A",
            },
            {
                "label": "B",
                "agent_type": AgentType.FULL_CG_DB.value,
                "episodes": EPISODES_PER_CELL,
                "seeds": f"{BASE_SEED}-{BASE_SEED + EPISODES_PER_CELL - 1}",
                "behavior_version": "10c",
            },
        ],
        "gradual_feedback_config": {
            "enabled": False,
        },
        "cost_summary": cost_tracker.get_summary(),
        "results": [],
    }

    for r in all_results:
        summary["results"].append({
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
        })

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # === Print summary ===
    logger.info(f"\n{'='*70}")
    logger.info("Run 11a: ORDER_STOCHASTIC Pilot Results")
    logger.info(f"{'='*70}")

    for cell_label, agent_type in [
        ("A", AgentType.BASE_LLM),
        ("B", AgentType.FULL_CG_DB),
    ]:
        condition_name = f"order_stochastic_{cell_label}"
        cell_results = [r for r in all_results if r.condition == condition_name]
        if cell_results:
            correct = sum(1 for r in cell_results if r.success)
            total = len(cell_results)
            submitted = sum(1 for r in cell_results if r.answer_submitted)
            obj_correct = sum(1 for r in cell_results if r.objects_correct)
            rt_correct = sum(1 for r in cell_results if r.rule_type_correct)
            order_correct = sum(
                1 for r in cell_results if r.order_correct
            )
            avg_steps = sum(r.steps_taken for r in cell_results) / total
            logger.info(
                f"\n  Cell {cell_label} ({agent_type.value}): "
                f"{correct}/{total} ({100*correct/total:.1f}%) | "
                f"Submitted: {submitted}/{total} | "
                f"RuleType: {rt_correct}/{total} | "
                f"Objects: {obj_correct}/{total} | "
                f"Order: {order_correct}/{total} | "
                f"Avg steps: {avg_steps:.1f}"
            )

    # Paired analysis (A vs B, same seeds)
    a_results = [r for r in all_results if r.condition == "order_stochastic_A"]
    b_results = [r for r in all_results if r.condition == "order_stochastic_B"]
    if len(a_results) == len(b_results) == EPISODES_PER_CELL:
        logger.info(f"\n--- Paired Analysis (A vs B) ---")
        concordant = sum(
            1 for a, b in zip(a_results, b_results) if a.success == b.success
        )
        a_only = sum(
            1 for a, b in zip(a_results, b_results) if a.success and not b.success
        )
        b_only = sum(
            1 for a, b in zip(a_results, b_results) if not a.success and b.success
        )
        logger.info(
            f"  Concordant: {concordant}, A-only: {a_only}, B-only: {b_only}"
        )

    logger.info(f"\nTotal cost: ${cost_tracker.total_cost:.2f}")
    logger.info(f"Summary saved: {summary_path}")

    return all_results


def main():
    asyncio.run(run_order_stochastic_pilot())


if __name__ == "__main__":
    main()
