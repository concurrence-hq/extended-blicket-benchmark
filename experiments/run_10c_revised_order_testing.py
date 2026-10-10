"""Run 10c: Revised ORDER_TESTING with Deconfounded Labels.

Two interventions:
  1. Revised ORDER_TESTING DB — systematic permutation elimination protocol
     (full_cg_db cells only; base_llm unchanged).
  2. Randomized label-to-role mapping — each episode independently selects
     which 3 of 5 labels are correct and in what order. Eliminates the
     alphabetical prior confound discovered in Run 10b.

4 cells × 15 episodes = 60 episodes total.

Cell A: base_llm    × 40 steps × randomized 3-obj order + gradual feedback
Cell B: full_cg_db  × 40 steps × randomized 3-obj order + gradual feedback [behaviors_10c]
Cell C: base_llm    × 55 steps × randomized 3-obj order + gradual feedback
Cell D: full_cg_db  × 55 steps × randomized 3-obj order + gradual feedback [behaviors_10c]
"""

from __future__ import annotations

RUN_NAME = "run10c"

import asyncio
import json
import logging
import random
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
from agents.dynamic_behaviors_10c import create_blicket_behaviors_10c
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
ALL_LABELS = ["A", "B", "C", "D", "E"]
N_CORRECT = 3
BASE_SEED = 1101
BUDGET_CAP = 80.0
EPISODES_PER_CELL = 15

# Cell definitions: (label, agent_type, max_steps)
CELLS = [
    ("A", AgentType.BASE_LLM, 40),
    ("B", AgentType.FULL_CG_DB, 40),
    ("C", AgentType.BASE_LLM, 55),
    ("D", AgentType.FULL_CG_DB, 55),
]


def generate_randomized_order_rule(
    seed: int,
    all_labels: list[str] | None = None,
    n_correct: int = 3,
) -> dict:
    """Generate a randomized order-sensitive rule for one episode.

    Args:
        seed: Episode seed for deterministic randomization.
        all_labels: Full set of object labels. Defaults to A-E.
        n_correct: Number of objects in the correct rule.

    Returns:
        dict with keys:
            'correct_objects': set of correct labels
            'required_order': list of correct labels in required sequence
            'distractors': set of distractor labels
    """
    if all_labels is None:
        all_labels = ALL_LABELS
    rng = random.Random(seed)
    correct = rng.sample(all_labels, n_correct)
    rng.shuffle(correct)  # Random ordering
    distractors = set(all_labels) - set(correct)
    return {
        "correct_objects": set(correct),
        "required_order": list(correct),
        "distractors": distractors,
    }


def create_agent(
    agent_type: AgentType,
    client: anthropic.AsyncAnthropic,
    config: ExperimentConfig,
    max_steps: int,
) -> BaseLLMAgent:
    ti = TextInterface5(include_order_sensitive=True)
    if agent_type == AgentType.BASE_LLM:
        return BaseLLMAgent(
            llm_client=client,
            model=config.action_model,
            text_interface=ti,
        )
    elif agent_type == AgentType.FULL_CG_DB:
        return FullAgent(
            llm_client=client,
            model=config.action_model,
            trigger_model=config.trigger_model,
            text_interface=ti,
            num_objects=NUM_OBJECTS,
            behaviors=create_blicket_behaviors_10c(),
        )
    raise ValueError(f"Unknown agent type: {agent_type}")


def create_env(
    required_objects: set[str],
    required_order: list[str],
    max_steps: int,
    seed: int,
) -> ExtendedBlicketEnv5:
    rule = CausalRule(
        rule_type=CausalRuleType.ORDER_SENSITIVE,
        required_objects=required_objects,
    )
    return ExtendedBlicketEnv5(
        causal_rule=rule,
        condition=ExtendedCondition.ORDER_SENSITIVE,
        max_steps=max_steps,
        order_config=OrderConfig(required_order=required_order),
        rng_seed=seed,
    )


async def run_revised_order_experiments(
    config: ExperimentConfig | None = None,
) -> list[EpisodeResult]:
    """Run all Run 10c revised ORDER_TESTING experiments."""
    if config is None:
        config = ExperimentConfig(
            episodes_per_condition=EPISODES_PER_CELL,
            max_steps_per_episode=55,  # max across cells; per-cell budget set via env
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

    # Pre-generate all episode rules (same seeds across all cells)
    episode_rules = []
    for ep in range(EPISODES_PER_CELL):
        rule_seed = BASE_SEED + ep
        rule = generate_randomized_order_rule(seed=rule_seed)
        episode_rules.append((rule_seed, rule))

    logger.info(f"Run 10c: {total_episodes} episodes across {len(CELLS)} cells")
    logger.info(f"Randomized rules for {EPISODES_PER_CELL} episodes (seeds {BASE_SEED}-{BASE_SEED + EPISODES_PER_CELL - 1})")
    for ep, (rule_seed, rule) in enumerate(episode_rules):
        logger.info(
            f"  Episode {ep}: seed={rule_seed}, "
            f"correct={sorted(rule['correct_objects'])}, "
            f"order={rule['required_order']}, "
            f"distractors={sorted(rule['distractors'])}"
        )

    for cell_label, agent_type, max_steps in CELLS:
        condition_name = f"3obj_order_rand_{cell_label}"

        for ep in range(EPISODES_PER_CELL):
            if not cost_tracker.check_budget():
                logger.warning(
                    f"Budget exhausted (${cost_tracker.total_cost:.2f}/{BUDGET_CAP}). Stopping."
                )
                break

            rule_seed, rule = episode_rules[ep]
            req_objects = rule["correct_objects"]
            req_order = rule["required_order"]

            gf_config = GradualFeedbackConfig(
                enabled=True,
                checkpoint_interval=5,
                correct_objects=sorted(req_objects),
            )

            logger.info(
                f"[{completed + 1}/{total_episodes}] "
                f"Cell {cell_label}: {agent_type.value} / {max_steps} steps / "
                f"order={req_order} / ep{ep}"
            )

            seed = config.get_episode_seed(condition_name, agent_type.value, ep)
            env = create_env(req_objects, req_order, max_steps, seed)
            agent = create_agent(agent_type, client, config, max_steps)

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
    summary_path = config.results_dir / "revised_order_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": RUN_NAME,
        "description": (
            "Revised ORDER_TESTING with deconfounded labels: systematic "
            "permutation elimination protocol + randomized label-to-role "
            "mapping per episode; 2 step budgets (40, 55)"
        ),
        "num_objects": NUM_OBJECTS,
        "n_correct": N_CORRECT,
        "base_seed": BASE_SEED,
        "gradual_feedback_config": {
            "enabled": True,
            "checkpoint_interval": 5,
        },
        "episode_rules": [
            {
                "episode": ep,
                "rule_seed": rule_seed,
                "rule_correct_objects": sorted(rule["correct_objects"]),
                "rule_required_order": rule["required_order"],
                "rule_distractors": sorted(rule["distractors"]),
            }
            for ep, (rule_seed, rule) in enumerate(episode_rules)
        ],
        "cells": [
            {
                "label": cell_label,
                "agent_type": agent_type.value,
                "max_steps": max_steps,
                "behavior_version": "10c" if agent_type == AgentType.FULL_CG_DB else "N/A",
            }
            for cell_label, agent_type, max_steps in CELLS
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
                "rule_correct_objects": sorted(episode_rules[r.episode_number][1]["correct_objects"]),
                "rule_required_order": episode_rules[r.episode_number][1]["required_order"],
                "rule_distractors": sorted(episode_rules[r.episode_number][1]["distractors"]),
                "rule_seed": episode_rules[r.episode_number][0],
            }
            for r in results
        ],
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    logger.info(f"\n{'='*70}")
    logger.info("Run 10c: Revised ORDER_TESTING + Randomized Labels Results")
    logger.info(f"{'='*70}")
    for cell_label, agent_type, max_steps in CELLS:
        condition_name = f"3obj_order_rand_{cell_label}"
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
                f"  Cell {cell_label} ({agent_type.value}/{max_steps}steps): "
                f"{correct}/{total} ({100*correct/total:.1f}%) | "
                f"Objects: {obj_correct}/{total} | Order: {order_correct}/{total} | "
                f"Avg GF events: {avg_gf:.1f}"
            )
    logger.info(f"Total cost: ${cost_tracker.total_cost:.2f}")
    logger.info(f"Summary saved: {summary_path}")

    return results


if __name__ == "__main__":
    asyncio.run(run_revised_order_experiments())
