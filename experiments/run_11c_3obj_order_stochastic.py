"""Run 11c: 3-Object ORDER_STOCHASTIC with v2 CG + Lowered Stochasticity Threshold.

Purpose:
  Tests two changes on top of Run 11b:
  - Change E: v2 CG with OBJECT_RELEVANCE_SCREENING (5-state base graph)
  - Change F: stochasticity_hypothesis fire_threshold lowered 7.0 → 6.0

Key differences from Run 11b:
  - cg_version="v2" for CG_ONLY and FULL_CG_DB agents
  - object_screening_max_steps=25 backstop for v2 screening state
  - behavior_version="11c" (lowered stochasticity threshold)
  - BASE_SEED=1301 (new seed range)
  - BUDGET_CAP=150.0 (slightly higher)

Cells (single session):
  A: base_llm     × 15 episodes (seeds 1301-1315)
  B: cg_only      × 15 episodes (seeds 1301-1315)
  C: full_cg_db   × 15 episodes (seeds 1301-1315)

Total: 45 episodes.

Usage:
  python experiments/run_11c_3obj_order_stochastic.py
"""

from __future__ import annotations

RUN_NAME = "run11c"

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
from environment.extended_env import ExtendedCondition, OrderConfig, StochasticConfig
from environment.extended_env_5obj import ExtendedBlicketEnv5
from environment.text_interface import TextInterface5
from agents.base_llm_agent import BaseLLMAgent
from agents.cg_agent import CGAgent
from agents.full_agent import FullAgent
from agents.dynamic_behaviors_11c import create_blicket_behaviors_11c
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
ALL_LABELS = ["A", "B", "C", "D", "E"]
N_CORRECT = 3
ACTIVATION_PROBABILITY = 0.7
MAX_STEPS = 100
EPISODES_PER_CELL = 15
BASE_SEED = 1301
BUDGET_CAP = 150.0
MAX_COST_PER_EPISODE = 3.0
INITIAL_EXPLORATION_MAX_STEPS = 15
OBJECT_SCREENING_MAX_STEPS = 25

# Cell definitions: (label, agent_type, trace_subdir)
CELLS = [
    ("A", AgentType.BASE_LLM, "base_llm"),
    ("B", AgentType.CG_ONLY, "cg_only"),
    ("C", AgentType.FULL_CG_DB, "full_cg_db"),
]


def generate_randomized_order_rule(
    seed: int,
    all_labels: list[str] | None = None,
    n_correct: int = N_CORRECT,
) -> dict:
    """Generate a randomized order-sensitive stochastic rule for one episode.

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
    text_interface: TextInterface5,
) -> BaseLLMAgent:
    """Create an agent of the specified type."""
    if agent_type == AgentType.BASE_LLM:
        return BaseLLMAgent(
            llm_client=client,
            model=config.action_model,
            text_interface=text_interface,
        )
    elif agent_type == AgentType.CG_ONLY:
        return CGAgent(
            llm_client=client,
            model=config.action_model,
            text_interface=text_interface,
            num_objects=NUM_OBJECTS,
            initial_exploration_max_steps=INITIAL_EXPLORATION_MAX_STEPS,
            cg_version="v2",
            object_list=ALL_LABELS,
            object_screening_max_steps=OBJECT_SCREENING_MAX_STEPS,
        )
    elif agent_type == AgentType.FULL_CG_DB:
        return FullAgent(
            llm_client=client,
            model=config.action_model,
            trigger_model=config.trigger_model,
            text_interface=text_interface,
            num_objects=NUM_OBJECTS,
            behaviors=create_blicket_behaviors_11c(),
            initial_exploration_max_steps=INITIAL_EXPLORATION_MAX_STEPS,
            cg_version="v2",
            object_list=ALL_LABELS,
            object_screening_max_steps=OBJECT_SCREENING_MAX_STEPS,
        )
    raise ValueError(f"Unknown agent type: {agent_type}")


def create_env(
    required_objects: set[str],
    required_order: list[str],
    seed: int,
) -> ExtendedBlicketEnv5:
    """Create a 5-object order-stochastic environment."""
    rule = CausalRule(
        rule_type=CausalRuleType.ORDER_SENSITIVE_STOCHASTIC,
        required_objects=required_objects,
    )
    return ExtendedBlicketEnv5(
        causal_rule=rule,
        condition=ExtendedCondition.ORDER_STOCHASTIC,
        max_steps=MAX_STEPS,
        order_config=OrderConfig(required_order=required_order),
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
    episode_rules: list[tuple[int, dict]],
    completed_ref: list[int],
    total_episodes: int,
) -> list[EpisodeResult]:
    """Run a single cell of episodes."""
    results: list[EpisodeResult] = []
    condition_name = f"3obj_order_stoch_{cell_label}"

    for ep in range(EPISODES_PER_CELL):
        if not cost_tracker.check_budget():
            logger.warning(
                f"Budget exhausted (${cost_tracker.total_cost:.2f}/{BUDGET_CAP}). Stopping."
            )
            break

        rule_seed, rule = episode_rules[ep]
        req_objects = rule["correct_objects"]
        req_order = rule["required_order"]
        seed = BASE_SEED + ep

        logger.info(
            f"[{completed_ref[0] + 1}/{total_episodes}] "
            f"Cell {cell_label}: {agent_type.value} / seed={seed} / "
            f"order={req_order} / ep{ep}"
        )

        env = create_env(req_objects, req_order, seed)
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


async def run_3obj_order_stochastic() -> list[EpisodeResult]:
    """Run all Run 11c cells in a single session."""

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

    total_episodes = EPISODES_PER_CELL * len(CELLS)  # 45
    completed_ref = [0]

    # Pre-generate all episode rules (same seeds across all cells)
    episode_rules: list[tuple[int, dict]] = []
    for ep in range(EPISODES_PER_CELL):
        rule_seed = BASE_SEED + ep
        rule = generate_randomized_order_rule(seed=rule_seed)
        episode_rules.append((rule_seed, rule))

    logger.info(f"Run 11c: 3-Object ORDER_STOCHASTIC — {total_episodes} episodes")
    logger.info(f"  Objects: {NUM_OBJECTS}, N_correct: {N_CORRECT}, p={ACTIVATION_PROBABILITY}")
    logger.info(f"  Max steps: {MAX_STEPS}, Budget cap: ${BUDGET_CAP}")
    logger.info(f"  Backstop: initial_exploration_max_steps={INITIAL_EXPLORATION_MAX_STEPS}")
    logger.info(f"  Screening backstop: object_screening_max_steps={OBJECT_SCREENING_MAX_STEPS}")
    logger.info(f"  CG version: v2 (5-state with OBJECT_RELEVANCE_SCREENING)")
    logger.info(f"  Seeds: {BASE_SEED}-{BASE_SEED + EPISODES_PER_CELL - 1}")
    logger.info(f"  Cells: {', '.join(f'{l}({t.value})' for l, t, _ in CELLS)}")

    logger.info("\nPre-generated episode rules:")
    for ep, (rule_seed, rule) in enumerate(episode_rules):
        logger.info(
            f"  Episode {ep}: seed={rule_seed}, "
            f"correct={sorted(rule['correct_objects'])}, "
            f"order={rule['required_order']}, "
            f"distractors={sorted(rule['distractors'])}"
        )

    for cell_label, agent_type, trace_subdir in CELLS:
        logger.info(f"\n{'='*70}")
        logger.info(f"CELL {cell_label}: {agent_type.value} × {EPISODES_PER_CELL} episodes")
        logger.info(f"{'='*70}")

        trace_dir = Path(f"data/traces_{RUN_NAME}/{trace_subdir}")

        cell_results = await run_cell(
            cell_label=cell_label,
            agent_type=agent_type,
            config=config,
            client=client,
            cost_tracker=cost_tracker,
            trace_dir=trace_dir,
            episode_rules=episode_rules,
            completed_ref=completed_ref,
            total_episodes=total_episodes,
        )
        all_results.extend(cell_results)

        if not cost_tracker.check_budget():
            logger.warning(f"Budget exhausted after Cell {cell_label}. Stopping.")
            break

    # === Save summary ===
    summary_path = config.results_dir / "3obj_order_stochastic.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": RUN_NAME,
        "description": (
            "Run 11c: 3-Object ORDER_STOCHASTIC with v2 CG (OBJECT_RELEVANCE_SCREENING) "
            "and lowered stochasticity threshold (6.0). Tests Change E (v2 CG) + Change F "
            "(lower threshold). 3 cells: A (base_llm), B (cg_only), C (full_cg_db). "
            f"{EPISODES_PER_CELL} episodes per cell, {total_episodes} total."
        ),
        "environment": {
            "num_objects": NUM_OBJECTS,
            "n_correct": N_CORRECT,
            "activation_probability": ACTIVATION_PROBABILITY,
            "max_steps": MAX_STEPS,
            "condition": "order_stochastic",
            "rule_type": "order_sensitive_stochastic",
            "label_randomization": True,
        },
        "instrumentation": {
            "temperature": 1.0,
            "action_model": config.action_model,
            "trigger_model": config.trigger_model,
            "initial_exploration_max_steps": INITIAL_EXPLORATION_MAX_STEPS,
            "object_screening_max_steps": OBJECT_SCREENING_MAX_STEPS,
            "cg_version": "v2",
        },
        "cells": [
            {
                "label": cell_label,
                "agent_type": agent_type.value,
                "episodes": EPISODES_PER_CELL,
                "seeds": f"{BASE_SEED}-{BASE_SEED + EPISODES_PER_CELL - 1}",
                "behavior_version": (
                    "11c" if agent_type == AgentType.FULL_CG_DB
                    else "N/A"
                ),
                "cg_version": (
                    "v2" if agent_type in (AgentType.CG_ONLY, AgentType.FULL_CG_DB)
                    else "N/A"
                ),
                "initial_exploration_max_steps": (
                    INITIAL_EXPLORATION_MAX_STEPS
                    if agent_type in (AgentType.CG_ONLY, AgentType.FULL_CG_DB)
                    else "N/A"
                ),
                "object_screening_max_steps": (
                    OBJECT_SCREENING_MAX_STEPS
                    if agent_type in (AgentType.CG_ONLY, AgentType.FULL_CG_DB)
                    else "N/A"
                ),
            }
            for cell_label, agent_type, _ in CELLS
        ],
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
        "gradual_feedback_config": {
            "enabled": False,
        },
        "cost_summary": cost_tracker.get_summary(),
        "results": [],
    }

    for r in all_results:
        ep_rule = episode_rules[r.episode_number]
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
            "rule_seed": ep_rule[0],
            "rule_correct_objects": sorted(ep_rule[1]["correct_objects"]),
            "rule_required_order": ep_rule[1]["required_order"],
            "rule_distractors": sorted(ep_rule[1]["distractors"]),
        })

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # === Print summary ===
    logger.info(f"\n{'='*70}")
    logger.info("Run 11c: 3-Object ORDER_STOCHASTIC Results")
    logger.info(f"{'='*70}")

    for cell_label, agent_type, _ in CELLS:
        condition_name = f"3obj_order_stoch_{cell_label}"
        cell_results = [r for r in all_results if r.condition == condition_name]
        if cell_results:
            correct = sum(1 for r in cell_results if r.success)
            total = len(cell_results)
            submitted = sum(1 for r in cell_results if r.answer_submitted)
            obj_correct = sum(1 for r in cell_results if r.objects_correct)
            rt_correct = sum(1 for r in cell_results if r.rule_type_correct)
            order_correct = sum(1 for r in cell_results if r.order_correct)
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

    # Paired analysis (A vs B vs C, same seeds)
    a_results = [r for r in all_results if r.condition == "3obj_order_stoch_A"]
    b_results = [r for r in all_results if r.condition == "3obj_order_stoch_B"]
    c_results = [r for r in all_results if r.condition == "3obj_order_stoch_C"]

    if len(a_results) == len(b_results) == len(c_results) == EPISODES_PER_CELL:
        logger.info(f"\n--- Paired Analysis ---")
        for label_x, label_y, res_x, res_y in [
            ("A(base_llm)", "B(cg_only)", a_results, b_results),
            ("A(base_llm)", "C(full_cg_db)", a_results, c_results),
            ("B(cg_only)", "C(full_cg_db)", b_results, c_results),
        ]:
            concordant = sum(
                1 for x, y in zip(res_x, res_y) if x.success == y.success
            )
            x_only = sum(
                1 for x, y in zip(res_x, res_y) if x.success and not y.success
            )
            y_only = sum(
                1 for x, y in zip(res_x, res_y) if not x.success and y.success
            )
            logger.info(
                f"  {label_x} vs {label_y}: "
                f"Concordant={concordant}, {label_x}-only={x_only}, {label_y}-only={y_only}"
            )

    logger.info(f"\nTotal cost: ${cost_tracker.total_cost:.2f}")
    logger.info(f"Summary saved: {summary_path}")

    return all_results


def main():
    asyncio.run(run_3obj_order_stochastic())


if __name__ == "__main__":
    main()
