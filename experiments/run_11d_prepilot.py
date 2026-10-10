"""Run 11d Pre-Pilot: Static 7-state CG ordering smoke test.

Purpose:
  Qualitative pre-pilot to determine the correct static CG design for Run 11d.
  Compares two v3 CG topologies:
    - v3_static_linear: 7-state linear chain (IE→ORS→CT→OT→RT→HE→V)
    - v3_static_branch: 7-state branching (CT can skip OT/RT)

Changes from Run 11c:
  - Change G: Revised ORS guidelines (elimination language, scorecard, individual-first)
  - Change H: Two new states (ORDER_TESTING, RELIABILITY_TESTING) with backstops
  - Change I: ORS backstop lowered from 25→20 steps
  - CG-only agents (no base_llm or full_cg_db)
  - 5 episodes per cell, 10 total

Cells:
  S1: static_linear  (CG_ONLY, v3_static_linear) × 5 episodes
  S2: static_branch  (CG_ONLY, v3_static_branch) × 5 episodes

Total: 10 episodes (~$30 estimated, $50 budget cap).

Usage:
  python experiments/run_11d_prepilot.py
"""

from __future__ import annotations

RUN_NAME = "run11d_prepilot"

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
from agents.cg_agent import CGAgent
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
EPISODES_PER_CELL = 5
BASE_SEED = 1401
BUDGET_CAP = 50.0
MAX_COST_PER_EPISODE = 5.0
INITIAL_EXPLORATION_MAX_STEPS = 15
OBJECT_SCREENING_MAX_STEPS = 20       # Change I: lowered from v2's 25
ORDER_TESTING_MAX_STEPS = 15
RELIABILITY_TESTING_MAX_STEPS = 10

# Cell definitions: (label, cg_version, trace_subdir)
CELLS = [
    ("S1", "v3_static_linear", "static_linear"),
    ("S2", "v3_static_branch", "static_branch"),
]


def generate_randomized_order_rule(
    seed: int,
    all_labels: list[str] | None = None,
    n_correct: int = N_CORRECT,
) -> dict:
    """Generate a randomized order-sensitive stochastic rule for one episode."""
    if all_labels is None:
        all_labels = ALL_LABELS
    rng = random.Random(seed)
    correct = rng.sample(all_labels, n_correct)
    rng.shuffle(correct)
    distractors = set(all_labels) - set(correct)
    return {
        "correct_objects": set(correct),
        "required_order": list(correct),
        "distractors": distractors,
    }


def create_agent(
    cg_version: str,
    client: anthropic.AsyncAnthropic,
    config: ExperimentConfig,
    text_interface: TextInterface5,
) -> CGAgent:
    """Create a CG-only agent with the specified CG version."""
    return CGAgent(
        llm_client=client,
        model=config.action_model,
        text_interface=text_interface,
        num_objects=NUM_OBJECTS,
        initial_exploration_max_steps=INITIAL_EXPLORATION_MAX_STEPS,
        cg_version=cg_version,
        object_list=ALL_LABELS,
        object_screening_max_steps=OBJECT_SCREENING_MAX_STEPS,
        order_testing_max_steps=ORDER_TESTING_MAX_STEPS,
        reliability_testing_max_steps=RELIABILITY_TESTING_MAX_STEPS,
    )


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
    cg_version: str,
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
    condition_name = f"prepilot_{cell_label}"

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
            f"Cell {cell_label}: cg_only({cg_version}) / seed={seed} / "
            f"order={req_order} / ep{ep}"
        )

        env = create_env(req_objects, req_order, seed)
        ti = TextInterface5(include_order_stochastic=True)
        agent = create_agent(cg_version, client, config, ti)

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
            agent="cg_only",
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


async def run_prepilot() -> list[EpisodeResult]:
    """Run all Run 11d pre-pilot cells."""

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

    total_episodes = EPISODES_PER_CELL * len(CELLS)  # 10
    completed_ref = [0]

    # Pre-generate all episode rules (same seeds across all cells)
    episode_rules: list[tuple[int, dict]] = []
    for ep in range(EPISODES_PER_CELL):
        rule_seed = BASE_SEED + ep
        rule = generate_randomized_order_rule(seed=rule_seed)
        episode_rules.append((rule_seed, rule))

    logger.info(f"Run 11d Pre-Pilot: Static 7-state CG Ordering Smoke Test")
    logger.info(f"  Objects: {NUM_OBJECTS}, N_correct: {N_CORRECT}, p={ACTIVATION_PROBABILITY}")
    logger.info(f"  Max steps: {MAX_STEPS}, Budget cap: ${BUDGET_CAP}")
    logger.info(f"  IE backstop: {INITIAL_EXPLORATION_MAX_STEPS} steps")
    logger.info(f"  ORS backstop: {OBJECT_SCREENING_MAX_STEPS} steps (Change I: lowered from 25)")
    logger.info(f"  OT backstop: {ORDER_TESTING_MAX_STEPS} steps")
    logger.info(f"  RT backstop: {RELIABILITY_TESTING_MAX_STEPS} steps")
    logger.info(f"  Seeds: {BASE_SEED}-{BASE_SEED + EPISODES_PER_CELL - 1}")
    logger.info(f"  Cells: {', '.join(f'{l}({v})' for l, v, _ in CELLS)}")

    logger.info("\nPre-generated episode rules:")
    for ep, (rule_seed, rule) in enumerate(episode_rules):
        logger.info(
            f"  Episode {ep}: seed={rule_seed}, "
            f"correct={sorted(rule['correct_objects'])}, "
            f"order={rule['required_order']}, "
            f"distractors={sorted(rule['distractors'])}"
        )

    for cell_label, cg_version, trace_subdir in CELLS:
        logger.info(f"\n{'='*70}")
        logger.info(f"CELL {cell_label}: cg_only({cg_version}) × {EPISODES_PER_CELL} episodes")
        logger.info(f"{'='*70}")

        trace_dir = Path(f"data/traces_{RUN_NAME}/{trace_subdir}")

        cell_results = await run_cell(
            cell_label=cell_label,
            cg_version=cg_version,
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
    summary_path = config.results_dir / "prepilot_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": RUN_NAME,
        "description": (
            "Run 11d Pre-Pilot: Static 7-state CG ordering smoke test. "
            "Tests Change G (revised ORS guidelines), Change H (ORDER_TESTING + "
            "RELIABILITY_TESTING states), and Change I (ORS backstop 25→20). "
            "2 cells: S1 (v3_static_linear), S2 (v3_static_branch). "
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
            "initial_exploration_max_steps": INITIAL_EXPLORATION_MAX_STEPS,
            "object_screening_max_steps": OBJECT_SCREENING_MAX_STEPS,
            "order_testing_max_steps": ORDER_TESTING_MAX_STEPS,
            "reliability_testing_max_steps": RELIABILITY_TESTING_MAX_STEPS,
        },
        "cells": [
            {
                "label": cell_label,
                "agent_type": "cg_only",
                "cg_version": cg_version,
                "episodes": EPISODES_PER_CELL,
                "seeds": f"{BASE_SEED}-{BASE_SEED + EPISODES_PER_CELL - 1}",
                "backstops": {
                    "initial_exploration_max_steps": INITIAL_EXPLORATION_MAX_STEPS,
                    "object_screening_max_steps": OBJECT_SCREENING_MAX_STEPS,
                    "order_testing_max_steps": ORDER_TESTING_MAX_STEPS,
                    "reliability_testing_max_steps": RELIABILITY_TESTING_MAX_STEPS,
                },
            }
            for cell_label, cg_version, _ in CELLS
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
    logger.info("Run 11d Pre-Pilot Results")
    logger.info(f"{'='*70}")

    for cell_label, cg_version, _ in CELLS:
        condition_name = f"prepilot_{cell_label}"
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
                f"\n  Cell {cell_label} ({cg_version}): "
                f"{correct}/{total} ({100*correct/total:.1f}%) | "
                f"Submitted: {submitted}/{total} | "
                f"RuleType: {rt_correct}/{total} | "
                f"Objects: {obj_correct}/{total} | "
                f"Order: {order_correct}/{total} | "
                f"Avg steps: {avg_steps:.1f}"
            )

    # Paired analysis
    s1_results = [r for r in all_results if r.condition == "prepilot_S1"]
    s2_results = [r for r in all_results if r.condition == "prepilot_S2"]

    if len(s1_results) == len(s2_results) == EPISODES_PER_CELL:
        logger.info(f"\n--- Paired Analysis ---")
        concordant = sum(
            1 for x, y in zip(s1_results, s2_results) if x.success == y.success
        )
        s1_only = sum(
            1 for x, y in zip(s1_results, s2_results) if x.success and not y.success
        )
        s2_only = sum(
            1 for x, y in zip(s1_results, s2_results) if not x.success and y.success
        )
        logger.info(
            f"  S1(linear) vs S2(branch): "
            f"Concordant={concordant}, S1-only={s1_only}, S2-only={s2_only}"
        )

    logger.info(f"\nTotal cost: ${cost_tracker.total_cost:.2f}")
    logger.info(f"Summary saved: {summary_path}")

    return all_results


def main():
    asyncio.run(run_prepilot())


if __name__ == "__main__":
    main()
