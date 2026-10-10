"""Run 10f: 4-Object Step Budget Pilot.

Purpose:
  Determine whether increasing the step budget from 75 to 100 or 125 steps
  lifts 4-object ORDER_SENSITIVE performance off the floor, and whether the
  lift is differential (favoring full_cg_db over base_llm).

Cells (single session):
  CAL: base_llm    × 40 steps  × randomized 3-obj order + gradual feedback (seeds 2001-2010)
  A:   base_llm    × 100 steps × randomized 4-obj order + gradual feedback (seeds 1411-1420)
  B:   full_cg_db  × 100 steps × randomized 4-obj order + gradual feedback (seeds 1411-1420)
  C:   base_llm    × 125 steps × randomized 4-obj order + gradual feedback (seeds 1421-1430)
  D:   full_cg_db  × 125 steps × randomized 4-obj order + gradual feedback (seeds 1421-1430)

Total: 50 episodes.

Usage:
  python experiments/run_10f_step_budget_pilot.py
"""

from __future__ import annotations

RUN_NAME = "run10f"

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
from environment.extended_env_6obj import ExtendedBlicketEnv6
from environment.text_interface import TextInterface5, TextInterface6
from agents.base_llm_agent import BaseLLMAgent
from agents.cg_agent import CGAgent
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

# --- Calibration constants (3-object) ---
CAL_NUM_OBJECTS = 5
CAL_ALL_LABELS = ["A", "B", "C", "D", "E"]
CAL_N_CORRECT = 3
CAL_BASE_SEED = 2001
CAL_EPISODES = 10
CAL_MAX_STEPS = 40

# --- Primary constants (4-object) ---
PRIMARY_NUM_OBJECTS = 6
PRIMARY_ALL_LABELS = ["A", "B", "C", "D", "E", "F"]
PRIMARY_N_CORRECT = 4
PRIMARY_EPISODES = 10  # per cell

# Step budget levels
STEPS_100 = 100
STEPS_125 = 125

# Seed ranges
SEEDS_100 = range(1411, 1421)  # Cells A, B
SEEDS_125 = range(1421, 1431)  # Cells C, D

BUDGET_CAP = 140.0
MAX_COST_PER_EPISODE = 4.0  # Raised from $2.00 for 125-step episodes


def generate_randomized_order_rule(
    seed: int,
    all_labels: list[str] | None = None,
    n_correct: int = 3,
) -> dict:
    """Generate a randomized order-sensitive rule for one episode."""
    if all_labels is None:
        all_labels = CAL_ALL_LABELS
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
    agent_type: AgentType,
    client: anthropic.AsyncAnthropic,
    config: ExperimentConfig,
    num_objects: int,
    text_interface,
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
            num_objects=num_objects,
            behaviors=create_blicket_behaviors_10c(),
        )
    raise ValueError(f"Unknown agent type: {agent_type}")


def create_env_5obj(
    required_objects: set[str],
    required_order: list[str],
    max_steps: int,
    seed: int,
) -> ExtendedBlicketEnv5:
    """Create a 5-object order-sensitive environment."""
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


def create_env_6obj(
    required_objects: set[str],
    required_order: list[str],
    max_steps: int,
    seed: int,
) -> ExtendedBlicketEnv6:
    """Create a 6-object order-sensitive environment."""
    rule = CausalRule(
        rule_type=CausalRuleType.ORDER_SENSITIVE,
        required_objects=required_objects,
    )
    return ExtendedBlicketEnv6(
        causal_rule=rule,
        condition=ExtendedCondition.ORDER_SENSITIVE,
        max_steps=max_steps,
        order_config=OrderConfig(required_order=required_order),
        rng_seed=seed,
    )


async def run_cell(
    cell_label: str,
    agent_type: AgentType,
    max_steps: int,
    seeds: range,
    num_objects: int,
    all_labels: list[str],
    n_correct: int,
    rules: list[tuple[int, dict]],
    config: ExperimentConfig,
    client: anthropic.AsyncAnthropic,
    cost_tracker: CostTracker,
    trace_dir: Path,
    completed_ref: list[int],
    total_episodes: int,
) -> list[EpisodeResult]:
    """Run a single cell of episodes."""
    results: list[EpisodeResult] = []
    condition_name = f"4obj_order_rand_{cell_label}" if num_objects == 6 else "cal_3obj_order_rand"

    for ep in range(len(rules)):
        if not cost_tracker.check_budget():
            logger.warning(f"Budget exhausted (${cost_tracker.total_cost:.2f}/{BUDGET_CAP}). Stopping.")
            break

        rule_seed, rule = rules[ep]
        req_objects = rule["correct_objects"]
        req_order = rule["required_order"]

        gf_config = GradualFeedbackConfig(
            enabled=True,
            checkpoint_interval=5,
            correct_objects=sorted(req_objects),
        )

        logger.info(
            f"[{completed_ref[0] + 1}/{total_episodes}] "
            f"Cell {cell_label}: {agent_type.value} / {max_steps} steps / "
            f"order={req_order} / ep{ep}"
        )

        seed = config.get_episode_seed(condition_name, agent_type.value, ep)

        if num_objects == 5:
            env = create_env_5obj(req_objects, req_order, max_steps, seed)
            ti = TextInterface5(include_order_sensitive=True)
        else:
            env = create_env_6obj(req_objects, req_order, max_steps, seed)
            ti = TextInterface6(include_order_sensitive=True)

        agent = create_agent(agent_type, client, config, num_objects, ti)

        result = await run_episode(
            env=env,
            agent=agent,
            condition=condition_name,
            episode_number=ep,
            trace_dir=trace_dir,
            max_cost_per_episode=MAX_COST_PER_EPISODE,
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
        completed_ref[0] += 1

        order_str = ""
        if result.order_correct is not None:
            order_str = f" | Order: {'CORRECT' if result.order_correct else 'WRONG'}"
        gf_str = f" | GF: {len(result.gradual_feedback_events)} objects revealed"

        logger.info(
            f"  Result: {'CORRECT' if result.success else 'WRONG'}{order_str}{gf_str} | "
            f"Steps: {result.steps_taken} | Cost: ${episode_cost:.3f} | "
            f"Total: ${cost_tracker.total_cost:.2f}"
        )

    return results


async def run_step_budget_pilot() -> list[EpisodeResult]:
    """Run all Run 10f cells in a single session."""

    config = ExperimentConfig(
        episodes_per_condition=PRIMARY_EPISODES,
        max_steps_per_episode=STEPS_125,  # Max across all cells
        trace_dir=Path(f"data/traces_{RUN_NAME}"),
        results_dir=Path(f"data/results_{RUN_NAME}"),
        base_seed=CAL_BASE_SEED,
        max_total_cost=BUDGET_CAP,
        max_cost_per_episode=MAX_COST_PER_EPISODE,
    )

    client = anthropic.AsyncAnthropic()
    cost_tracker = CostTracker(max_total=config.max_total_cost)
    all_results: list[EpisodeResult] = []

    total_episodes = CAL_EPISODES + PRIMARY_EPISODES * 4  # CAL + A + B + C + D = 50
    completed_ref = [0]  # Mutable counter

    # --- Pre-generate all episode rules ---

    # Calibration: 3-object (seeds 2001-2010)
    cal_rules = []
    for ep in range(CAL_EPISODES):
        rule_seed = CAL_BASE_SEED + ep
        rule = generate_randomized_order_rule(
            seed=rule_seed, all_labels=CAL_ALL_LABELS, n_correct=CAL_N_CORRECT,
        )
        cal_rules.append((rule_seed, rule))

    # 100-step rules (seeds 1411-1420)
    rules_100 = []
    for ep in range(PRIMARY_EPISODES):
        rule_seed = 1411 + ep
        rule = generate_randomized_order_rule(
            seed=rule_seed, all_labels=PRIMARY_ALL_LABELS, n_correct=PRIMARY_N_CORRECT,
        )
        rules_100.append((rule_seed, rule))

    # 125-step rules (seeds 1421-1430)
    rules_125 = []
    for ep in range(PRIMARY_EPISODES):
        rule_seed = 1421 + ep
        rule = generate_randomized_order_rule(
            seed=rule_seed, all_labels=PRIMARY_ALL_LABELS, n_correct=PRIMARY_N_CORRECT,
        )
        rules_125.append((rule_seed, rule))

    # Log all rules
    logger.info(f"Run 10f: Step Budget Pilot — {total_episodes} episodes")
    logger.info(f"  Budget cap: ${BUDGET_CAP}, max cost/episode: ${MAX_COST_PER_EPISODE}")
    logger.info(f"\nCalibration: {CAL_EPISODES} episodes (seeds 2001-2010, 3-obj, 40 steps)")
    for ep, (rule_seed, rule) in enumerate(cal_rules):
        logger.info(
            f"  Episode {ep}: seed={rule_seed}, "
            f"correct={sorted(rule['correct_objects'])}, "
            f"order={rule['required_order']}, "
            f"distractors={sorted(rule['distractors'])}"
        )

    logger.info(f"\n100-step cells (A, B): {PRIMARY_EPISODES} episodes each (seeds 1411-1420, 4-obj)")
    for ep, (rule_seed, rule) in enumerate(rules_100):
        logger.info(
            f"  Episode {ep}: seed={rule_seed}, "
            f"correct={sorted(rule['correct_objects'])}, "
            f"order={rule['required_order']}, "
            f"distractors={sorted(rule['distractors'])}"
        )

    logger.info(f"\n125-step cells (C, D): {PRIMARY_EPISODES} episodes each (seeds 1421-1430, 4-obj)")
    for ep, (rule_seed, rule) in enumerate(rules_125):
        logger.info(
            f"  Episode {ep}: seed={rule_seed}, "
            f"correct={sorted(rule['correct_objects'])}, "
            f"order={rule['required_order']}, "
            f"distractors={sorted(rule['distractors'])}"
        )

    # === CELL DEFINITIONS ===
    cells = [
        {
            "label": "CAL",
            "agent_type": AgentType.BASE_LLM,
            "max_steps": CAL_MAX_STEPS,
            "num_objects": CAL_NUM_OBJECTS,
            "all_labels": CAL_ALL_LABELS,
            "n_correct": CAL_N_CORRECT,
            "rules": cal_rules,
            "trace_subdir": "calibration",
        },
        {
            "label": "A",
            "agent_type": AgentType.BASE_LLM,
            "max_steps": STEPS_100,
            "num_objects": PRIMARY_NUM_OBJECTS,
            "all_labels": PRIMARY_ALL_LABELS,
            "n_correct": PRIMARY_N_CORRECT,
            "rules": rules_100,
            "trace_subdir": "4obj_100steps",
        },
        {
            "label": "B",
            "agent_type": AgentType.FULL_CG_DB,
            "max_steps": STEPS_100,
            "num_objects": PRIMARY_NUM_OBJECTS,
            "all_labels": PRIMARY_ALL_LABELS,
            "n_correct": PRIMARY_N_CORRECT,
            "rules": rules_100,
            "trace_subdir": "4obj_100steps",
        },
        {
            "label": "C",
            "agent_type": AgentType.BASE_LLM,
            "max_steps": STEPS_125,
            "num_objects": PRIMARY_NUM_OBJECTS,
            "all_labels": PRIMARY_ALL_LABELS,
            "n_correct": PRIMARY_N_CORRECT,
            "rules": rules_125,
            "trace_subdir": "4obj_125steps",
        },
        {
            "label": "D",
            "agent_type": AgentType.FULL_CG_DB,
            "max_steps": STEPS_125,
            "num_objects": PRIMARY_NUM_OBJECTS,
            "all_labels": PRIMARY_ALL_LABELS,
            "n_correct": PRIMARY_N_CORRECT,
            "rules": rules_125,
            "trace_subdir": "4obj_125steps",
        },
    ]

    # === RUN ALL CELLS ===
    for cell in cells:
        label = cell["label"]
        agent_type = cell["agent_type"]
        max_steps = cell["max_steps"]

        logger.info(f"\n{'='*70}")
        logger.info(
            f"CELL {label}: {agent_type.value} × "
            f"{cell['n_correct']}-obj × {max_steps} steps"
        )
        logger.info(f"{'='*70}")

        trace_dir = Path(f"data/traces_{RUN_NAME}/{cell['trace_subdir']}")

        cell_results = await run_cell(
            cell_label=label,
            agent_type=agent_type,
            max_steps=max_steps,
            seeds=SEEDS_100 if max_steps == STEPS_100 else SEEDS_125,
            num_objects=cell["num_objects"],
            all_labels=cell["all_labels"],
            n_correct=cell["n_correct"],
            rules=cell["rules"],
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
    summary_path = config.results_dir / "step_budget_pilot.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": RUN_NAME,
        "description": (
            "Run 10f: Step Budget Pilot. Tests whether increasing step budget "
            "from 75 to 100/125 steps lifts 4-object ORDER_SENSITIVE performance. "
            "Single session. 5 cells: CAL (base_llm, 3-obj, 40 steps), "
            "A (base_llm, 4-obj, 100 steps), B (full_cg_db, 4-obj, 100 steps), "
            "C (base_llm, 4-obj, 125 steps), D (full_cg_db, 4-obj, 125 steps). "
            "Instrumented: temperature=1.0 pinned, API response metadata logged."
        ),
        "instrumentation": {
            "temperature": 1.0,
            "action_model": config.action_model,
            "trigger_model": config.trigger_model,
        },
        "calibration": {
            "num_objects": CAL_NUM_OBJECTS,
            "n_correct": CAL_N_CORRECT,
            "max_steps": CAL_MAX_STEPS,
            "episodes": CAL_EPISODES,
            "base_seed": CAL_BASE_SEED,
            "agent_type": AgentType.BASE_LLM.value,
            "episode_rules": [
                {
                    "episode": ep,
                    "rule_seed": rule_seed,
                    "rule_correct_objects": sorted(rule["correct_objects"]),
                    "rule_required_order": rule["required_order"],
                    "rule_distractors": sorted(rule["distractors"]),
                }
                for ep, (rule_seed, rule) in enumerate(cal_rules)
            ],
        },
        "primary_cells": {
            "num_objects": PRIMARY_NUM_OBJECTS,
            "n_correct": PRIMARY_N_CORRECT,
            "episodes_per_cell": PRIMARY_EPISODES,
            "cells": [
                {"label": "A", "agent_type": AgentType.BASE_LLM.value, "max_steps": STEPS_100, "seeds": "1411-1420", "behavior_version": "N/A"},
                {"label": "B", "agent_type": AgentType.FULL_CG_DB.value, "max_steps": STEPS_100, "seeds": "1411-1420", "behavior_version": "10c"},
                {"label": "C", "agent_type": AgentType.BASE_LLM.value, "max_steps": STEPS_125, "seeds": "1421-1430", "behavior_version": "N/A"},
                {"label": "D", "agent_type": AgentType.FULL_CG_DB.value, "max_steps": STEPS_125, "seeds": "1421-1430", "behavior_version": "10c"},
            ],
            "100_step_rules": [
                {
                    "episode": ep,
                    "rule_seed": rule_seed,
                    "rule_correct_objects": sorted(rule["correct_objects"]),
                    "rule_required_order": rule["required_order"],
                    "rule_distractors": sorted(rule["distractors"]),
                }
                for ep, (rule_seed, rule) in enumerate(rules_100)
            ],
            "125_step_rules": [
                {
                    "episode": ep,
                    "rule_seed": rule_seed,
                    "rule_correct_objects": sorted(rule["correct_objects"]),
                    "rule_required_order": rule["required_order"],
                    "rule_distractors": sorted(rule["distractors"]),
                }
                for ep, (rule_seed, rule) in enumerate(rules_125)
            ],
        },
        "gradual_feedback_config": {
            "enabled": True,
            "checkpoint_interval": 5,
        },
        "cost_summary": cost_tracker.get_summary(),
        "results": [],
    }

    # Attach results
    for r in all_results:
        if r.condition.startswith("cal_"):
            # Find matching rule from cal_rules
            rule_seed, rule = cal_rules[r.episode_number]
        elif r.condition.endswith("_A") or r.condition.endswith("_B"):
            rule_seed, rule = rules_100[r.episode_number]
        else:
            rule_seed, rule = rules_125[r.episode_number]

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
            "gradual_feedback_events": r.gradual_feedback_events,
            "gradual_feedback_confirmed_objects": r.gradual_feedback_confirmed_objects,
            "rule_correct_objects": sorted(rule["correct_objects"]),
            "rule_required_order": rule["required_order"],
            "rule_distractors": sorted(rule["distractors"]),
            "rule_seed": rule_seed,
        })

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # === Print summary ===
    logger.info(f"\n{'='*70}")
    logger.info(f"Run 10f: Step Budget Pilot Results")
    logger.info(f"{'='*70}")

    # Calibration summary
    cal_results = [r for r in all_results if r.condition.startswith("cal_")]
    if cal_results:
        correct = sum(1 for r in cal_results if r.success)
        total = len(cal_results)
        submitted = sum(1 for r in cal_results if r.answer_submitted)
        obj_correct = sum(1 for r in cal_results if r.objects_correct)
        order_correct = sum(1 for r in cal_results if r.order_correct)
        avg_gf = sum(len(r.gradual_feedback_events) for r in cal_results) / total
        logger.info(
            f"\n  CAL (base_llm / 3-obj / {CAL_MAX_STEPS} steps): "
            f"{correct}/{total} ({100*correct/total:.1f}%) | "
            f"Submitted: {submitted}/{total} | "
            f"Objects: {obj_correct}/{total} | Order: {order_correct}/{total} | "
            f"Avg GF events: {avg_gf:.1f}"
        )

    # Primary cell summaries
    for cell_label, agent_type, step_budget in [
        ("A", AgentType.BASE_LLM, STEPS_100),
        ("B", AgentType.FULL_CG_DB, STEPS_100),
        ("C", AgentType.BASE_LLM, STEPS_125),
        ("D", AgentType.FULL_CG_DB, STEPS_125),
    ]:
        condition_name = f"4obj_order_rand_{cell_label}"
        cell_results = [r for r in all_results if r.condition == condition_name]
        if cell_results:
            correct = sum(1 for r in cell_results if r.success)
            total = len(cell_results)
            submitted = sum(1 for r in cell_results if r.answer_submitted)
            obj_correct = sum(1 for r in cell_results if r.objects_correct)
            order_correct = sum(1 for r in cell_results if r.order_correct)
            avg_gf = sum(len(r.gradual_feedback_events) for r in cell_results) / total
            logger.info(
                f"  Cell {cell_label} ({agent_type.value} / 4-obj / {step_budget} steps): "
                f"{correct}/{total} ({100*correct/total:.1f}%) | "
                f"Submitted: {submitted}/{total} | "
                f"Objects: {obj_correct}/{total} | Order: {order_correct}/{total} | "
                f"Avg GF events: {avg_gf:.1f}"
            )

    # Paired episode analysis for 100-step cells (A vs B)
    logger.info(f"\n--- Paired Analysis: 100-step (A vs B) ---")
    a_results = [r for r in all_results if r.condition == "4obj_order_rand_A"]
    b_results = [r for r in all_results if r.condition == "4obj_order_rand_B"]
    if len(a_results) == len(b_results) == PRIMARY_EPISODES:
        concordant = sum(1 for a, b in zip(a_results, b_results) if a.success == b.success)
        a_only = sum(1 for a, b in zip(a_results, b_results) if a.success and not b.success)
        b_only = sum(1 for a, b in zip(a_results, b_results) if not a.success and b.success)
        logger.info(f"  Concordant: {concordant}, A-only: {a_only}, B-only: {b_only}")

    # Paired episode analysis for 125-step cells (C vs D)
    logger.info(f"\n--- Paired Analysis: 125-step (C vs D) ---")
    c_results = [r for r in all_results if r.condition == "4obj_order_rand_C"]
    d_results = [r for r in all_results if r.condition == "4obj_order_rand_D"]
    if len(c_results) == len(d_results) == PRIMARY_EPISODES:
        concordant = sum(1 for c, d in zip(c_results, d_results) if c.success == d.success)
        c_only = sum(1 for c, d in zip(c_results, d_results) if c.success and not d.success)
        d_only = sum(1 for c, d in zip(c_results, d_results) if not c.success and d.success)
        logger.info(f"  Concordant: {concordant}, C-only: {c_only}, D-only: {d_only}")

    logger.info(f"\nTotal cost: ${cost_tracker.total_cost:.2f}")
    logger.info(f"Summary saved: {summary_path}")

    return all_results


def main():
    asyncio.run(run_step_budget_pilot())


if __name__ == "__main__":
    main()
