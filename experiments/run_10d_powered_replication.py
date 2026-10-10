"""Run 10d: Powered 3-Object Replication + Attribution + 4-Object Pilot.

Phase 1 — Powered 3-object replication (n=50, 3 agent types):
  Cell A: base_llm    × 40 steps × randomized 3-obj order + gradual feedback
  Cell B: cg_only     × 40 steps × randomized 3-obj order + gradual feedback
  Cell C: full_cg_db  × 40 steps × randomized 3-obj order + gradual feedback [behaviors_10c]

Phase 2 — 4-object pilot (n=5, 2 agent types):
  Cell D: base_llm    × 75 steps × randomized 4-obj order + gradual feedback
  Cell E: full_cg_db  × 75 steps × randomized 4-obj order + gradual feedback [behaviors_10c]

Total: 160 episodes across 5 cells.
"""

from __future__ import annotations

RUN_NAME = "run10d"

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

# --- Phase 1 constants ---
P1_NUM_OBJECTS = 5
P1_ALL_LABELS = ["A", "B", "C", "D", "E"]
P1_N_CORRECT = 3
P1_BASE_SEED = 1201
P1_EPISODES = 50
P1_MAX_STEPS = 40

# --- Phase 2 constants ---
P2_NUM_OBJECTS = 6
P2_ALL_LABELS = ["A", "B", "C", "D", "E", "F"]
P2_N_CORRECT = 4
P2_BASE_SEED = 1301
P2_EPISODES = 5
P2_MAX_STEPS = 75

BUDGET_CAP = 150.0

# Cell definitions: (label, agent_type, max_steps, phase)
PHASE1_CELLS = [
    ("A", AgentType.BASE_LLM, P1_MAX_STEPS),
    ("B", AgentType.CG_ONLY, P1_MAX_STEPS),
    ("C", AgentType.FULL_CG_DB, P1_MAX_STEPS),
]

PHASE2_CELLS = [
    ("D", AgentType.BASE_LLM, P2_MAX_STEPS),
    ("E", AgentType.FULL_CG_DB, P2_MAX_STEPS),
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
        all_labels = P1_ALL_LABELS
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
    elif agent_type == AgentType.CG_ONLY:
        return CGAgent(
            llm_client=client,
            model=config.action_model,
            text_interface=text_interface,
            num_objects=num_objects,
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


async def run_powered_replication(
    config: ExperimentConfig | None = None,
) -> list[EpisodeResult]:
    """Run all Run 10d experiments: Phase 1 (powered replication) + Phase 2 (4-obj pilot)."""
    if config is None:
        config = ExperimentConfig(
            episodes_per_condition=P1_EPISODES,
            max_steps_per_episode=P2_MAX_STEPS,  # max across all cells
            trace_dir=Path(f"data/traces_{RUN_NAME}"),
            results_dir=Path(f"data/results_{RUN_NAME}"),
            base_seed=P1_BASE_SEED,
            max_total_cost=BUDGET_CAP,
        )

    client = anthropic.AsyncAnthropic()
    cost_tracker = CostTracker(max_total=config.max_total_cost)
    results: list[EpisodeResult] = []

    total_phase1 = len(PHASE1_CELLS) * P1_EPISODES
    total_phase2 = len(PHASE2_CELLS) * P2_EPISODES
    total_episodes = total_phase1 + total_phase2
    completed = 0

    # --- Pre-generate all episode rules ---

    # Phase 1: 3-object rules (seeds 1201-1250)
    p1_rules = []
    for ep in range(P1_EPISODES):
        rule_seed = P1_BASE_SEED + ep
        rule = generate_randomized_order_rule(
            seed=rule_seed, all_labels=P1_ALL_LABELS, n_correct=P1_N_CORRECT,
        )
        p1_rules.append((rule_seed, rule))

    # Phase 2: 4-object rules (seeds 1301-1305)
    p2_rules = []
    for ep in range(P2_EPISODES):
        rule_seed = P2_BASE_SEED + ep
        rule = generate_randomized_order_rule(
            seed=rule_seed, all_labels=P2_ALL_LABELS, n_correct=P2_N_CORRECT,
        )
        p2_rules.append((rule_seed, rule))

    # Log rules
    logger.info(f"Run 10d: {total_episodes} episodes across 5 cells (Phase 1: {total_phase1}, Phase 2: {total_phase2})")
    logger.info(f"\nPhase 1: {P1_EPISODES} episodes × 3 cells (seeds {P1_BASE_SEED}-{P1_BASE_SEED + P1_EPISODES - 1})")
    for ep, (rule_seed, rule) in enumerate(p1_rules):
        logger.info(
            f"  Episode {ep}: seed={rule_seed}, "
            f"correct={sorted(rule['correct_objects'])}, "
            f"order={rule['required_order']}, "
            f"distractors={sorted(rule['distractors'])}"
        )

    logger.info(f"\nPhase 2: {P2_EPISODES} episodes × 2 cells (seeds {P2_BASE_SEED}-{P2_BASE_SEED + P2_EPISODES - 1})")
    for ep, (rule_seed, rule) in enumerate(p2_rules):
        logger.info(
            f"  Episode {ep}: seed={rule_seed}, "
            f"correct={sorted(rule['correct_objects'])}, "
            f"order={rule['required_order']}, "
            f"distractors={sorted(rule['distractors'])}"
        )

    # === PHASE 1: Powered 3-object replication ===
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 1: Powered 3-Object Replication (3 cells × 50 episodes)")
    logger.info(f"{'='*70}")

    for cell_label, agent_type, max_steps in PHASE1_CELLS:
        condition_name = f"3obj_order_rand_{cell_label}"

        for ep in range(P1_EPISODES):
            if not cost_tracker.check_budget():
                logger.warning(
                    f"Budget exhausted (${cost_tracker.total_cost:.2f}/{BUDGET_CAP}). Stopping."
                )
                break

            rule_seed, rule = p1_rules[ep]
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
            env = create_env_5obj(req_objects, req_order, max_steps, seed)
            ti = TextInterface5(include_order_sensitive=True)
            agent = create_agent(agent_type, client, config, P1_NUM_OBJECTS, ti)

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

    # === PHASE 2: 4-Object Pilot ===
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 2: 4-Object Pilot (2 cells × 5 episodes)")
    logger.info(f"{'='*70}")

    for cell_label, agent_type, max_steps in PHASE2_CELLS:
        condition_name = f"4obj_order_rand_{cell_label}"

        for ep in range(P2_EPISODES):
            if not cost_tracker.check_budget():
                logger.warning(
                    f"Budget exhausted (${cost_tracker.total_cost:.2f}/{BUDGET_CAP}). Stopping."
                )
                break

            rule_seed, rule = p2_rules[ep]
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
            env = create_env_6obj(req_objects, req_order, max_steps, seed)
            ti = TextInterface6(include_order_sensitive=True)
            agent = create_agent(agent_type, client, config, P2_NUM_OBJECTS, ti)

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

    # === Save summary ===
    summary_path = config.results_dir / "powered_replication_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": RUN_NAME,
        "description": (
            "Run 10d: Powered 3-object replication (n=50, 3 agent types: "
            "base_llm, cg_only, full_cg_db) at 40 steps + 4-object pilot "
            "(n=5, 2 agent types: base_llm, full_cg_db) at 75 steps. "
            "Randomized label-to-role mapping, gradual periodic feedback."
        ),
        "phase1": {
            "num_objects": P1_NUM_OBJECTS,
            "n_correct": P1_N_CORRECT,
            "max_steps": P1_MAX_STEPS,
            "episodes_per_cell": P1_EPISODES,
            "base_seed": P1_BASE_SEED,
            "cells": [
                {
                    "label": cell_label,
                    "agent_type": agent_type.value,
                    "max_steps": max_steps,
                    "behavior_version": (
                        "10c" if agent_type == AgentType.FULL_CG_DB
                        else "none" if agent_type == AgentType.CG_ONLY
                        else "N/A"
                    ),
                }
                for cell_label, agent_type, max_steps in PHASE1_CELLS
            ],
            "episode_rules": [
                {
                    "episode": ep,
                    "rule_seed": rule_seed,
                    "rule_correct_objects": sorted(rule["correct_objects"]),
                    "rule_required_order": rule["required_order"],
                    "rule_distractors": sorted(rule["distractors"]),
                }
                for ep, (rule_seed, rule) in enumerate(p1_rules)
            ],
        },
        "phase2": {
            "num_objects": P2_NUM_OBJECTS,
            "n_correct": P2_N_CORRECT,
            "max_steps": P2_MAX_STEPS,
            "episodes_per_cell": P2_EPISODES,
            "base_seed": P2_BASE_SEED,
            "cells": [
                {
                    "label": cell_label,
                    "agent_type": agent_type.value,
                    "max_steps": max_steps,
                    "behavior_version": "10c" if agent_type == AgentType.FULL_CG_DB else "N/A",
                }
                for cell_label, agent_type, max_steps in PHASE2_CELLS
            ],
            "episode_rules": [
                {
                    "episode": ep,
                    "rule_seed": rule_seed,
                    "rule_correct_objects": sorted(rule["correct_objects"]),
                    "rule_required_order": rule["required_order"],
                    "rule_distractors": sorted(rule["distractors"]),
                }
                for ep, (rule_seed, rule) in enumerate(p2_rules)
            ],
        },
        "gradual_feedback_config": {
            "enabled": True,
            "checkpoint_interval": 5,
        },
        "cost_summary": cost_tracker.get_summary(),
        "results": [],
    }

    # Attach results with per-episode rule info
    for r in results:
        # Determine which phase/rules this result belongs to
        if r.condition.startswith("3obj"):
            ep_rules = p1_rules
        else:
            ep_rules = p2_rules

        rule_seed, rule = ep_rules[r.episode_number]
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

    # === Print summary tables ===
    logger.info(f"\n{'='*70}")
    logger.info("Run 10d: Powered Replication + Attribution + 4-Object Pilot Results")
    logger.info(f"{'='*70}")

    # Phase 1 summary
    logger.info("\n--- Phase 1: 3-Object Replication (40 steps, n=50) ---")
    for cell_label, agent_type, max_steps in PHASE1_CELLS:
        condition_name = f"3obj_order_rand_{cell_label}"
        cell_results = [
            r for r in results
            if r.condition == condition_name and r.agent_type == agent_type.value
        ]
        if cell_results:
            correct = sum(1 for r in cell_results if r.success)
            total = len(cell_results)
            submitted = sum(1 for r in cell_results if r.answer_submitted)
            obj_correct = sum(1 for r in cell_results if r.objects_correct)
            order_correct = sum(1 for r in cell_results if r.order_correct)
            avg_gf = sum(len(r.gradual_feedback_events) for r in cell_results) / total
            logger.info(
                f"  Cell {cell_label} ({agent_type.value}/{max_steps}steps): "
                f"{correct}/{total} ({100*correct/total:.1f}%) | "
                f"Submitted: {submitted}/{total} | "
                f"Objects: {obj_correct}/{total} | Order: {order_correct}/{total} | "
                f"Avg GF events: {avg_gf:.1f}"
            )

    # Phase 2 summary
    logger.info("\n--- Phase 2: 4-Object Pilot (75 steps, n=5) ---")
    for cell_label, agent_type, max_steps in PHASE2_CELLS:
        condition_name = f"4obj_order_rand_{cell_label}"
        cell_results = [
            r for r in results
            if r.condition == condition_name and r.agent_type == agent_type.value
        ]
        if cell_results:
            correct = sum(1 for r in cell_results if r.success)
            total = len(cell_results)
            submitted = sum(1 for r in cell_results if r.answer_submitted)
            obj_correct = sum(1 for r in cell_results if r.objects_correct)
            order_correct = sum(1 for r in cell_results if r.order_correct)
            avg_gf = sum(len(r.gradual_feedback_events) for r in cell_results) / total
            logger.info(
                f"  Cell {cell_label} ({agent_type.value}/{max_steps}steps): "
                f"{correct}/{total} ({100*correct/total:.1f}%) | "
                f"Submitted: {submitted}/{total} | "
                f"Objects: {obj_correct}/{total} | Order: {order_correct}/{total} | "
                f"Avg GF events: {avg_gf:.1f}"
            )

    logger.info(f"\nTotal cost: ${cost_tracker.total_cost:.2f}")
    logger.info(f"Summary saved: {summary_path}")

    return results


if __name__ == "__main__":
    asyncio.run(run_powered_replication())
