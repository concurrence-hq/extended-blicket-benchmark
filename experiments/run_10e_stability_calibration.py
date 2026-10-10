"""Run 10e: Stability Calibration + 4-Object Signal Check.

Purpose:
  (1) Establish whether the experimental environment is stable across sessions
  (2) Get a directional signal on whether architectural separation exists at
      4-object ordering before committing to a powered experiment

Cells (per session):
  CAL: base_llm    × 40 steps × randomized 3-obj order + gradual feedback (seeds 2001-2010)
  A:   base_llm    × 75 steps × randomized 4-obj order + gradual feedback (seeds 1401-1410)
  B:   full_cg_db  × 75 steps × randomized 4-obj order + gradual feedback (seeds 1401-1410)

Total per session: 30 episodes.
Total across both sessions: 60 episodes.
Sessions must be run on separate days.

Usage:
  python experiments/run_10e_stability_calibration.py --session 1
  python experiments/run_10e_stability_calibration.py --session 2
"""

from __future__ import annotations

RUN_NAME = "run10e"

import argparse
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
CAL_BASE_SEED = 2001  # Seeds 2001-2010: permanently reserved for calibration
CAL_EPISODES = 10
CAL_MAX_STEPS = 40

# --- Signal check constants (4-object) ---
SIG_NUM_OBJECTS = 6
SIG_ALL_LABELS = ["A", "B", "C", "D", "E", "F"]
SIG_N_CORRECT = 4
SIG_BASE_SEED = 1401  # Seeds 1401-1410
SIG_EPISODES = 10
SIG_MAX_STEPS = 75

BUDGET_CAP = 110.0


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
        all_labels = CAL_ALL_LABELS
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


async def run_stability_calibration(session: int) -> list[EpisodeResult]:
    """Run all Run 10e cells for one session (day).

    Args:
        session: 1 or 2 (which day).
    """
    assert session in (1, 2), f"Session must be 1 or 2, got {session}"
    day_label = f"day{session}"

    config = ExperimentConfig(
        episodes_per_condition=SIG_EPISODES,
        max_steps_per_episode=SIG_MAX_STEPS,
        trace_dir=Path(f"data/traces_{RUN_NAME}"),  # Overridden per cell below
        results_dir=Path(f"data/results_{RUN_NAME}"),
        base_seed=CAL_BASE_SEED,
        max_total_cost=BUDGET_CAP,
    )

    client = anthropic.AsyncAnthropic()
    cost_tracker = CostTracker(max_total=config.max_total_cost)
    results: list[EpisodeResult] = []

    total_episodes = CAL_EPISODES + SIG_EPISODES * 2  # CAL + Cell A + Cell B
    completed = 0

    # --- Pre-generate all episode rules ---

    # Calibration: 3-object rules (seeds 2001-2010)
    cal_rules = []
    for ep in range(CAL_EPISODES):
        rule_seed = CAL_BASE_SEED + ep
        rule = generate_randomized_order_rule(
            seed=rule_seed, all_labels=CAL_ALL_LABELS, n_correct=CAL_N_CORRECT,
        )
        cal_rules.append((rule_seed, rule))

    # Signal check: 4-object rules (seeds 1401-1410)
    sig_rules = []
    for ep in range(SIG_EPISODES):
        rule_seed = SIG_BASE_SEED + ep
        rule = generate_randomized_order_rule(
            seed=rule_seed, all_labels=SIG_ALL_LABELS, n_correct=SIG_N_CORRECT,
        )
        sig_rules.append((rule_seed, rule))

    # Log rules
    logger.info(f"Run 10e Session {session}: {total_episodes} episodes (10 CAL + 10 Cell A + 10 Cell B)")
    logger.info(f"\nCalibration: {CAL_EPISODES} episodes (seeds {CAL_BASE_SEED}-{CAL_BASE_SEED + CAL_EPISODES - 1})")
    for ep, (rule_seed, rule) in enumerate(cal_rules):
        logger.info(
            f"  Episode {ep}: seed={rule_seed}, "
            f"correct={sorted(rule['correct_objects'])}, "
            f"order={rule['required_order']}, "
            f"distractors={sorted(rule['distractors'])}"
        )

    logger.info(f"\nSignal check: {SIG_EPISODES} episodes (seeds {SIG_BASE_SEED}-{SIG_BASE_SEED + SIG_EPISODES - 1})")
    for ep, (rule_seed, rule) in enumerate(sig_rules):
        logger.info(
            f"  Episode {ep}: seed={rule_seed}, "
            f"correct={sorted(rule['correct_objects'])}, "
            f"order={rule['required_order']}, "
            f"distractors={sorted(rule['distractors'])}"
        )

    # === CALIBRATION: 3-object base_llm ===
    logger.info(f"\n{'='*70}")
    logger.info(f"CALIBRATION: base_llm × 3-obj × 40 steps (Session {session})")
    logger.info(f"{'='*70}")

    cal_trace_dir = Path(f"data/traces_{RUN_NAME}/calibration_{day_label}")
    condition_name = f"cal_3obj_order_rand"

    for ep in range(CAL_EPISODES):
        if not cost_tracker.check_budget():
            logger.warning(f"Budget exhausted (${cost_tracker.total_cost:.2f}/{BUDGET_CAP}). Stopping.")
            break

        rule_seed, rule = cal_rules[ep]
        req_objects = rule["correct_objects"]
        req_order = rule["required_order"]

        gf_config = GradualFeedbackConfig(
            enabled=True,
            checkpoint_interval=5,
            correct_objects=sorted(req_objects),
        )

        logger.info(
            f"[{completed + 1}/{total_episodes}] "
            f"CAL: base_llm / {CAL_MAX_STEPS} steps / "
            f"order={req_order} / ep{ep}"
        )

        seed = config.get_episode_seed(condition_name, AgentType.BASE_LLM.value, ep)
        env = create_env_5obj(req_objects, req_order, CAL_MAX_STEPS, seed)
        ti = TextInterface5(include_order_sensitive=True)
        agent = create_agent(AgentType.BASE_LLM, client, config, CAL_NUM_OBJECTS, ti)

        result = await run_episode(
            env=env,
            agent=agent,
            condition=condition_name,
            episode_number=ep,
            trace_dir=cal_trace_dir,
            max_cost_per_episode=config.max_cost_per_episode,
            gradual_feedback_config=gf_config,
        )

        episode_cost = cost_tracker.add_episode_cost(
            condition=condition_name,
            agent=AgentType.BASE_LLM.value,
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

    # === CELL A: 4-object base_llm ===
    logger.info(f"\n{'='*70}")
    logger.info(f"CELL A: base_llm × 4-obj × 75 steps (Session {session})")
    logger.info(f"{'='*70}")

    sig_trace_dir = Path(f"data/traces_{RUN_NAME}/4obj_{day_label}")

    for cell_label, agent_type in [("A", AgentType.BASE_LLM), ("B", AgentType.FULL_CG_DB)]:
        cell_header = "CELL A: base_llm" if cell_label == "A" else "CELL B: full_cg_db"
        if cell_label == "B":
            logger.info(f"\n{'='*70}")
            logger.info(f"CELL B: full_cg_db × 4-obj × 75 steps (Session {session})")
            logger.info(f"{'='*70}")

        condition_name = f"4obj_order_rand_{cell_label}"

        for ep in range(SIG_EPISODES):
            if not cost_tracker.check_budget():
                logger.warning(f"Budget exhausted (${cost_tracker.total_cost:.2f}/{BUDGET_CAP}). Stopping.")
                break

            rule_seed, rule = sig_rules[ep]
            req_objects = rule["correct_objects"]
            req_order = rule["required_order"]

            gf_config = GradualFeedbackConfig(
                enabled=True,
                checkpoint_interval=5,
                correct_objects=sorted(req_objects),
            )

            logger.info(
                f"[{completed + 1}/{total_episodes}] "
                f"Cell {cell_label}: {agent_type.value} / {SIG_MAX_STEPS} steps / "
                f"order={req_order} / ep{ep}"
            )

            seed = config.get_episode_seed(condition_name, agent_type.value, ep)
            env = create_env_6obj(req_objects, req_order, SIG_MAX_STEPS, seed)
            ti = TextInterface6(include_order_sensitive=True)
            agent = create_agent(agent_type, client, config, SIG_NUM_OBJECTS, ti)

            result = await run_episode(
                env=env,
                agent=agent,
                condition=condition_name,
                episode_number=ep,
                trace_dir=sig_trace_dir,
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
    summary_path = config.results_dir / f"stability_calibration_session{session}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": RUN_NAME,
        "session": session,
        "description": (
            f"Run 10e Session {session}: Stability calibration (CAL: base_llm, "
            f"3-obj, 40 steps, seeds 2001-2010) + 4-object signal check "
            f"(Cell A: base_llm, Cell B: full_cg_db, 4-obj, 75 steps, seeds 1401-1410). "
            f"Instrumented: temperature=1.0 pinned, API response metadata logged."
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
        "signal_check": {
            "num_objects": SIG_NUM_OBJECTS,
            "n_correct": SIG_N_CORRECT,
            "max_steps": SIG_MAX_STEPS,
            "episodes_per_cell": SIG_EPISODES,
            "base_seed": SIG_BASE_SEED,
            "cells": [
                {"label": "A", "agent_type": AgentType.BASE_LLM.value, "behavior_version": "N/A"},
                {"label": "B", "agent_type": AgentType.FULL_CG_DB.value, "behavior_version": "10c"},
            ],
            "episode_rules": [
                {
                    "episode": ep,
                    "rule_seed": rule_seed,
                    "rule_correct_objects": sorted(rule["correct_objects"]),
                    "rule_required_order": rule["required_order"],
                    "rule_distractors": sorted(rule["distractors"]),
                }
                for ep, (rule_seed, rule) in enumerate(sig_rules)
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
    for r in results:
        if r.condition.startswith("cal_"):
            ep_rules = cal_rules
        else:
            ep_rules = sig_rules

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

    # === Print summary ===
    logger.info(f"\n{'='*70}")
    logger.info(f"Run 10e Session {session}: Stability Calibration Results")
    logger.info(f"{'='*70}")

    # Calibration summary
    cal_results = [r for r in results if r.condition.startswith("cal_")]
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

    # Signal check summary
    for cell_label, agent_type in [("A", AgentType.BASE_LLM), ("B", AgentType.FULL_CG_DB)]:
        condition_name = f"4obj_order_rand_{cell_label}"
        cell_results = [r for r in results if r.condition == condition_name]
        if cell_results:
            correct = sum(1 for r in cell_results if r.success)
            total = len(cell_results)
            submitted = sum(1 for r in cell_results if r.answer_submitted)
            obj_correct = sum(1 for r in cell_results if r.objects_correct)
            order_correct = sum(1 for r in cell_results if r.order_correct)
            avg_gf = sum(len(r.gradual_feedback_events) for r in cell_results) / total
            logger.info(
                f"  Cell {cell_label} ({agent_type.value} / 4-obj / {SIG_MAX_STEPS} steps): "
                f"{correct}/{total} ({100*correct/total:.1f}%) | "
                f"Submitted: {submitted}/{total} | "
                f"Objects: {obj_correct}/{total} | Order: {order_correct}/{total} | "
                f"Avg GF events: {avg_gf:.1f}"
            )

    logger.info(f"\nTotal cost: ${cost_tracker.total_cost:.2f}")
    logger.info(f"Summary saved: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run 10e: Stability Calibration + 4-Object Signal Check")
    parser.add_argument(
        "--session", type=int, required=True, choices=[1, 2],
        help="Session number (1 = Day 1, 2 = Day 2). Must be run on separate days.",
    )
    args = parser.parse_args()
    asyncio.run(run_stability_calibration(args.session))


if __name__ == "__main__":
    main()
