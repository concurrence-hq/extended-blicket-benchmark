"""Run 10g: 4-Object ORDER_SENSITIVE Powered Replication at 100 Steps.

Purpose:
  Powered replication of the Run 10f pilot finding that 100 steps produces
  architectural separation on 4-object ORDER_SENSITIVE (full_cg_db 40% vs
  base_llm 20%). Pools 10f pilot data (n=10) with 20 fresh episodes per
  cell for N=30 pooled analysis.

Cells (single session):
  COMPAT: base_llm × 100 steps × seed 1411 (backward compat check)
  CAL:    base_llm × 40 steps  × randomized 3-obj order + gradual feedback (seeds 2001-2005)
  A:      base_llm    × 100 steps × randomized 4-obj order + gradual feedback (seeds 1421-1440)
  B:      full_cg_db  × 100 steps × randomized 4-obj order + gradual feedback (seeds 1421-1440)

Total new episodes: 46 (1 COMPAT + 5 CAL + 20 A + 20 B).

Usage:
  python experiments/run_10g_powered_replication.py
"""

from __future__ import annotations

RUN_NAME = "run10g"

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
CAL_EPISODES = 5
CAL_MAX_STEPS = 40

# --- Primary constants (4-object) ---
PRIMARY_NUM_OBJECTS = 6
PRIMARY_ALL_LABELS = ["A", "B", "C", "D", "E", "F"]
PRIMARY_N_CORRECT = 4
PRIMARY_EPISODES = 20  # per cell
PRIMARY_BASE_SEED = 1421
MAX_STEPS = 100

# --- Budget ---
BUDGET_CAP = 150.0
MAX_COST_PER_EPISODE = 4.0

# --- Compat check (10f Cell A ep0) ---
COMPAT_SEED = 1411
COMPAT_EXPECTED_OBJECTS = sorted(["A", "D", "E", "F"])
COMPAT_EXPECTED_ORDER = ["A", "D", "E", "F"]

# Resume flag — set True to skip compat+CAL (already completed on 2026-03-19,
# compat PASSED, CAL 3/10 30% — investigator override per Option A)
SKIP_CALIBRATION = False


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


def _count_desync_activations(agent: BaseLLMAgent) -> int:
    """Count desync fix activations (rejected transitions) from agent's transition log."""
    if not hasattr(agent, "_transition_log"):
        return 0
    return sum(1 for entry in agent._transition_log if not entry.get("success", True))


async def run_compat_check(
    client: anthropic.AsyncAnthropic,
    config: ExperimentConfig,
    cost_tracker: CostTracker,
    trace_dir: Path,
) -> tuple[bool, EpisodeResult | None]:
    """Run backward compatibility check: 1 episode of base_llm on seed 1411."""
    logger.info(f"\n{'='*70}")
    logger.info(f"COMPAT CHECK: base_llm / seed {COMPAT_SEED} / {MAX_STEPS} steps")
    logger.info(f"{'='*70}")

    # Verify rule generation matches 10f
    rule = generate_randomized_order_rule(COMPAT_SEED, PRIMARY_ALL_LABELS, PRIMARY_N_CORRECT)
    rule_objects = sorted(rule["correct_objects"])
    rule_order = rule["required_order"]

    if rule_objects != COMPAT_EXPECTED_OBJECTS:
        logger.error(
            f"COMPAT FAIL: rule objects {rule_objects} != expected {COMPAT_EXPECTED_OBJECTS}"
        )
        return False, None
    if rule_order != COMPAT_EXPECTED_ORDER:
        logger.error(
            f"COMPAT FAIL: rule order {rule_order} != expected {COMPAT_EXPECTED_ORDER}"
        )
        return False, None
    logger.info(f"  Rule generation matches 10f: objects={rule_objects}, order={rule_order}")

    # Run episode
    gf_config = GradualFeedbackConfig(
        enabled=True,
        checkpoint_interval=5,
        correct_objects=sorted(rule["correct_objects"]),
    )

    condition_name = "compat_check"
    seed = config.get_episode_seed(condition_name, AgentType.BASE_LLM.value, 0)
    env = create_env_6obj(rule["correct_objects"], rule_order, MAX_STEPS, seed)
    ti = TextInterface6(include_order_sensitive=True)
    agent = create_agent(AgentType.BASE_LLM, client, config, PRIMARY_NUM_OBJECTS, ti)

    result = await run_episode(
        env=env,
        agent=agent,
        condition=condition_name,
        episode_number=0,
        trace_dir=trace_dir,
        max_cost_per_episode=MAX_COST_PER_EPISODE,
        gradual_feedback_config=gf_config,
    )

    episode_cost = cost_tracker.add_episode_cost(
        condition=condition_name,
        agent=AgentType.BASE_LLM.value,
        episode=0,
        input_tokens=result.total_input_tokens,
        output_tokens=result.total_output_tokens,
        trigger_tokens=result.trigger_tokens,
    )

    # Check desync activations
    desync_count = _count_desync_activations(agent)

    logger.info(f"  Result: {'CORRECT' if result.success else 'WRONG'}")
    logger.info(f"  Steps: {result.steps_taken}, Cost: ${episode_cost:.3f}")
    logger.info(f"  Desync fix activations: {desync_count} (expected: 0)")

    if result.error:
        logger.error(f"  COMPAT FAIL: Episode error: {result.error}")
        return False, result

    if desync_count > 0:
        logger.warning(f"  COMPAT WARNING: {desync_count} desync activations on v1 CG")

    logger.info(f"  COMPAT CHECK PASSED (rule match, no errors)")
    return True, result


async def run_cell(
    cell_label: str,
    agent_type: AgentType,
    max_steps: int,
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
) -> list[tuple[EpisodeResult, int]]:
    """Run a single cell. Returns list of (result, desync_count) tuples."""
    results: list[tuple[EpisodeResult, int]] = []
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

        desync_count = _count_desync_activations(agent)

        completed_ref[0] += 1

        order_str = ""
        if result.order_correct is not None:
            order_str = f" | Order: {'CORRECT' if result.order_correct else 'WRONG'}"
        gf_str = f" | GF: {len(result.gradual_feedback_events)} objects revealed"
        desync_str = f" | Desyncs: {desync_count}" if desync_count > 0 else ""

        logger.info(
            f"  Result: {'CORRECT' if result.success else 'WRONG'}{order_str}{gf_str}{desync_str} | "
            f"Steps: {result.steps_taken} | Cost: ${episode_cost:.3f} | "
            f"Total: ${cost_tracker.total_cost:.2f}"
        )

        results.append((result, desync_count))

    return results


async def run_powered_replication() -> list[EpisodeResult]:
    """Run all Run 10g cells in a single session."""

    config = ExperimentConfig(
        episodes_per_condition=PRIMARY_EPISODES,
        max_steps_per_episode=MAX_STEPS,
        trace_dir=Path(f"data/traces_{RUN_NAME}"),
        results_dir=Path(f"data/results_{RUN_NAME}"),
        base_seed=CAL_BASE_SEED,
        max_total_cost=BUDGET_CAP,
        max_cost_per_episode=MAX_COST_PER_EPISODE,
    )

    client = anthropic.AsyncAnthropic()
    cost_tracker = CostTracker(max_total=config.max_total_cost)
    all_results: list[EpisodeResult] = []
    all_desync_counts: dict[str, list[int]] = {}

    # Primary rules (seeds 1421-1440)
    primary_rules = []
    for ep in range(PRIMARY_EPISODES):
        rule_seed = PRIMARY_BASE_SEED + ep
        rule = generate_randomized_order_rule(
            seed=rule_seed, all_labels=PRIMARY_ALL_LABELS, n_correct=PRIMARY_N_CORRECT,
        )
        primary_rules.append((rule_seed, rule))

    # Compat rule (for summary JSON)
    compat_rule = generate_randomized_order_rule(
        COMPAT_SEED, PRIMARY_ALL_LABELS, PRIMARY_N_CORRECT,
    )

    compat_ok = False
    compat_result = None

    if SKIP_CALIBRATION:
        # --- Resume mode: skip compat+CAL, go straight to primary cells ---
        total_episodes = PRIMARY_EPISODES * 2  # 20 A + 20 B = 40
        completed_ref = [0]
        logger.info(f"Run 10g: Powered Replication — RESUME MODE (skip compat+CAL)")
        logger.info(f"  Compat+CAL already completed 2026-03-19: compat PASSED, CAL 3/10 (30%)")
        logger.info(f"  Investigator override: proceeding with primary cells")
        logger.info(f"  Budget cap: ${BUDGET_CAP}, max cost/episode: ${MAX_COST_PER_EPISODE}")
        logger.info(f"\nPrimary cells (A, B): {PRIMARY_EPISODES} episodes each (seeds {PRIMARY_BASE_SEED}-{PRIMARY_BASE_SEED + PRIMARY_EPISODES - 1}, 4-obj, {MAX_STEPS} steps)")
        for ep, (rule_seed, rule) in enumerate(primary_rules):
            logger.info(
                f"  Episode {ep}: seed={rule_seed}, "
                f"correct={sorted(rule['correct_objects'])}, "
                f"order={rule['required_order']}, "
                f"distractors={sorted(rule['distractors'])}"
            )
        compat_ok = True  # Already verified
    else:
        # --- Full mode: compat → CAL → primary cells ---
        total_episodes = 1 + CAL_EPISODES + PRIMARY_EPISODES * 2
        completed_ref = [0]

        # Calibration rules
        cal_rules = []
        for ep in range(CAL_EPISODES):
            rule_seed = CAL_BASE_SEED + ep
            rule = generate_randomized_order_rule(
                seed=rule_seed, all_labels=CAL_ALL_LABELS, n_correct=CAL_N_CORRECT,
            )
            cal_rules.append((rule_seed, rule))

        logger.info(f"Run 10g: Powered Replication — {total_episodes} episodes")
        logger.info(f"  Budget cap: ${BUDGET_CAP}, max cost/episode: ${MAX_COST_PER_EPISODE}")

        logger.info(f"\nCompat check: seed {COMPAT_SEED}, 4-obj, {MAX_STEPS} steps")
        logger.info(
            f"  Expected: correct={COMPAT_EXPECTED_OBJECTS}, order={COMPAT_EXPECTED_ORDER}"
        )

        logger.info(f"\nCalibration: {CAL_EPISODES} episodes (seeds 2001-2005, 3-obj, {CAL_MAX_STEPS} steps)")
        for ep, (rule_seed, rule) in enumerate(cal_rules):
            logger.info(
                f"  Episode {ep}: seed={rule_seed}, "
                f"correct={sorted(rule['correct_objects'])}, "
                f"order={rule['required_order']}, "
                f"distractors={sorted(rule['distractors'])}"
            )

        logger.info(f"\nPrimary cells (A, B): {PRIMARY_EPISODES} episodes each (seeds {PRIMARY_BASE_SEED}-{PRIMARY_BASE_SEED + PRIMARY_EPISODES - 1}, 4-obj, {MAX_STEPS} steps)")
        for ep, (rule_seed, rule) in enumerate(primary_rules):
            logger.info(
                f"  Episode {ep}: seed={rule_seed}, "
                f"correct={sorted(rule['correct_objects'])}, "
                f"order={rule['required_order']}, "
                f"distractors={sorted(rule['distractors'])}"
            )

        # === STEP 0: COMPAT CHECK ===
        compat_ok, compat_result = await run_compat_check(
            client, config, cost_tracker,
            trace_dir=Path(f"data/traces_{RUN_NAME}/compat"),
        )
        if compat_result:
            all_results.append(compat_result)
        completed_ref[0] += 1

        if not compat_ok:
            logger.error("COMPAT CHECK FAILED. Aborting run.")
            return all_results

        # === STEP 1: CALIBRATION ===
        logger.info(f"\n{'='*70}")
        logger.info(f"CELL CAL: base_llm × 3-obj × {CAL_MAX_STEPS} steps")
        logger.info(f"{'='*70}")

        cal_cell_results = await run_cell(
            cell_label="CAL",
            agent_type=AgentType.BASE_LLM,
            max_steps=CAL_MAX_STEPS,
            num_objects=CAL_NUM_OBJECTS,
            all_labels=CAL_ALL_LABELS,
            n_correct=CAL_N_CORRECT,
            rules=cal_rules,
            config=config,
            client=client,
            cost_tracker=cost_tracker,
            trace_dir=Path(f"data/traces_{RUN_NAME}/calibration"),
            completed_ref=completed_ref,
            total_episodes=total_episodes,
        )

        cal_results = [r for r, _ in cal_cell_results]
        cal_desyncs = [d for _, d in cal_cell_results]
        all_results.extend(cal_results)
        all_desync_counts["CAL"] = cal_desyncs

        # CAL go/no-go
        cal_correct = sum(1 for r in cal_results if r.success)
        logger.info(f"\n  CAL result: {cal_correct}/{len(cal_results)}")

        if cal_correct <= 1:
            logger.warning(
                f"  CAL ≤ 1/{len(cal_results)} correct — triggering enhanced calibration"
            )
            enhanced_rules = []
            for ep in range(5):
                rule_seed = 2006 + ep
                rule = generate_randomized_order_rule(
                    seed=rule_seed, all_labels=CAL_ALL_LABELS, n_correct=CAL_N_CORRECT,
                )
                enhanced_rules.append((rule_seed, rule))

            logger.info(f"  Running enhanced calibration: seeds 2006-2010")
            enhanced_results_raw = await run_cell(
                cell_label="CAL_ENHANCED",
                agent_type=AgentType.BASE_LLM,
                max_steps=CAL_MAX_STEPS,
                num_objects=CAL_NUM_OBJECTS,
                all_labels=CAL_ALL_LABELS,
                n_correct=CAL_N_CORRECT,
                rules=enhanced_rules,
                config=config,
                client=client,
                cost_tracker=cost_tracker,
                trace_dir=Path(f"data/traces_{RUN_NAME}/calibration_enhanced"),
                completed_ref=completed_ref,
                total_episodes=total_episodes + 5,
            )
            enhanced_results = [r for r, _ in enhanced_results_raw]
            all_results.extend(enhanced_results)

            full_cal_correct = cal_correct + sum(1 for r in enhanced_results if r.success)
            full_cal_total = len(cal_results) + len(enhanced_results)
            logger.info(f"  Full CAL: {full_cal_correct}/{full_cal_total}")

            if full_cal_correct / full_cal_total < 0.4:
                logger.error(
                    f"  Full CAL accuracy {full_cal_correct}/{full_cal_total} "
                    f"({100*full_cal_correct/full_cal_total:.0f}%) below historical range (50-70%). "
                    f"Aborting run — investigate before proceeding."
                )
                return all_results

    # === STEP 2: CELL A (base_llm) ===
    logger.info(f"\n{'='*70}")
    logger.info(f"CELL A: base_llm × 4-obj × {MAX_STEPS} steps")
    logger.info(f"{'='*70}")

    cell_a_results_raw = await run_cell(
        cell_label="A",
        agent_type=AgentType.BASE_LLM,
        max_steps=MAX_STEPS,
        num_objects=PRIMARY_NUM_OBJECTS,
        all_labels=PRIMARY_ALL_LABELS,
        n_correct=PRIMARY_N_CORRECT,
        rules=primary_rules,
        config=config,
        client=client,
        cost_tracker=cost_tracker,
        trace_dir=Path(f"data/traces_{RUN_NAME}/4obj_100steps"),
        completed_ref=completed_ref,
        total_episodes=total_episodes,
    )

    cell_a_results = [r for r, _ in cell_a_results_raw]
    cell_a_desyncs = [d for _, d in cell_a_results_raw]
    all_results.extend(cell_a_results)
    all_desync_counts["A"] = cell_a_desyncs

    if not cost_tracker.check_budget():
        logger.warning(f"Budget exhausted after Cell A. Stopping.")
    else:
        # === STEP 3: CELL B (full_cg_db) ===
        logger.info(f"\n{'='*70}")
        logger.info(f"CELL B: full_cg_db × 4-obj × {MAX_STEPS} steps")
        logger.info(f"{'='*70}")

        cell_b_results_raw = await run_cell(
            cell_label="B",
            agent_type=AgentType.FULL_CG_DB,
            max_steps=MAX_STEPS,
            num_objects=PRIMARY_NUM_OBJECTS,
            all_labels=PRIMARY_ALL_LABELS,
            n_correct=PRIMARY_N_CORRECT,
            rules=primary_rules,
            config=config,
            client=client,
            cost_tracker=cost_tracker,
            trace_dir=Path(f"data/traces_{RUN_NAME}/4obj_100steps"),
            completed_ref=completed_ref,
            total_episodes=total_episodes,
        )

        cell_b_results = [r for r, _ in cell_b_results_raw]
        cell_b_desyncs = [d for _, d in cell_b_results_raw]
        all_results.extend(cell_b_results)
        all_desync_counts["B"] = cell_b_desyncs

    # === Save summary ===
    summary_path = config.results_dir / "powered_replication.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": RUN_NAME,
        "description": (
            "Run 10g: Powered replication of 10f 100-step finding. "
            "Tests whether full_cg_db > base_llm at 100 steps on 4-object ORDER_SENSITIVE. "
            "Pools with 10f pilot (n=10 per cell) for N=30 pooled analysis. "
            "Single session. Cells: COMPAT (1 ep), CAL (5 ep base_llm 3-obj), "
            "A (20 ep base_llm 4-obj 100 steps), B (20 ep full_cg_db 4-obj 100 steps). "
            "Instrumented: temperature=1.0 pinned, cost limiter enforced, desync logging."
        ),
        "instrumentation": {
            "temperature": 1.0,
            "action_model": config.action_model,
            "trigger_model": config.trigger_model,
            "cost_limiter_enforced": True,
            "max_cost_per_episode": MAX_COST_PER_EPISODE,
        },
        "compat_check": {
            "seed": COMPAT_SEED,
            "expected_objects": COMPAT_EXPECTED_OBJECTS,
            "expected_order": COMPAT_EXPECTED_ORDER,
            "passed": compat_ok,
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
            "max_steps": MAX_STEPS,
            "cells": [
                {
                    "label": "A",
                    "agent_type": AgentType.BASE_LLM.value,
                    "max_steps": MAX_STEPS,
                    "seeds": f"{PRIMARY_BASE_SEED}-{PRIMARY_BASE_SEED + PRIMARY_EPISODES - 1}",
                    "behavior_version": "N/A",
                },
                {
                    "label": "B",
                    "agent_type": AgentType.FULL_CG_DB.value,
                    "max_steps": MAX_STEPS,
                    "seeds": f"{PRIMARY_BASE_SEED}-{PRIMARY_BASE_SEED + PRIMARY_EPISODES - 1}",
                    "behavior_version": "10c",
                },
            ],
            "episode_rules": [
                {
                    "episode": ep,
                    "rule_seed": rule_seed,
                    "rule_correct_objects": sorted(rule["correct_objects"]),
                    "rule_required_order": rule["required_order"],
                    "rule_distractors": sorted(rule["distractors"]),
                }
                for ep, (rule_seed, rule) in enumerate(primary_rules)
            ],
        },
        "gradual_feedback_config": {
            "enabled": True,
            "checkpoint_interval": 5,
        },
        "desync_counts": {
            cell: counts for cell, counts in all_desync_counts.items()
        },
        "cost_summary": cost_tracker.get_summary(),
        "results": [],
    }

    # Attach results
    for r in all_results:
        # Find matching rule
        if r.condition == "compat_check":
            rule_seed = COMPAT_SEED
            rule = compat_rule
        elif r.condition.startswith("cal_"):
            rule_seed, rule = cal_rules[r.episode_number]
        elif r.condition == "4obj_order_rand_A" or r.condition == "4obj_order_rand_B":
            rule_seed, rule = primary_rules[r.episode_number]
        else:
            # Enhanced calibration or other
            rule_seed = 0
            rule = {"correct_objects": set(), "required_order": [], "distractors": set()}

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
    logger.info(f"Run 10g: Powered Replication Results")
    logger.info(f"{'='*70}")

    # Compat
    if compat_result:
        logger.info(
            f"\n  COMPAT (seed {COMPAT_SEED}): "
            f"{'CORRECT' if compat_result.success else 'WRONG'} | "
            f"Steps: {compat_result.steps_taken}"
        )

    # Calibration summary
    if cal_results:
        correct = sum(1 for r in cal_results if r.success)
        total = len(cal_results)
        submitted = sum(1 for r in cal_results if r.answer_submitted)
        obj_correct = sum(1 for r in cal_results if r.objects_correct)
        order_correct_count = sum(1 for r in cal_results if r.order_correct)
        avg_gf = sum(len(r.gradual_feedback_events) for r in cal_results) / total
        logger.info(
            f"\n  CAL (base_llm / 3-obj / {CAL_MAX_STEPS} steps): "
            f"{correct}/{total} ({100*correct/total:.1f}%) | "
            f"Submitted: {submitted}/{total} | "
            f"Objects: {obj_correct}/{total} | Order: {order_correct_count}/{total} | "
            f"Avg GF events: {avg_gf:.1f}"
        )

    # Primary cell summaries
    for cell_label, agent_type_val in [("A", AgentType.BASE_LLM), ("B", AgentType.FULL_CG_DB)]:
        condition_name = f"4obj_order_rand_{cell_label}"
        cell_results = [r for r in all_results if r.condition == condition_name]
        if cell_results:
            correct = sum(1 for r in cell_results if r.success)
            total = len(cell_results)
            submitted = sum(1 for r in cell_results if r.answer_submitted)
            obj_correct = sum(1 for r in cell_results if r.objects_correct)
            order_correct_count = sum(1 for r in cell_results if r.order_correct)
            avg_gf = sum(len(r.gradual_feedback_events) for r in cell_results) / total
            desync_total = sum(all_desync_counts.get(cell_label, []))
            logger.info(
                f"  Cell {cell_label} ({agent_type_val.value} / 4-obj / {MAX_STEPS} steps): "
                f"{correct}/{total} ({100*correct/total:.1f}%) | "
                f"Submitted: {submitted}/{total} | "
                f"Objects: {obj_correct}/{total} | Order: {order_correct_count}/{total} | "
                f"Avg GF events: {avg_gf:.1f} | Desyncs: {desync_total}"
            )

    # Paired episode analysis (A vs B on fresh seeds)
    a_results = [r for r in all_results if r.condition == "4obj_order_rand_A"]
    b_results = [r for r in all_results if r.condition == "4obj_order_rand_B"]
    if len(a_results) == len(b_results) == PRIMARY_EPISODES:
        logger.info(f"\n--- Paired Analysis: A vs B (fresh seeds, n={PRIMARY_EPISODES}) ---")
        concordant = sum(1 for a, b in zip(a_results, b_results) if a.success == b.success)
        a_only = sum(1 for a, b in zip(a_results, b_results) if a.success and not b.success)
        b_only = sum(1 for a, b in zip(a_results, b_results) if not a.success and b.success)
        logger.info(f"  Concordant: {concordant}, A-only: {a_only}, B-only: {b_only}")

    logger.info(f"\nTotal cost: ${cost_tracker.total_cost:.2f}")
    logger.info(f"Summary saved: {summary_path}")

    return all_results


def main():
    asyncio.run(run_powered_replication())


if __name__ == "__main__":
    main()
