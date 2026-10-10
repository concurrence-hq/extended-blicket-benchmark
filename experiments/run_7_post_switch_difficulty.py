"""Run 7: Post-Switch Difficulty Pilot — 6 cells × 20 episodes (120 total).

Calibration pilot testing whether harder post-switch rules create measurable
differences in reasoning-eligible accuracy between agents. Only base_llm and
full_cg_db (the two architectural endpoints).

Three post-switch difficulty levels, all conjunctive:
  Medium:              Conjunctive {C, D}     — new pair, no overlap
  Hard (interference): Conjunctive {A, C}     — partial overlap with pre-switch
  Hard (search space): Conjunctive {C, D, E}  — three-way conjunction

Cells:
  A1: base_llm,   post_switch=Conj{C,D},    switch=3, 20 episodes
  A2: full_cg_db, post_switch=Conj{C,D},    switch=3, 20 episodes
  B1: base_llm,   post_switch=Conj{A,C},    switch=3, 20 episodes
  B2: full_cg_db, post_switch=Conj{A,C},    switch=3, 20 episodes
  C1: base_llm,   post_switch=Conj{C,D,E},  switch=3, 20 episodes
  C2: full_cg_db, post_switch=Conj{C,D,E},  switch=3, 20 episodes

Seed 501 (independent from prior runs: 42, 137, 251, 307, 401).

Supports --cell flag for process-level parallelism:
  python -m experiments.run_7_post_switch_difficulty --cell A1
  python -m experiments.run_7_post_switch_difficulty --cell A2
  ...etc (run multiple in separate terminals / background processes)
  python -m experiments.run_7_post_switch_difficulty --merge  (combine per-cell results)
"""

from __future__ import annotations

RUN_NAME = "run7"

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.blicket_env import CausalRule, CausalRuleType
from environment.extended_env import (
    ExtendedCondition,
    HiddenModeratorConfig,
)
from environment.extended_env_5obj import ExtendedBlicketEnv5
from environment.text_interface import TextInterface5
from agents.base_llm_agent import BaseLLMAgent
from agents.full_agent import FullAgent
from experiments.config import AgentType, CostTracker, ExperimentConfig
from experiments.episode_runner import EpisodeResult, run_episode

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

NUM_OBJECTS = 5
PRE_SWITCH_BLICKETS = ["A", "B"]
MAX_STEPS = 75
SWITCH_AFTER_N = 3
BASE_SEED = 501
BUDGET_CAP = 150.0

# Post-switch rules by difficulty level
POST_SWITCH_RULES = {
    "medium": CausalRule(
        rule_type=CausalRuleType.CONJUNCTIVE,
        required_objects={"C", "D"},
    ),
    "hard_interference": CausalRule(
        rule_type=CausalRuleType.CONJUNCTIVE,
        required_objects={"A", "C"},
    ),
    "hard_search_space": CausalRule(
        rule_type=CausalRuleType.CONJUNCTIVE,
        required_objects={"C", "D", "E"},
    ),
}

# Cell definitions: (cell_name, agent_type, difficulty_key)
CELL_DEFS = {
    "A1": (AgentType.BASE_LLM, "medium"),
    "A2": (AgentType.FULL_CG_DB, "medium"),
    "B1": (AgentType.BASE_LLM, "hard_interference"),
    "B2": (AgentType.FULL_CG_DB, "hard_interference"),
    "C1": (AgentType.BASE_LLM, "hard_search_space"),
    "C2": (AgentType.FULL_CG_DB, "hard_search_space"),
}

# Ordered list for sequential execution
CELL_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]


def create_agent(
    agent_type: AgentType, client: anthropic.AsyncAnthropic, config: ExperimentConfig
) -> BaseLLMAgent:
    ti = TextInterface5()
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


def create_env(config: ExperimentConfig, seed: int, difficulty: str) -> ExtendedBlicketEnv5:
    """Create hidden moderator environment with configurable post-switch rule."""
    base_rule = CausalRule(
        rule_type=CausalRuleType.CONJUNCTIVE,
        required_objects=set(PRE_SWITCH_BLICKETS),
    )
    return ExtendedBlicketEnv5(
        causal_rule=base_rule,
        condition=ExtendedCondition.HIDDEN_MODERATOR,
        max_steps=MAX_STEPS,
        moderator_config=HiddenModeratorConfig(
            switch_after_n=SWITCH_AFTER_N,
            alternate_rule=POST_SWITCH_RULES[difficulty],
        ),
        rng_seed=seed,
    )


def difficulty_label(difficulty: str) -> str:
    """Human-readable label for condition naming."""
    return {
        "medium": "conj_CD",
        "hard_interference": "conj_AC",
        "hard_search_space": "conj_CDE",
    }[difficulty]


async def run_cell(
    cell_name: str,
    config: ExperimentConfig,
) -> list[EpisodeResult]:
    """Run all episodes for a single cell."""
    agent_type, difficulty = CELL_DEFS[cell_name]
    condition = f"hidden_moderator_{difficulty_label(difficulty)}"

    client = anthropic.AsyncAnthropic()
    cost_tracker = CostTracker(max_total=config.max_total_cost)
    results: list[EpisodeResult] = []

    for ep in range(config.episodes_per_condition):
        if not cost_tracker.check_budget():
            logger.warning(
                f"Budget exhausted (${cost_tracker.total_cost:.2f}/{config.max_total_cost}). Stopping."
            )
            break

        logger.info(
            f"[{cell_name}] [{ep + 1}/{config.episodes_per_condition}] "
            f"{condition}/{agent_type.value}/ep{ep}"
        )

        seed = config.get_episode_seed(condition, agent_type.value, ep)
        env = create_env(config, seed, difficulty)
        agent = create_agent(agent_type, client, config)

        trace_subdir = config.trace_dir / cell_name

        result = await run_episode(
            env=env,
            agent=agent,
            condition=condition,
            episode_number=ep,
            trace_dir=trace_subdir,
            max_cost_per_episode=config.max_cost_per_episode,
        )

        episode_cost = cost_tracker.add_episode_cost(
            condition=condition,
            agent=agent_type.value,
            episode=ep,
            input_tokens=result.total_input_tokens,
            output_tokens=result.total_output_tokens,
            trigger_tokens=result.trigger_tokens,
        )

        results.append(result)

        logger.info(
            f"  Result: {'CORRECT' if result.success else 'WRONG'} | "
            f"Steps: {result.steps_taken} | Cost: ${episode_cost:.3f} | "
            f"Total: ${cost_tracker.total_cost:.2f}"
        )

    # Save per-cell results
    cell_results_path = config.results_dir / f"cell_{cell_name}.json"
    cell_results_path.parent.mkdir(parents=True, exist_ok=True)

    cell_summary = {
        "run_name": RUN_NAME,
        "cell": cell_name,
        "agent_type": agent_type.value,
        "difficulty": difficulty,
        "post_switch_rule": {
            "type": POST_SWITCH_RULES[difficulty].rule_type.value,
            "objects": sorted(POST_SWITCH_RULES[difficulty].required_objects),
        },
        "condition": condition,
        "base_seed": BASE_SEED,
        "switch_after_n": SWITCH_AFTER_N,
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
                "true_rule_type": r.true_rule_type,
                "true_objects": r.true_objects,
                "guessed_rule_type": r.guessed_rule_type,
                "guessed_objects": r.guessed_objects,
                "wall_time_seconds": r.wall_time_seconds,
            }
            for r in results
        ],
    }

    with open(cell_results_path, "w") as f:
        json.dump(cell_summary, f, indent=2)

    # Print cell summary
    correct = sum(1 for r in results if r.success)
    total = len(results)
    logger.info(f"\n{'='*50}")
    logger.info(f"Cell {cell_name}: {agent_type.value} / {difficulty}")
    logger.info(f"  Post-switch rule: {POST_SWITCH_RULES[difficulty].rule_type.value} {sorted(POST_SWITCH_RULES[difficulty].required_objects)}")
    logger.info(f"  Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    logger.info(f"  Cost: ${cost_tracker.total_cost:.2f}")
    logger.info(f"  Saved: {cell_results_path}")
    logger.info(f"{'='*50}")

    return results


async def run_all_cells(config: ExperimentConfig) -> list[EpisodeResult]:
    """Run all 6 cells sequentially (no --cell flag)."""
    all_results: list[EpisodeResult] = []
    for cell_name in CELL_ORDER:
        results = await run_cell(cell_name, config)
        all_results.extend(results)
    merge_results(config)
    return all_results


def merge_results(config: ExperimentConfig) -> None:
    """Merge per-cell result files into a single combined summary."""
    results_dir = config.results_dir
    cell_files = sorted(results_dir.glob("cell_*.json"))

    if not cell_files:
        logger.warning(f"No cell result files found in {results_dir}")
        return

    all_results = []
    all_costs = []
    total_cost = 0.0

    for cf in cell_files:
        with open(cf) as f:
            data = json.load(f)
        for r in data["results"]:
            r["cell"] = data["cell"]
            r["difficulty"] = data["difficulty"]
            r["post_switch_rule"] = data["post_switch_rule"]
        all_results.extend(data["results"])
        cell_cost = data["cost_summary"]["total_cost_usd"]
        total_cost += cell_cost
        all_costs.append({
            "cell": data["cell"],
            "agent_type": data["agent_type"],
            "difficulty": data["difficulty"],
            "episodes": data["cost_summary"]["episodes_run"],
            "cost_usd": cell_cost,
        })

    merged = {
        "run_name": RUN_NAME,
        "description": "Run 7: Post-Switch Difficulty Pilot (seed 501): 6 cells x 20 episodes",
        "num_objects": NUM_OBJECTS,
        "base_seed": BASE_SEED,
        "switch_after_n": SWITCH_AFTER_N,
        "pre_switch_rule": {
            "type": "conjunctive",
            "objects": sorted(PRE_SWITCH_BLICKETS),
        },
        "post_switch_rules": {
            k: {"type": v.rule_type.value, "objects": sorted(v.required_objects)}
            for k, v in POST_SWITCH_RULES.items()
        },
        "cells_merged": [cf.name for cf in cell_files],
        "cost_by_cell": all_costs,
        "total_cost_usd": total_cost,
        "results": all_results,
    }

    summary_path = results_dir / "extended_results.json"
    with open(summary_path, "w") as f:
        json.dump(merged, f, indent=2)

    # Print summary table
    logger.info(f"\n{'='*60}")
    logger.info(f"Run 7: Post-Switch Difficulty — Merged Results")
    logger.info(f"{'='*60}")
    for cell_name in CELL_ORDER:
        agent_type, difficulty = CELL_DEFS[cell_name]
        cell_results = [r for r in all_results if r.get("cell") == cell_name]
        if cell_results:
            correct = sum(1 for r in cell_results if r["success"])
            total = len(cell_results)
            rule_objs = sorted(POST_SWITCH_RULES[difficulty].required_objects)
            logger.info(
                f"  {cell_name}: {agent_type.value} / {difficulty} "
                f"(post={rule_objs}): {correct}/{total} ({100*correct/total:.1f}%)"
            )
    logger.info(f"Total cost: ${total_cost:.2f}")
    logger.info(f"Summary saved: {summary_path}")

    return merged


def make_config(cell_name: str | None = None) -> ExperimentConfig:
    """Create experiment config. Budget is per-cell when running a single cell."""
    if cell_name:
        # Per-cell budget: ~$25 for base_llm cells, ~$30 for full_cg_db cells
        per_cell_budget = 30.0
    else:
        per_cell_budget = BUDGET_CAP

    return ExperimentConfig(
        episodes_per_condition=20,
        max_steps_per_episode=MAX_STEPS,
        trace_dir=Path(f"data/traces_{RUN_NAME}"),
        results_dir=Path(f"data/results_{RUN_NAME}"),
        base_seed=BASE_SEED,
        max_total_cost=per_cell_budget,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 7: Post-Switch Difficulty Pilot (6 cells × 20 episodes)"
    )
    parser.add_argument(
        "--cell",
        choices=list(CELL_DEFS.keys()),
        help="Run a single cell (for parallel execution). "
        "Options: A1, A2, B1, B2, C1, C2",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge per-cell result files into combined summary (no episodes run)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.merge:
        config = make_config()
        merge_results(config)
    elif args.cell:
        config = make_config(cell_name=args.cell)
        asyncio.run(run_cell(args.cell, config))
    else:
        config = make_config()
        asyncio.run(run_all_cells(config))
