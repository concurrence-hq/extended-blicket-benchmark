"""Run 8: Powered Follow-Up — Triple Conjunction {C,D,E} — 5 cells, 200 episodes.

Powered replication of Run 07's finding that full_cg_db outperforms base_llm
on the triple conjunction, plus cg_only to determine whether the advantage
is DB-specific or CG-structural. Includes exploratory compressed-runway arm.

Cells:
  A: base_llm,   max_steps=75, 50 episodes  (replicates Run 07 C1)
  B: cg_only,    max_steps=75, 50 episodes  (NEW — attribution test)
  C: full_cg_db, max_steps=75, 50 episodes  (replicates Run 07 C2)
  D: base_llm,   max_steps=50, 25 episodes  (exploratory: runway compression)
  E: full_cg_db, max_steps=50, 25 episodes  (exploratory: runway compression)

Seed 601 (independent from Run 07: 501, and prior runs: 42, 137, 251, 307, 401).

Supports --cell flag for process-level parallelism:
  python -m experiments.run_8_powered_followup --cell A
  python -m experiments.run_8_powered_followup --cell B
  ...etc
  python -m experiments.run_8_powered_followup --merge
"""

from __future__ import annotations

RUN_NAME = "run8"

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
from agents.cg_agent import CGAgent
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
SWITCH_AFTER_N = 3
BASE_SEED = 601
BUDGET_CAP = 150.0

POST_SWITCH_RULE = CausalRule(
    rule_type=CausalRuleType.CONJUNCTIVE,
    required_objects={"C", "D", "E"},
)

# Cell definitions: (agent_type, max_steps, episodes)
CELL_DEFS = {
    "A": (AgentType.BASE_LLM, 75, 50),
    "B": (AgentType.CG_ONLY, 75, 50),
    "C": (AgentType.FULL_CG_DB, 75, 50),
    "D": (AgentType.BASE_LLM, 50, 25),
    "E": (AgentType.FULL_CG_DB, 50, 25),
}

CELL_ORDER = ["A", "B", "C", "D", "E"]


def create_agent(
    agent_type: AgentType, client: anthropic.AsyncAnthropic, config: ExperimentConfig
) -> BaseLLMAgent:
    ti = TextInterface5()
    if agent_type == AgentType.BASE_LLM:
        return BaseLLMAgent(llm_client=client, model=config.action_model, text_interface=ti)
    elif agent_type == AgentType.CG_ONLY:
        return CGAgent(llm_client=client, model=config.action_model, text_interface=ti, num_objects=NUM_OBJECTS)
    elif agent_type == AgentType.FULL_CG_DB:
        return FullAgent(
            llm_client=client,
            model=config.action_model,
            trigger_model=config.trigger_model,
            text_interface=ti,
            num_objects=NUM_OBJECTS,
        )
    raise ValueError(f"Unknown agent type: {agent_type}")


def create_env(seed: int, max_steps: int) -> ExtendedBlicketEnv5:
    """Create hidden moderator environment with triple conjunction post-switch rule."""
    base_rule = CausalRule(
        rule_type=CausalRuleType.CONJUNCTIVE,
        required_objects=set(PRE_SWITCH_BLICKETS),
    )
    return ExtendedBlicketEnv5(
        causal_rule=base_rule,
        condition=ExtendedCondition.HIDDEN_MODERATOR,
        max_steps=max_steps,
        moderator_config=HiddenModeratorConfig(
            switch_after_n=SWITCH_AFTER_N,
            alternate_rule=POST_SWITCH_RULE,
        ),
        rng_seed=seed,
    )


def condition_label(max_steps: int) -> str:
    return f"hidden_moderator_conj_CDE_s{max_steps}"


async def run_cell(
    cell_name: str,
    config: ExperimentConfig,
) -> list[EpisodeResult]:
    """Run all episodes for a single cell."""
    agent_type, max_steps, num_episodes = CELL_DEFS[cell_name]
    condition = condition_label(max_steps)

    client = anthropic.AsyncAnthropic()
    cost_tracker = CostTracker(max_total=config.max_total_cost)
    results: list[EpisodeResult] = []

    for ep in range(num_episodes):
        if not cost_tracker.check_budget():
            logger.warning(
                f"Budget exhausted (${cost_tracker.total_cost:.2f}/{config.max_total_cost}). Stopping."
            )
            break

        logger.info(
            f"[{cell_name}] [{ep + 1}/{num_episodes}] "
            f"{condition}/{agent_type.value}/ep{ep}"
        )

        seed = config.get_episode_seed(condition, agent_type.value, ep)
        env = create_env(seed, max_steps)
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
        "max_steps": max_steps,
        "post_switch_rule": {
            "type": POST_SWITCH_RULE.rule_type.value,
            "objects": sorted(POST_SWITCH_RULE.required_objects),
        },
        "condition": condition,
        "base_seed": BASE_SEED,
        "switch_after_n": SWITCH_AFTER_N,
        "num_episodes": num_episodes,
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

    correct = sum(1 for r in results if r.success)
    total = len(results)
    logger.info(f"\n{'='*50}")
    logger.info(f"Cell {cell_name}: {agent_type.value} / max_steps={max_steps} / n={num_episodes}")
    logger.info(f"  Post-switch rule: conjunctive {sorted(POST_SWITCH_RULE.required_objects)}")
    logger.info(f"  Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    logger.info(f"  Cost: ${cost_tracker.total_cost:.2f}")
    logger.info(f"  Saved: {cell_results_path}")
    logger.info(f"{'='*50}")

    return results


async def run_all_cells(config: ExperimentConfig) -> list[EpisodeResult]:
    """Run all 5 cells sequentially."""
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
            r["max_steps"] = data["max_steps"]
        all_results.extend(data["results"])
        cell_cost = data["cost_summary"]["total_cost_usd"]
        total_cost += cell_cost
        all_costs.append({
            "cell": data["cell"],
            "agent_type": data["agent_type"],
            "max_steps": data["max_steps"],
            "episodes": data["cost_summary"]["episodes_run"],
            "cost_usd": cell_cost,
        })

    merged = {
        "run_name": RUN_NAME,
        "description": "Run 8: Powered Follow-Up — Triple Conjunction {C,D,E} (seed 601): 5 cells, 200 episodes",
        "num_objects": NUM_OBJECTS,
        "base_seed": BASE_SEED,
        "switch_after_n": SWITCH_AFTER_N,
        "pre_switch_rule": {
            "type": "conjunctive",
            "objects": sorted(PRE_SWITCH_BLICKETS),
        },
        "post_switch_rule": {
            "type": POST_SWITCH_RULE.rule_type.value,
            "objects": sorted(POST_SWITCH_RULE.required_objects),
        },
        "cells": {
            name: {
                "agent_type": CELL_DEFS[name][0].value,
                "max_steps": CELL_DEFS[name][1],
                "episodes": CELL_DEFS[name][2],
            }
            for name in CELL_ORDER
        },
        "cells_merged": [cf.name for cf in cell_files],
        "cost_by_cell": all_costs,
        "total_cost_usd": total_cost,
        "results": all_results,
    }

    summary_path = results_dir / "extended_results.json"
    with open(summary_path, "w") as f:
        json.dump(merged, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Run 8: Powered Follow-Up — Merged Results")
    logger.info(f"{'='*60}")
    for cell_name in CELL_ORDER:
        agent_type, max_steps, num_eps = CELL_DEFS[cell_name]
        cell_results = [r for r in all_results if r.get("cell") == cell_name]
        if cell_results:
            correct = sum(1 for r in cell_results if r["success"])
            total = len(cell_results)
            logger.info(
                f"  {cell_name}: {agent_type.value} (steps={max_steps}, n={num_eps}): "
                f"{correct}/{total} ({100*correct/total:.1f}%)"
            )
    logger.info(f"Total cost: ${total_cost:.2f}")
    logger.info(f"Summary saved: {summary_path}")

    return merged


def make_config(cell_name: str | None = None) -> ExperimentConfig:
    """Create experiment config."""
    if cell_name:
        _, _, num_eps = CELL_DEFS[cell_name]
        per_cell_budget = 55.0 if num_eps == 50 else 25.0
    else:
        per_cell_budget = BUDGET_CAP

    return ExperimentConfig(
        episodes_per_condition=50,  # not used directly; per-cell counts in CELL_DEFS
        max_steps_per_episode=75,   # not used directly; per-cell max_steps in CELL_DEFS
        trace_dir=Path(f"data/traces_{RUN_NAME}"),
        results_dir=Path(f"data/results_{RUN_NAME}"),
        base_seed=BASE_SEED,
        max_total_cost=per_cell_budget,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 8: Powered Follow-Up — Triple Conjunction {C,D,E} (5 cells, 200 episodes)"
    )
    parser.add_argument(
        "--cell",
        choices=list(CELL_DEFS.keys()),
        help="Run a single cell (for parallel execution). Options: A, B, C, D, E",
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
