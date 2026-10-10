"""Run 6: Differential Exploration Depth and Architectural Rescue — 6 cells × 50 episodes (300 total).

Tests whether compositional architecture enables coordinate system revision
through deeper exploration and mechanistic detection. Two switch points
(sw=3 shared challenge, sw=5 CG-selective) across all three agent types.

Cells:
  A1: base_llm,    switch=3, 50 episodes
  A2: cg_only,     switch=3, 50 episodes
  A3: full_cg_db,  switch=3, 50 episodes
  B1: base_llm,    switch=5, 50 episodes
  B2: cg_only,     switch=5, 50 episodes
  B3: full_cg_db,  switch=5, 50 episodes

Seed 401 (independent from prior runs: 42, 137, 251, 307).
"""

from __future__ import annotations

RUN_NAME = "run6"

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
BLICKETS = ["A", "B"]
MAX_STEPS = 75
BASE_SEED = 401  # Independent from Run 3 (137), Run 3B (251), Run 4 pilot (307), Runs 1-2 (42)
BUDGET_CAP = 300.0

# 6 cells: (agent_type, switch_after_n)
CELLS = [
    (AgentType.BASE_LLM, 3),
    (AgentType.CG_ONLY, 3),
    (AgentType.FULL_CG_DB, 3),
    (AgentType.BASE_LLM, 5),
    (AgentType.CG_ONLY, 5),
    (AgentType.FULL_CG_DB, 5),
]


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


def create_hidden_moderator_env(config: ExperimentConfig, seed: int, switch_after_n: int) -> ExtendedBlicketEnv5:
    """Create hidden moderator environment with parameterized switch point."""
    base_rule = CausalRule(
        rule_type=CausalRuleType.CONJUNCTIVE,
        required_objects=set(BLICKETS),
    )
    alternate_rule = CausalRule(
        rule_type=CausalRuleType.DISJUNCTIVE,
        required_objects={"C"},
    )
    return ExtendedBlicketEnv5(
        causal_rule=base_rule,
        condition=ExtendedCondition.HIDDEN_MODERATOR,
        max_steps=MAX_STEPS,
        moderator_config=HiddenModeratorConfig(
            switch_after_n=switch_after_n,
            alternate_rule=alternate_rule,
        ),
        rng_seed=seed,
    )


async def run_6_experiments(config: ExperimentConfig | None = None) -> list[EpisodeResult]:
    """Run 6: Differential exploration depth — 6 cells × 50 episodes."""
    if config is None:
        config = ExperimentConfig(
            episodes_per_condition=50,
            max_steps_per_episode=MAX_STEPS,
            trace_dir=Path(f"data/traces_{RUN_NAME}"),
            results_dir=Path(f"data/results_{RUN_NAME}"),
            base_seed=BASE_SEED,
            max_total_cost=BUDGET_CAP,
        )

    client = anthropic.AsyncAnthropic()
    cost_tracker = CostTracker(max_total=config.max_total_cost)
    results: list[EpisodeResult] = []

    total_episodes = len(CELLS) * config.episodes_per_condition
    completed = 0

    for agent_type, switch_n in CELLS:
        condition = f"hidden_moderator_sw{switch_n}"

        for ep in range(config.episodes_per_condition):
            if not cost_tracker.check_budget():
                logger.warning(f"Budget exhausted (${cost_tracker.total_cost:.2f}/{BUDGET_CAP}). Stopping.")
                break

            logger.info(
                f"[{completed + 1}/{total_episodes}] "
                f"{condition}/{agent_type.value}/ep{ep} (switch={switch_n})"
            )

            seed = config.get_episode_seed(condition, agent_type.value, ep)
            env = create_hidden_moderator_env(config, seed, switch_n)
            agent = create_agent(agent_type, client, config)

            trace_subdir = config.trace_dir / f"sw{switch_n}"

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
            completed += 1

            logger.info(
                f"  Result: {'CORRECT' if result.success else 'WRONG'} | "
                f"Steps: {result.steps_taken} | Cost: ${episode_cost:.3f} | "
                f"Total: ${cost_tracker.total_cost:.2f}"
            )

    # Save summary with switch_after_n field
    summary_path = config.results_dir / "extended_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a lookup from (condition, agent_type, episode) -> switch_n
    cell_lookup = {}
    for agent_type, switch_n in CELLS:
        condition = f"hidden_moderator_sw{switch_n}"
        for ep in range(config.episodes_per_condition):
            cell_lookup[(condition, agent_type.value, ep)] = switch_n

    summary = {
        "run_name": RUN_NAME,
        "description": "Run 6: Differential exploration depth (seed 401): 6 cells × 50 episodes",
        "num_objects": NUM_OBJECTS,
        "base_seed": BASE_SEED,
        "cells": [
            {"agent_type": at.value, "switch_after_n": sn, "episodes": config.episodes_per_condition}
            for at, sn in CELLS
        ],
        "cost_summary": cost_tracker.get_summary(),
        "results": [
            {
                "condition": r.condition,
                "agent_type": r.agent_type,
                "episode": r.episode_number,
                "switch_after_n": cell_lookup.get((r.condition, r.agent_type, r.episode_number), None),
                "success": r.success,
                "steps": r.steps_taken,
                "answer_submitted": r.answer_submitted,
                "rule_type_correct": r.rule_type_correct,
                "objects_correct": r.objects_correct,
            }
            for r in results
        ],
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    logger.info(f"\n{'='*60}")
    logger.info(f"Run 6: Differential Exploration Depth Results")
    logger.info(f"{'='*60}")
    for agent_type, switch_n in CELLS:
        condition = f"hidden_moderator_sw{switch_n}"
        cond_results = [
            r for r in results
            if r.condition == condition and r.agent_type == agent_type.value
        ]
        if cond_results:
            correct = sum(1 for r in cond_results if r.success)
            total = len(cond_results)
            logger.info(f"  {agent_type.value}/sw{switch_n}: {correct}/{total} ({100*correct/total:.1f}%)")
    logger.info(f"Total cost: ${cost_tracker.total_cost:.2f}")
    logger.info(f"Summary saved: {summary_path}")

    return results


if __name__ == "__main__":
    asyncio.run(run_6_experiments())
