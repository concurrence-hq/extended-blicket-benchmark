"""Run 3 Phase B RESUME: Complete remaining extended condition episodes.

Resumes from where run_extended_5obj.py stopped after hitting the $200 budget cap.

Completed (176 episodes):
  - order_sensitive: 30 base_llm + 30 cg_only + 30 full_cg_db = 90
  - stochastic:      30 base_llm + 30 cg_only + 26 full_cg_db = 86

Remaining (~94 episodes):
  - stochastic/full_cg_db: ep26-29 (4 episodes)
  - hidden_moderator: 30 base_llm + 30 cg_only + 30 full_cg_db (90 episodes)

Loads existing results, runs remaining episodes, merges, and saves.
"""

from __future__ import annotations

RUN_NAME = "run3"

import asyncio
import json
import logging
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.blicket_env import CausalRule, CausalRuleType
from environment.blicket_env_5obj import BlicketEnv5
from environment.extended_env import (
    ExtendedCondition,
    HiddenModeratorConfig,
    OrderConfig,
    StochasticConfig,
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
BASE_SEED = 137
BUDGET_CAP = 320.0  # Raised from 200 to cover remaining ~94 episodes


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


def create_extended_env(
    condition: str, config: ExperimentConfig, seed: int
) -> ExtendedBlicketEnv5:
    base_rule = CausalRule(
        rule_type=CausalRuleType.CONJUNCTIVE,
        required_objects=set(BLICKETS),
    )

    if condition == "order_sensitive":
        return ExtendedBlicketEnv5(
            causal_rule=base_rule,
            condition=ExtendedCondition.ORDER_SENSITIVE,
            max_steps=MAX_STEPS,
            order_config=OrderConfig(required_order=["A", "B"]),
            rng_seed=seed,
        )
    elif condition == "stochastic":
        return ExtendedBlicketEnv5(
            causal_rule=base_rule,
            condition=ExtendedCondition.STOCHASTIC,
            max_steps=MAX_STEPS,
            stochastic_config=StochasticConfig(activation_probability=0.7),
            rng_seed=seed,
        )
    elif condition == "hidden_moderator":
        alternate_rule = CausalRule(
            rule_type=CausalRuleType.DISJUNCTIVE,
            required_objects={"C"},
        )
        return ExtendedBlicketEnv5(
            causal_rule=base_rule,
            condition=ExtendedCondition.HIDDEN_MODERATOR,
            max_steps=MAX_STEPS,
            moderator_config=HiddenModeratorConfig(
                switch_after_n=3,
                alternate_rule=alternate_rule,
            ),
            rng_seed=seed,
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")


def load_existing_results(results_path: Path) -> tuple[dict, list[dict], set[tuple[str, str, int]]]:
    """Load existing results and return (full_data, results_list, completed_keys)."""
    with open(results_path) as f:
        data = json.load(f)

    completed = set()
    for r in data["results"]:
        completed.add((r["condition"], r["agent_type"], r["episode"]))

    return data, data["results"], completed


# Define remaining work explicitly
REMAINING_EPISODES: list[tuple[str, AgentType, int]] = []

# stochastic/full_cg_db ep26-29
for ep in range(26, 30):
    REMAINING_EPISODES.append(("stochastic", AgentType.FULL_CG_DB, ep))

# hidden_moderator: all agents, all episodes
for agent_type in [AgentType.BASE_LLM, AgentType.CG_ONLY, AgentType.FULL_CG_DB]:
    for ep in range(30):
        REMAINING_EPISODES.append(("hidden_moderator", agent_type, ep))


async def run_resume() -> list[EpisodeResult]:
    """Run remaining episodes and merge with existing results."""
    results_path = Path(f"data/results_{RUN_NAME}/extended_results.json")

    # Load existing
    existing_data, existing_results, completed_keys = load_existing_results(results_path)
    prior_cost = existing_data["cost_summary"]["total_cost_usd"]
    logger.info(f"Loaded {len(existing_results)} existing results (prior cost: ${prior_cost:.2f})")

    config = ExperimentConfig(
        episodes_per_condition=30,
        max_steps_per_episode=MAX_STEPS,
        trace_dir=Path(f"data/traces_{RUN_NAME}"),
        results_dir=Path(f"data/results_{RUN_NAME}"),
        base_seed=BASE_SEED,
        max_total_cost=BUDGET_CAP,
    )

    client = anthropic.AsyncAnthropic()
    # Initialize cost tracker with prior spend
    cost_tracker = CostTracker(max_total=BUDGET_CAP - prior_cost)
    new_results: list[EpisodeResult] = []

    total_remaining = len(REMAINING_EPISODES)
    completed = 0
    skipped = 0

    for condition, agent_type, ep in REMAINING_EPISODES:
        key = (condition, agent_type.value, ep)
        if key in completed_keys:
            logger.info(f"  SKIP (already done): {condition}/{agent_type.value}/ep{ep}")
            skipped += 1
            continue

        if not cost_tracker.check_budget():
            logger.warning(
                f"Budget exhausted (${prior_cost + cost_tracker.total_cost:.2f}/{BUDGET_CAP}). Stopping."
            )
            break

        logger.info(
            f"[{completed + 1}/{total_remaining - skipped}] "
            f"{condition}/{agent_type.value}/ep{ep}"
        )

        seed = config.get_episode_seed(condition, agent_type.value, ep)
        env = create_extended_env(condition, config, seed)
        agent = create_agent(agent_type, client, config)

        result = await run_episode(
            env=env,
            agent=agent,
            condition=condition,
            episode_number=ep,
            trace_dir=config.trace_dir / "extended",
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

        new_results.append(result)
        completed += 1

        logger.info(
            f"  Result: {'CORRECT' if result.success else 'WRONG'} | "
            f"Steps: {result.steps_taken} | Cost: ${episode_cost:.3f} | "
            f"Session: ${cost_tracker.total_cost:.2f} | "
            f"Cumulative: ${prior_cost + cost_tracker.total_cost:.2f}"
        )

    # Merge results
    new_result_dicts = [
        {
            "condition": r.condition,
            "agent_type": r.agent_type,
            "episode": r.episode_number,
            "success": r.success,
            "steps": r.steps_taken,
            "answer_submitted": r.answer_submitted,
            "rule_type_correct": r.rule_type_correct,
            "objects_correct": r.objects_correct,
        }
        for r in new_results
    ]

    merged_results = existing_results + new_result_dicts

    # Merge cost data
    resume_cost_summary = cost_tracker.get_summary()
    merged_episode_costs = (
        existing_data["cost_summary"]["episode_costs"]
        + resume_cost_summary["episode_costs"]
    )
    total_cost = prior_cost + cost_tracker.total_cost

    merged_data = {
        "run_name": RUN_NAME,
        "description": "5-object extended conditions (order + stochastic + hidden_moderator)",
        "num_objects": NUM_OBJECTS,
        "base_seed": BASE_SEED,
        "cost_summary": {
            "total_cost_usd": total_cost,
            "episodes_run": len(merged_results),
            "avg_cost_per_episode": total_cost / len(merged_results) if merged_results else 0,
            "remaining_budget_usd": BUDGET_CAP - total_cost,
            "episode_costs": merged_episode_costs,
        },
        "results": merged_results,
    }

    # Save merged results
    with open(results_path, "w") as f:
        json.dump(merged_data, f, indent=2)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"RESUME RUN COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"New episodes: {completed}")
    logger.info(f"Skipped (already done): {skipped}")
    logger.info(f"Total episodes now: {len(merged_results)}")
    logger.info(f"Session cost: ${cost_tracker.total_cost:.2f}")
    logger.info(f"Cumulative cost: ${total_cost:.2f}")
    logger.info(f"{'='*60}")

    # Print per-condition/agent breakdown
    from collections import Counter
    counts = Counter()
    correct_counts = Counter()
    for r in merged_results:
        key = f"{r['condition']}/{r['agent_type']}"
        counts[key] += 1
        if r["success"]:
            correct_counts[key] += 1

    for condition in ["order_sensitive", "stochastic", "hidden_moderator"]:
        for agent in ["base_llm", "cg_only", "full_cg_db"]:
            key = f"{condition}/{agent}"
            total = counts.get(key, 0)
            correct = correct_counts.get(key, 0)
            if total > 0:
                logger.info(f"  {key}: {correct}/{total} ({100*correct/total:.1f}%)")
            else:
                logger.info(f"  {key}: 0/0 (not run)")

    logger.info(f"\nResults saved: {results_path}")

    return new_results


if __name__ == "__main__":
    asyncio.run(run_resume())
