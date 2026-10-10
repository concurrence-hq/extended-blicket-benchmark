"""Run 3 Phase A: Competitive inhibition validation on 3-object space.

Tests the new competitive inhibition + raised thresholds (7.0) on
full_cg_db agent only, 3 conditions × 10 episodes = 30 episodes.
"""

from __future__ import annotations

RUN_NAME = "run3_phaseA"

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
    ExtendedBlicketEnv,
    ExtendedCondition,
    HiddenModeratorConfig,
    OrderConfig,
    StochasticConfig,
)
from agents.full_agent import FullAgent
from experiments.config import CostTracker, ExperimentConfig
from experiments.episode_runner import EpisodeResult, run_episode

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

EPISODES_PER_CONDITION = 10


def create_extended_env(
    condition: str, config: ExperimentConfig, seed: int
) -> ExtendedBlicketEnv:
    base_rule = CausalRule(
        rule_type=CausalRuleType.CONJUNCTIVE,
        required_objects=set(config.conjunctive_blickets),
    )

    if condition == "order_sensitive":
        return ExtendedBlicketEnv(
            causal_rule=base_rule,
            condition=ExtendedCondition.ORDER_SENSITIVE,
            max_steps=config.max_steps_per_episode,
            order_config=OrderConfig(required_order=config.order_required_order),
            rng_seed=seed,
        )
    elif condition == "stochastic":
        return ExtendedBlicketEnv(
            causal_rule=base_rule,
            condition=ExtendedCondition.STOCHASTIC,
            max_steps=config.max_steps_per_episode,
            stochastic_config=StochasticConfig(
                activation_probability=config.stochastic_probability
            ),
            rng_seed=seed,
        )
    elif condition == "hidden_moderator":
        alternate_rule = CausalRule(
            rule_type=CausalRuleType.DISJUNCTIVE,
            required_objects={"C"},
        )
        return ExtendedBlicketEnv(
            causal_rule=base_rule,
            condition=ExtendedCondition.HIDDEN_MODERATOR,
            max_steps=config.max_steps_per_episode,
            moderator_config=HiddenModeratorConfig(
                switch_after_n=config.moderator_switch_after,
                alternate_rule=alternate_rule,
            ),
            rng_seed=seed,
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")


async def run_phaseA_experiments() -> list[EpisodeResult]:
    """Run Phase A: competitive inhibition validation (full_cg_db only)."""
    config = ExperimentConfig(
        episodes_per_condition=EPISODES_PER_CONDITION,
        trace_dir=Path(f"data/traces_{RUN_NAME}"),
        results_dir=Path(f"data/results_{RUN_NAME}"),
        base_seed=42,  # Same seeds as Run 2 (first 10)
    )

    client = anthropic.AsyncAnthropic()
    cost_tracker = CostTracker(max_total=config.max_total_cost)
    results: list[EpisodeResult] = []

    conditions = ["order_sensitive", "stochastic", "hidden_moderator"]
    total_episodes = len(conditions) * EPISODES_PER_CONDITION
    completed = 0

    for condition in conditions:
        for ep in range(EPISODES_PER_CONDITION):
            if not cost_tracker.check_budget():
                logger.warning("Budget exhausted. Stopping.")
                break

            logger.info(
                f"[{completed + 1}/{total_episodes}] "
                f"{condition}/full_cg_db/ep{ep}"
            )

            seed = config.get_episode_seed(condition, "full_cg_db", ep)
            env = create_extended_env(condition, config, seed)
            agent = FullAgent(
                llm_client=client,
                model=config.action_model,
                trigger_model=config.trigger_model,
            )

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
                agent="full_cg_db",
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

    # Save summary
    summary_path = config.results_dir / "phaseA_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": RUN_NAME,
        "description": "Phase A: competitive inhibition validation (full_cg_db only, 3-object)",
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
            }
            for r in results
        ],
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase A Results Summary")
    logger.info(f"{'='*60}")
    for condition in conditions:
        cond_results = [r for r in results if r.condition == condition]
        correct = sum(1 for r in cond_results if r.success)
        total = len(cond_results)
        logger.info(f"  {condition}: {correct}/{total} ({100*correct/total:.1f}%)")
    logger.info(f"Total cost: ${cost_tracker.total_cost:.2f}")
    logger.info(f"Summary saved: {summary_path}")

    return results


if __name__ == "__main__":
    asyncio.run(run_phaseA_experiments())
