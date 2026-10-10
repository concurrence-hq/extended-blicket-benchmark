"""Run 3 Phase B: Extended conditions — order + stochastic + moderator × 3 agents (5-object).

3 conditions × 3 agents × 30 episodes = 270 episodes total.
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
BASE_SEED = 137  # New seed set — different problem space, no seed-level comparison
BUDGET_CAP = 200.0


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
        # 5-object: switches from conjunctive {A,B} to disjunctive {C}
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


async def run_extended_5obj_experiments(config: ExperimentConfig | None = None) -> list[EpisodeResult]:
    """Run all 5-object extended condition experiments."""
    if config is None:
        config = ExperimentConfig(
            episodes_per_condition=30,
            max_steps_per_episode=MAX_STEPS,
            trace_dir=Path(f"data/traces_{RUN_NAME}"),
            results_dir=Path(f"data/results_{RUN_NAME}"),
            base_seed=BASE_SEED,
            max_total_cost=BUDGET_CAP,
        )

    client = anthropic.AsyncAnthropic()
    cost_tracker = CostTracker(max_total=config.max_total_cost)
    results: list[EpisodeResult] = []

    conditions = ["order_sensitive", "stochastic", "hidden_moderator"]
    agents = [AgentType.BASE_LLM, AgentType.CG_ONLY, AgentType.FULL_CG_DB]

    total_episodes = len(conditions) * len(agents) * config.episodes_per_condition
    completed = 0

    for condition in conditions:
        for agent_type in agents:
            for ep in range(config.episodes_per_condition):
                if not cost_tracker.check_budget():
                    logger.warning(f"Budget exhausted (${cost_tracker.total_cost:.2f}/{BUDGET_CAP}). Stopping.")
                    break

                logger.info(
                    f"[{completed + 1}/{total_episodes}] "
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

                results.append(result)
                completed += 1

                logger.info(
                    f"  Result: {'CORRECT' if result.success else 'WRONG'} | "
                    f"Steps: {result.steps_taken} | Cost: ${episode_cost:.3f} | "
                    f"Total: ${cost_tracker.total_cost:.2f}"
                )

    # Save summary
    summary_path = config.results_dir / "extended_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": RUN_NAME,
        "description": "5-object extended conditions (order + stochastic + hidden_moderator)",
        "num_objects": NUM_OBJECTS,
        "base_seed": BASE_SEED,
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
    logger.info(f"5-Object Extended Results Summary")
    logger.info(f"{'='*60}")
    for condition in conditions:
        for agent_type in agents:
            cond_results = [
                r for r in results
                if r.condition == condition and r.agent_type == agent_type.value
            ]
            if cond_results:
                correct = sum(1 for r in cond_results if r.success)
                total = len(cond_results)
                logger.info(f"  {condition}/{agent_type.value}: {correct}/{total} ({100*correct/total:.1f}%)")
    logger.info(f"Total cost: ${cost_tracker.total_cost:.2f}")
    logger.info(f"Summary saved: {summary_path}")

    return results


if __name__ == "__main__":
    asyncio.run(run_extended_5obj_experiments())
