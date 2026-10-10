"""Phase 3: Standard conditions — conjunctive + disjunctive × 3 agents.

Run 2: 2 conditions × 3 agents × 5 episodes = 30 episodes (smoke test).
"""

from __future__ import annotations

RUN_NAME = "run2"

import asyncio
import json
import logging
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.blicket_env import BlicketEnv, CausalRule, CausalRuleType
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


def create_agent(
    agent_type: AgentType, client: anthropic.AsyncAnthropic, config: ExperimentConfig
) -> BaseLLMAgent:
    if agent_type == AgentType.BASE_LLM:
        return BaseLLMAgent(llm_client=client, model=config.action_model)
    elif agent_type == AgentType.CG_ONLY:
        return CGAgent(llm_client=client, model=config.action_model)
    elif agent_type == AgentType.FULL_CG_DB:
        return FullAgent(
            llm_client=client,
            model=config.action_model,
            trigger_model=config.trigger_model,
        )
    raise ValueError(f"Unknown agent type: {agent_type}")


def create_standard_env(
    condition: str, config: ExperimentConfig
) -> BlicketEnv:
    if condition == "conjunctive":
        rule = CausalRule(
            rule_type=CausalRuleType.CONJUNCTIVE,
            required_objects=set(config.conjunctive_blickets),
        )
    elif condition == "disjunctive":
        rule = CausalRule(
            rule_type=CausalRuleType.DISJUNCTIVE,
            required_objects=set(config.disjunctive_blickets),
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return BlicketEnv(causal_rule=rule, max_steps=config.max_steps_per_episode)


async def run_standard_experiments(config: ExperimentConfig | None = None) -> list[EpisodeResult]:
    """Run all standard condition experiments."""
    if config is None:
        config = ExperimentConfig(
            episodes_per_condition=5,
            trace_dir=Path(f"data/traces_{RUN_NAME}"),
            results_dir=Path(f"data/results_{RUN_NAME}"),
        )

    client = anthropic.AsyncAnthropic()
    cost_tracker = CostTracker(max_total=config.max_total_cost)
    results: list[EpisodeResult] = []

    conditions = ["conjunctive", "disjunctive"]
    agents = [AgentType.BASE_LLM, AgentType.CG_ONLY, AgentType.FULL_CG_DB]

    total_episodes = len(conditions) * len(agents) * config.episodes_per_condition
    completed = 0

    for condition in conditions:
        for agent_type in agents:
            for ep in range(config.episodes_per_condition):
                if not cost_tracker.check_budget():
                    logger.warning("Budget exhausted. Stopping.")
                    break

                logger.info(
                    f"[{completed + 1}/{total_episodes}] "
                    f"{condition}/{agent_type.value}/ep{ep}"
                )

                env = create_standard_env(condition, config)
                agent = create_agent(agent_type, client, config)

                result = await run_episode(
                    env=env,
                    agent=agent,
                    condition=condition,
                    episode_number=ep,
                    trace_dir=config.trace_dir / "standard",
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
    summary_path = config.results_dir / "standard_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
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

    logger.info(f"Standard experiments complete. Summary: {summary_path}")
    logger.info(f"Total cost: ${cost_tracker.total_cost:.2f}")

    return results


if __name__ == "__main__":
    asyncio.run(run_standard_experiments())
