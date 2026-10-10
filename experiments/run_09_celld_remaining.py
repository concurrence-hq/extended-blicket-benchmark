"""Run 09 Cell D Completion: Finish remaining episodes after budget cap.

Reads existing traces to find which Cell D episodes are already done,
then runs only the missing ones. Merges results into the existing
order_complexity_results.json.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.blicket_env import CausalRule, CausalRuleType
from environment.extended_env import ExtendedCondition, OrderConfig
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

RUN_NAME = "run9"
NUM_OBJECTS = 5
BASE_SEED = 901
BUDGET_CAP = 30.0  # Enough for ~5 episodes at ~$3 each

CELL_D_LABEL = "D"
CELL_D_AGENT = AgentType.FULL_CG_DB
CELL_D_OBJECTS = {"A", "D", "C", "B"}
CELL_D_ORDER = ["A", "D", "C", "B"]
CELL_D_MAX_STEPS = 100
EPISODES_PER_CELL = 15


def find_completed_episodes(trace_dir: Path) -> set[int]:
    """Find which Cell D episodes already have trace files."""
    completed = set()
    for f in trace_dir.glob("4obj_order_D_full_cg_db_ep*.json"):
        ep_str = f.stem.split("_ep")[-1]
        try:
            completed.add(int(ep_str))
        except ValueError:
            pass
    return completed


def create_agent(
    client: anthropic.AsyncAnthropic, config: ExperimentConfig
) -> FullAgent:
    ti = TextInterface5(include_order_sensitive=True)
    return FullAgent(
        llm_client=client,
        model=config.action_model,
        trigger_model=config.trigger_model,
        text_interface=ti,
        num_objects=NUM_OBJECTS,
    )


def create_env(seed: int) -> ExtendedBlicketEnv5:
    rule = CausalRule(
        rule_type=CausalRuleType.ORDER_SENSITIVE,
        required_objects=CELL_D_OBJECTS,
    )
    return ExtendedBlicketEnv5(
        causal_rule=rule,
        condition=ExtendedCondition.ORDER_SENSITIVE,
        max_steps=CELL_D_MAX_STEPS,
        order_config=OrderConfig(required_order=CELL_D_ORDER),
        rng_seed=seed,
    )


async def run_remaining_celld():
    config = ExperimentConfig(
        episodes_per_condition=EPISODES_PER_CELL,
        max_steps_per_episode=CELL_D_MAX_STEPS,
        trace_dir=Path(f"data/traces_{RUN_NAME}"),
        results_dir=Path(f"data/results_{RUN_NAME}"),
        base_seed=BASE_SEED,
        max_total_cost=BUDGET_CAP,
    )

    completed = find_completed_episodes(config.trace_dir)
    remaining = sorted(set(range(EPISODES_PER_CELL)) - completed)

    if not remaining:
        logger.info("All 15 Cell D episodes already complete. Nothing to do.")
        return []

    logger.info(f"Cell D: {len(completed)}/15 done. Running remaining: {remaining}")

    client = anthropic.AsyncAnthropic()
    cost_tracker = CostTracker(max_total=BUDGET_CAP)
    results: list[EpisodeResult] = []

    condition_name = "4obj_order_D"

    for ep in remaining:
        if not cost_tracker.check_budget():
            logger.warning(
                f"Budget exhausted (${cost_tracker.total_cost:.2f}/{BUDGET_CAP}). Stopping."
            )
            break

        logger.info(
            f"[{len(results) + 1}/{len(remaining)}] "
            f"Cell D: full_cg_db / 4-obj order / ep{ep}"
        )

        seed = config.get_episode_seed(condition_name, CELL_D_AGENT.value, ep)
        env = create_env(seed)
        agent = create_agent(client, config)

        result = await run_episode(
            env=env,
            agent=agent,
            condition=condition_name,
            episode_number=ep,
            trace_dir=config.trace_dir,
            max_cost_per_episode=config.max_cost_per_episode,
        )

        episode_cost = cost_tracker.add_episode_cost(
            condition=condition_name,
            agent=CELL_D_AGENT.value,
            episode=ep,
            input_tokens=result.total_input_tokens,
            output_tokens=result.total_output_tokens,
            trigger_tokens=result.trigger_tokens,
        )

        results.append(result)

        order_str = ""
        if result.order_correct is not None:
            order_str = f" | Order: {'CORRECT' if result.order_correct else 'WRONG'}"

        logger.info(
            f"  Result: {'CORRECT' if result.success else 'WRONG'}{order_str} | "
            f"Steps: {result.steps_taken} | Cost: ${episode_cost:.3f} | "
            f"Total: ${cost_tracker.total_cost:.2f}"
        )

    # Merge into existing results JSON
    main_results_path = config.results_dir / "order_complexity_results.json"
    if main_results_path.exists():
        with open(main_results_path) as f:
            summary = json.load(f)

        # Add new results
        for r in results:
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
            })

        # Update cost summary
        old_cost = summary.get("cost_summary", {}).get("total_cost", 0)
        summary["cost_summary"]["total_cost"] = old_cost + cost_tracker.total_cost
        summary["cost_summary"]["note_celld_extension"] = (
            f"Extended budget by ${cost_tracker.total_cost:.2f} to complete Cell D"
        )

        with open(main_results_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Merged {len(results)} new results into {main_results_path}")
    else:
        logger.warning(f"Main results file not found at {main_results_path}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Cell D completion: {len(results)} episodes run")
    correct = sum(1 for r in results if r.success)
    logger.info(f"Results: {correct}/{len(results)} correct")
    logger.info(f"Cost: ${cost_tracker.total_cost:.2f}")
    logger.info(f"{'='*60}")

    return results


if __name__ == "__main__":
    asyncio.run(run_remaining_celld())
