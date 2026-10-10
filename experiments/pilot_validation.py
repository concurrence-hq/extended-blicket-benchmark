"""Pilot validation: 5 episodes per agent × 2 standard conditions × 3 agents = 30 total.

Checks:
- Parse success rate >95%
- Costs within estimates (~$0.33/episode)
- DB fires at reasonable times (not too early, not never)
- Answer submission rate
- Basic correctness patterns
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.blicket_env import BlicketEnv, CausalRule, CausalRuleType
from agents.base_llm_agent import BaseLLMAgent
from agents.cg_agent import CGAgent
from agents.full_agent import FullAgent
from experiments.config import (
    AgentType,
    CostTracker,
    ExperimentConfig,
    SONNET_INPUT_COST_PER_TOKEN,
    SONNET_OUTPUT_COST_PER_TOKEN,
    HAIKU_INPUT_COST_PER_TOKEN,
)
from experiments.episode_runner import EpisodeResult, run_episode

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

EPISODES_PER_CELL = 5


def create_agent(agent_type: AgentType, client, config):
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


async def run_pilot_validation():
    config = ExperimentConfig(
        episodes_per_condition=EPISODES_PER_CELL,
        max_steps_per_episode=50,
    )
    client = anthropic.AsyncAnthropic()
    cost_tracker = CostTracker(max_total=config.max_total_cost)
    trace_dir = config.trace_dir / "pilot_validation"

    conditions = {
        "conjunctive": CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"}),
        "disjunctive": CausalRule(CausalRuleType.DISJUNCTIVE, {"A", "B"}),
    }
    agents = [AgentType.BASE_LLM, AgentType.CG_ONLY, AgentType.FULL_CG_DB]

    total = len(conditions) * len(agents) * EPISODES_PER_CELL
    results: list[EpisodeResult] = []
    i = 0
    start = time.time()

    for cond_name, rule in conditions.items():
        for agent_type in agents:
            for ep in range(EPISODES_PER_CELL):
                i += 1
                logger.info(f"[{i}/{total}] {cond_name}/{agent_type.value}/ep{ep}")

                env = BlicketEnv(causal_rule=rule, max_steps=config.max_steps_per_episode)
                agent = create_agent(agent_type, client, config)

                result = await run_episode(
                    env=env,
                    agent=agent,
                    condition=cond_name,
                    episode_number=ep,
                    trace_dir=trace_dir,
                )

                ep_cost = cost_tracker.add_episode_cost(
                    condition=cond_name,
                    agent=agent_type.value,
                    episode=ep,
                    input_tokens=result.total_input_tokens,
                    output_tokens=result.total_output_tokens,
                    trigger_tokens=result.trigger_tokens,
                )

                results.append(result)
                status = "OK" if result.success else ("WRONG" if result.answer_submitted else "NO_ANS")
                logger.info(
                    f"  {status} | Steps:{result.steps_taken} | "
                    f"Cost:${ep_cost:.3f} | Parse fails:{result.parse_failures}"
                )

    wall_time = time.time() - start

    # === VALIDATION REPORT ===
    print("\n" + "=" * 70)
    print("PILOT VALIDATION REPORT")
    print("=" * 70)

    # 1. Completion & Parse Success
    total_parse_failures = sum(r.parse_failures for r in results)
    total_steps = sum(r.steps_taken for r in results)
    parse_success_rate = 1.0 - (total_parse_failures / max(total_steps, 1))
    errors = [r for r in results if r.error]

    print(f"\n1. COMPLETION")
    print(f"   Episodes run:       {len(results)}/{total}")
    print(f"   Errors/crashes:     {len(errors)}")
    print(f"   Parse success rate: {parse_success_rate:.1%} (target: >95%)")
    print(f"   {'PASS' if parse_success_rate > 0.95 and not errors else 'FAIL'}")

    # 2. Answer Submission
    answered = sum(1 for r in results if r.answer_submitted)
    print(f"\n2. ANSWER SUBMISSION")
    print(f"   Submitted answers:  {answered}/{len(results)} ({answered/len(results):.0%})")

    # 3. Correctness by cell
    print(f"\n3. CORRECTNESS")
    print(f"   {'Condition':<15} {'Agent':<12} {'Correct':>7} {'of':>3} {'N':>3} {'Rate':>6}")
    print(f"   {'-'*50}")
    for cond_name in conditions:
        for agent_type in agents:
            cell = [r for r in results if r.condition == cond_name and r.agent_type == agent_type.value]
            correct = sum(1 for r in cell if r.success)
            print(f"   {cond_name:<15} {agent_type.value:<12} {correct:>7} {'of':>3} {len(cell):>3} {correct/len(cell):>5.0%}")

    # 4. Steps
    print(f"\n4. STEPS TAKEN")
    print(f"   {'Condition':<15} {'Agent':<12} {'Mean':>6} {'Min':>4} {'Max':>4}")
    print(f"   {'-'*45}")
    for cond_name in conditions:
        for agent_type in agents:
            cell = [r for r in results if r.condition == cond_name and r.agent_type == agent_type.value]
            steps = [r.steps_taken for r in cell]
            print(f"   {cond_name:<15} {agent_type.value:<12} {sum(steps)/len(steps):>5.1f} {min(steps):>4} {max(steps):>4}")

    # 5. Cost
    print(f"\n5. COST")
    print(f"   Total cost:         ${cost_tracker.total_cost:.2f}")
    print(f"   Avg per episode:    ${cost_tracker.total_cost/len(results):.3f}")
    print(f"   {'Condition':<15} {'Agent':<12} {'Avg Cost':>9}")
    print(f"   {'-'*40}")
    for cond_name in conditions:
        for agent_type in agents:
            cell_costs = [
                c for c in cost_tracker._episode_costs
                if c["condition"] == cond_name and c["agent"] == agent_type.value
            ]
            avg = sum(c["cost_usd"] for c in cell_costs) / len(cell_costs)
            print(f"   {cond_name:<15} {agent_type.value:<12} ${avg:>7.3f}")

    # Projected full-run cost
    avg_cost = cost_tracker.total_cost / len(results)
    projected_standard = avg_cost * 180
    projected_extended = avg_cost * 270 * 1.1  # 10% buffer for extended complexity
    print(f"\n   Projected Phase 3 (180 ep): ${projected_standard:.0f}")
    print(f"   Projected Phase 4 (270 ep): ${projected_extended:.0f}")
    print(f"   Projected total:            ${projected_standard + projected_extended:.0f}")

    # 6. DB activations (Full agent only)
    print(f"\n6. DYNAMIC BEHAVIORS (full_cg_db only)")
    full_results = [r for r in results if r.agent_type == "full_cg_db"]
    if full_results:
        # Load traces to inspect DB logs
        for r in full_results:
            if r.trace_path:
                try:
                    with open(r.trace_path) as f:
                        trace = json.load(f)
                    db_log = trace["agent_trace"].get("db_activation_log", [])
                    fired = trace["agent_trace"].get("db_fired_behaviors", [])
                    print(f"   {r.condition}/ep{r.episode_number}: "
                          f"fired={fired} at steps={[d['step'] for d in db_log]} "
                          f"trigger_tokens={r.trigger_tokens}")
                except Exception:
                    print(f"   {r.condition}/ep{r.episode_number}: (trace read failed)")

    # 7. Timing
    print(f"\n7. TIMING")
    print(f"   Total wall time:    {wall_time:.0f}s ({wall_time/60:.1f}min)")
    print(f"   Avg per episode:    {wall_time/len(results):.1f}s")
    projected_full = (wall_time / len(results)) * 450
    print(f"   Projected full run: {projected_full/60:.0f}min ({projected_full/3600:.1f}hrs)")

    print(f"\n{'='*70}")
    all_pass = parse_success_rate > 0.95 and not errors and answered == len(results)
    print(f"OVERALL: {'PASS' if all_pass else 'REVIEW NEEDED'}")
    print(f"{'='*70}")

    # Save report
    report_path = config.results_dir / "pilot_validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "episodes": len(results),
        "parse_success_rate": parse_success_rate,
        "answer_rate": answered / len(results),
        "total_cost": cost_tracker.total_cost,
        "avg_cost_per_episode": cost_tracker.total_cost / len(results),
        "wall_time_seconds": wall_time,
        "results": [
            {
                "condition": r.condition,
                "agent_type": r.agent_type,
                "episode": r.episode_number,
                "success": r.success,
                "steps": r.steps_taken,
                "parse_failures": r.parse_failures,
                "input_tokens": r.total_input_tokens,
                "output_tokens": r.total_output_tokens,
                "trigger_tokens": r.trigger_tokens,
                "wall_time": r.wall_time_seconds,
                "error": r.error,
            }
            for r in results
        ],
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")

    return all_pass


if __name__ == "__main__":
    success = asyncio.run(run_pilot_validation())
    sys.exit(0 if success else 1)
