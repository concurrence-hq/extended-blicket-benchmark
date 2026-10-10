"""Smoke test: 1 episode per agent per standard condition (6 total).

Verifies end-to-end: API calls, trace saving, parse success, no crashes.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.blicket_env import BlicketEnv, CausalRule, CausalRuleType
from agents.base_llm_agent import BaseLLMAgent
from agents.cg_agent import CGAgent
from agents.full_agent import FullAgent
from experiments.config import ExperimentConfig
from experiments.episode_runner import run_episode

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def smoke_test():
    config = ExperimentConfig(
        episodes_per_condition=1,
        max_steps_per_episode=25,  # Shorter for smoke test
    )
    client = anthropic.AsyncAnthropic()
    trace_dir = config.trace_dir / "smoke_test"

    conditions = {
        "conjunctive": CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"}),
        "disjunctive": CausalRule(CausalRuleType.DISJUNCTIVE, {"A", "B"}),
    }

    agents_spec = [
        ("base_llm", lambda: BaseLLMAgent(llm_client=client, model=config.action_model)),
        ("cg_only", lambda: CGAgent(llm_client=client, model=config.action_model)),
        ("full_cg_db", lambda: FullAgent(
            llm_client=client,
            model=config.action_model,
            trigger_model=config.trigger_model,
        )),
    ]

    results = []
    total = len(conditions) * len(agents_spec)
    i = 0

    for cond_name, rule in conditions.items():
        for agent_name, agent_factory in agents_spec:
            i += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"[{i}/{total}] {cond_name} / {agent_name}")
            logger.info(f"{'='*60}")

            env = BlicketEnv(causal_rule=rule, max_steps=config.max_steps_per_episode)
            agent = agent_factory()

            result = await run_episode(
                env=env,
                agent=agent,
                condition=cond_name,
                episode_number=0,
                trace_dir=trace_dir,
            )

            results.append(result)
            status = "CORRECT" if result.success else ("SUBMITTED WRONG" if result.answer_submitted else "NO ANSWER")
            logger.info(
                f"  {status} | Steps: {result.steps_taken} | "
                f"Tokens: {result.total_input_tokens}+{result.total_output_tokens} | "
                f"Time: {result.wall_time_seconds:.1f}s"
            )
            if result.error:
                logger.error(f"  ERROR: {result.error}")

    # Summary
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    print(f"{'Condition':<15} {'Agent':<12} {'Result':<15} {'Steps':>5} {'Tokens':>8} {'Time':>6}")
    print("-" * 60)
    for r in results:
        status = "CORRECT" if r.success else ("WRONG" if r.answer_submitted else "NO ANS")
        tokens = r.total_input_tokens + r.total_output_tokens
        print(f"{r.condition:<15} {r.agent_type:<12} {status:<15} {r.steps_taken:>5} {tokens:>8} {r.wall_time_seconds:>5.1f}s")

    print("-" * 60)
    passed = sum(1 for r in results if r.error is None)
    print(f"Completed without error: {passed}/{total}")
    answered = sum(1 for r in results if r.answer_submitted)
    print(f"Submitted answers: {answered}/{total}")
    correct = sum(1 for r in results if r.success)
    print(f"Correct answers: {correct}/{total}")

    # Check traces saved
    traces = list(trace_dir.glob("*.json"))
    print(f"Trace files saved: {len(traces)}")

    if passed < total:
        print("\nFAILED - some episodes crashed")
        return False
    print("\nPASSED - all episodes completed")
    return True


if __name__ == "__main__":
    success = asyncio.run(smoke_test())
    sys.exit(0 if success else 1)
