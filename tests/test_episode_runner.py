"""Episode runner integration tests (audit item 5).

Tests end-to-end episode execution with scripted mock agents
and real environments. Validates:
- Correct/incorrect answer reporting
- Timeout (step budget exhaustion)
- Trace file completeness
- Answer checking for hidden moderator pre/post switch
"""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.blicket_env import Action, BlicketEnv, CausalRule, CausalRuleType
from environment.blicket_env_5obj import Action5, BlicketEnv5
from environment.extended_env import ExtendedCondition, HiddenModeratorConfig
from environment.extended_env_5obj import ExtendedBlicketEnv5
from agents.base_llm_agent import BaseLLMAgent
from experiments.episode_runner import run_episode


class ScriptedAgent(BaseLLMAgent):
    """A mock agent that follows a scripted sequence of actions and a final answer.

    Does not call any LLM — all responses are predetermined.
    """

    def __init__(self, actions: list[Action], answer_text: str, text_interface=None):
        # Create a dummy client that will never be called
        mock_client = MagicMock()
        super().__init__(llm_client=mock_client, text_interface=text_interface)
        self._scripted_actions = list(actions)
        self._action_index = 0
        self._answer_text = answer_text

    def reset(self):
        super().reset()
        self._action_index = 0

    async def get_action(self, env):
        if self._action_index >= len(self._scripted_actions):
            return None
        action = self._scripted_actions[self._action_index]
        self._action_index += 1
        return action

    async def get_answer(self):
        parsed = self.text_interface.parse_answer(self._answer_text)
        # Normalize: TextInterface returns 2-tuple, TextInterface5 returns 3-tuple
        if len(parsed) == 2:
            rule_type, blickets = parsed
            return rule_type, blickets, None
        return parsed


# ── A. Minimal end-to-end tests ─────────────────────────────────────────


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_correct_answer_reports_success(self):
        """Agent places A and B, submits correct conjunctive answer."""
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env = BlicketEnv(causal_rule=rule, max_steps=50)

        actions = [Action.PLACE_A, Action.PLACE_B, Action.CHECK]
        answer = "RULE_TYPE: conjunctive\nBLICKETS: A, B"
        agent = ScriptedAgent(actions=actions, answer_text=answer)

        with tempfile.TemporaryDirectory() as tmp:
            result = await run_episode(
                env=env, agent=agent, condition="test",
                episode_number=0, trace_dir=Path(tmp),
            )

        assert result.success is True
        assert result.answer_submitted is True
        assert result.rule_type_correct is True
        assert result.objects_correct is True

    @pytest.mark.asyncio
    async def test_incorrect_answer_reports_failure(self):
        """Agent submits wrong rule type."""
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env = BlicketEnv(causal_rule=rule, max_steps=50)

        actions = [Action.PLACE_A, Action.CHECK]
        answer = "RULE_TYPE: disjunctive\nBLICKETS: A, B"
        agent = ScriptedAgent(actions=actions, answer_text=answer)

        with tempfile.TemporaryDirectory() as tmp:
            result = await run_episode(
                env=env, agent=agent, condition="test",
                episode_number=0, trace_dir=Path(tmp),
            )

        assert result.success is False
        assert result.answer_submitted is True
        assert result.rule_type_correct is False
        assert result.objects_correct is True

    @pytest.mark.asyncio
    async def test_timeout_reports_failure(self):
        """Agent exhausts step budget without answering."""
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env = BlicketEnv(causal_rule=rule, max_steps=3)

        # 3 actions with no CHECK — will hit max_steps
        actions = [Action.PLACE_A, Action.PLACE_B, Action.REMOVE_A]
        answer = "RULE_TYPE: conjunctive\nBLICKETS: A, B"
        agent = ScriptedAgent(actions=actions, answer_text=answer)

        with tempfile.TemporaryDirectory() as tmp:
            result = await run_episode(
                env=env, agent=agent, condition="test",
                episode_number=0, trace_dir=Path(tmp),
            )

        assert result.success is False
        assert result.answer_submitted is False
        assert result.steps_taken == 3


# ── B. Trace completeness ───────────────────────────────────────────────


class TestTraceCompleteness:
    @pytest.mark.asyncio
    async def test_trace_file_written(self):
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env = BlicketEnv(causal_rule=rule, max_steps=50)

        actions = [Action.PLACE_A, Action.CHECK]
        answer = "RULE_TYPE: conjunctive\nBLICKETS: A, B"
        agent = ScriptedAgent(actions=actions, answer_text=answer)

        with tempfile.TemporaryDirectory() as tmp:
            result = await run_episode(
                env=env, agent=agent, condition="test",
                episode_number=0, trace_dir=Path(tmp),
            )
            trace_path = Path(result.trace_path)
            assert trace_path.exists()

    @pytest.mark.asyncio
    async def test_trace_has_required_keys(self):
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env = BlicketEnv(causal_rule=rule, max_steps=50)

        actions = [Action.PLACE_A, Action.CHECK]
        answer = "RULE_TYPE: conjunctive\nBLICKETS: A, B"
        agent = ScriptedAgent(actions=actions, answer_text=answer)

        with tempfile.TemporaryDirectory() as tmp:
            result = await run_episode(
                env=env, agent=agent, condition="test",
                episode_number=0, trace_dir=Path(tmp),
            )
            with open(result.trace_path) as f:
                trace = json.load(f)

        assert "result" in trace
        assert "steps" in trace
        assert "agent_trace" in trace
        assert "env_info" in trace


# ── C. Answer checking flow (hidden moderator) ──────────────────────────


class TestAnswerChecking:
    @pytest.mark.asyncio
    async def test_post_switch_evaluates_against_new_rule(self):
        """After hidden moderator switches, check_answer uses the NEW rule."""
        from environment.text_interface import TextInterface5

        original_rule = CausalRule(CausalRuleType.DISJUNCTIVE, {"A"})
        alternate_rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"B", "C"})
        moderator_config = HiddenModeratorConfig(
            switch_after_n=1, alternate_rule=alternate_rule,
        )
        env = ExtendedBlicketEnv5(
            causal_rule=original_rule,
            condition=ExtendedCondition.HIDDEN_MODERATOR,
            moderator_config=moderator_config,
            max_steps=50,
        )
        env.reset()

        # Trigger the switch: place A (disjunctive with {A}) -> detector activates
        env.step(Action5.PLACE_A)  # Activates -> successful_activations = 1 -> switch!

        assert env._rule_switched is True

        # Now check_answer should evaluate against the NEW rule (conjunctive {B, C})
        result = env.check_answer(
            rule_type=CausalRuleType.CONJUNCTIVE,
            required_objects={"B", "C"},
        )
        assert result["correct"] is True
        assert result["rule_type_correct"] is True

    @pytest.mark.asyncio
    async def test_pre_switch_evaluates_against_original_rule(self):
        """Before hidden moderator switches, check_answer uses the ORIGINAL rule."""
        original_rule = CausalRule(CausalRuleType.DISJUNCTIVE, {"A"})
        alternate_rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"B", "C"})
        moderator_config = HiddenModeratorConfig(
            switch_after_n=5, alternate_rule=alternate_rule,  # won't switch in 1 step
        )
        env = ExtendedBlicketEnv5(
            causal_rule=original_rule,
            condition=ExtendedCondition.HIDDEN_MODERATOR,
            moderator_config=moderator_config,
            max_steps=50,
        )
        env.reset()

        assert env._rule_switched is False

        # check_answer should evaluate against the ORIGINAL rule (disjunctive {A})
        result = env.check_answer(
            rule_type=CausalRuleType.DISJUNCTIVE,
            required_objects={"A"},
        )
        assert result["correct"] is True

    @pytest.mark.asyncio
    async def test_correct_blickets_wrong_rule_type_is_incorrect(self):
        """Right objects but wrong rule type should be marked incorrect."""
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env = BlicketEnv(causal_rule=rule, max_steps=50)
        env.reset()

        result = env.check_answer(
            rule_type=CausalRuleType.DISJUNCTIVE,
            required_objects={"A", "B"},
        )
        assert result["correct"] is False
        assert result["rule_type_correct"] is False
        assert result["objects_correct"] is True
