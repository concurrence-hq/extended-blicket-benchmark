"""Agent integration tests (audit item 4).

Tests for BaseLLMAgent, CGAgent, and FullAgent covering:
- System prompt construction
- CG transition processing and desync fix
- DB evaluation ordering and activation logging
- Ablation hierarchy integrity
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.blicket_env import BlicketEnv, CausalRule, CausalRuleType
from environment.text_interface import TextInterface
from agents.base_llm_agent import BaseLLMAgent
from agents.cg_agent import CGAgent
from agents.full_agent import FullAgent
from agents.context_graph import create_blicket_cg


def _make_mock_client(response_text: str = "REASONING: test\nACTION: 1"):
    """Create a mock Anthropic client that returns a fixed response."""
    mock_response = SimpleNamespace(
        content=[SimpleNamespace(text=response_text)],
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        model="claude-sonnet-4-5-20250929",
        id="msg_test_123",
    )
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=mock_response)
    return client


def _make_env():
    """Create a simple conjunctive environment for testing."""
    rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
    return BlicketEnv(causal_rule=rule, max_steps=50)


# ── A. BaseLLMAgent basics ──────────────────────────────────────────────


class TestBaseLLMAgent:
    def test_constructs_without_error(self):
        client = _make_mock_client()
        agent = BaseLLMAgent(llm_client=client)
        assert agent is not None

    def test_system_prompt_is_task_description_only(self):
        client = _make_mock_client()
        agent = BaseLLMAgent(llm_client=client)
        prompt = agent._build_system_prompt()
        # Should contain task description
        assert "blicket detector" in prompt
        # Should NOT contain CG content
        assert "Context Graph" not in prompt
        assert "CG_STATE" not in prompt
        assert "TRANSITION" not in prompt

    @pytest.mark.asyncio
    async def test_get_action_returns_parsed_action(self):
        client = _make_mock_client("REASONING: placing A\nACTION: 1")
        agent = BaseLLMAgent(llm_client=client)
        env = _make_env()
        env.reset()
        action = await agent.get_action(env)
        assert action is not None
        # Action 1 from valid_actions should be PLACE_A
        from environment.blicket_env import Action
        assert action == Action.PLACE_A

    def test_agent_type_in_trace(self):
        client = _make_mock_client()
        agent = BaseLLMAgent(llm_client=client)
        trace = agent.get_trace_data()
        assert trace["agent_type"] == "base_llm"


# ── B. CGAgent CG integration ───────────────────────────────────────────


class TestCGAgent:
    def test_system_prompt_includes_cg_context(self):
        client = _make_mock_client()
        agent = CGAgent(llm_client=client)
        prompt = agent._build_system_prompt()
        assert "CG_STATE" in prompt
        assert "TRANSITION" in prompt
        # Should include CG rendered content
        assert "INITIAL_EXPLORATION" in prompt

    @pytest.mark.asyncio
    async def test_valid_transition_updates_cg(self):
        response = (
            "CG_STATE: INITIAL_EXPLORATION\n"
            "TRANSITION: COMBINATION_TESTING\n"
            "REASONING: ready to test combos\n"
            "ACTION: 1"
        )
        client = _make_mock_client(response)
        agent = CGAgent(llm_client=client)
        env = _make_env()
        env.reset()

        assert agent.context_graph.current_state == "INITIAL_EXPLORATION"
        await agent.get_action(env)
        assert agent.context_graph.current_state == "COMBINATION_TESTING"

    @pytest.mark.asyncio
    async def test_invalid_transition_does_not_update_cg(self):
        response = (
            "CG_STATE: INITIAL_EXPLORATION\n"
            "TRANSITION: NONEXISTENT_STATE\n"
            "REASONING: trying invalid\n"
            "ACTION: 1"
        )
        client = _make_mock_client(response)
        agent = CGAgent(llm_client=client)
        env = _make_env()
        env.reset()

        assert agent.context_graph.current_state == "INITIAL_EXPLORATION"
        await agent.get_action(env)
        # Should remain in original state
        assert agent.context_graph.current_state == "INITIAL_EXPLORATION"

    @pytest.mark.asyncio
    async def test_desync_correction_fires_on_invalid_transition(self):
        """When a transition is invalid, _pending_desync_correction is set
        and injected as a user message on the next get_action() call."""
        response_1 = (
            "CG_STATE: INITIAL_EXPLORATION\n"
            "TRANSITION: NONEXISTENT_STATE\n"
            "REASONING: trying invalid\n"
            "ACTION: 1"
        )
        response_2 = (
            "CG_STATE: INITIAL_EXPLORATION\n"
            "REASONING: ok staying here\n"
            "ACTION: 2"
        )
        client = _make_mock_client(response_1)
        agent = CGAgent(llm_client=client)
        env = _make_env()
        env.reset()

        # First call: invalid transition sets _pending_desync_correction
        await agent.get_action(env)
        assert agent._pending_desync_correction is not None
        assert "NONEXISTENT_STATE" in agent._pending_desync_correction
        assert "not accepted" in agent._pending_desync_correction

        # Second call: desync correction is injected into conversation
        client.messages.create = AsyncMock(
            return_value=SimpleNamespace(
                content=[SimpleNamespace(text=response_2)],
                usage=SimpleNamespace(input_tokens=100, output_tokens=50),
                model="claude-sonnet-4-5-20250929",
                id="msg_test_456",
            )
        )
        await agent.get_action(env)
        # Correction should have been consumed
        assert agent._pending_desync_correction is None
        # Check it was injected into conversation
        desync_messages = [
            m for m in agent.conversation
            if m["role"] == "user" and "not accepted" in m.get("content", "")
        ]
        assert len(desync_messages) == 1

    def test_agent_type_in_trace(self):
        client = _make_mock_client()
        agent = CGAgent(llm_client=client)
        trace = agent.get_trace_data()
        assert trace["agent_type"] == "cg_only"


# ── C. FullAgent DB integration ─────────────────────────────────────────


class TestFullAgent:
    def test_db_engine_initialized_with_behaviors(self):
        client = _make_mock_client()
        agent = FullAgent(llm_client=client)
        assert agent.db_engine is not None
        assert len(agent.db_engine.behaviors) > 0

    @pytest.mark.asyncio
    async def test_db_evaluation_before_llm_call(self):
        """DB evaluation happens BEFORE the LLM action call.

        When a DB fires, its notification appears in the conversation
        history that the LLM sees for action selection.
        """
        response = "REASONING: test\nACTION: 1"
        client = _make_mock_client(response)
        agent = FullAgent(llm_client=client)
        env = _make_env()
        env.reset()

        # Mock db_engine.evaluate_and_apply to return a notification
        agent.db_engine.evaluate_and_apply = AsyncMock(
            return_value=["[DB] exploration_stagnation fired: try new approach"]
        )
        agent.db_engine._fired = {"exploration_stagnation"}

        await agent.get_action(env)

        # The notification should be in conversation BEFORE the state text
        # Find the notification message and the state message
        notification_idx = None
        state_idx = None
        for i, msg in enumerate(agent.conversation):
            if msg["role"] == "user" and "[DB]" in msg.get("content", ""):
                notification_idx = i
            if msg["role"] == "user" and "Blicket Detector Status" in msg.get("content", ""):
                state_idx = i

        assert notification_idx is not None, "DB notification not found in conversation"
        assert state_idx is not None, "State text not found in conversation"
        assert notification_idx < state_idx, "DB notification must appear before state text"

    @pytest.mark.asyncio
    async def test_db_activation_log_records_events(self):
        response = "REASONING: test\nACTION: 1"
        client = _make_mock_client(response)
        agent = FullAgent(llm_client=client)
        env = _make_env()
        env.reset()

        agent.db_engine.evaluate_and_apply = AsyncMock(
            return_value=["[DB] stagnation fired"]
        )
        agent.db_engine._fired = {"exploration_stagnation"}

        await agent.get_action(env)

        assert len(agent._db_activation_log) == 1
        log_entry = agent._db_activation_log[0]
        assert log_entry["step"] == 0
        assert "[DB]" in log_entry["notification"]

    def test_db_fired_behaviors_in_trace(self):
        client = _make_mock_client()
        agent = FullAgent(llm_client=client)
        # Simulate a fired behavior
        agent.db_engine._fired.add("exploration_stagnation")
        trace = agent.get_trace_data()
        assert "exploration_stagnation" in trace["db_fired_behaviors"]

    def test_agent_type_in_trace(self):
        client = _make_mock_client()
        agent = FullAgent(llm_client=client)
        trace = agent.get_trace_data()
        assert trace["agent_type"] == "full_cg_db"


# ── D. Ablation hierarchy integrity ─────────────────────────────────────


class TestAblationHierarchy:
    def test_base_llm_has_no_cg_or_db(self):
        client = _make_mock_client()
        agent = BaseLLMAgent(llm_client=client)
        assert not hasattr(agent, "context_graph")
        assert not hasattr(agent, "db_engine")

    def test_cg_agent_has_cg_but_no_db(self):
        client = _make_mock_client()
        agent = CGAgent(llm_client=client)
        assert hasattr(agent, "context_graph")
        assert agent.context_graph is not None
        assert not hasattr(agent, "db_engine")

    def test_full_agent_has_cg_and_db(self):
        client = _make_mock_client()
        agent = FullAgent(llm_client=client)
        assert hasattr(agent, "context_graph")
        assert agent.context_graph is not None
        assert hasattr(agent, "db_engine")
        assert agent.db_engine is not None
