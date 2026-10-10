"""Unit tests for competitive inhibition in Dynamic Behaviors engine.

Tests the Run 3 two-phase evaluation: stagnation (independent) + diagnostic
(competitive winner-take-most with 1.0 proximity window).
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.context_graph import ContextGraph, create_blicket_cg
from agents.dynamic_behaviors import (
    DynamicBehavior,
    DynamicBehaviorEngine,
    TriggerCondition,
    CGModification,
    create_blicket_behaviors,
)


def _make_mock_client():
    """Create a mock LLM client."""
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock()
    return client


def _make_mock_response(score_text: str):
    """Create a mock LLM response with a score."""
    response = MagicMock()
    response.content = [MagicMock(text=score_text)]
    response.usage = MagicMock(input_tokens=100, output_tokens=5)
    return response


def _make_behavior(name: str, stagnation_mode: bool = False, fire_threshold: float = 7.0, min_steps: int = 1) -> DynamicBehavior:
    """Create a minimal test behavior."""
    return DynamicBehavior(
        name=name,
        description=f"Test behavior {name}",
        trigger=TriggerCondition(
            evaluation_prompt=f"Test prompt for {name}",
            min_steps=min_steps,
            stagnation_mode=stagnation_mode,
            stagnation_threshold=2 if stagnation_mode else 6,
            fire_threshold=fire_threshold,
        ),
        modifications=[],
        notification_message=f"[SYSTEM] {name} fired",
    )


class TestCompetitiveInhibition:
    """Test the two-phase competitive inhibition logic."""

    @pytest.mark.asyncio
    async def test_winner_take_most_top_scorer_only(self):
        """Scores 8.0, 6.0, 5.0 → only 8.0 fires (others below threshold or too far)."""
        client = _make_mock_client()
        behaviors = [
            _make_behavior("b1", fire_threshold=7.0),
            _make_behavior("b2", fire_threshold=7.0),
            _make_behavior("b3", fire_threshold=7.0),
        ]
        engine = DynamicBehaviorEngine(behaviors=behaviors, llm_client=client)

        # Return scores: 8.0, 6.0, 5.0
        client.messages.create.side_effect = [
            _make_mock_response("8"),
            _make_mock_response("6"),
            _make_mock_response("5"),
        ]

        cg = create_blicket_cg()
        notifications = await engine.evaluate_and_apply(
            context_graph=cg,
            conversation_history=[{"role": "user", "content": "test"}],
            step_number=10,
        )

        assert len(notifications) == 1
        assert "b1" in notifications[0]
        assert engine.fired_behaviors == {"b1"}

    @pytest.mark.asyncio
    async def test_near_tie_both_fire(self):
        """Scores 8.0, 7.5, 5.0 → both 8.0 and 7.5 fire (within 1.0 proximity)."""
        client = _make_mock_client()
        behaviors = [
            _make_behavior("b1", fire_threshold=7.0),
            _make_behavior("b2", fire_threshold=7.0),
            _make_behavior("b3", fire_threshold=7.0),
        ]
        engine = DynamicBehaviorEngine(behaviors=behaviors, llm_client=client)

        client.messages.create.side_effect = [
            _make_mock_response("8"),
            _make_mock_response("7.5"),
            _make_mock_response("5"),
        ]

        cg = create_blicket_cg()
        notifications = await engine.evaluate_and_apply(
            context_graph=cg,
            conversation_history=[{"role": "user", "content": "test"}],
            step_number=10,
        )

        assert len(notifications) == 2
        assert engine.fired_behaviors == {"b1", "b2"}

    @pytest.mark.asyncio
    async def test_all_below_threshold_none_fire(self):
        """Scores 6.5, 5.0, 4.0 (threshold 7.0) → none fire."""
        client = _make_mock_client()
        behaviors = [
            _make_behavior("b1", fire_threshold=7.0),
            _make_behavior("b2", fire_threshold=7.0),
            _make_behavior("b3", fire_threshold=7.0),
        ]
        engine = DynamicBehaviorEngine(behaviors=behaviors, llm_client=client)

        client.messages.create.side_effect = [
            _make_mock_response("6.5"),
            _make_mock_response("5"),
            _make_mock_response("4"),
        ]

        cg = create_blicket_cg()
        notifications = await engine.evaluate_and_apply(
            context_graph=cg,
            conversation_history=[{"role": "user", "content": "test"}],
            step_number=10,
        )

        assert len(notifications) == 0
        assert engine.fired_behaviors == set()

    @pytest.mark.asyncio
    async def test_stagnation_fires_independently_alongside_diagnostic(self):
        """Stagnation fires on its own, then diagnostic winner also fires."""
        client = _make_mock_client()
        behaviors = [
            _make_behavior("stagnation", stagnation_mode=True, fire_threshold=6.0),
            _make_behavior("diag1", fire_threshold=7.0),
            _make_behavior("diag2", fire_threshold=7.0),
        ]
        engine = DynamicBehaviorEngine(behaviors=behaviors, llm_client=client)

        # Stagnation scores 8.0, diag1 scores 8.0, diag2 scores 5.0
        client.messages.create.side_effect = [
            _make_mock_response("8"),   # stagnation
            _make_mock_response("8"),   # diag1
            _make_mock_response("5"),   # diag2
        ]

        cg = create_blicket_cg()
        notifications = await engine.evaluate_and_apply(
            context_graph=cg,
            conversation_history=[{"role": "user", "content": "test"}],
            step_number=10,
            activation_history=[],  # No activations for stagnation to pass pre-screen
        )

        assert len(notifications) == 2
        assert engine.fired_behaviors == {"stagnation", "diag1"}

    @pytest.mark.asyncio
    async def test_one_shot_respected(self):
        """Previously-fired behavior is skipped."""
        client = _make_mock_client()
        behaviors = [
            _make_behavior("b1", fire_threshold=7.0),
            _make_behavior("b2", fire_threshold=7.0),
        ]
        engine = DynamicBehaviorEngine(behaviors=behaviors, llm_client=client)

        # First evaluation: b1 wins
        client.messages.create.side_effect = [
            _make_mock_response("9"),
            _make_mock_response("5"),
        ]

        cg = create_blicket_cg()
        await engine.evaluate_and_apply(
            context_graph=cg,
            conversation_history=[{"role": "user", "content": "test"}],
            step_number=10,
        )
        assert engine.fired_behaviors == {"b1"}

        # Second evaluation: b1 should be skipped, b2 wins
        client.messages.create.side_effect = [
            _make_mock_response("8"),  # Only b2 evaluated
        ]

        notifications = await engine.evaluate_and_apply(
            context_graph=cg,
            conversation_history=[{"role": "user", "content": "test"}],
            step_number=15,
        )

        assert len(notifications) == 1
        assert "b2" in notifications[0]
        assert engine.fired_behaviors == {"b1", "b2"}

    @pytest.mark.asyncio
    async def test_score_log_records_all_evaluations(self):
        """All evaluated scores are recorded in the score log."""
        client = _make_mock_client()
        behaviors = [
            _make_behavior("stagnation", stagnation_mode=True, fire_threshold=6.0),
            _make_behavior("diag1", fire_threshold=7.0),
            _make_behavior("diag2", fire_threshold=7.0),
        ]
        engine = DynamicBehaviorEngine(behaviors=behaviors, llm_client=client)

        client.messages.create.side_effect = [
            _make_mock_response("7"),   # stagnation → fires (≥6.0)
            _make_mock_response("8"),   # diag1 → fires (≥7.0)
            _make_mock_response("6"),   # diag2 → doesn't fire (<7.0)
        ]

        cg = create_blicket_cg()
        await engine.evaluate_and_apply(
            context_graph=cg,
            conversation_history=[{"role": "user", "content": "test"}],
            step_number=10,
            activation_history=[],
        )

        log = engine.score_log
        assert len(log) == 3

        # Check stagnation entry
        stag_entry = next(e for e in log if e["behavior"] == "stagnation")
        assert stag_entry["score"] == 7.0
        assert stag_entry["fired"] is True
        assert stag_entry["phase"] == "stagnation"

        # Check diagnostic entries
        diag1_entry = next(e for e in log if e["behavior"] == "diag1")
        assert diag1_entry["score"] == 8.0
        assert diag1_entry["fired"] is True
        assert diag1_entry["phase"] == "diagnostic"

        diag2_entry = next(e for e in log if e["behavior"] == "diag2")
        assert diag2_entry["score"] == 6.0
        assert diag2_entry["fired"] is False
        assert diag2_entry["phase"] == "diagnostic"

    @pytest.mark.asyncio
    async def test_score_log_reset(self):
        """Score log resets between episodes."""
        client = _make_mock_client()
        behaviors = [_make_behavior("b1", fire_threshold=7.0)]
        engine = DynamicBehaviorEngine(behaviors=behaviors, llm_client=client)

        client.messages.create.side_effect = [_make_mock_response("8")]

        cg = create_blicket_cg()
        await engine.evaluate_and_apply(
            context_graph=cg,
            conversation_history=[{"role": "user", "content": "test"}],
            step_number=10,
        )
        assert len(engine.score_log) == 1

        engine.reset()
        assert len(engine.score_log) == 0
        assert engine.fired_behaviors == set()

    @pytest.mark.asyncio
    async def test_proximity_boundary_exactly_one_point(self):
        """Score exactly 1.0 below top → fires. Score 1.1 below → doesn't."""
        client = _make_mock_client()
        behaviors = [
            _make_behavior("b1", fire_threshold=7.0),
            _make_behavior("b2", fire_threshold=7.0),
            _make_behavior("b3", fire_threshold=7.0),
        ]
        engine = DynamicBehaviorEngine(behaviors=behaviors, llm_client=client)

        # 9.0, 8.0 (exactly 1.0 below), 7.5 (1.5 below but above threshold)
        client.messages.create.side_effect = [
            _make_mock_response("9"),
            _make_mock_response("8"),
            _make_mock_response("7.5"),
        ]

        cg = create_blicket_cg()
        notifications = await engine.evaluate_and_apply(
            context_graph=cg,
            conversation_history=[{"role": "user", "content": "test"}],
            step_number=10,
        )

        # b1 (9.0) fires, b2 (8.0, within 1.0) fires, b3 (7.5, 1.5 away) doesn't
        assert len(notifications) == 2
        assert engine.fired_behaviors == {"b1", "b2"}

    @pytest.mark.asyncio
    async def test_evaluate_trigger_returns_none_on_error(self):
        """Trigger evaluation returns None on LLM error, behavior doesn't fire."""
        client = _make_mock_client()
        behaviors = [_make_behavior("b1", fire_threshold=7.0)]
        engine = DynamicBehaviorEngine(behaviors=behaviors, llm_client=client)

        client.messages.create.side_effect = Exception("API error")

        cg = create_blicket_cg()
        notifications = await engine.evaluate_and_apply(
            context_graph=cg,
            conversation_history=[{"role": "user", "content": "test"}],
            step_number=10,
        )

        assert len(notifications) == 0
        assert engine.fired_behaviors == set()


class TestBlicketBehaviorThresholds:
    """Verify the actual blicket behavior thresholds are correct for Run 3."""

    def test_stagnation_threshold_unchanged(self):
        behaviors = create_blicket_behaviors()
        stagnation = next(b for b in behaviors if b.name == "exploration_stagnation")
        assert stagnation.trigger.fire_threshold == 6.0
        assert stagnation.trigger.stagnation_mode is True

    def test_diagnostic_thresholds_raised_to_7(self):
        behaviors = create_blicket_behaviors()
        for name in ["order_hypothesis", "stochasticity_hypothesis", "rule_change_hypothesis"]:
            behavior = next(b for b in behaviors if b.name == name)
            assert behavior.trigger.fire_threshold == 7.0, f"{name} threshold should be 7.0"
            assert behavior.trigger.stagnation_mode is False
