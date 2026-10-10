"""Unit tests for the gradual periodic feedback mechanism.

Tests checkpoint triggering, object eligibility, first-tested ordering,
activation suppression, message content, backward compatibility, and
progressive reveal simulation.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.config import GradualFeedbackConfig, PartialFeedbackConfig
from experiments.episode_runner import (
    _check_gradual_feedback_checkpoint,
    _render_gradual_feedback_message,
)


class TestCheckpointTrigger:
    """Checkpoints fire at correct step intervals."""

    def _config(self) -> GradualFeedbackConfig:
        return GradualFeedbackConfig(
            enabled=True, checkpoint_interval=5, correct_objects=["A", "C", "D"]
        )

    def test_fires_at_step_5(self):
        result = _check_gradual_feedback_checkpoint(
            step_count=5,
            tested_objects_ordered=["A"],
            confirmed_objects=[],
            any_activation=False,
            config=self._config(),
        )
        assert result == "A"

    def test_fires_at_step_10(self):
        result = _check_gradual_feedback_checkpoint(
            step_count=10,
            tested_objects_ordered=["A", "D"],
            confirmed_objects=["A"],
            any_activation=False,
            config=self._config(),
        )
        assert result == "D"

    def test_fires_at_step_15(self):
        result = _check_gradual_feedback_checkpoint(
            step_count=15,
            tested_objects_ordered=["A", "D", "C"],
            confirmed_objects=["A", "D"],
            any_activation=False,
            config=self._config(),
        )
        assert result == "C"

    def test_does_not_fire_at_step_0(self):
        result = _check_gradual_feedback_checkpoint(
            step_count=0,
            tested_objects_ordered=["A"],
            confirmed_objects=[],
            any_activation=False,
            config=self._config(),
        )
        assert result is None

    def test_does_not_fire_at_step_7(self):
        result = _check_gradual_feedback_checkpoint(
            step_count=7,
            tested_objects_ordered=["A"],
            confirmed_objects=[],
            any_activation=False,
            config=self._config(),
        )
        assert result is None


class TestOnlyTestedObjectsEligible:
    """Only objects the agent has actually tested are eligible for confirmation."""

    def _config(self) -> GradualFeedbackConfig:
        return GradualFeedbackConfig(
            enabled=True, checkpoint_interval=5, correct_objects=["A", "C", "D"]
        )

    def test_untested_correct_objects_not_revealed(self):
        """A and D are correct but only A was tested — only A is eligible."""
        result = _check_gradual_feedback_checkpoint(
            step_count=5,
            tested_objects_ordered=["A"],
            confirmed_objects=[],
            any_activation=False,
            config=self._config(),
        )
        assert result == "A"

    def test_no_feedback_if_only_distractors_tested(self):
        """B and E are distractors — no eligible objects."""
        result = _check_gradual_feedback_checkpoint(
            step_count=5,
            tested_objects_ordered=["B", "E"],
            confirmed_objects=[],
            any_activation=False,
            config=self._config(),
        )
        assert result is None

    def test_no_feedback_if_nothing_tested(self):
        result = _check_gradual_feedback_checkpoint(
            step_count=5,
            tested_objects_ordered=[],
            confirmed_objects=[],
            any_activation=False,
            config=self._config(),
        )
        assert result is None


class TestOneObjectPerCheckpoint:
    """Only one object is revealed per checkpoint, even with multiple eligible."""

    def _config(self) -> GradualFeedbackConfig:
        return GradualFeedbackConfig(
            enabled=True, checkpoint_interval=5, correct_objects=["A", "C", "D"]
        )

    def test_single_string_returned(self):
        result = _check_gradual_feedback_checkpoint(
            step_count=5,
            tested_objects_ordered=["A", "D", "C"],
            confirmed_objects=[],
            any_activation=False,
            config=self._config(),
        )
        assert isinstance(result, str)
        assert result in {"A", "C", "D"}

    def test_deterministic_first_tested_priority(self):
        """When multiple are eligible, first-tested wins."""
        result = _check_gradual_feedback_checkpoint(
            step_count=5,
            tested_objects_ordered=["D", "A", "C"],
            confirmed_objects=[],
            any_activation=False,
            config=self._config(),
        )
        assert result == "D"


class TestNoFeedbackAfterActivation:
    """Once the detector has activated, no further feedback is delivered."""

    def _config(self) -> GradualFeedbackConfig:
        return GradualFeedbackConfig(
            enabled=True, checkpoint_interval=5, correct_objects=["A", "C", "D"]
        )

    def test_activation_suppresses_feedback(self):
        result = _check_gradual_feedback_checkpoint(
            step_count=5,
            tested_objects_ordered=["A", "D", "C"],
            confirmed_objects=[],
            any_activation=True,
            config=self._config(),
        )
        assert result is None

    def test_activation_suppresses_even_at_later_checkpoints(self):
        result = _check_gradual_feedback_checkpoint(
            step_count=10,
            tested_objects_ordered=["A", "D", "C"],
            confirmed_objects=["A"],
            any_activation=True,
            config=self._config(),
        )
        assert result is None


class TestAlreadyConfirmedExcluded:
    """Confirmed objects are never re-revealed."""

    def _config(self) -> GradualFeedbackConfig:
        return GradualFeedbackConfig(
            enabled=True, checkpoint_interval=5, correct_objects=["A", "C", "D"]
        )

    def test_confirmed_objects_skipped(self):
        result = _check_gradual_feedback_checkpoint(
            step_count=10,
            tested_objects_ordered=["A", "D", "C"],
            confirmed_objects=["A"],
            any_activation=False,
            config=self._config(),
        )
        assert result == "D"
        assert result != "A"

    def test_all_confirmed_returns_none(self):
        result = _check_gradual_feedback_checkpoint(
            step_count=20,
            tested_objects_ordered=["A", "D", "C"],
            confirmed_objects=["A", "D", "C"],
            any_activation=False,
            config=self._config(),
        )
        assert result is None


class TestGradualMessageContent:
    """Message names a single object, uses OBSERVATION prefix, avoids forbidden words."""

    FORBIDDEN_WORDS = [
        "order", "sequence", "arrangement", "first", "before", "after",
        "total", "remaining", "three", "four", "all", "only",
    ]

    def test_names_single_object(self):
        msg = _render_gradual_feedback_message("A")
        assert "object A" in msg

    def test_observation_prefix(self):
        msg = _render_gradual_feedback_message("D")
        assert msg.startswith("OBSERVATION:")

    def test_no_forbidden_words(self):
        msg = _render_gradual_feedback_message("C")
        msg_lower = msg.lower()
        for word in self.FORBIDDEN_WORDS:
            assert word not in msg_lower, f"Forbidden word '{word}' found in message: {msg}"

    def test_different_objects_produce_different_messages(self):
        msg_a = _render_gradual_feedback_message("A")
        msg_d = _render_gradual_feedback_message("D")
        assert msg_a != msg_d
        assert "object A" in msg_a
        assert "object D" in msg_d


class TestGradualBackwardCompatibility:
    """Default disabled config never triggers; partial feedback is independent."""

    def test_disabled_config_never_triggers(self):
        config = GradualFeedbackConfig()
        assert config.enabled is False
        result = _check_gradual_feedback_checkpoint(
            step_count=5,
            tested_objects_ordered=["A", "D", "C"],
            confirmed_objects=[],
            any_activation=False,
            config=config,
        )
        assert result is None

    def test_partial_feedback_config_still_independent(self):
        """PartialFeedbackConfig is unaffected by GradualFeedbackConfig existence."""
        pf = PartialFeedbackConfig(enabled=True, correct_objects=["A", "C", "D"])
        gf = GradualFeedbackConfig(enabled=False)
        assert pf.enabled is True
        assert gf.enabled is False

    def test_mutual_exclusivity_error(self):
        """Both enabled should raise ValueError in run_episode, but we test the
        config objects can coexist when not both enabled."""
        pf = PartialFeedbackConfig(enabled=True, correct_objects=["A"])
        gf = GradualFeedbackConfig(enabled=True, correct_objects=["A"])
        # Both enabled is fine at config level — the validation is in run_episode
        assert pf.enabled and gf.enabled


class TestProgressiveRevealSimulation:
    """Simulate a 3-checkpoint sequence confirming A→C→D."""

    def _config(self) -> GradualFeedbackConfig:
        return GradualFeedbackConfig(
            enabled=True, checkpoint_interval=5, correct_objects=["A", "C", "D"]
        )

    def test_full_sequence(self):
        config = self._config()
        tested = ["A", "C", "D", "B", "E"]  # A tested first, then C, etc.
        confirmed: list[str] = []

        # Checkpoint 1 (step 5): A is first eligible
        r1 = _check_gradual_feedback_checkpoint(5, tested, confirmed, False, config)
        assert r1 == "A"
        confirmed.append(r1)

        # Checkpoint 2 (step 10): C is next first-tested eligible
        r2 = _check_gradual_feedback_checkpoint(10, tested, confirmed, False, config)
        assert r2 == "C"
        confirmed.append(r2)

        # Checkpoint 3 (step 15): D is last eligible
        r3 = _check_gradual_feedback_checkpoint(15, tested, confirmed, False, config)
        assert r3 == "D"
        confirmed.append(r3)

        # Checkpoint 4 (step 20): all confirmed → None
        r4 = _check_gradual_feedback_checkpoint(20, tested, confirmed, False, config)
        assert r4 is None

    def test_sequence_with_distractors_tested_first(self):
        """If distractors are tested first, they don't count; first correct object wins."""
        config = self._config()
        tested = ["B", "E", "D", "A", "C"]  # Distractors B, E tested first
        confirmed: list[str] = []

        # Step 5: D is the first correct object tested (index 2)
        r1 = _check_gradual_feedback_checkpoint(5, tested, confirmed, False, config)
        assert r1 == "D"
        confirmed.append(r1)

        # Step 10: A is next (index 3)
        r2 = _check_gradual_feedback_checkpoint(10, tested, confirmed, False, config)
        assert r2 == "A"
        confirmed.append(r2)


class TestFirstTestedOrdering:
    """Verify that first-tested priority determines selection, not alphabetical."""

    def _config(self) -> GradualFeedbackConfig:
        return GradualFeedbackConfig(
            enabled=True, checkpoint_interval=5, correct_objects=["A", "C", "D"]
        )

    def test_d_first_when_tested_first(self):
        """D is tested before A and C — D should be revealed first."""
        result = _check_gradual_feedback_checkpoint(
            step_count=5,
            tested_objects_ordered=["D", "C", "A"],
            confirmed_objects=[],
            any_activation=False,
            config=self._config(),
        )
        assert result == "D"

    def test_c_first_when_tested_first(self):
        """C is tested before A and D — C should be revealed first."""
        result = _check_gradual_feedback_checkpoint(
            step_count=5,
            tested_objects_ordered=["C", "A", "D"],
            confirmed_objects=[],
            any_activation=False,
            config=self._config(),
        )
        assert result == "C"

    def test_order_preserved_across_checkpoints(self):
        config = self._config()
        tested = ["C", "D", "A"]
        confirmed: list[str] = []

        r1 = _check_gradual_feedback_checkpoint(5, tested, confirmed, False, config)
        assert r1 == "C"
        confirmed.append(r1)

        r2 = _check_gradual_feedback_checkpoint(10, tested, confirmed, False, config)
        assert r2 == "D"
        confirmed.append(r2)

        r3 = _check_gradual_feedback_checkpoint(15, tested, confirmed, False, config)
        assert r3 == "A"
