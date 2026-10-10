"""Unit tests for partial feedback mechanism (Run 10)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.config import PartialFeedbackConfig
from experiments.episode_runner import (
    _check_partial_feedback_trigger,
    _render_partial_feedback_message,
)


# --- 1. Trigger fires correctly ---


class TestTriggerFiresCorrectly:
    """Trigger should fire when all conditions are met."""

    def _make_config(self, **kwargs):
        defaults = dict(enabled=True, min_steps=5, min_unique_combos=3, correct_objects=["A", "C", "D"])
        defaults.update(kwargs)
        return PartialFeedbackConfig(**defaults)

    def test_fires_at_threshold(self):
        config = self._make_config()
        combos = {frozenset({"A"}), frozenset({"B"}), frozenset({"A", "B"})}
        assert _check_partial_feedback_trigger(
            step_count=5, unique_combinations=combos,
            any_activation=False, already_delivered=False, config=config,
        ) is True

    def test_fires_above_threshold(self):
        config = self._make_config()
        combos = {frozenset({"A"}), frozenset({"B"}), frozenset({"C"}), frozenset({"A", "B"})}
        assert _check_partial_feedback_trigger(
            step_count=10, unique_combinations=combos,
            any_activation=False, already_delivered=False, config=config,
        ) is True

    def test_below_step_threshold(self):
        config = self._make_config()
        combos = {frozenset({"A"}), frozenset({"B"}), frozenset({"A", "B"})}
        assert _check_partial_feedback_trigger(
            step_count=4, unique_combinations=combos,
            any_activation=False, already_delivered=False, config=config,
        ) is False

    def test_below_combo_threshold(self):
        config = self._make_config()
        combos = {frozenset({"A"}), frozenset({"B"})}
        assert _check_partial_feedback_trigger(
            step_count=10, unique_combinations=combos,
            any_activation=False, already_delivered=False, config=config,
        ) is False


# --- 2. Message content ---


class TestMessageContent:
    """Message should name correct objects, be sorted, and avoid forbidden words."""

    FORBIDDEN_WORDS = ["order", "sequence", "arrangement", "first", "before", "after"]

    def test_names_correct_objects(self):
        msg = _render_partial_feedback_message(["A", "C", "D"])
        assert "A" in msg
        assert "C" in msg
        assert "D" in msg

    def test_objects_sorted_alphabetically(self):
        msg = _render_partial_feedback_message(["D", "A", "C"])
        # Should appear as "A, C, D" not "D, A, C"
        assert "A, C, D" in msg

    def test_no_forbidden_words(self):
        msg = _render_partial_feedback_message(["A", "C", "D"])
        msg_lower = msg.lower()
        for word in self.FORBIDDEN_WORDS:
            assert word not in msg_lower, f"Forbidden word '{word}' found in message"

    def test_starts_with_partial_insight(self):
        msg = _render_partial_feedback_message(["A", "C", "D"])
        assert msg.startswith("PARTIAL INSIGHT:")

    def test_single_object(self):
        msg = _render_partial_feedback_message(["B"])
        assert "B" in msg

    def test_two_objects_sorted(self):
        msg = _render_partial_feedback_message(["C", "A"])
        assert "A, C" in msg


# --- 3. No trigger on activation ---


class TestNoTriggerOnActivation:
    """If detector activated at any point, trigger should not fire."""

    def test_activation_blocks_trigger(self):
        config = PartialFeedbackConfig(
            enabled=True, min_steps=5, min_unique_combos=3,
            correct_objects=["A", "C", "D"],
        )
        combos = {frozenset({"A"}), frozenset({"B"}), frozenset({"A", "B"})}
        assert _check_partial_feedback_trigger(
            step_count=10, unique_combinations=combos,
            any_activation=True, already_delivered=False, config=config,
        ) is False


# --- 4. No double delivery ---


class TestNoDoubleDelivery:
    """Once delivered, trigger should not fire again."""

    def test_already_delivered_blocks_trigger(self):
        config = PartialFeedbackConfig(
            enabled=True, min_steps=5, min_unique_combos=3,
            correct_objects=["A", "C", "D"],
        )
        combos = {frozenset({"A"}), frozenset({"B"}), frozenset({"A", "B"})}
        assert _check_partial_feedback_trigger(
            step_count=10, unique_combinations=combos,
            any_activation=False, already_delivered=True, config=config,
        ) is False


# --- 5. Backward compatibility ---


class TestBackwardCompatibility:
    """Default config (enabled=False) should never trigger."""

    def test_default_config_never_triggers(self):
        config = PartialFeedbackConfig()
        assert config.enabled is False
        combos = {frozenset({"A"}), frozenset({"B"}), frozenset({"A", "B"})}
        assert _check_partial_feedback_trigger(
            step_count=100, unique_combinations=combos,
            any_activation=False, already_delivered=False, config=config,
        ) is False

    def test_explicit_disabled_never_triggers(self):
        config = PartialFeedbackConfig(
            enabled=False, min_steps=1, min_unique_combos=1,
            correct_objects=["A"],
        )
        combos = {frozenset({"A"}), frozenset({"B"}), frozenset({"A", "B"})}
        assert _check_partial_feedback_trigger(
            step_count=100, unique_combinations=combos,
            any_activation=False, already_delivered=False, config=config,
        ) is False


# --- 6. Integration: frozenset combo tracking ---


class TestFrozensetComboTracking:
    """Validate that frozenset tracking matches env behavior."""

    def test_frozenset_deduplication(self):
        """Same objects placed in different order should be same combo."""
        combos: set[frozenset[str]] = set()
        combos.add(frozenset(["A", "B"]))
        combos.add(frozenset(["B", "A"]))
        assert len(combos) == 1

    def test_empty_not_tracked(self):
        """Empty detector should not add a combo (matches episode_runner logic)."""
        combos: set[frozenset[str]] = set()
        objects_on_detector: list[str] = []
        if objects_on_detector:
            combos.add(frozenset(objects_on_detector))
        assert len(combos) == 0

    def test_progressive_combo_building(self):
        """Simulate PLACE actions building unique combos."""
        combos: set[frozenset[str]] = set()

        # Step 1: PLACE A → {A}
        combos.add(frozenset(["A"]))
        assert len(combos) == 1

        # Step 2: PLACE B → {A, B}
        combos.add(frozenset(["A", "B"]))
        assert len(combos) == 2

        # Step 3: REMOVE A → {B}
        combos.add(frozenset(["B"]))
        assert len(combos) == 3

        # Step 4: PLACE A again → {A, B} (duplicate)
        combos.add(frozenset(["A", "B"]))
        assert len(combos) == 3  # Still 3

    def test_config_correct_objects_matches_env(self):
        """Verify PartialFeedbackConfig can hold the same objects as a causal rule."""
        from environment.blicket_env import CausalRule, CausalRuleType

        rule = CausalRule(
            rule_type=CausalRuleType.ORDER_SENSITIVE,
            required_objects={"A", "C", "D"},
        )
        config = PartialFeedbackConfig(
            enabled=True,
            correct_objects=sorted(rule.required_objects),
        )
        assert set(config.correct_objects) == rule.required_objects
