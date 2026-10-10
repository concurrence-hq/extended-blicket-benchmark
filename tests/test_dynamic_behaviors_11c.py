"""Unit tests for Dynamic Behaviors v4 (Run 11c).

Tests the lowered stochasticity_hypothesis threshold (6.0 vs 7.0)
and validates that base/10c/11b behaviors are NOT modified.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.context_graph import (
    CGState,
    CGStateType,
    CGTransition,
    ContextGraph,
    create_blicket_cg,
)
from agents.dynamic_behaviors import create_blicket_behaviors
from agents.dynamic_behaviors_10c import create_blicket_behaviors_10c
from agents.dynamic_behaviors_11b import create_blicket_behaviors_11b
from agents.dynamic_behaviors_11c import create_blicket_behaviors_11c


class TestBehaviorStructure:
    """Test that 11c returns 4 behaviors with correct names."""

    def test_returns_four_behaviors(self):
        behaviors = create_blicket_behaviors_11c()
        assert len(behaviors) == 4

    def test_behavior_names(self):
        behaviors = create_blicket_behaviors_11c()
        names = [b.name for b in behaviors]
        assert names == [
            "exploration_stagnation",
            "order_hypothesis",
            "stochasticity_hypothesis",
            "rule_change_hypothesis",
        ]

    def test_order_hypothesis_inherited_from_10c(self):
        """order_hypothesis should be the 10c revision (inherited via 11b)."""
        behaviors_11c = create_blicket_behaviors_11c()
        behaviors_10c = create_blicket_behaviors_10c()

        oh_11c = next(b for b in behaviors_11c if b.name == "order_hypothesis")
        oh_10c = next(b for b in behaviors_10c if b.name == "order_hypothesis")

        assert oh_11c.modifications[0].state.guidelines == oh_10c.modifications[0].state.guidelines

    def test_exploration_stagnation_inherited(self):
        """exploration_stagnation should be identical to 11b."""
        behaviors_11c = create_blicket_behaviors_11c()
        behaviors_11b = create_blicket_behaviors_11b()

        es_11c = next(b for b in behaviors_11c if b.name == "exploration_stagnation")
        es_11b = next(b for b in behaviors_11b if b.name == "exploration_stagnation")

        assert es_11c.trigger.fire_threshold == es_11b.trigger.fire_threshold

    def test_rule_change_hypothesis_inherited(self):
        """rule_change_hypothesis should be identical to 11b."""
        behaviors_11c = create_blicket_behaviors_11c()
        behaviors_11b = create_blicket_behaviors_11b()

        rc_11c = next(b for b in behaviors_11c if b.name == "rule_change_hypothesis")
        rc_11b = next(b for b in behaviors_11b if b.name == "rule_change_hypothesis")

        assert rc_11c.trigger.fire_threshold == rc_11b.trigger.fire_threshold


class TestStochasticityThreshold:
    """Test the lowered fire_threshold for stochasticity_hypothesis."""

    def test_threshold_is_6_0(self):
        behaviors = create_blicket_behaviors_11c()
        sh = next(b for b in behaviors if b.name == "stochasticity_hypothesis")
        assert sh.trigger.fire_threshold == 6.0

    def test_threshold_differs_from_11b(self):
        behaviors_11c = create_blicket_behaviors_11c()
        behaviors_11b = create_blicket_behaviors_11b()

        sh_11c = next(b for b in behaviors_11c if b.name == "stochasticity_hypothesis")
        sh_11b = next(b for b in behaviors_11b if b.name == "stochasticity_hypothesis")

        assert sh_11c.trigger.fire_threshold < sh_11b.trigger.fire_threshold
        assert sh_11b.trigger.fire_threshold == 7.0
        assert sh_11c.trigger.fire_threshold == 6.0

    def test_min_steps_unchanged(self):
        behaviors = create_blicket_behaviors_11c()
        sh = next(b for b in behaviors if b.name == "stochasticity_hypothesis")
        assert sh.trigger.min_steps == 8

    def test_connect_from_preserved_from_11b(self):
        """connect_from should still have 5 states (inherited from 11b)."""
        behaviors = create_blicket_behaviors_11c()
        sh = next(b for b in behaviors if b.name == "stochasticity_hypothesis")
        connect_from = sh.modifications[0].connect_from
        assert len(connect_from) == 5
        assert set(connect_from) == {
            "DIMENSION_DISCOVERY",
            "COMBINATION_TESTING",
            "VERIFICATION",
            "HYPOTHESIS_EVALUATION",
            "ORDER_TESTING",
        }

    def test_reliability_testing_has_two_transitions(self):
        behaviors = create_blicket_behaviors_11c()
        sh = next(b for b in behaviors if b.name == "stochasticity_hypothesis")
        state = sh.modifications[0].state
        assert state.name == "RELIABILITY_TESTING"
        assert len(state.transitions) == 2
        targets = {t.target for t in state.transitions}
        assert targets == {"VERIFICATION", "HYPOTHESIS_EVALUATION"}


class TestBaseNotModified:
    """Regression: base/10c/11b behaviors should NOT be modified by 11c creation."""

    def test_base_stochasticity_threshold_unchanged(self):
        """Base stochasticity_hypothesis should still have threshold 7.0."""
        _ = create_blicket_behaviors_11c()

        base = create_blicket_behaviors()
        sh = next(b for b in base if b.name == "stochasticity_hypothesis")
        assert sh.trigger.fire_threshold == 7.0

    def test_base_stochasticity_connect_from_unchanged(self):
        _ = create_blicket_behaviors_11c()

        base = create_blicket_behaviors()
        sh = next(b for b in base if b.name == "stochasticity_hypothesis")
        assert len(sh.modifications[0].connect_from) == 2

    def test_11b_stochasticity_threshold_unchanged(self):
        """11b stochasticity_hypothesis should still have threshold 7.0."""
        _ = create_blicket_behaviors_11c()

        behaviors_11b = create_blicket_behaviors_11b()
        sh = next(b for b in behaviors_11b if b.name == "stochasticity_hypothesis")
        assert sh.trigger.fire_threshold == 7.0

    def test_10c_behaviors_unchanged(self):
        _ = create_blicket_behaviors_11c()

        behaviors_10c = create_blicket_behaviors_10c()
        sh = next(b for b in behaviors_10c if b.name == "stochasticity_hypothesis")
        assert len(sh.modifications[0].connect_from) == 2
