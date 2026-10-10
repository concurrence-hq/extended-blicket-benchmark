"""Unit tests for Dynamic Behaviors v3 (Run 11b).

Tests the revised stochasticity_hypothesis with expanded reachability
and validates that base behaviors are NOT modified.
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


class TestBehaviorStructure:
    """Test that 11b returns 4 behaviors with correct names."""

    def test_returns_four_behaviors(self):
        behaviors = create_blicket_behaviors_11b()
        assert len(behaviors) == 4

    def test_behavior_names(self):
        behaviors = create_blicket_behaviors_11b()
        names = [b.name for b in behaviors]
        assert names == [
            "exploration_stagnation",
            "order_hypothesis",
            "stochasticity_hypothesis",
            "rule_change_hypothesis",
        ]

    def test_order_hypothesis_inherited_from_10c(self):
        """order_hypothesis should be the 10c revision, not the base."""
        behaviors_11b = create_blicket_behaviors_11b()
        behaviors_10c = create_blicket_behaviors_10c()

        oh_11b = next(b for b in behaviors_11b if b.name == "order_hypothesis")
        oh_10c = next(b for b in behaviors_10c if b.name == "order_hypothesis")

        # Should be identical — 11b inherits 10c's order_hypothesis
        assert oh_11b.modifications[0].state.guidelines == oh_10c.modifications[0].state.guidelines

    def test_stochasticity_hypothesis_is_revised(self):
        """stochasticity_hypothesis should differ from base and 10c."""
        behaviors_11b = create_blicket_behaviors_11b()
        behaviors_base = create_blicket_behaviors()

        sh_11b = next(b for b in behaviors_11b if b.name == "stochasticity_hypothesis")
        sh_base = next(b for b in behaviors_base if b.name == "stochasticity_hypothesis")

        # connect_from should be different
        assert sh_11b.modifications[0].connect_from != sh_base.modifications[0].connect_from


class TestStochasticityReachability:
    """Test the revised stochasticity_hypothesis connect_from and transitions."""

    def test_connect_from_has_five_states(self):
        behaviors = create_blicket_behaviors_11b()
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

    def test_reliability_testing_has_verification_transition(self):
        behaviors = create_blicket_behaviors_11b()
        sh = next(b for b in behaviors if b.name == "stochasticity_hypothesis")
        state = sh.modifications[0].state
        assert state.name == "RELIABILITY_TESTING"
        targets = {t.target for t in state.transitions}
        assert "VERIFICATION" in targets

    def test_reliability_testing_has_hypothesis_evaluation_escape(self):
        behaviors = create_blicket_behaviors_11b()
        sh = next(b for b in behaviors if b.name == "stochasticity_hypothesis")
        state = sh.modifications[0].state
        targets = {t.target for t in state.transitions}
        assert "HYPOTHESIS_EVALUATION" in targets

    def test_reliability_testing_has_two_transitions(self):
        behaviors = create_blicket_behaviors_11b()
        sh = next(b for b in behaviors if b.name == "stochasticity_hypothesis")
        state = sh.modifications[0].state
        assert len(state.transitions) == 2


class TestIntegrationReachability:
    """Test that applying 11b stochasticity behavior makes RELIABILITY_TESTING reachable."""

    def _apply_stochasticity_to_cg(self, cg: ContextGraph) -> None:
        """Apply the stochasticity_hypothesis modifications to a CG."""
        behaviors = create_blicket_behaviors_11b()
        sh = next(b for b in behaviors if b.name == "stochasticity_hypothesis")
        mod = sh.modifications[0]
        cg.add_state(mod.state, connect_from=mod.connect_from, step_number=22)

    def test_reachable_from_order_testing(self):
        """When CG is in ORDER_TESTING, RELIABILITY_TESTING should be reachable."""
        cg = create_blicket_cg()

        # First add ORDER_TESTING (as order_hypothesis would)
        behaviors = create_blicket_behaviors_11b()
        oh = next(b for b in behaviors if b.name == "order_hypothesis")
        oh_mod = oh.modifications[0]
        cg.add_state(oh_mod.state, connect_from=oh_mod.connect_from, step_number=10)

        # Navigate to ORDER_TESTING
        cg.transition_to("COMBINATION_TESTING")
        assert cg.transition_to("ORDER_TESTING")

        # Now apply stochasticity
        self._apply_stochasticity_to_cg(cg)

        # ORDER_TESTING should now have a transition to RELIABILITY_TESTING
        ot = cg.states["ORDER_TESTING"]
        targets = {t.target for t in ot.transitions}
        assert "RELIABILITY_TESTING" in targets

        # And we can transition there
        assert cg.transition_to("RELIABILITY_TESTING")
        assert cg.current_state == "RELIABILITY_TESTING"

    def test_reachable_from_verification(self):
        """When CG is in VERIFICATION, RELIABILITY_TESTING should be reachable."""
        cg = create_blicket_cg()
        self._apply_stochasticity_to_cg(cg)

        # Navigate to VERIFICATION
        cg.transition_to("COMBINATION_TESTING")
        cg.transition_to("HYPOTHESIS_EVALUATION")
        cg.transition_to("VERIFICATION")

        targets = {t.target for t in cg.states["VERIFICATION"].transitions}
        assert "RELIABILITY_TESTING" in targets
        assert cg.transition_to("RELIABILITY_TESTING")

    def test_reachable_from_hypothesis_evaluation(self):
        """When CG is in HYPOTHESIS_EVALUATION, RELIABILITY_TESTING should be reachable."""
        cg = create_blicket_cg()
        self._apply_stochasticity_to_cg(cg)

        cg.transition_to("COMBINATION_TESTING")
        cg.transition_to("HYPOTHESIS_EVALUATION")

        targets = {t.target for t in cg.states["HYPOTHESIS_EVALUATION"].transitions}
        assert "RELIABILITY_TESTING" in targets
        assert cg.transition_to("RELIABILITY_TESTING")

    def test_reachable_from_combination_testing(self):
        cg = create_blicket_cg()
        self._apply_stochasticity_to_cg(cg)

        cg.transition_to("COMBINATION_TESTING")
        targets = {t.target for t in cg.states["COMBINATION_TESTING"].transitions}
        assert "RELIABILITY_TESTING" in targets

    def test_can_exit_to_verification(self):
        """RELIABILITY_TESTING -> VERIFICATION should work."""
        cg = create_blicket_cg()
        self._apply_stochasticity_to_cg(cg)

        cg.transition_to("COMBINATION_TESTING")
        cg.transition_to("RELIABILITY_TESTING")
        assert cg.transition_to("VERIFICATION")

    def test_can_exit_to_hypothesis_evaluation(self):
        """RELIABILITY_TESTING -> HYPOTHESIS_EVALUATION should work."""
        cg = create_blicket_cg()
        self._apply_stochasticity_to_cg(cg)

        cg.transition_to("COMBINATION_TESTING")
        cg.transition_to("RELIABILITY_TESTING")
        assert cg.transition_to("HYPOTHESIS_EVALUATION")


class TestBaseNotModified:
    """Regression: base behaviors should NOT be modified by 11b creation."""

    def test_base_stochasticity_connect_from_unchanged(self):
        """Base stochasticity_hypothesis should still have 2 connect_from states."""
        # Create 11b first to ensure no side effects
        _ = create_blicket_behaviors_11b()

        base = create_blicket_behaviors()
        sh = next(b for b in base if b.name == "stochasticity_hypothesis")
        assert len(sh.modifications[0].connect_from) == 2
        assert set(sh.modifications[0].connect_from) == {
            "DIMENSION_DISCOVERY",
            "COMBINATION_TESTING",
        }

    def test_base_reliability_testing_has_one_transition(self):
        """Base RELIABILITY_TESTING should still have only 1 transition."""
        _ = create_blicket_behaviors_11b()

        base = create_blicket_behaviors()
        sh = next(b for b in base if b.name == "stochasticity_hypothesis")
        assert len(sh.modifications[0].state.transitions) == 1

    def test_10c_behaviors_unchanged(self):
        """10c behaviors should not be modified."""
        _ = create_blicket_behaviors_11b()

        behaviors_10c = create_blicket_behaviors_10c()
        sh = next(b for b in behaviors_10c if b.name == "stochasticity_hypothesis")
        # 10c inherits base stochasticity, so still 2 connect_from
        assert len(sh.modifications[0].connect_from) == 2
