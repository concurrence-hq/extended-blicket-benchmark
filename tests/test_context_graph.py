"""Unit tests for the Context Graph data structure."""

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


class TestContextGraph:
    def test_create_blicket_cg(self):
        cg = create_blicket_cg()
        assert len(cg.states) == 4
        assert cg.current_state == "INITIAL_EXPLORATION"
        assert "COMBINATION_TESTING" in cg.states
        assert "HYPOTHESIS_EVALUATION" in cg.states
        assert "VERIFICATION" in cg.states

    def test_valid_transition(self):
        cg = create_blicket_cg()
        assert cg.current_state == "INITIAL_EXPLORATION"
        success = cg.transition_to("COMBINATION_TESTING")
        assert success is True
        assert cg.current_state == "COMBINATION_TESTING"

    def test_invalid_transition(self):
        cg = create_blicket_cg()
        # Can't go directly from INITIAL_EXPLORATION to VERIFICATION
        success = cg.transition_to("VERIFICATION")
        assert success is False
        assert cg.current_state == "INITIAL_EXPLORATION"

    def test_nonexistent_state_transition(self):
        cg = create_blicket_cg()
        success = cg.transition_to("NONEXISTENT")
        assert success is False

    def test_transition_chain(self):
        cg = create_blicket_cg()
        assert cg.transition_to("COMBINATION_TESTING")
        assert cg.transition_to("HYPOTHESIS_EVALUATION")
        assert cg.transition_to("VERIFICATION")
        assert cg.current_state == "VERIFICATION"

    def test_back_transition(self):
        cg = create_blicket_cg()
        cg.transition_to("COMBINATION_TESTING")
        cg.transition_to("HYPOTHESIS_EVALUATION")
        # Go back to COMBINATION_TESTING
        success = cg.transition_to("COMBINATION_TESTING")
        assert success is True

    def test_add_state(self):
        cg = create_blicket_cg()
        new_state = CGState(
            name="NEW_STATE",
            state_type=CGStateType.REFLECTION,
            objective="Test objective",
        )
        cg.add_state(new_state, connect_from=["COMBINATION_TESTING"])
        assert "NEW_STATE" in cg.states

        # Check transition was added
        ct = cg.states["COMBINATION_TESTING"]
        targets = {t.target for t in ct.transitions}
        assert "NEW_STATE" in targets

    def test_add_state_no_duplicates(self):
        cg = create_blicket_cg()
        new_state = CGState(
            name="NEW_STATE",
            state_type=CGStateType.REFLECTION,
            objective="Test",
        )
        cg.add_state(new_state, connect_from=["COMBINATION_TESTING"])
        # Add again — should not create duplicate transition
        cg.add_state(new_state, connect_from=["COMBINATION_TESTING"])
        ct = cg.states["COMBINATION_TESTING"]
        count = sum(1 for t in ct.transitions if t.target == "NEW_STATE")
        assert count == 1

    def test_modify_guidelines(self):
        cg = create_blicket_cg()
        original_count = len(cg.states["INITIAL_EXPLORATION"].guidelines)
        success = cg.modify_guidelines(
            "INITIAL_EXPLORATION",
            ["New guideline 1", "New guideline 2"],
        )
        assert success is True
        assert len(cg.states["INITIAL_EXPLORATION"].guidelines) == original_count + 2

    def test_modify_guidelines_nonexistent(self):
        cg = create_blicket_cg()
        success = cg.modify_guidelines("NONEXISTENT", ["Guideline"])
        assert success is False

    def test_expose_tools(self):
        cg = create_blicket_cg()
        success = cg.expose_tools("INITIAL_EXPLORATION", ["tool_a", "tool_b"])
        assert success is True
        assert "tool_a" in cg.states["INITIAL_EXPLORATION"].tools

    def test_modification_log(self):
        cg = create_blicket_cg()
        assert len(cg.modification_log) == 0

        new_state = CGState(
            name="TEST", state_type=CGStateType.ACTION, objective="Test"
        )
        cg.add_state(new_state, step_number=5)
        cg.modify_guidelines("INITIAL_EXPLORATION", ["Guide"], step_number=6)

        assert len(cg.modification_log) == 2
        assert cg.modification_log[0].action == "add_state"
        assert cg.modification_log[0].step_number == 5
        assert cg.modification_log[1].action == "modify_guidelines"

    def test_render_for_prompt(self):
        cg = create_blicket_cg()
        text = cg.render_for_prompt()
        assert "Context Graph" in text
        assert "INITIAL_EXPLORATION" in text
        assert "YOU ARE HERE" in text
        assert "Objective:" in text
        assert "Guidelines:" in text

    def test_render_with_no_state(self):
        cg = ContextGraph()
        text = cg.render_for_prompt()
        assert "No active state" in text

    def test_state_types(self):
        cg = create_blicket_cg()
        assert cg.states["INITIAL_EXPLORATION"].state_type == CGStateType.ACTION
        assert cg.states["HYPOTHESIS_EVALUATION"].state_type == CGStateType.REFLECTION
        assert cg.states["VERIFICATION"].state_type == CGStateType.DECISION

    def test_serialization_roundtrip(self):
        cg = create_blicket_cg()
        cg.transition_to("COMBINATION_TESTING")
        data = cg.model_dump()
        restored = ContextGraph.model_validate(data)
        assert restored.current_state == "COMBINATION_TESTING"
        assert len(restored.states) == 4


class TestBackstop:
    """Tests for the INITIAL_EXPLORATION step-count backstop."""

    def test_backstop_disabled_by_default(self):
        """With default initial_exploration_max_steps=None, backstop never fires."""
        cg = create_blicket_cg()
        assert cg.initial_exploration_max_steps is None
        for _ in range(100):
            result = cg.advance_step()
            assert result is None
        assert cg.current_state == "INITIAL_EXPLORATION"

    def test_backstop_fires_at_threshold(self):
        """Backstop fires when step count reaches the threshold."""
        cg = create_blicket_cg()
        cg.initial_exploration_max_steps = 10

        for i in range(9):
            result = cg.advance_step()
            assert result is None
            assert cg.current_state == "INITIAL_EXPLORATION"

        # 10th step should fire
        result = cg.advance_step()
        assert result == ("INITIAL_EXPLORATION", "COMBINATION_TESTING")
        assert cg.current_state == "COMBINATION_TESTING"
        assert cg.steps_in_current_state == 0

    def test_backstop_fires_only_once(self):
        """Backstop does not re-fire if agent returns to INITIAL_EXPLORATION."""
        cg = create_blicket_cg()
        cg.initial_exploration_max_steps = 5

        # Fire the backstop
        for _ in range(5):
            cg.advance_step()
        assert cg.current_state == "COMBINATION_TESTING"

        # Return to INITIAL_EXPLORATION via valid transition
        assert cg.transition_to("INITIAL_EXPLORATION")

        # Advance many steps — backstop should NOT fire again
        for _ in range(20):
            result = cg.advance_step()
            assert result is None
        assert cg.current_state == "INITIAL_EXPLORATION"

    def test_backstop_only_affects_initial_exploration(self):
        """Backstop has no effect when agent is in other states."""
        cg = create_blicket_cg()
        cg.initial_exploration_max_steps = 3

        # Move to COMBINATION_TESTING manually
        cg.transition_to("COMBINATION_TESTING")

        # Advance many steps — backstop should not fire
        for _ in range(20):
            result = cg.advance_step()
            assert result is None
        assert cg.current_state == "COMBINATION_TESTING"

    def test_backstop_logged_in_modification_log(self):
        """Backstop transition should NOT add a modification_log entry (that's cg_agent's job)."""
        cg = create_blicket_cg()
        cg.initial_exploration_max_steps = 3

        initial_log_len = len(cg.modification_log)
        for _ in range(3):
            cg.advance_step()

        # advance_step itself does not log — the cg_agent layer handles logging
        assert len(cg.modification_log) == initial_log_len

    def test_transition_resets_step_counter(self):
        """transition_to() should reset steps_in_current_state."""
        cg = create_blicket_cg()

        # Accumulate steps
        for _ in range(5):
            cg.advance_step()
        assert cg.steps_in_current_state == 5

        # Transition resets
        cg.transition_to("COMBINATION_TESTING")
        assert cg.steps_in_current_state == 0

    def test_serialization_with_backstop_fields(self):
        """Backstop fields should survive serialization roundtrip."""
        cg = create_blicket_cg()
        cg.initial_exploration_max_steps = 15
        cg.steps_in_current_state = 7

        data = cg.model_dump()
        restored = ContextGraph.model_validate(data)

        assert restored.initial_exploration_max_steps == 15
        assert restored.steps_in_current_state == 7

    def test_old_serialization_loads_with_defaults(self):
        """Data without backstop fields should load with defaults."""
        # Simulate old serialization format (no backstop fields)
        old_data = {
            "states": {
                "S1": {
                    "name": "S1",
                    "state_type": "action",
                    "objective": "Test",
                    "guidelines": [],
                    "tools": [],
                    "transitions": [],
                }
            },
            "current_state": "S1",
            "modification_log": [],
        }
        cg = ContextGraph.model_validate(old_data)
        assert cg.steps_in_current_state == 0
        assert cg.initial_exploration_max_steps is None


class TestCGv2:
    """Tests for v2 Context Graph with OBJECT_RELEVANCE_SCREENING."""

    def test_v2_creates_five_states(self):
        cg = create_blicket_cg(num_objects=5, cg_version="v2", object_list=["A", "B", "C", "D", "E"])
        assert len(cg.states) == 5
        assert "OBJECT_RELEVANCE_SCREENING" in cg.states

    def test_v2_initial_exploration_transitions_to_screening(self):
        cg = create_blicket_cg(cg_version="v2", object_list=["A", "B", "C"])
        ie = cg.states["INITIAL_EXPLORATION"]
        targets = {t.target for t in ie.transitions}
        assert "OBJECT_RELEVANCE_SCREENING" in targets
        assert "COMBINATION_TESTING" not in targets

    def test_v2_full_transition_chain(self):
        cg = create_blicket_cg(cg_version="v2", object_list=["A", "B", "C"])
        assert cg.transition_to("OBJECT_RELEVANCE_SCREENING")
        assert cg.transition_to("COMBINATION_TESTING")
        assert cg.transition_to("HYPOTHESIS_EVALUATION")
        assert cg.transition_to("VERIFICATION")
        assert cg.current_state == "VERIFICATION"

    def test_v1_still_produces_four_states(self):
        """Regression: v1 should be unchanged."""
        cg = create_blicket_cg(cg_version="v1")
        assert len(cg.states) == 4
        assert "OBJECT_RELEVANCE_SCREENING" not in cg.states

    def test_default_is_v1(self):
        """No version arg should produce v1 (backward compat)."""
        cg = create_blicket_cg()
        assert len(cg.states) == 4
        assert "OBJECT_RELEVANCE_SCREENING" not in cg.states

    def test_object_relevance_screening_is_action_type(self):
        cg = create_blicket_cg(cg_version="v2", object_list=["A", "B", "C"])
        assert cg.states["OBJECT_RELEVANCE_SCREENING"].state_type == CGStateType.ACTION

    def test_object_list_appears_in_v2_guidelines(self):
        cg = create_blicket_cg(cg_version="v2", object_list=["X", "Y", "Z"])
        ie_guidelines = " ".join(cg.states["INITIAL_EXPLORATION"].guidelines)
        assert "X" in ie_guidelines
        assert "Y" in ie_guidelines
        assert "Z" in ie_guidelines

    def test_invalid_cg_version_raises(self):
        with pytest.raises(ValueError, match="Invalid cg_version"):
            create_blicket_cg(cg_version="v99")


class TestBackstopV2:
    """Tests for backstop behavior in v2 Context Graphs."""

    def test_ie_backstop_targets_screening_in_v2(self):
        """In v2, IE backstop should go to OBJECT_RELEVANCE_SCREENING."""
        cg = create_blicket_cg(cg_version="v2", object_list=["A", "B", "C"])
        cg.initial_exploration_max_steps = 10

        for _ in range(10):
            result = cg.advance_step()
            if result is not None:
                break

        assert result == ("INITIAL_EXPLORATION", "OBJECT_RELEVANCE_SCREENING")
        assert cg.current_state == "OBJECT_RELEVANCE_SCREENING"

    def test_screening_backstop_fires_at_threshold(self):
        """Screening backstop fires after object_screening_max_steps."""
        cg = create_blicket_cg(cg_version="v2", object_list=["A", "B", "C"])
        cg.object_screening_max_steps = 5

        # Move to OBJECT_RELEVANCE_SCREENING manually
        cg.transition_to("OBJECT_RELEVANCE_SCREENING")

        for i in range(4):
            result = cg.advance_step()
            assert result is None

        result = cg.advance_step()
        assert result == ("OBJECT_RELEVANCE_SCREENING", "COMBINATION_TESTING")
        assert cg.current_state == "COMBINATION_TESTING"

    def test_screening_backstop_fires_only_once(self):
        """Screening backstop should not re-fire."""
        cg = create_blicket_cg(cg_version="v2", object_list=["A", "B", "C"])
        cg.object_screening_max_steps = 3

        cg.transition_to("OBJECT_RELEVANCE_SCREENING")
        for _ in range(3):
            cg.advance_step()
        assert cg.current_state == "COMBINATION_TESTING"

        # Return to screening via COMBINATION_TESTING -> OBJECT_RELEVANCE_SCREENING
        cg.transition_to("OBJECT_RELEVANCE_SCREENING")
        for _ in range(10):
            result = cg.advance_step()
            assert result is None
        assert cg.current_state == "OBJECT_RELEVANCE_SCREENING"

    def test_both_backstops_fire_in_sequence(self):
        """IE backstop → screening backstop → COMBINATION_TESTING."""
        cg = create_blicket_cg(cg_version="v2", object_list=["A", "B", "C"])
        cg.initial_exploration_max_steps = 5
        cg.object_screening_max_steps = 3

        # Fire IE backstop
        for _ in range(5):
            result = cg.advance_step()
        assert result == ("INITIAL_EXPLORATION", "OBJECT_RELEVANCE_SCREENING")
        assert cg.current_state == "OBJECT_RELEVANCE_SCREENING"

        # Fire screening backstop
        for _ in range(3):
            result = cg.advance_step()
        assert result == ("OBJECT_RELEVANCE_SCREENING", "COMBINATION_TESTING")
        assert cg.current_state == "COMBINATION_TESTING"

    def test_v1_backstop_returns_tuple(self):
        """v1 backstop should also return the new tuple format."""
        cg = create_blicket_cg(cg_version="v1")
        cg.initial_exploration_max_steps = 3

        for _ in range(3):
            result = cg.advance_step()
        assert result == ("INITIAL_EXPLORATION", "COMBINATION_TESTING")


class TestCGv3:
    """Tests for v3 Context Graph with revised ORS guidelines."""

    def test_v3_creates_five_states(self):
        cg = create_blicket_cg(cg_version="v3", object_list=["A", "B", "C", "D", "E"])
        assert len(cg.states) == 5
        assert "OBJECT_RELEVANCE_SCREENING" in cg.states

    def test_v3_ors_has_elimination_language(self):
        cg = create_blicket_cg(cg_version="v3", object_list=["A", "B", "C"])
        ors_guidelines = " ".join(cg.states["OBJECT_RELEVANCE_SCREENING"].guidelines)
        assert "ELIMINATE" in ors_guidelines

    def test_v3_ors_has_scorecard_language(self):
        cg = create_blicket_cg(cg_version="v3", object_list=["A", "B", "C"])
        ors_guidelines = " ".join(cg.states["OBJECT_RELEVANCE_SCREENING"].guidelines)
        assert "scorecard" in ors_guidelines

    def test_v3_individual_first_guideline(self):
        cg = create_blicket_cg(cg_version="v3", object_list=["A", "B", "C"])
        ors_guidelines = " ".join(cg.states["OBJECT_RELEVANCE_SCREENING"].guidelines)
        assert "individually first" in ors_guidelines

    def test_v3_full_transition_chain(self):
        cg = create_blicket_cg(cg_version="v3", object_list=["A", "B", "C"])
        assert cg.transition_to("OBJECT_RELEVANCE_SCREENING")
        assert cg.transition_to("COMBINATION_TESTING")
        assert cg.transition_to("HYPOTHESIS_EVALUATION")
        assert cg.transition_to("VERIFICATION")
        assert cg.current_state == "VERIFICATION"


class TestBackwardCompat:
    """Regression tests ensuring v1 and v2 are unchanged."""

    def test_v1_unchanged(self):
        cg = create_blicket_cg(cg_version="v1")
        assert len(cg.states) == 4
        assert cg.current_state == "INITIAL_EXPLORATION"
        assert cg.transition_to("COMBINATION_TESTING")
        assert cg.transition_to("HYPOTHESIS_EVALUATION")
        assert cg.transition_to("VERIFICATION")

    def test_v2_unchanged(self):
        cg = create_blicket_cg(cg_version="v2", object_list=["A", "B", "C"])
        assert len(cg.states) == 5
        assert "OBJECT_RELEVANCE_SCREENING" in cg.states
        assert cg.transition_to("OBJECT_RELEVANCE_SCREENING")
        assert cg.transition_to("COMBINATION_TESTING")

    def test_invalid_version_raises(self):
        with pytest.raises(ValueError, match="Invalid cg_version"):
            create_blicket_cg(cg_version="v99_nonexistent")
