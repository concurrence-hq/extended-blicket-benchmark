"""Unit tests for the text interface."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.blicket_env import (
    Action,
    BlicketEnv,
    CausalRule,
    CausalRuleType,
)
from environment.text_interface import TextInterface


@pytest.fixture
def ti():
    return TextInterface()


@pytest.fixture
def env():
    rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
    e = BlicketEnv(causal_rule=rule, max_steps=50)
    e.reset()
    return e


class TestRenderState:
    def test_initial_state(self, ti, env):
        text = ti.render_state(env)
        assert "none" in text.lower()
        assert "INACTIVE" in text
        assert "Place object A" in text

    def test_after_place(self, ti, env):
        env.step(Action.PLACE_A)
        text = ti.render_state(env)
        assert "A" in text
        assert "Remove object A" in text
        assert "Place object A" not in text


class TestParseAction:
    def test_numbered_action(self, ti, env):
        valid = env.get_valid_actions()
        result = ti.parse_action("REASONING: testing\nACTION: 1", valid)
        assert result == valid[0]

    def test_keyword_action(self, ti, env):
        valid = env.get_valid_actions()
        result = ti.parse_action("ACTION: place a", valid)
        assert result == Action.PLACE_A

    def test_keyword_case_insensitive(self, ti, env):
        valid = env.get_valid_actions()
        result = ti.parse_action("ACTION: Place B on detector", valid)
        assert result == Action.PLACE_B

    def test_submit_keyword(self, ti, env):
        valid = env.get_valid_actions()
        result = ti.parse_action("ACTION: submit", valid)
        assert result == Action.CHECK

    def test_full_text_scan(self, ti, env):
        valid = env.get_valid_actions()
        result = ti.parse_action(
            "I think I should place A on the detector to see what happens.",
            valid,
        )
        assert result == Action.PLACE_A

    def test_unparseable(self, ti, env):
        valid = env.get_valid_actions()
        result = ti.parse_action("I'm not sure what to do next.", valid)
        assert result is None

    def test_respects_valid_actions(self, ti, env):
        env.step(Action.PLACE_A)
        valid = env.get_valid_actions()
        # "place a" should not match since A is already placed
        result = ti.parse_action("ACTION: place a", valid)
        assert result is None

    def test_remove_keyword(self, ti, env):
        env.step(Action.PLACE_A)
        valid = env.get_valid_actions()
        result = ti.parse_action("ACTION: remove a", valid)
        assert result == Action.REMOVE_A

    def test_numbered_out_of_range(self, ti, env):
        valid = env.get_valid_actions()
        result = ti.parse_action("ACTION: 99", valid)
        assert result is None

    def test_last_line_fallback(self, ti, env):
        valid = env.get_valid_actions()
        result = ti.parse_action(
            "Let me think about this carefully.\n"
            "I want to test each object.\n"
            "place b",
            valid,
        )
        assert result == Action.PLACE_B


class TestParseAnswer:
    def test_standard_format(self, ti):
        rule, objs = ti.parse_answer(
            "RULE_TYPE: conjunctive\nBLICKETS: A, B"
        )
        assert rule == CausalRuleType.CONJUNCTIVE
        assert objs == {"A", "B"}

    def test_disjunctive(self, ti):
        rule, objs = ti.parse_answer(
            "RULE_TYPE: disjunctive\nBLICKETS: A"
        )
        assert rule == CausalRuleType.DISJUNCTIVE
        assert objs == {"A"}

    def test_natural_language_fallback(self, ti):
        rule, objs = ti.parse_answer(
            "I believe the rule is conjunctive. "
            "The blickets are A and B."
        )
        assert rule == CausalRuleType.CONJUNCTIVE
        assert objs == {"A", "B"}

    def test_ambiguous_rule_type(self, ti):
        rule, objs = ti.parse_answer(
            "It could be conjunctive or disjunctive. BLICKETS: A"
        )
        assert rule is None

    def test_missing_objects(self, ti):
        rule, objs = ti.parse_answer("RULE_TYPE: conjunctive")
        assert rule is None
        assert objs is None

    def test_three_blickets(self, ti):
        rule, objs = ti.parse_answer(
            "RULE_TYPE: conjunctive\nBLICKETS: A, B, C"
        )
        assert rule == CausalRuleType.CONJUNCTIVE
        assert objs == {"A", "B", "C"}

    def test_invalid_objects_filtered(self, ti):
        rule, objs = ti.parse_answer(
            "RULE_TYPE: disjunctive\nBLICKETS: A, D, B"
        )
        assert objs == {"A", "B"}  # D is not a valid object
