"""Unit tests for the 5-object text interface."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.blicket_env import CausalRule, CausalRuleType
from environment.blicket_env_5obj import Action5, BlicketEnv5
from environment.text_interface import TextInterface5


class TestTextInterface5:
    def setup_method(self):
        self.ti = TextInterface5()

    def test_task_description_mentions_five_objects(self):
        desc = self.ti.render_task_description()
        assert "5 objects" in desc
        assert "A, B, C, D, and E" in desc

    def test_parse_action_place_d(self):
        all_actions = [
            Action5.PLACE_A, Action5.PLACE_B, Action5.PLACE_C,
            Action5.PLACE_D, Action5.PLACE_E, Action5.CHECK,
        ]
        result = self.ti.parse_action("REASONING: testing D\nACTION: place d", all_actions)
        assert result == Action5.PLACE_D

    def test_parse_action_remove_e(self):
        valid = [Action5.REMOVE_A, Action5.REMOVE_E, Action5.CHECK]
        result = self.ti.parse_action("REASONING: removing\nACTION: remove e", valid)
        assert result == Action5.REMOVE_E

    def test_parse_action_by_number(self):
        valid = [
            Action5.PLACE_A, Action5.PLACE_B, Action5.PLACE_C,
            Action5.PLACE_D, Action5.PLACE_E, Action5.CHECK,
        ]
        # Action 4 = PLACE_D
        result = self.ti.parse_action("ACTION: 4", valid)
        assert result == Action5.PLACE_D

    def test_parse_action_numbered_up_to_11(self):
        """All 11 actions available (5 place + 5 remove + check)."""
        valid = [
            Action5.PLACE_A, Action5.PLACE_B, Action5.PLACE_C,
            Action5.PLACE_D, Action5.PLACE_E,
            Action5.REMOVE_A, Action5.REMOVE_B, Action5.REMOVE_C,
            Action5.REMOVE_D, Action5.REMOVE_E,
            Action5.CHECK,
        ]
        # Action 11 = CHECK
        result = self.ti.parse_action("ACTION: 11", valid)
        assert result == Action5.CHECK

    def test_parse_answer_abd(self):
        rule_type, objects, order = self.ti.parse_answer(
            "RULE_TYPE: conjunctive\nBLICKETS: A, B, D"
        )
        assert rule_type == CausalRuleType.CONJUNCTIVE
        assert objects == {"A", "B", "D"}
        assert order is None

    def test_parse_answer_rejects_f(self):
        """F is not a valid 5-object identifier."""
        rule_type, objects, order = self.ti.parse_answer(
            "RULE_TYPE: disjunctive\nBLICKETS: A, F"
        )
        # F should be filtered out, leaving only A
        assert rule_type == CausalRuleType.DISJUNCTIVE
        assert objects == {"A"}
        assert order is None

    def test_parse_answer_all_five(self):
        rule_type, objects, order = self.ti.parse_answer(
            "RULE_TYPE: conjunctive\nBLICKETS: A, B, C, D, E"
        )
        assert objects == {"A", "B", "C", "D", "E"}
        assert order is None

    def test_render_state(self):
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env = BlicketEnv5(causal_rule=rule, max_steps=75)
        env.reset()
        env.step(Action5.PLACE_A)
        env.step(Action5.PLACE_D)

        state_text = self.ti.render_state(env)
        assert "A" in state_text
        assert "D" in state_text
        assert "Place object E" in state_text  # E should be placeable
        assert "Remove object A" in state_text  # A is on detector

    def test_render_observation(self):
        from environment.blicket_env import Observation
        obs = Observation(
            action_taken=Action5.PLACE_D,
            action_valid=True,
            objects_on_detector={"D"},
            detector_active=False,
            detector_changed=False,
            message="You placed object D on the detector. The detector remains INACTIVE.",
        )
        text = self.ti.render_observation(obs)
        assert "object D" in text


class TestTextInterface5OrderSensitive:
    """Tests for TextInterface5 with order-sensitive mode enabled."""

    def setup_method(self):
        self.ti = TextInterface5(include_order_sensitive=True)

    def test_task_description_mentions_order_sensitive(self):
        desc = self.ti.render_task_description()
        assert "ORDER_SENSITIVE" in desc
        assert "specific sequence" in desc

    def test_answer_prompt_includes_order_field(self):
        prompt = self.ti.render_answer_prompt()
        assert "ORDER:" in prompt
        assert "order_sensitive" in prompt

    def test_parse_3obj_order_answer(self):
        rule_type, objects, order = self.ti.parse_answer(
            "RULE_TYPE: order_sensitive\nBLICKETS: A, D, C\nORDER: A, D, C"
        )
        assert rule_type == CausalRuleType.ORDER_SENSITIVE
        assert objects == {"A", "D", "C"}
        assert order == ["A", "D", "C"]

    def test_parse_4obj_order_answer(self):
        rule_type, objects, order = self.ti.parse_answer(
            "RULE_TYPE: order_sensitive\nBLICKETS: A, D, C, B\nORDER: A, D, C, B"
        )
        assert rule_type == CausalRuleType.ORDER_SENSITIVE
        assert objects == {"A", "D", "C", "B"}
        assert order == ["A", "D", "C", "B"]

    def test_conjunctive_answer_order_is_none(self):
        rule_type, objects, order = self.ti.parse_answer(
            "RULE_TYPE: conjunctive\nBLICKETS: A, B"
        )
        assert rule_type == CausalRuleType.CONJUNCTIVE
        assert objects == {"A", "B"}
        assert order is None

    def test_order_sensitive_without_order_field(self):
        """Agent says order_sensitive but forgets ORDER: line."""
        rule_type, objects, order = self.ti.parse_answer(
            "RULE_TYPE: order_sensitive\nBLICKETS: A, D, C"
        )
        assert rule_type == CausalRuleType.ORDER_SENSITIVE
        assert objects == {"A", "D", "C"}
        assert order is None

    def test_default_interface_has_no_order(self):
        """Default TextInterface5 (include_order_sensitive=False) still works."""
        ti = TextInterface5()
        desc = ti.render_task_description()
        assert "ORDER_SENSITIVE" not in desc
        prompt = ti.render_answer_prompt()
        assert "ORDER:" not in prompt

    def test_retry_prompt_includes_order(self):
        retry = self.ti.render_answer_retry_prompt()
        assert "order_sensitive" in retry
        assert "ORDER:" in retry


class TestTextInterface5OrderStochastic:
    """Tests for TextInterface5 with order-stochastic mode enabled."""

    def setup_method(self):
        self.ti = TextInterface5(include_order_stochastic=True)

    def test_task_description_mentions_order_sensitive_stochastic(self):
        desc = self.ti.render_task_description()
        assert "ORDER_SENSITIVE_STOCHASTIC" in desc
        assert "ORDER_SENSITIVE" in desc
        assert "STOCHASTIC" in desc
        assert "probabilistic" in desc

    def test_answer_prompt_includes_all_rule_types(self):
        prompt = self.ti.render_answer_prompt()
        assert "order_sensitive_stochastic" in prompt
        assert "order_sensitive" in prompt
        assert "conjunctive" in prompt
        assert "disjunctive" in prompt

    def test_parse_order_sensitive_stochastic(self):
        rule_type, objects, order = self.ti.parse_answer(
            "RULE_TYPE: order_sensitive_stochastic\nBLICKETS: A, B\nORDER: A, B"
        )
        assert rule_type == CausalRuleType.ORDER_SENSITIVE_STOCHASTIC
        assert objects == {"A", "B"}
        assert order == ["A", "B"]

    def test_parse_order_sensitive_still_works(self):
        """Existing order_sensitive parsing is not broken."""
        rule_type, objects, order = self.ti.parse_answer(
            "RULE_TYPE: order_sensitive\nBLICKETS: A, D, C\nORDER: A, D, C"
        )
        assert rule_type == CausalRuleType.ORDER_SENSITIVE
        assert objects == {"A", "D", "C"}
        assert order == ["A", "D", "C"]

    def test_parse_conjunctive_still_works(self):
        """Existing conjunctive parsing is not broken."""
        rule_type, objects, order = self.ti.parse_answer(
            "RULE_TYPE: conjunctive\nBLICKETS: A, B"
        )
        assert rule_type == CausalRuleType.CONJUNCTIVE
        assert objects == {"A", "B"}
        assert order is None

    def test_parse_disjunctive_still_works(self):
        """Existing disjunctive parsing is not broken."""
        rule_type, objects, order = self.ti.parse_answer(
            "RULE_TYPE: disjunctive\nBLICKETS: A"
        )
        assert rule_type == CausalRuleType.DISJUNCTIVE
        assert objects == {"A"}
        assert order is None

    def test_retry_prompt_includes_order_stochastic(self):
        retry = self.ti.render_answer_retry_prompt()
        assert "order_sensitive_stochastic" in retry
        assert "ORDER:" in retry

    def test_default_interface_unaffected(self):
        """Default TextInterface5 (no flags) still works without ORDER_SENSITIVE_STOCHASTIC."""
        ti = TextInterface5()
        desc = ti.render_task_description()
        assert "ORDER_SENSITIVE_STOCHASTIC" not in desc
        assert "STOCHASTIC" not in desc

    def test_parse_plain_stochastic(self):
        """Plain 'stochastic' answer parses to CausalRuleType.STOCHASTIC (audit item 3)."""
        rule_type, objects, order = self.ti.parse_answer(
            "RULE_TYPE: stochastic\nBLICKETS: A, B"
        )
        assert rule_type == CausalRuleType.STOCHASTIC
        assert objects == {"A", "B"}
        assert order is None

    def test_plain_stochastic_check_answer(self):
        """A stochastic answer evaluates correctly against a stochastic environment."""
        from environment.extended_env import ExtendedCondition, StochasticConfig
        from environment.extended_env_5obj import ExtendedBlicketEnv5

        rule = CausalRule(CausalRuleType.STOCHASTIC, {"A", "B"})
        env = ExtendedBlicketEnv5(
            causal_rule=rule,
            condition=ExtendedCondition.STOCHASTIC,
            stochastic_config=StochasticConfig(activation_probability=0.7),
            rng_seed=42,
        )
        env.reset()

        result = env.check_answer(
            rule_type=CausalRuleType.STOCHASTIC,
            required_objects={"A", "B"},
        )
        assert result["correct"] is True
        assert result["rule_type_correct"] is True
        assert result["objects_correct"] is True
