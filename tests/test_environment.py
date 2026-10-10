"""Unit tests for the blicket detector environment."""

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
from environment.extended_env import (
    ExtendedBlicketEnv,
    ExtendedCondition,
    HiddenModeratorConfig,
    OrderConfig,
    StochasticConfig,
)


# --- CausalRule tests ---


class TestCausalRule:
    def test_conjunctive_all_present(self):
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        assert rule.is_satisfied({"A", "B"}) is True
        assert rule.is_satisfied({"A", "B", "C"}) is True

    def test_conjunctive_partial(self):
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        assert rule.is_satisfied({"A"}) is False
        assert rule.is_satisfied({"B"}) is False
        assert rule.is_satisfied({"C"}) is False

    def test_conjunctive_empty(self):
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        assert rule.is_satisfied(set()) is False

    def test_disjunctive_any_present(self):
        rule = CausalRule(CausalRuleType.DISJUNCTIVE, {"A", "B"})
        assert rule.is_satisfied({"A"}) is True
        assert rule.is_satisfied({"B"}) is True
        assert rule.is_satisfied({"A", "B"}) is True
        assert rule.is_satisfied({"A", "C"}) is True

    def test_disjunctive_none_present(self):
        rule = CausalRule(CausalRuleType.DISJUNCTIVE, {"A", "B"})
        assert rule.is_satisfied({"C"}) is False
        assert rule.is_satisfied(set()) is False

    def test_single_blicket_conjunctive(self):
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A"})
        assert rule.is_satisfied({"A"}) is True
        assert rule.is_satisfied({"B"}) is False

    def test_all_blickets_conjunctive(self):
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B", "C"})
        assert rule.is_satisfied({"A", "B"}) is False
        assert rule.is_satisfied({"A", "B", "C"}) is True

    def test_empty_required_objects(self):
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, set())
        assert rule.is_satisfied({"A"}) is False


# --- BlicketEnv tests ---


class TestBlicketEnv:
    def make_env(self, rule_type=CausalRuleType.CONJUNCTIVE, required=None):
        if required is None:
            required = {"A", "B"}
        rule = CausalRule(rule_type, required)
        return BlicketEnv(causal_rule=rule, max_steps=50)

    def test_initial_state(self):
        env = self.make_env()
        state = env.reset()
        assert state.objects_on_detector == set()
        assert state.detector_active is False
        assert state.step_count == 0

    def test_place_object(self):
        env = self.make_env()
        env.reset()
        obs = env.step(Action.PLACE_A)
        assert obs.action_valid is True
        assert "A" in obs.objects_on_detector
        assert "placed object a" in obs.message.lower()

    def test_place_duplicate(self):
        env = self.make_env()
        env.reset()
        env.step(Action.PLACE_A)
        obs = env.step(Action.PLACE_A)
        assert obs.action_valid is False
        assert "already" in obs.message.lower()

    def test_remove_object(self):
        env = self.make_env()
        env.reset()
        env.step(Action.PLACE_A)
        obs = env.step(Action.REMOVE_A)
        assert obs.action_valid is True
        assert "A" not in obs.objects_on_detector

    def test_remove_absent(self):
        env = self.make_env()
        env.reset()
        obs = env.step(Action.REMOVE_A)
        assert obs.action_valid is False
        assert "not on" in obs.message.lower()

    def test_conjunctive_activation(self):
        env = self.make_env(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env.reset()
        obs = env.step(Action.PLACE_A)
        assert obs.detector_active is False
        obs = env.step(Action.PLACE_B)
        assert obs.detector_active is True

    def test_conjunctive_deactivation(self):
        env = self.make_env(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env.reset()
        env.step(Action.PLACE_A)
        env.step(Action.PLACE_B)
        obs = env.step(Action.REMOVE_A)
        assert obs.detector_active is False
        assert obs.detector_changed is True

    def test_disjunctive_activation(self):
        env = self.make_env(CausalRuleType.DISJUNCTIVE, {"A", "B"})
        env.reset()
        obs = env.step(Action.PLACE_A)
        assert obs.detector_active is True

    def test_check_ends_episode(self):
        env = self.make_env()
        env.reset()
        env.step(Action.CHECK)
        assert env.done is True

    def test_max_steps(self):
        env = BlicketEnv(
            causal_rule=CausalRule(CausalRuleType.CONJUNCTIVE, {"A"}),
            max_steps=3,
        )
        env.reset()
        env.step(Action.PLACE_A)
        env.step(Action.REMOVE_A)
        obs = env.step(Action.PLACE_A)
        assert env.done is True
        assert "maximum steps" in obs.message.lower()

    def test_step_after_done_raises(self):
        env = self.make_env()
        env.reset()
        env.step(Action.CHECK)
        with pytest.raises(RuntimeError):
            env.step(Action.PLACE_A)

    def test_valid_actions(self):
        env = self.make_env()
        env.reset()
        valid = env.get_valid_actions()
        # Should be: place A, B, C + check
        assert Action.PLACE_A in valid
        assert Action.PLACE_B in valid
        assert Action.PLACE_C in valid
        assert Action.REMOVE_A not in valid
        assert Action.CHECK in valid

    def test_valid_actions_after_place(self):
        env = self.make_env()
        env.reset()
        env.step(Action.PLACE_A)
        valid = env.get_valid_actions()
        assert Action.PLACE_A not in valid
        assert Action.REMOVE_A in valid

    def test_check_answer_correct(self):
        env = self.make_env(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env.reset()
        result = env.check_answer(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        assert result["correct"] is True
        assert result["rule_type_correct"] is True
        assert result["objects_correct"] is True

    def test_check_answer_wrong_type(self):
        env = self.make_env(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env.reset()
        result = env.check_answer(CausalRuleType.DISJUNCTIVE, {"A", "B"})
        assert result["correct"] is False
        assert result["rule_type_correct"] is False
        assert result["objects_correct"] is True

    def test_check_answer_wrong_objects(self):
        env = self.make_env(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env.reset()
        result = env.check_answer(CausalRuleType.CONJUNCTIVE, {"A"})
        assert result["correct"] is False
        assert result["rule_type_correct"] is True
        assert result["objects_correct"] is False

    def test_detector_changed_tracking(self):
        env = self.make_env(CausalRuleType.DISJUNCTIVE, {"A"})
        env.reset()
        obs1 = env.step(Action.PLACE_A)  # inactive -> active
        assert obs1.detector_changed is True
        obs2 = env.step(Action.PLACE_B)  # stays active
        assert obs2.detector_changed is False
        obs3 = env.step(Action.REMOVE_A)  # active -> inactive
        assert obs3.detector_changed is True

    def test_action_history(self):
        env = self.make_env()
        env.reset()
        env.step(Action.PLACE_A)
        env.step(Action.PLACE_B)
        env.step(Action.REMOVE_A)
        assert env.state.action_history == [Action.PLACE_A, Action.PLACE_B, Action.REMOVE_A]


# --- ExtendedBlicketEnv tests ---


class TestExtendedBlicketEnv:
    def test_order_sensitive_correct_order(self):
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env = ExtendedBlicketEnv(
            causal_rule=rule,
            condition=ExtendedCondition.ORDER_SENSITIVE,
            order_config=OrderConfig(required_order=["A", "B"]),
        )
        env.reset()
        env.step(Action.PLACE_A)
        obs = env.step(Action.PLACE_B)
        assert obs.detector_active is True

    def test_order_sensitive_wrong_order(self):
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env = ExtendedBlicketEnv(
            causal_rule=rule,
            condition=ExtendedCondition.ORDER_SENSITIVE,
            order_config=OrderConfig(required_order=["A", "B"]),
        )
        env.reset()
        env.step(Action.PLACE_B)
        obs = env.step(Action.PLACE_A)
        assert obs.detector_active is False

    def test_stochastic_deterministic_with_seed(self):
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A"})
        # Run with seed and check we get consistent results
        results = []
        for _ in range(5):
            env = ExtendedBlicketEnv(
                causal_rule=rule,
                condition=ExtendedCondition.STOCHASTIC,
                stochastic_config=StochasticConfig(activation_probability=0.7),
                rng_seed=42,
            )
            env.reset()
            obs = env.step(Action.PLACE_A)
            results.append(obs.detector_active)
        # All runs with same seed should give same result
        assert all(r == results[0] for r in results)

    def test_stochastic_never_activates_when_rule_not_satisfied(self):
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env = ExtendedBlicketEnv(
            causal_rule=rule,
            condition=ExtendedCondition.STOCHASTIC,
            stochastic_config=StochasticConfig(activation_probability=1.0),
            rng_seed=42,
        )
        env.reset()
        obs = env.step(Action.PLACE_A)  # Only A, need A+B
        assert obs.detector_active is False

    def test_hidden_moderator_switches(self):
        initial_rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        alt_rule = CausalRule(CausalRuleType.DISJUNCTIVE, {"C"})
        env = ExtendedBlicketEnv(
            causal_rule=initial_rule,
            condition=ExtendedCondition.HIDDEN_MODERATOR,
            moderator_config=HiddenModeratorConfig(
                switch_after_n=2,
                alternate_rule=alt_rule,
            ),
        )
        env.reset()

        # First activation
        env.step(Action.PLACE_A)
        env.step(Action.PLACE_B)  # Should activate (count=1)

        # Remove and redo for second activation
        env.step(Action.REMOVE_A)
        env.step(Action.REMOVE_B)
        env.step(Action.PLACE_A)
        env.step(Action.PLACE_B)  # Should activate (count=2, triggers switch)

        # Now rule should have switched to disjunctive {C}
        env.step(Action.REMOVE_A)
        env.step(Action.REMOVE_B)
        obs = env.step(Action.PLACE_C)
        assert obs.detector_active is True

    def test_hidden_moderator_original_stops_working(self):
        initial_rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        alt_rule = CausalRule(CausalRuleType.DISJUNCTIVE, {"C"})
        env = ExtendedBlicketEnv(
            causal_rule=initial_rule,
            condition=ExtendedCondition.HIDDEN_MODERATOR,
            moderator_config=HiddenModeratorConfig(
                switch_after_n=1,
                alternate_rule=alt_rule,
            ),
        )
        env.reset()

        # Trigger switch with 1 activation
        env.step(Action.PLACE_A)
        env.step(Action.PLACE_B)  # Activates, count=1, switches

        # Original rule should no longer work by itself after clearing
        env.step(Action.REMOVE_A)
        env.step(Action.REMOVE_B)
        env.step(Action.PLACE_A)
        obs = env.step(Action.PLACE_B)
        # Under new rule (disjunctive {C}), A+B shouldn't activate
        assert obs.detector_active is False

    def test_validation_missing_config(self):
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A"})
        with pytest.raises(ValueError, match="OrderConfig"):
            ExtendedBlicketEnv(
                causal_rule=rule,
                condition=ExtendedCondition.ORDER_SENSITIVE,
            )
        with pytest.raises(ValueError, match="StochasticConfig"):
            ExtendedBlicketEnv(
                causal_rule=rule,
                condition=ExtendedCondition.STOCHASTIC,
            )
        with pytest.raises(ValueError, match="HiddenModeratorConfig"):
            ExtendedBlicketEnv(
                causal_rule=rule,
                condition=ExtendedCondition.HIDDEN_MODERATOR,
            )

    def test_order_sensitive_recovery_after_individual_testing(self):
        """Key regression test: place A, remove A, place B, remove B, place A, place B -> ACTIVE."""
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env = ExtendedBlicketEnv(
            causal_rule=rule,
            condition=ExtendedCondition.ORDER_SENSITIVE,
            order_config=OrderConfig(required_order=["A", "B"]),
        )
        env.reset()
        env.step(Action.PLACE_A)   # Test A individually
        env.step(Action.REMOVE_A)  # Remove A
        env.step(Action.PLACE_B)   # Test B individually
        env.step(Action.REMOVE_B)  # Remove B
        env.step(Action.PLACE_A)   # Place A first (correct order)
        obs = env.step(Action.PLACE_B)  # Place B second
        assert obs.detector_active is True

    def test_order_sensitive_wrong_order_after_removal(self):
        """Place A, place B (active), remove A, place A -> NOT active (B, A order after removal)."""
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env = ExtendedBlicketEnv(
            causal_rule=rule,
            condition=ExtendedCondition.ORDER_SENSITIVE,
            order_config=OrderConfig(required_order=["A", "B"]),
        )
        env.reset()
        env.step(Action.PLACE_A)
        env.step(Action.PLACE_B)   # Active (A, B correct order)
        env.step(Action.REMOVE_A)  # Now order is ["B"]
        obs = env.step(Action.PLACE_A)  # Now order is ["B", "A"] — wrong
        assert obs.detector_active is False

    def test_order_sensitive_clean_reorder(self):
        """Place B, remove B, place A, place B -> ACTIVE (clean slate after full removal)."""
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env = ExtendedBlicketEnv(
            causal_rule=rule,
            condition=ExtendedCondition.ORDER_SENSITIVE,
            order_config=OrderConfig(required_order=["A", "B"]),
        )
        env.reset()
        env.step(Action.PLACE_B)   # Wrong order start
        env.step(Action.REMOVE_B)  # Clear
        env.step(Action.PLACE_A)   # Correct first
        obs = env.step(Action.PLACE_B)  # Correct second
        assert obs.detector_active is True

    def test_order_sensitive_removal_clears_from_sequence(self):
        """Place A, place B, remove A -> internal order is ['B']."""
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env = ExtendedBlicketEnv(
            causal_rule=rule,
            condition=ExtendedCondition.ORDER_SENSITIVE,
            order_config=OrderConfig(required_order=["A", "B"]),
        )
        env.reset()
        env.step(Action.PLACE_A)
        env.step(Action.PLACE_B)
        env.step(Action.REMOVE_A)
        assert env._placement_order == ["B"]

    def test_reset_clears_extended_state(self):
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env = ExtendedBlicketEnv(
            causal_rule=rule,
            condition=ExtendedCondition.ORDER_SENSITIVE,
            order_config=OrderConfig(required_order=["A", "B"]),
        )
        env.reset()
        env.step(Action.PLACE_B)  # Wrong order
        env.reset()
        # After reset, order tracking should be fresh
        env.step(Action.PLACE_A)
        obs = env.step(Action.PLACE_B)
        assert obs.detector_active is True
