"""Unit tests for the 5-object blicket detector environment."""

import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.blicket_env import CausalRule, CausalRuleType
from environment.blicket_env_5obj import (
    Action5,
    BlicketEnv5,
    OBJECT_NAMES_5,
    PLACE_ACTIONS_5,
    REMOVE_ACTIONS_5,
)
from environment.extended_env import (
    ExtendedCondition,
    HiddenModeratorConfig,
    OrderConfig,
    StochasticConfig,
)
from environment.extended_env_5obj import ExtendedBlicketEnv5


class TestBlicketEnv5:
    def make_env(self, rule_type=CausalRuleType.CONJUNCTIVE, required=None):
        if required is None:
            required = {"A", "B"}
        rule = CausalRule(rule_type, required)
        return BlicketEnv5(causal_rule=rule, max_steps=75)

    def test_five_objects_available(self):
        assert OBJECT_NAMES_5 == ["A", "B", "C", "D", "E"]
        assert len(PLACE_ACTIONS_5) == 5
        assert len(REMOVE_ACTIONS_5) == 5

    def test_initial_state(self):
        env = self.make_env()
        state = env.reset()
        assert state.objects_on_detector == set()
        assert state.detector_active is False
        assert state.step_count == 0

    def test_place_all_five_objects(self):
        env = self.make_env()
        env.reset()
        for action in [Action5.PLACE_A, Action5.PLACE_B, Action5.PLACE_C, Action5.PLACE_D, Action5.PLACE_E]:
            obs = env.step(action)
            assert obs.action_valid is True
        assert env.state.objects_on_detector == {"A", "B", "C", "D", "E"}

    def test_remove_all_five_objects(self):
        env = self.make_env()
        env.reset()
        for action in [Action5.PLACE_A, Action5.PLACE_B, Action5.PLACE_C, Action5.PLACE_D, Action5.PLACE_E]:
            env.step(action)
        for action in [Action5.REMOVE_A, Action5.REMOVE_B, Action5.REMOVE_C, Action5.REMOVE_D, Action5.REMOVE_E]:
            obs = env.step(action)
            assert obs.action_valid is True
        assert env.state.objects_on_detector == set()

    def test_valid_actions_initial(self):
        env = self.make_env()
        env.reset()
        valid = env.get_valid_actions()
        # Should be: place A-E + check = 6 actions
        assert len(valid) == 6
        assert Action5.PLACE_D in valid
        assert Action5.PLACE_E in valid
        assert Action5.REMOVE_D not in valid
        assert Action5.CHECK in valid

    def test_conjunctive_ab_activation(self):
        """Conjunctive {A,B}: C, D, E are distractors."""
        env = self.make_env(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env.reset()
        obs = env.step(Action5.PLACE_A)
        assert obs.detector_active is False
        obs = env.step(Action5.PLACE_B)
        assert obs.detector_active is True

    def test_conjunctive_distractors_dont_activate(self):
        """C, D, E alone don't activate conjunctive {A,B}."""
        env = self.make_env(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env.reset()
        env.step(Action5.PLACE_C)
        env.step(Action5.PLACE_D)
        obs = env.step(Action5.PLACE_E)
        assert obs.detector_active is False

    def test_conjunctive_requires_both(self):
        """{A} alone doesn't activate, {A,B} does, {A,B,C,D,E} does."""
        env = self.make_env(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env.reset()
        obs = env.step(Action5.PLACE_A)
        assert obs.detector_active is False
        obs = env.step(Action5.PLACE_B)
        assert obs.detector_active is True
        obs = env.step(Action5.PLACE_C)
        assert obs.detector_active is True  # Extra objects don't deactivate
        obs = env.step(Action5.PLACE_D)
        assert obs.detector_active is True
        obs = env.step(Action5.PLACE_E)
        assert obs.detector_active is True

    def test_disjunctive_activation(self):
        env = self.make_env(CausalRuleType.DISJUNCTIVE, {"A", "B"})
        env.reset()
        obs = env.step(Action5.PLACE_A)
        assert obs.detector_active is True

    def test_disjunctive_non_blickets(self):
        env = self.make_env(CausalRuleType.DISJUNCTIVE, {"A", "B"})
        env.reset()
        env.step(Action5.PLACE_C)
        env.step(Action5.PLACE_D)
        obs = env.step(Action5.PLACE_E)
        assert obs.detector_active is False

    def test_check_ends_episode(self):
        env = self.make_env()
        env.reset()
        env.step(Action5.CHECK)
        assert env.done is True

    def test_max_steps_75(self):
        env = BlicketEnv5(
            causal_rule=CausalRule(CausalRuleType.CONJUNCTIVE, {"A"}),
            max_steps=3,
        )
        env.reset()
        env.step(Action5.PLACE_A)
        env.step(Action5.REMOVE_A)
        obs = env.step(Action5.PLACE_A)
        assert env.done is True
        assert "maximum steps" in obs.message.lower()

    def test_check_answer_correct(self):
        env = self.make_env(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env.reset()
        result = env.check_answer(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        assert result["correct"] is True

    def test_check_answer_wrong(self):
        env = self.make_env(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env.reset()
        result = env.check_answer(CausalRuleType.CONJUNCTIVE, {"A", "D"})
        assert result["correct"] is False
        assert result["objects_correct"] is False

    def test_get_info_includes_num_objects(self):
        env = self.make_env()
        env.reset()
        info = env.get_info()
        assert info["num_objects"] == 5

    def test_place_duplicate(self):
        env = self.make_env()
        env.reset()
        env.step(Action5.PLACE_D)
        obs = env.step(Action5.PLACE_D)
        assert obs.action_valid is False

    def test_remove_absent(self):
        env = self.make_env()
        env.reset()
        obs = env.step(Action5.REMOVE_E)
        assert obs.action_valid is False


class TestExtendedBlicketEnv5:
    def test_order_sensitive_correct_order(self):
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env = ExtendedBlicketEnv5(
            causal_rule=rule,
            condition=ExtendedCondition.ORDER_SENSITIVE,
            order_config=OrderConfig(required_order=["A", "B"]),
        )
        env.reset()
        env.step(Action5.PLACE_A)
        obs = env.step(Action5.PLACE_B)
        assert obs.detector_active is True

    def test_order_sensitive_wrong_order(self):
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env = ExtendedBlicketEnv5(
            causal_rule=rule,
            condition=ExtendedCondition.ORDER_SENSITIVE,
            order_config=OrderConfig(required_order=["A", "B"]),
        )
        env.reset()
        env.step(Action5.PLACE_B)
        obs = env.step(Action5.PLACE_A)
        assert obs.detector_active is False

    def test_order_sensitive_with_distractors(self):
        """Placing distractors between A and B doesn't affect order check."""
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env = ExtendedBlicketEnv5(
            causal_rule=rule,
            condition=ExtendedCondition.ORDER_SENSITIVE,
            order_config=OrderConfig(required_order=["A", "B"]),
        )
        env.reset()
        env.step(Action5.PLACE_A)
        env.step(Action5.PLACE_D)  # Distractor
        env.step(Action5.PLACE_E)  # Distractor
        obs = env.step(Action5.PLACE_B)
        assert obs.detector_active is True

    def test_stochastic_with_seed(self):
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A"})
        results = []
        for _ in range(5):
            env = ExtendedBlicketEnv5(
                causal_rule=rule,
                condition=ExtendedCondition.STOCHASTIC,
                stochastic_config=StochasticConfig(activation_probability=0.7),
                rng_seed=42,
            )
            env.reset()
            obs = env.step(Action5.PLACE_A)
            results.append(obs.detector_active)
        assert all(r == results[0] for r in results)

    def test_hidden_moderator_switches(self):
        initial_rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        alt_rule = CausalRule(CausalRuleType.DISJUNCTIVE, {"C"})
        env = ExtendedBlicketEnv5(
            causal_rule=initial_rule,
            condition=ExtendedCondition.HIDDEN_MODERATOR,
            moderator_config=HiddenModeratorConfig(
                switch_after_n=2,
                alternate_rule=alt_rule,
            ),
        )
        env.reset()

        # First activation
        env.step(Action5.PLACE_A)
        env.step(Action5.PLACE_B)  # count=1

        env.step(Action5.REMOVE_A)
        env.step(Action5.REMOVE_B)
        env.step(Action5.PLACE_A)
        env.step(Action5.PLACE_B)  # count=2, triggers switch

        # Now rule is disjunctive {C}
        env.step(Action5.REMOVE_A)
        env.step(Action5.REMOVE_B)
        obs = env.step(Action5.PLACE_C)
        assert obs.detector_active is True

    def test_hidden_moderator_original_stops_working(self):
        initial_rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        alt_rule = CausalRule(CausalRuleType.DISJUNCTIVE, {"C"})
        env = ExtendedBlicketEnv5(
            causal_rule=initial_rule,
            condition=ExtendedCondition.HIDDEN_MODERATOR,
            moderator_config=HiddenModeratorConfig(
                switch_after_n=1,
                alternate_rule=alt_rule,
            ),
        )
        env.reset()

        env.step(Action5.PLACE_A)
        env.step(Action5.PLACE_B)  # Activates, count=1, switches

        env.step(Action5.REMOVE_A)
        env.step(Action5.REMOVE_B)
        env.step(Action5.PLACE_A)
        obs = env.step(Action5.PLACE_B)
        # Under new rule (disjunctive {C}), A+B shouldn't activate
        assert obs.detector_active is False

    def test_order_sensitive_recovery_after_individual_testing(self):
        """5-object version of the key regression test."""
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env = ExtendedBlicketEnv5(
            causal_rule=rule,
            condition=ExtendedCondition.ORDER_SENSITIVE,
            order_config=OrderConfig(required_order=["A", "B"]),
        )
        env.reset()
        env.step(Action5.PLACE_A)   # Test A
        env.step(Action5.REMOVE_A)
        env.step(Action5.PLACE_B)   # Test B
        env.step(Action5.REMOVE_B)
        env.step(Action5.PLACE_C)   # Test C (distractor)
        env.step(Action5.REMOVE_C)
        env.step(Action5.PLACE_D)   # Test D (distractor)
        env.step(Action5.REMOVE_D)
        env.step(Action5.PLACE_E)   # Test E (distractor)
        env.step(Action5.REMOVE_E)
        env.step(Action5.PLACE_A)   # Place A first (correct order)
        obs = env.step(Action5.PLACE_B)  # Place B second
        assert obs.detector_active is True

    def test_order_sensitive_removal_clears_from_sequence(self):
        """Place A, place B, remove A -> internal order is ['B']."""
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env = ExtendedBlicketEnv5(
            causal_rule=rule,
            condition=ExtendedCondition.ORDER_SENSITIVE,
            order_config=OrderConfig(required_order=["A", "B"]),
        )
        env.reset()
        env.step(Action5.PLACE_A)
        env.step(Action5.PLACE_B)
        env.step(Action5.REMOVE_A)
        assert env._placement_order == ["B"]

    def test_reset_clears_extended_state(self):
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env = ExtendedBlicketEnv5(
            causal_rule=rule,
            condition=ExtendedCondition.ORDER_SENSITIVE,
            order_config=OrderConfig(required_order=["A", "B"]),
        )
        env.reset()
        env.step(Action5.PLACE_B)
        env.reset()
        env.step(Action5.PLACE_A)
        obs = env.step(Action5.PLACE_B)
        assert obs.detector_active is True

    def test_validation_missing_config(self):
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A"})
        with pytest.raises(ValueError, match="OrderConfig"):
            ExtendedBlicketEnv5(
                causal_rule=rule,
                condition=ExtendedCondition.ORDER_SENSITIVE,
            )


class TestOrderSensitiveCheckAnswer:
    """Tests for check_answer() with ORDER_SENSITIVE rule type."""

    def _make_env(self, required_objects, required_order):
        rule = CausalRule(CausalRuleType.ORDER_SENSITIVE, set(required_objects))
        return ExtendedBlicketEnv5(
            causal_rule=rule,
            condition=ExtendedCondition.ORDER_SENSITIVE,
            order_config=OrderConfig(required_order=required_order),
        )

    def test_3obj_correct(self):
        """Rule + objects + order all match."""
        env = self._make_env({"A", "D", "C"}, ["A", "D", "C"])
        env.reset()
        result = env.check_answer(
            CausalRuleType.ORDER_SENSITIVE, {"A", "D", "C"}, ["A", "D", "C"]
        )
        assert result["correct"] is True
        assert result["order_correct"] is True
        assert result["true_order"] == ["A", "D", "C"]
        assert result["guessed_order"] == ["A", "D", "C"]

    def test_3obj_wrong_order(self):
        """Objects correct but order wrong."""
        env = self._make_env({"A", "D", "C"}, ["A", "D", "C"])
        env.reset()
        result = env.check_answer(
            CausalRuleType.ORDER_SENSITIVE, {"A", "D", "C"}, ["A", "C", "D"]
        )
        assert result["correct"] is False
        assert result["objects_correct"] is True
        assert result["order_correct"] is False

    def test_3obj_no_order_guessed(self):
        """Agent guessed order_sensitive but didn't provide ORDER."""
        env = self._make_env({"A", "D", "C"}, ["A", "D", "C"])
        env.reset()
        result = env.check_answer(
            CausalRuleType.ORDER_SENSITIVE, {"A", "D", "C"}, None
        )
        assert result["correct"] is False
        assert result["order_correct"] is False

    def test_4obj_correct(self):
        """4-object order fully correct."""
        env = self._make_env({"A", "D", "C", "B"}, ["A", "D", "C", "B"])
        env.reset()
        result = env.check_answer(
            CausalRuleType.ORDER_SENSITIVE, {"A", "D", "C", "B"}, ["A", "D", "C", "B"]
        )
        assert result["correct"] is True
        assert result["order_correct"] is True

    def test_4obj_partial_order(self):
        """4-object: first 3 correct but last element wrong position."""
        env = self._make_env({"A", "D", "C", "B"}, ["A", "D", "C", "B"])
        env.reset()
        result = env.check_answer(
            CausalRuleType.ORDER_SENSITIVE, {"A", "D", "C", "B"}, ["A", "D", "B", "C"]
        )
        assert result["correct"] is False
        assert result["objects_correct"] is True
        assert result["order_correct"] is False

    def test_order_sensitive_activation_conjunctive_like(self):
        """ORDER_SENSITIVE in CausalRule.is_satisfied() checks all objects present."""
        rule = CausalRule(CausalRuleType.ORDER_SENSITIVE, {"A", "D", "C"})
        assert rule.is_satisfied({"A", "D", "C"}) is True
        assert rule.is_satisfied({"A", "D", "C", "E"}) is True
        assert rule.is_satisfied({"A", "D"}) is False
        assert rule.is_satisfied({"B", "E"}) is False

    def test_non_order_env_ignores_guessed_order(self):
        """Standard BlicketEnv5 ignores guessed_order param."""
        rule = CausalRule(CausalRuleType.CONJUNCTIVE, {"A", "B"})
        env = BlicketEnv5(causal_rule=rule)
        env.reset()
        result = env.check_answer(
            CausalRuleType.CONJUNCTIVE, {"A", "B"}, ["A", "B"]
        )
        assert result["correct"] is True
        assert "order_correct" not in result


class TestFourObjectOrderActivationABEC:
    """Tests for 4-object order-sensitive activation with required order [A, B, E, C]."""

    def _make_env(self) -> ExtendedBlicketEnv5:
        rule = CausalRule(CausalRuleType.ORDER_SENSITIVE, {"A", "B", "E", "C"})
        return ExtendedBlicketEnv5(
            causal_rule=rule,
            condition=ExtendedCondition.ORDER_SENSITIVE,
            order_config=OrderConfig(required_order=["A", "B", "E", "C"]),
        )

    def test_correct_order_abec_activates(self):
        """[A, B, E, C] in correct order activates detector."""
        env = self._make_env()
        env.reset()
        env.step(Action5.PLACE_A)
        env.step(Action5.PLACE_B)
        env.step(Action5.PLACE_E)
        obs = env.step(Action5.PLACE_C)
        assert obs.detector_active is True

    def test_wrong_order_abec_does_not_activate(self):
        """[C, E, B, A] — completely reversed — does not activate."""
        env = self._make_env()
        env.reset()
        env.step(Action5.PLACE_C)
        env.step(Action5.PLACE_E)
        env.step(Action5.PLACE_B)
        obs = env.step(Action5.PLACE_A)
        assert obs.detector_active is False

    def test_partial_correct_swapped_tail(self):
        """[A, B, C, E] — last two swapped — does not activate."""
        env = self._make_env()
        env.reset()
        env.step(Action5.PLACE_A)
        env.step(Action5.PLACE_B)
        env.step(Action5.PLACE_C)
        obs = env.step(Action5.PLACE_E)
        assert obs.detector_active is False

    def test_correct_order_with_distractor_d(self):
        """A, D, B, E, C — distractor D interleaved — still activates."""
        env = self._make_env()
        env.reset()
        env.step(Action5.PLACE_A)
        env.step(Action5.PLACE_D)  # Distractor
        env.step(Action5.PLACE_B)
        env.step(Action5.PLACE_E)
        obs = env.step(Action5.PLACE_C)
        assert obs.detector_active is True

    def test_recovery_after_removal(self):
        """Wrong order, clear, correct order — activates."""
        env = self._make_env()
        env.reset()
        # Wrong order
        env.step(Action5.PLACE_C)
        env.step(Action5.PLACE_B)
        obs = env.step(Action5.PLACE_A)
        assert obs.detector_active is False
        # Clear
        env.step(Action5.REMOVE_C)
        env.step(Action5.REMOVE_B)
        env.step(Action5.REMOVE_A)
        # Correct order
        env.step(Action5.PLACE_A)
        env.step(Action5.PLACE_B)
        env.step(Action5.PLACE_E)
        obs = env.step(Action5.PLACE_C)
        assert obs.detector_active is True

    def test_check_answer_abec_correct(self):
        """check_answer with correct [A, B, E, C] order."""
        env = self._make_env()
        env.reset()
        result = env.check_answer(
            CausalRuleType.ORDER_SENSITIVE, {"A", "B", "E", "C"}, ["A", "B", "E", "C"]
        )
        assert result["correct"] is True
        assert result["order_correct"] is True

    def test_check_answer_abec_wrong_order(self):
        """check_answer with wrong order [A, E, B, C] — objects correct, order wrong."""
        env = self._make_env()
        env.reset()
        result = env.check_answer(
            CausalRuleType.ORDER_SENSITIVE, {"A", "B", "E", "C"}, ["A", "E", "B", "C"]
        )
        assert result["correct"] is False
        assert result["objects_correct"] is True
        assert result["order_correct"] is False


class TestOrderStochastic:
    """Tests for ORDER_STOCHASTIC combined condition activation."""

    def _make_env(self, seed=42):
        rule = CausalRule(CausalRuleType.ORDER_SENSITIVE_STOCHASTIC, {"A", "B"})
        return ExtendedBlicketEnv5(
            causal_rule=rule,
            condition=ExtendedCondition.ORDER_STOCHASTIC,
            max_steps=75,
            order_config=OrderConfig(required_order=["A", "B"]),
            stochastic_config=StochasticConfig(activation_probability=0.7),
            rng_seed=seed,
        )

    def test_wrong_order_deterministic_failure(self):
        """Wrong order always fails regardless of stochastic roll."""
        for seed in range(10):
            env = self._make_env(seed=seed)
            env.reset()
            env.step(Action5.PLACE_B)
            obs = env.step(Action5.PLACE_A)
            assert obs.detector_active is False

    def test_wrong_objects_deterministic_failure(self):
        """Non-blicket objects never activate."""
        env = self._make_env()
        env.reset()
        env.step(Action5.PLACE_C)
        env.step(Action5.PLACE_D)
        obs = env.step(Action5.PLACE_E)
        assert obs.detector_active is False

    def test_single_object_inactive(self):
        """A single blicket is insufficient."""
        env = self._make_env()
        env.reset()
        obs = env.step(Action5.PLACE_A)
        assert obs.detector_active is False

    def test_correct_order_seed_reproducibility(self):
        """Same seed produces same stochastic result."""
        results = []
        for _ in range(5):
            env = self._make_env(seed=42)
            env.reset()
            env.step(Action5.PLACE_A)
            obs = env.step(Action5.PLACE_B)
            results.append(obs.detector_active)
        assert all(r == results[0] for r in results)

    def test_correct_order_with_distractor(self):
        """Distractor between A and B doesn't break order check."""
        env = self._make_env(seed=42)
        env.reset()
        env.step(Action5.PLACE_A)
        env.step(Action5.PLACE_D)  # Distractor
        obs = env.step(Action5.PLACE_B)
        # Result is stochastic but should be reproducible with seed
        env2 = self._make_env(seed=42)
        env2.reset()
        env2.step(Action5.PLACE_A)
        env2.step(Action5.PLACE_D)
        obs2 = env2.step(Action5.PLACE_B)
        assert obs.detector_active == obs2.detector_active

    def test_recovery_after_removal(self):
        """Remove and re-place in correct order recovers opportunity."""
        env = self._make_env(seed=100)
        env.reset()
        # Wrong order
        env.step(Action5.PLACE_B)
        env.step(Action5.PLACE_A)
        # Clear
        env.step(Action5.REMOVE_B)
        env.step(Action5.REMOVE_A)
        # Correct order — stochastic result
        env.step(Action5.PLACE_A)
        obs = env.step(Action5.PLACE_B)
        # Just verify no crash; result is stochastic
        assert isinstance(obs.detector_active, bool)

    def test_statistical_activation_rate(self):
        """Over 1000 trials with correct order, activation rate ~ 0.7 (95% CI: 0.65-0.75)."""
        activations = 0
        trials = 1000
        for seed in range(trials):
            env = self._make_env(seed=seed + 10000)
            env.reset()
            env.step(Action5.PLACE_A)
            obs = env.step(Action5.PLACE_B)
            if obs.detector_active:
                activations += 1
        rate = activations / trials
        assert 0.65 <= rate <= 0.75, f"Activation rate {rate:.3f} outside 95% CI [0.65, 0.75]"

    def test_validation_requires_both_configs(self):
        """ORDER_STOCHASTIC requires both OrderConfig and StochasticConfig."""
        rule = CausalRule(CausalRuleType.ORDER_SENSITIVE_STOCHASTIC, {"A", "B"})
        with pytest.raises(ValueError, match="OrderConfig"):
            ExtendedBlicketEnv5(
                causal_rule=rule,
                condition=ExtendedCondition.ORDER_STOCHASTIC,
                stochastic_config=StochasticConfig(activation_probability=0.7),
            )
        with pytest.raises(ValueError, match="StochasticConfig"):
            ExtendedBlicketEnv5(
                causal_rule=rule,
                condition=ExtendedCondition.ORDER_STOCHASTIC,
                order_config=OrderConfig(required_order=["A", "B"]),
            )

    def test_get_info_includes_combined_fields(self):
        """get_info() returns both required_order and activation_probability."""
        env = self._make_env()
        env.reset()
        info = env.get_info()
        assert info["condition"] == "order_stochastic"
        assert info["required_order"] == ["A", "B"]
        assert info["activation_probability"] == 0.7


class TestOrderStochasticCheckAnswer:
    """Tests for check_answer() with ORDER_SENSITIVE_STOCHASTIC rule type."""

    def _make_env(self):
        rule = CausalRule(CausalRuleType.ORDER_SENSITIVE_STOCHASTIC, {"A", "B"})
        return ExtendedBlicketEnv5(
            causal_rule=rule,
            condition=ExtendedCondition.ORDER_STOCHASTIC,
            order_config=OrderConfig(required_order=["A", "B"]),
            stochastic_config=StochasticConfig(activation_probability=0.7),
        )

    def test_fully_correct(self):
        """Correct rule type + objects + order."""
        env = self._make_env()
        env.reset()
        result = env.check_answer(
            CausalRuleType.ORDER_SENSITIVE_STOCHASTIC, {"A", "B"}, ["A", "B"]
        )
        assert result["correct"] is True
        assert result["rule_type_correct"] is True
        assert result["objects_correct"] is True
        assert result["order_correct"] is True

    def test_correct_objects_wrong_order(self):
        """Correct objects but wrong order."""
        env = self._make_env()
        env.reset()
        result = env.check_answer(
            CausalRuleType.ORDER_SENSITIVE_STOCHASTIC, {"A", "B"}, ["B", "A"]
        )
        assert result["correct"] is False
        assert result["objects_correct"] is True
        assert result["order_correct"] is False

    def test_wrong_rule_type(self):
        """Agent guesses ORDER_SENSITIVE instead of ORDER_SENSITIVE_STOCHASTIC."""
        env = self._make_env()
        env.reset()
        result = env.check_answer(
            CausalRuleType.ORDER_SENSITIVE, {"A", "B"}, ["A", "B"]
        )
        assert result["correct"] is False
        assert result["rule_type_correct"] is False

    def test_no_order_guessed(self):
        """Agent provides no order."""
        env = self._make_env()
        env.reset()
        result = env.check_answer(
            CausalRuleType.ORDER_SENSITIVE_STOCHASTIC, {"A", "B"}, None
        )
        assert result["correct"] is False
        assert result["order_correct"] is False
