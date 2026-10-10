"""Analysis metrics layer tests (audit item 6).

Tests compute_episode_metrics and compute_aggregate_metrics with
synthetic trace data (no real trace files loaded).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.metrics import compute_episode_metrics, compute_aggregate_metrics


def _make_trace(
    *,
    condition="hidden_moderator",
    agent_type="base_llm",
    episode=0,
    success=True,
    answer_submitted=True,
    steps_taken=10,
    rule_type_correct=True,
    objects_correct=True,
    parse_failures=0,
    total_input_tokens=500,
    total_output_tokens=200,
    trigger_tokens=0,
    wall_time_seconds=5.0,
    steps=None,
    agent_trace=None,
    env_info=None,
):
    """Build a synthetic trace dict matching the real trace format."""
    if steps is None:
        steps = [
            {
                "step": i,
                "action": "place_a" if i % 2 == 0 else "place_b",
                "action_valid": True,
                "objects_on_detector": ["A"] if i % 2 == 0 else ["A", "B"],
                "detector_active": i % 2 == 1,
                "detector_changed": True,
                "observation_text": "test obs",
                "cg_state": "",
                "db_notifications": [],
                "partial_feedback_delivered": False,
                "gradual_feedback_object": None,
            }
            for i in range(steps_taken)
        ]
    if agent_trace is None:
        agent_trace = {"agent_type": agent_type}
    if env_info is None:
        env_info = {"rule_type": "conjunctive", "required_objects": ["A", "B"]}

    return {
        "result": {
            "condition": condition,
            "agent_type": agent_type,
            "episode_number": episode,
            "success": success,
            "answer_submitted": answer_submitted,
            "steps_taken": steps_taken,
            "rule_type_correct": rule_type_correct,
            "objects_correct": objects_correct,
            "parse_failures": parse_failures,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "trigger_tokens": trigger_tokens,
            "wall_time_seconds": wall_time_seconds,
        },
        "steps": steps,
        "agent_trace": agent_trace,
        "env_info": env_info,
    }


# ── A. Episode metric computation ───────────────────────────────────────


class TestEpisodeMetrics:
    def test_successful_episode(self):
        trace = _make_trace(success=True, rule_type_correct=True, objects_correct=True)
        m = compute_episode_metrics(trace)
        assert m["success"] is True
        assert m["rule_type_correct"] is True
        assert m["objects_correct"] is True
        assert m["steps_taken"] == 10
        assert m["answer_submitted"] is True

    def test_failed_episode(self):
        trace = _make_trace(success=False, rule_type_correct=False, objects_correct=False)
        m = compute_episode_metrics(trace)
        assert m["success"] is False

    def test_unique_configurations_counted(self):
        steps = [
            {"step": 0, "action": "place_a", "action_valid": True,
             "objects_on_detector": ["A"], "detector_active": False,
             "detector_changed": False, "observation_text": "", "cg_state": "",
             "db_notifications": [], "partial_feedback_delivered": False,
             "gradual_feedback_object": None},
            {"step": 1, "action": "place_b", "action_valid": True,
             "objects_on_detector": ["A", "B"], "detector_active": True,
             "detector_changed": True, "observation_text": "", "cg_state": "",
             "db_notifications": [], "partial_feedback_delivered": False,
             "gradual_feedback_object": None},
            {"step": 2, "action": "remove_b", "action_valid": True,
             "objects_on_detector": ["A"], "detector_active": False,
             "detector_changed": True, "observation_text": "", "cg_state": "",
             "db_notifications": [], "partial_feedback_delivered": False,
             "gradual_feedback_object": None},
        ]
        trace = _make_trace(steps_taken=3, steps=steps)
        m = compute_episode_metrics(trace)
        # Two unique configs: ("A",) and ("A","B")
        assert m["unique_configurations"] == 2

    def test_db_firing_metrics_extracted(self):
        agent_trace = {
            "agent_type": "full_cg_db",
            "db_activation_log": [
                {"step": 5, "notification": "[DB] stagnation", "fired_behaviors": ["exploration_stagnation"]},
                {"step": 8, "notification": "[DB] hypothesis", "fired_behaviors": ["exploration_stagnation", "hypothesis_failure"]},
            ],
            "db_fired_behaviors": ["exploration_stagnation", "hypothesis_failure"],
            "transition_log": [],
        }
        trace = _make_trace(agent_type="full_cg_db", agent_trace=agent_trace)
        m = compute_episode_metrics(trace)
        assert m["db_activations"] == 2
        assert "exploration_stagnation" in m["db_behaviors_fired"]
        assert "hypothesis_failure" in m["db_behaviors_fired"]

    def test_cg_transitions_counted(self):
        agent_trace = {
            "agent_type": "cg_only",
            "transition_log": [
                {"step": 3, "from": "INITIAL_EXPLORATION", "to": "COMBINATION_TESTING", "success": True},
                {"step": 7, "from": "COMBINATION_TESTING", "to": "NONEXISTENT", "success": False},
            ],
        }
        trace = _make_trace(agent_type="cg_only", agent_trace=agent_trace)
        m = compute_episode_metrics(trace)
        assert m["cg_transitions"] == 2
        assert m["cg_valid_transitions"] == 1


# ── B. Aggregate metrics ────────────────────────────────────────────────


class TestAggregateMetrics:
    def test_accuracy_computation(self):
        episodes = [
            compute_episode_metrics(_make_trace(condition="hm", agent_type="base_llm", episode=0, success=True)),
            compute_episode_metrics(_make_trace(condition="hm", agent_type="base_llm", episode=1, success=False)),
            compute_episode_metrics(_make_trace(condition="hm", agent_type="base_llm", episode=2, success=True)),
        ]
        agg = compute_aggregate_metrics(episodes)
        row = agg[(agg["condition"] == "hm") & (agg["agent_type"] == "base_llm")].iloc[0]
        assert row["n_episodes"] == 3
        assert abs(row["success_rate"] - 2 / 3) < 0.001

    def test_multi_condition_grouping(self):
        episodes = [
            compute_episode_metrics(_make_trace(condition="hm", agent_type="base_llm", episode=0, success=True)),
            compute_episode_metrics(_make_trace(condition="hm", agent_type="cg_only", episode=0, success=False)),
            compute_episode_metrics(_make_trace(condition="order", agent_type="base_llm", episode=0, success=True)),
        ]
        agg = compute_aggregate_metrics(episodes)
        assert len(agg) == 3  # 3 unique condition × agent_type groups

    def test_empty_input(self):
        agg = compute_aggregate_metrics([])
        assert len(agg) == 0


# ── C. Trace format robustness ──────────────────────────────────────────


class TestTraceRobustness:
    def test_missing_db_activation_log(self):
        """base_llm traces have no db_activation_log — should not crash."""
        agent_trace = {"agent_type": "base_llm"}
        trace = _make_trace(agent_trace=agent_trace)
        m = compute_episode_metrics(trace)
        assert m["db_activations"] == 0
        assert m["db_behaviors_fired"] == []

    def test_missing_transition_log(self):
        """base_llm traces have no transition_log — should not crash."""
        agent_trace = {"agent_type": "base_llm"}
        trace = _make_trace(agent_trace=agent_trace)
        m = compute_episode_metrics(trace)
        assert m["cg_transitions"] == 0
        assert m["cg_valid_transitions"] == 0
