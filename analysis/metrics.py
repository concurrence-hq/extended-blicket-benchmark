"""Metrics computation for blicket pilot episodes.

Computes per-episode and aggregate metrics from saved trace files.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_traces(trace_dir: Path) -> list[dict]:
    """Load all JSON trace files from a directory."""
    traces = []
    for path in sorted(trace_dir.glob("*.json")):
        with open(path) as f:
            traces.append(json.load(f))
    return traces


def compute_episode_metrics(trace: dict) -> dict[str, Any]:
    """Compute metrics for a single episode trace."""
    result = trace["result"]
    steps = trace["steps"]

    # Basic outcomes
    metrics = {
        "condition": result["condition"],
        "agent_type": result["agent_type"],
        "episode": result["episode_number"],
        "success": result["success"],
        "answer_submitted": result["answer_submitted"],
        "steps_taken": result["steps_taken"],
        "rule_type_correct": result["rule_type_correct"],
        "objects_correct": result["objects_correct"],
        "parse_failures": result["parse_failures"],
        "input_tokens": result["total_input_tokens"],
        "output_tokens": result["total_output_tokens"],
        "trigger_tokens": result["trigger_tokens"],
        "wall_time": result["wall_time_seconds"],
    }

    # Action diversity: unique object configurations tested
    configs_seen = set()
    for step in steps:
        config = tuple(sorted(step["objects_on_detector"]))
        configs_seen.add(config)
    metrics["unique_configurations"] = len(configs_seen)

    # Action entropy: how spread out are the actions?
    action_counts = Counter(s["action"] for s in steps)
    metrics["action_entropy"] = _entropy(list(action_counts.values()))

    # Steps to first activation
    first_active = None
    for step in steps:
        if step["detector_active"]:
            first_active = step["step"]
            break
    metrics["steps_to_first_activation"] = first_active

    # Steps to answer (= total steps if answer submitted, else None)
    metrics["steps_to_answer"] = result["steps_taken"] if result["answer_submitted"] else None

    # Exploration after first success: how many steps after first activation?
    if first_active is not None:
        metrics["exploration_after_activation"] = result["steps_taken"] - first_active
    else:
        metrics["exploration_after_activation"] = None

    # CG transitions (for CG and Full agents)
    agent_trace = trace.get("agent_trace", {})
    transition_log = agent_trace.get("transition_log", [])
    metrics["cg_transitions"] = len(transition_log)
    metrics["cg_valid_transitions"] = sum(1 for t in transition_log if t.get("success"))

    # DB activations (for Full agent)
    db_log = agent_trace.get("db_activation_log", [])
    metrics["db_activations"] = len(db_log)
    fired = agent_trace.get("db_fired_behaviors", [])
    metrics["db_behaviors_fired"] = fired

    return metrics


def compute_aggregate_metrics(
    episode_metrics: list[dict],
) -> pd.DataFrame:
    """Compute aggregate metrics grouped by condition × agent."""
    df = pd.DataFrame(episode_metrics)

    if df.empty:
        return df

    grouped = df.groupby(["condition", "agent_type"]).agg(
        n_episodes=("success", "count"),
        success_rate=("success", "mean"),
        mean_steps=("steps_taken", "mean"),
        std_steps=("steps_taken", "std"),
        answer_rate=("answer_submitted", "mean"),
        rule_type_accuracy=("rule_type_correct", "mean"),
        objects_accuracy=("objects_correct", "mean"),
        mean_unique_configs=("unique_configurations", "mean"),
        mean_action_entropy=("action_entropy", "mean"),
        mean_cg_transitions=("cg_transitions", "mean"),
        mean_db_activations=("db_activations", "mean"),
        total_input_tokens=("input_tokens", "sum"),
        total_output_tokens=("output_tokens", "sum"),
        mean_wall_time=("wall_time", "mean"),
    ).reset_index()

    return grouped


def _entropy(counts: list[int]) -> float:
    """Compute Shannon entropy of a count distribution."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def process_all_traces(trace_base_dir: Path, results_dir: Path) -> None:
    """Process all trace directories and save computed metrics."""
    results_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for subdir in ["standard", "extended"]:
        trace_dir = trace_base_dir / subdir
        if not trace_dir.exists():
            continue

        traces = load_traces(trace_dir)
        for trace in traces:
            metrics = compute_episode_metrics(trace)
            all_metrics.append(metrics)

    if not all_metrics:
        print("No traces found.")
        return

    # Save per-episode metrics
    df = pd.DataFrame(all_metrics)
    df.to_csv(results_dir / "episode_metrics.csv", index=False)
    print(f"Saved {len(df)} episode metrics to episode_metrics.csv")

    # Save aggregate metrics
    agg = compute_aggregate_metrics(all_metrics)
    agg.to_csv(results_dir / "aggregate_metrics.csv", index=False)
    print(f"Saved aggregate metrics to aggregate_metrics.csv")
    print("\nAggregate results:")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    import sys

    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/traces")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/results")
    process_all_traces(base, out)
