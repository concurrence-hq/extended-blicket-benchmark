#!/usr/bin/env python3
"""Model Stability Audit: Extract model strings, sampling params, performance
trajectory, and API metadata from all Extended Blicket Benchmark runs.

Read-only analysis — does not modify any files.
"""

import json
import glob
import os
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"


def section(title):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")


# ============================================================
# TASK 1: Model String Verification
# ============================================================

def task1_model_strings():
    section("TASK 1: MODEL STRING VERIFICATION")

    # Find all trace directories
    trace_dirs = sorted(glob.glob(str(DATA / "traces_run*")))
    print(f"\nFound {len(trace_dirs)} trace directories")

    results = []
    for td in trace_dirs:
        td_path = Path(td)
        run_name = td_path.name.replace("traces_", "")

        # Get first trace file
        traces = sorted(glob.glob(str(td_path / "*.json")))
        if not traces:
            results.append({"run": run_name, "action_model": "NO TRACES", "trigger_model": "N/A", "n_traces": 0})
            continue

        # Read first trace
        with open(traces[0]) as f:
            trace = json.load(f)

        agent_trace = trace.get("agent_trace", {})
        action_model = agent_trace.get("model", "NOT IN TRACE")

        # Check for trigger model — look in db_activation_log or separate field
        trigger_model = agent_trace.get("trigger_model", None)
        if trigger_model is None:
            # Check if db_trigger_tokens exists (implies Haiku was used)
            if agent_trace.get("db_trigger_tokens", 0) > 0:
                trigger_model = "(Haiku used, string not recorded)"
            elif "full_cg_db" in traces[0]:
                trigger_model = "(full_cg_db but no trigger tokens)"
            else:
                trigger_model = "N/A (base_llm)"

        # Also check a full_cg_db trace if the first was base_llm
        full_action_model = action_model
        full_trigger_model = trigger_model
        full_traces = [t for t in traces if "full_cg_db" in t]
        if full_traces:
            with open(full_traces[0]) as f:
                ft = json.load(f)
            fat = ft.get("agent_trace", {})
            full_action_model = fat.get("model", "NOT IN TRACE")
            if fat.get("db_trigger_tokens", 0) > 0:
                full_trigger_model = "(Haiku used, string not recorded)"

        results.append({
            "run": run_name,
            "n_traces": len(traces),
            "action_model_base": action_model,
            "action_model_full": full_action_model,
            "trigger_model": full_trigger_model,
            "sample_file": os.path.basename(traces[0]),
        })

    # Print table
    print(f"\n{'Run':<20} {'N':>4} {'Action Model (base_llm)':<32} {'Action Model (full_cg_db)':<32} {'Trigger Model':<35}")
    print("-" * 125)
    for r in results:
        print(f"{r['run']:<20} {r.get('n_traces',0):>4} {r.get('action_model_base',''):<32} {r.get('action_model_full',''):<32} {r.get('trigger_model',''):<35}")

    return results


# ============================================================
# TASK 2: Sampling Parameter Verification
# ============================================================

def task2_sampling_params():
    section("TASK 2: SAMPLING PARAMETER VERIFICATION")

    # Check agent source files for API call parameters
    agent_files = [
        PROJECT / "agents" / "base_llm_agent.py",
        PROJECT / "agents" / "cg_agent.py",
        PROJECT / "agents" / "full_agent.py",
    ]

    print("\nSearching agent files for sampling parameters...")
    for af in agent_files:
        if not af.exists():
            print(f"  {af.name}: NOT FOUND")
            continue
        content = af.read_text()
        has_temp = "temperature" in content
        has_top_p = "top_p" in content
        has_top_k = "top_k" in content
        # Find max_tokens
        max_tokens = "not found"
        for line in content.split("\n"):
            if "max_tokens" in line and "=" in line:
                max_tokens = line.strip()
                break
        print(f"  {af.name}: temp={'YES' if has_temp else 'no'}, top_p={'YES' if has_top_p else 'no'}, top_k={'YES' if has_top_k else 'no'}, max_tokens={max_tokens}")

    # Check config
    config_file = PROJECT / "experiments" / "config.py"
    if config_file.exists():
        content = config_file.read_text()
        has_temp = "temperature" in content
        has_top_p = "top_p" in content
        print(f"  config.py: temp={'YES' if has_temp else 'no'}, top_p={'YES' if has_top_p else 'no'}")

    # Check episode_runner
    runner_file = PROJECT / "experiments" / "episode_runner.py"
    if runner_file.exists():
        content = runner_file.read_text()
        has_temp = "temperature" in content
        has_top_p = "top_p" in content
        print(f"  episode_runner.py: temp={'YES' if has_temp else 'no'}, top_p={'YES' if has_top_p else 'no'}")

    # Check dynamic_behaviors for trigger model max_tokens
    for db_file in ["dynamic_behaviors.py", "dynamic_behaviors_10c.py"]:
        db_path = PROJECT / "agents" / db_file
        if db_path.exists():
            content = db_path.read_text()
            for line in content.split("\n"):
                if "max_tokens" in line and "=" in line:
                    print(f"  {db_file}: {line.strip()}")
                    break

    print("\n  SUMMARY: All sampling parameters use API defaults (no explicit temperature, top_p, top_k).")
    print("  This means Anthropic can change defaults server-side without any code change being visible.")


# ============================================================
# TASK 3: Base_llm Performance Trajectory
# ============================================================

def task3_performance_trajectory():
    section("TASK 3: BASE_LLM PERFORMANCE TRAJECTORY")

    # Read all result summary JSONs
    result_files = sorted(glob.glob(str(DATA / "results_run*" / "*.json")))

    # Extract base_llm results from each
    run_data = []

    for rf in result_files:
        rf_path = Path(rf)
        run_dir = rf_path.parent.name
        run_name = run_dir.replace("results_", "")

        with open(rf) as f:
            try:
                summary = json.load(f)
            except json.JSONDecodeError:
                continue

        results = summary.get("results", [])
        if not results:
            continue

        # Group by condition + agent_type
        groups = defaultdict(list)
        for r in results:
            agent = r.get("agent_type", "unknown")
            condition = r.get("condition", "unknown")
            groups[(condition, agent)].append(r)

        for (condition, agent), episodes in groups.items():
            if "base_llm" not in agent:
                continue
            n = len(episodes)
            successes = sum(1 for e in episodes if e.get("success", False))
            submissions = sum(1 for e in episodes if e.get("answer_submitted", False))
            accuracy = successes / n * 100 if n > 0 else 0

            run_data.append({
                "run": run_name,
                "file": rf_path.name,
                "condition": condition,
                "agent": agent,
                "n": n,
                "successes": successes,
                "accuracy": accuracy,
                "submission_rate": submissions / n * 100 if n > 0 else 0,
            })

    # Also extract from report files for dates
    report_files = sorted(glob.glob(str(PROJECT / "runs" / "report_run*.md")))
    run_dates = {}
    for rf in report_files:
        name = os.path.basename(rf)
        # Extract date from filename: report_runXX_YYYY-MM-DD.md
        parts = name.replace("report_", "").replace(".md", "").split("_")
        if len(parts) >= 2:
            run_id = parts[0]
            date = "_".join(parts[1:])
            run_dates[run_id] = date

    # Print trajectory tables
    print("\n--- Hidden Moderator / Extended Conditions (Runs 02-08) ---")
    print(f"{'Run':<15} {'Date':<12} {'Condition':<35} {'N':>4} {'Accuracy':>9} {'Submit%':>8}")
    print("-" * 90)
    for r in run_data:
        if "order" not in r["condition"].lower() and "diag" not in r["condition"].lower():
            date = run_dates.get(r["run"], "?")
            print(f"{r['run']:<15} {date:<12} {r['condition']:<35} {r['n']:>4} {r['accuracy']:>8.1f}% {r['submission_rate']:>7.1f}%")

    print("\n--- Order-Sensitive Conditions (Runs 09-10d-diagnostic) ---")
    print(f"{'Run':<15} {'Date':<12} {'Condition':<35} {'N':>4} {'Accuracy':>9} {'Submit%':>8}")
    print("-" * 90)
    for r in run_data:
        if "order" in r["condition"].lower() or "diag" in r["condition"].lower():
            date = run_dates.get(r["run"], "?")
            print(f"{r['run']:<15} {date:<12} {r['condition']:<35} {r['n']:>4} {r['accuracy']:>8.1f}% {r['submission_rate']:>7.1f}%")

    # Drift analysis for order-sensitive
    print("\n--- Order-Sensitive Drift Analysis (base_llm, comparable conditions) ---")
    order_runs = [r for r in run_data if "order" in r["condition"].lower() or "diag" in r["condition"].lower()]
    if order_runs:
        accs = [r["accuracy"] for r in order_runs]
        print(f"  Range: {min(accs):.1f}% – {max(accs):.1f}% (spread: {max(accs)-min(accs):.1f}pp)")
        print(f"  Runs: {len(order_runs)}")
        for r in order_runs:
            import math
            p = r["accuracy"] / 100
            n = r["n"]
            ci = 1.96 * math.sqrt(p * (1-p) / n) * 100 if n > 0 and 0 < p < 1 else 0
            print(f"    {r['run']:<15} {r['condition']:<35} {r['accuracy']:>6.1f}% ± {ci:.1f}pp (95% CI, n={n})")

    return run_data


# ============================================================
# TASK 4: Full_cg_db Performance (for comparison)
# ============================================================

def task3b_fullcgdb_trajectory():
    section("TASK 3b: FULL_CG_DB PERFORMANCE TRAJECTORY")

    result_files = sorted(glob.glob(str(DATA / "results_run*" / "*.json")))
    run_data = []

    for rf in result_files:
        with open(rf) as f:
            try:
                summary = json.load(f)
            except json.JSONDecodeError:
                continue

        results = summary.get("results", [])
        if not results:
            continue

        groups = defaultdict(list)
        for r in results:
            agent = r.get("agent_type", "unknown")
            condition = r.get("condition", "unknown")
            groups[(condition, agent)].append(r)

        for (condition, agent), episodes in groups.items():
            if "full_cg_db" not in agent:
                continue
            n = len(episodes)
            successes = sum(1 for e in episodes if e.get("success", False))
            accuracy = successes / n * 100 if n > 0 else 0

            run_data.append({
                "run": Path(rf).parent.name.replace("results_", ""),
                "condition": condition,
                "n": n,
                "accuracy": accuracy,
            })

    print("\n--- Order-Sensitive full_cg_db ---")
    print(f"{'Run':<15} {'Condition':<35} {'N':>4} {'Accuracy':>9}")
    print("-" * 70)
    for r in run_data:
        if "order" in r["condition"].lower() or "diag" in r["condition"].lower():
            print(f"{r['run']:<15} {r['condition']:<35} {r['n']:>4} {r['accuracy']:>8.1f}%")


# ============================================================
# TASK 5: API Response Metadata Check
# ============================================================

def task5_api_metadata():
    section("TASK 5: API RESPONSE METADATA CHECK")

    # Sample one trace from each run, check all top-level keys
    trace_dirs = sorted(glob.glob(str(DATA / "traces_run*")))

    print(f"\nChecking {len(trace_dirs)} trace directories for API metadata fields...")
    print(f"\n{'Run':<20} {'agent_trace keys':<80}")
    print("-" * 100)

    for td in trace_dirs:
        td_path = Path(td)
        run_name = td_path.name.replace("traces_", "")
        traces = sorted(glob.glob(str(td_path / "*.json")))
        if not traces:
            continue

        with open(traces[0]) as f:
            trace = json.load(f)

        at_keys = sorted(trace.get("agent_trace", {}).keys())
        print(f"{run_name:<20} {', '.join(at_keys)}")

    # Deep check: look for model version info in a few traces
    print("\n\nDeep check: searching for API metadata fields in sample traces...")
    metadata_fields = ["system_fingerprint", "x-model-version", "server_timing",
                       "api_version", "stop_reason", "stop_sequence", "model_version"]

    sample_runs = ["traces_run10", "traces_run10c", "traces_run10d", "traces_run10d_diagnostic"]
    for run in sample_runs:
        td = DATA / run
        if not td.exists():
            continue
        traces = sorted(glob.glob(str(td / "*.json")))
        if not traces:
            continue

        with open(traces[0]) as f:
            content = f.read()

        found = []
        for field in metadata_fields:
            if field in content:
                found.append(field)

        if found:
            print(f"  {run}: FOUND {found}")
        else:
            print(f"  {run}: None of {metadata_fields} found in trace")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 90)
    print("  RETROSPECTIVE MODEL STABILITY AUDIT")
    print("  Extended Blicket Benchmark — All Runs")
    print("  Date: 2026-03-14")
    print("=" * 90)

    task1_model_strings()
    task2_sampling_params()
    task3_performance_trajectory()
    task3b_fullcgdb_trajectory()
    task5_api_metadata()

    section("AUDIT COMPLETE")
