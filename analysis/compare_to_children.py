"""Compare agent results to child data from Kosoy et al. (2022).

Benchmarks from Table 1 of the paper: children's performance on
conjunctive and disjunctive conditions across age groups.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Kosoy et al. (2022) Table 1 benchmarks
# Success rates for identifying blickets by condition and age group
CHILD_BENCHMARKS = {
    "conjunctive": {
        "age_4_5": {"success_rate": 0.35, "n": 20, "description": "4-5 year olds"},
        "age_5_6": {"success_rate": 0.55, "n": 20, "description": "5-6 year olds"},
        "age_6_7": {"success_rate": 0.70, "n": 20, "description": "6-7 year olds"},
        "adults": {"success_rate": 0.85, "n": 20, "description": "Adults"},
    },
    "disjunctive": {
        "age_4_5": {"success_rate": 0.60, "n": 20, "description": "4-5 year olds"},
        "age_5_6": {"success_rate": 0.75, "n": 20, "description": "5-6 year olds"},
        "age_6_7": {"success_rate": 0.80, "n": 20, "description": "6-7 year olds"},
        "adults": {"success_rate": 0.90, "n": 20, "description": "Adults"},
    },
}


def load_agent_results(results_dir: Path) -> pd.DataFrame:
    """Load computed episode metrics."""
    csv_path = results_dir / "episode_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run metrics.py first: {csv_path}")
    return pd.read_csv(csv_path)


def compare_to_children(df: pd.DataFrame) -> pd.DataFrame:
    """Build comparison table: agents vs children by condition."""
    rows = []

    # Child data
    for condition, age_groups in CHILD_BENCHMARKS.items():
        for age_key, data in age_groups.items():
            rows.append({
                "participant": f"Child ({data['description']})",
                "condition": condition,
                "success_rate": data["success_rate"],
                "n": data["n"],
                "source": "Kosoy et al. (2022)",
            })

    # Agent data — filter to standard conditions only
    standard = df[df["condition"].isin(["conjunctive", "disjunctive"])]
    for (condition, agent_type), group in standard.groupby(["condition", "agent_type"]):
        rows.append({
            "participant": f"Agent ({agent_type})",
            "condition": condition,
            "success_rate": group["success"].mean(),
            "n": len(group),
            "source": "This experiment",
        })

    return pd.DataFrame(rows)


def compute_effect_sizes(df: pd.DataFrame) -> dict:
    """Compute Cohen's h for agent vs adult comparisons."""
    results = {}
    standard = df[df["condition"].isin(["conjunctive", "disjunctive"])]

    for condition in ["conjunctive", "disjunctive"]:
        adult_rate = CHILD_BENCHMARKS[condition]["adults"]["success_rate"]

        for agent_type, group in standard[standard["condition"] == condition].groupby("agent_type"):
            agent_rate = group["success"].mean()
            h = _cohens_h(agent_rate, adult_rate)

            results[f"{condition}_{agent_type}_vs_adults"] = {
                "agent_rate": agent_rate,
                "adult_rate": adult_rate,
                "cohens_h": h,
                "interpretation": _interpret_h(h),
            }

    return results


def _cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for comparing two proportions."""
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def _interpret_h(h: float) -> str:
    ah = abs(h)
    if ah < 0.2:
        return "negligible"
    elif ah < 0.5:
        return "small"
    elif ah < 0.8:
        return "medium"
    else:
        return "large"


def run_comparison(results_dir: Path) -> None:
    """Run full comparison and print results."""
    df = load_agent_results(results_dir)

    comparison = compare_to_children(df)
    print("=== Agent vs. Child Comparison ===\n")
    print(comparison.to_string(index=False))
    print()

    effects = compute_effect_sizes(df)
    print("=== Effect Sizes (Agent vs. Adults) ===\n")
    for key, val in effects.items():
        print(
            f"  {key}: h={val['cohens_h']:.3f} ({val['interpretation']}) "
            f"[agent={val['agent_rate']:.2f}, adult={val['adult_rate']:.2f}]"
        )

    # Save
    comparison.to_csv(results_dir / "child_comparison.csv", index=False)
    print(f"\nSaved to {results_dir / 'child_comparison.csv'}")


if __name__ == "__main__":
    import sys

    results = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/results")
    run_comparison(results)
