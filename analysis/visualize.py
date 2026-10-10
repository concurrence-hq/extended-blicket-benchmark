"""Visualization for blicket pilot results.

Generates plots: success rate heatmaps, action timelines, hypothesis
diversity bars, DB activation scatter, CG state diagrams.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_metrics(results_dir: Path) -> pd.DataFrame:
    csv_path = results_dir / "episode_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run metrics.py first: {csv_path}")
    return pd.read_csv(csv_path)


def plot_success_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap of success rates: conditions (rows) × agents (columns)."""
    pivot = df.pivot_table(
        index="condition", columns="agent_type", values="success", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Success Rate by Condition and Agent Architecture")
    ax.set_ylabel("Condition")
    ax.set_xlabel("Agent Type")
    plt.tight_layout()
    plt.savefig(output_dir / "success_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved success_heatmap.png")


def plot_steps_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Box plots of steps taken per condition × agent."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="condition",
        y="steps_taken",
        hue="agent_type",
        ax=ax,
    )
    ax.set_title("Steps Taken by Condition and Agent")
    ax.set_ylabel("Steps")
    ax.set_xlabel("Condition")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "steps_distribution.png", dpi=150)
    plt.close()
    print(f"Saved steps_distribution.png")


def plot_exploration_diversity(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of unique configurations explored."""
    fig, ax = plt.subplots(figsize=(10, 6))
    means = df.groupby(["condition", "agent_type"])["unique_configurations"].mean().unstack()
    means.plot(kind="bar", ax=ax)
    ax.set_title("Mean Unique Configurations Explored")
    ax.set_ylabel("Unique Configurations")
    ax.set_xlabel("Condition")
    plt.xticks(rotation=15)
    plt.legend(title="Agent Type")
    plt.tight_layout()
    plt.savefig(output_dir / "exploration_diversity.png", dpi=150)
    plt.close()
    print(f"Saved exploration_diversity.png")


def plot_action_entropy(df: pd.DataFrame, output_dir: Path) -> None:
    """Action entropy comparison across agents."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=df,
        x="condition",
        y="action_entropy",
        hue="agent_type",
        ax=ax,
        inner="quart",
    )
    ax.set_title("Action Entropy by Condition and Agent")
    ax.set_ylabel("Shannon Entropy (bits)")
    ax.set_xlabel("Condition")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "action_entropy.png", dpi=150)
    plt.close()
    print(f"Saved action_entropy.png")


def plot_db_activations(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter of DB activation counts for the Full agent across conditions."""
    full_agent = df[df["agent_type"] == "full_cg_db"]
    if full_agent.empty:
        print("No full_cg_db data for DB activations plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for condition in full_agent["condition"].unique():
        subset = full_agent[full_agent["condition"] == condition]
        ax.scatter(
            subset["steps_taken"],
            subset["db_activations"],
            label=condition,
            alpha=0.6,
        )

    ax.set_title("Dynamic Behavior Activations (Full Agent)")
    ax.set_xlabel("Steps Taken")
    ax.set_ylabel("DB Activations")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "db_activations.png", dpi=150)
    plt.close()
    print(f"Saved db_activations.png")


def plot_child_comparison(results_dir: Path, output_dir: Path) -> None:
    """Bar chart comparing agent success rates to child benchmarks."""
    csv_path = results_dir / "child_comparison.csv"
    if not csv_path.exists():
        print("Run compare_to_children.py first.")
        return

    df = pd.read_csv(csv_path)

    for condition in ["conjunctive", "disjunctive"]:
        subset = df[df["condition"] == condition]
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#b3cde3"] * 4 + ["#fbb4ae", "#ccebc5", "#decbe4"]
        bars = ax.bar(
            range(len(subset)),
            subset["success_rate"],
            color=colors[: len(subset)],
            edgecolor="gray",
        )
        ax.set_xticks(range(len(subset)))
        ax.set_xticklabels(subset["participant"], rotation=25, ha="right")
        ax.set_ylabel("Success Rate")
        ax.set_title(f"Agent vs. Child Performance ({condition.title()} Condition)")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

        for bar, rate in zip(bars, subset["success_rate"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{rate:.2f}",
                ha="center",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(output_dir / f"child_comparison_{condition}.png", dpi=150)
        plt.close()
        print(f"Saved child_comparison_{condition}.png")


def generate_all_plots(results_dir: Path, output_dir: Path | None = None) -> None:
    """Generate all plots from computed metrics."""
    if output_dir is None:
        output_dir = results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(results_dir)

    plot_success_heatmap(df, output_dir)
    plot_steps_distribution(df, output_dir)
    plot_exploration_diversity(df, output_dir)
    plot_action_entropy(df, output_dir)
    plot_db_activations(df, output_dir)
    plot_child_comparison(results_dir, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    import sys

    results = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/results")
    generate_all_plots(results)
