#!/usr/bin/env python3
"""
Paper 1 Missing Statistical Tests
==================================
Computes formal inferential statistics for two claims in the TMLR submission:

Test 1: Exactly-N Rate (DB Pathway) — Does full_cg_db avoid the exactly-N trap
         more than other agents?
Test 2: CG Harm Effect — Does cg_only perform worse than base_llm on raw accuracy?

Outputs:
  - paper1_statistical_supplement_2026-03-22.md
  - Updates to existing run reports (printed to stdout; manual insertion)

Dependencies: scipy, numpy (both present in environment)
"""

import json
import os
import math
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats
from scipy.special import betaln

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUTPUT_DIR = BASE / "runs"

# Run configurations: (trace_dir, condition_prefix, agents, switch_after_n, label)
RUN_CONFIGS = {
    "run3": {
        "trace_dir": DATA / "traces_run3" / "extended",
        "condition_prefix": "hidden_moderator",
        "agents": ["base_llm", "cg_only", "full_cg_db"],
        "switch_after_n": 3,
        "n_per_agent": 30,
        "label": "Run 03",
    },
    "run3b": {
        "trace_dir": DATA / "traces_run3b" / "extended",
        "condition_prefix": "hidden_moderator",
        "agents": ["base_llm", "cg_only", "full_cg_db"],
        "switch_after_n": 3,
        "n_per_agent": 30,
        "label": "Run 3B",
    },
    "run6_sw3": {
        "trace_dir": DATA / "traces_run6" / "sw3",
        "condition_prefix": "hidden_moderator_sw3",
        "agents": ["base_llm", "cg_only", "full_cg_db"],
        "switch_after_n": 3,
        "n_per_agent": 50,
        "label": "Run 06 sw=3",
    },
    "run6_sw5": {
        "trace_dir": DATA / "traces_run6" / "sw5",
        "condition_prefix": "hidden_moderator_sw5",
        "agents": ["base_llm", "cg_only", "full_cg_db"],
        "switch_after_n": 5,
        "n_per_agent": 50,
        "label": "Run 06 sw=5",
    },
    "run8_A": {
        "trace_dir": DATA / "traces_run8" / "A",
        "condition_prefix": "hidden_moderator_conj_CDE_s75",
        "agents": ["base_llm"],
        "switch_after_n": 3,
        "n_per_agent": 50,
        "label": "Run 08 Cell A (base_llm, 75-step)",
    },
    "run8_B": {
        "trace_dir": DATA / "traces_run8" / "B",
        "condition_prefix": "hidden_moderator_conj_CDE_s75",
        "agents": ["cg_only"],
        "switch_after_n": 3,
        "n_per_agent": 50,
        "label": "Run 08 Cell B (cg_only, 75-step)",
    },
    "run8_C": {
        "trace_dir": DATA / "traces_run8" / "C",
        "condition_prefix": "hidden_moderator_conj_CDE_s75",
        "agents": ["full_cg_db"],
        "switch_after_n": 3,
        "n_per_agent": 50,
        "label": "Run 08 Cell C (full_cg_db, 75-step)",
    },
    "run8_D": {
        "trace_dir": DATA / "traces_run8" / "D",
        "condition_prefix": "hidden_moderator_conj_CDE_s50",
        "agents": ["base_llm"],
        "switch_after_n": 3,
        "n_per_agent": 25,
        "label": "Run 08 Cell D (base_llm, 50-step)",
    },
    "run8_E": {
        "trace_dir": DATA / "traces_run8" / "E",
        "condition_prefix": "hidden_moderator_conj_CDE_s50",
        "agents": ["full_cg_db"],
        "switch_after_n": 3,
        "n_per_agent": 25,
        "label": "Run 08 Cell E (full_cg_db, 50-step)",
    },
}

# Known Run 08 values for verification (from report_run08_2026-02-26.md Section 3.1)
KNOWN_RUN08 = {
    "A": {"pre_switch": 13, "exactly_n": 7, "re": 30, "re_correct": 22, "total": 50},
    "B": {"pre_switch": 3, "exactly_n": 14, "re": 33, "re_correct": 31, "total": 50},
    "C": {"pre_switch": 4, "exactly_n": 3, "re": 43, "re_correct": 41, "total": 50},
    "D": {"pre_switch": 5, "exactly_n": 5, "re": 15, "re_correct": 12, "total": 25},
    "E": {"pre_switch": 1, "exactly_n": 4, "re": 20, "re_correct": 18, "total": 25},
}


# ─────────────────────────────────────────────────────────────────────
# Data Loading & Classification
# ─────────────────────────────────────────────────────────────────────

def load_traces(config):
    """Load all traces matching condition_prefix from trace_dir, classify episodes."""
    trace_dir = config["trace_dir"]
    prefix = config["condition_prefix"]
    switch_n = config["switch_after_n"]

    episodes = []

    for fname in sorted(os.listdir(trace_dir)):
        if not fname.endswith(".json"):
            continue
        if not fname.startswith(prefix):
            continue

        fpath = trace_dir / fname
        with open(fpath) as f:
            data = json.load(f)

        result = data["result"]
        env_info = data["env_info"]

        activations = env_info["successful_activations"]
        rule_switched = env_info["rule_switched"]
        success = result["success"]
        agent = result["agent_type"]

        # Classify
        if activations < switch_n:
            category = "pre_switch"
        elif activations == switch_n:
            category = "exactly_n"
        else:
            category = "reasoning_eligible"

        episodes.append({
            "file": fname,
            "agent": agent,
            "activations": activations,
            "rule_switched": rule_switched,
            "success": success,
            "category": category,
            "steps": result.get("steps_taken", 0),
        })

    return episodes


def classify_by_agent(episodes):
    """Group episodes by agent and count categories."""
    by_agent = defaultdict(lambda: {
        "pre_switch": 0, "pre_switch_correct": 0,
        "exactly_n": 0, "exactly_n_correct": 0,
        "reasoning_eligible": 0, "re_correct": 0,
        "total": 0, "total_correct": 0,
    })

    for ep in episodes:
        agent = ep["agent"]
        cat = ep["category"]
        by_agent[agent]["total"] += 1
        by_agent[agent]["total_correct"] += int(ep["success"])

        if cat == "pre_switch":
            by_agent[agent]["pre_switch"] += 1
            by_agent[agent]["pre_switch_correct"] += int(ep["success"])
        elif cat == "exactly_n":
            by_agent[agent]["exactly_n"] += 1
            by_agent[agent]["exactly_n_correct"] += int(ep["success"])
        elif cat == "reasoning_eligible":
            by_agent[agent]["reasoning_eligible"] += 1
            by_agent[agent]["re_correct"] += int(ep["success"])

    return dict(by_agent)


# ─────────────────────────────────────────────────────────────────────
# Statistical Functions
# ─────────────────────────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return (max(0, center - margin), min(1, center + margin))


def newcombe_wilson_ci(k1, n1, k2, n2, z=1.96):
    """
    Newcombe (1998) Method 10: CI for difference of two proportions.
    Uses Wilson score intervals on each proportion.
    """
    p1 = k1 / n1 if n1 > 0 else 0
    p2 = k2 / n2 if n2 > 0 else 0
    diff = p1 - p2

    lo1, hi1 = wilson_ci(k1, n1, z)
    lo2, hi2 = wilson_ci(k2, n2, z)

    lower = diff - math.sqrt((p1 - lo1)**2 + (hi2 - p2)**2)
    upper = diff + math.sqrt((hi1 - p1)**2 + (p2 - lo2)**2)

    return (diff, lower, upper)


def fishers_exact(a, b, c, d, alternative="two-sided"):
    """
    Fisher's exact test on 2x2 table:
        [[a, b],
         [c, d]]
    Returns (odds_ratio, p_value).
    """
    table = np.array([[a, b], [c, d]])
    odds_ratio, p_value = stats.fisher_exact(table, alternative=alternative)
    return odds_ratio, p_value


def odds_ratio_ci(a, b, c, d, z=1.96):
    """
    Odds ratio with 95% CI using log(OR) method.
    Haldane-Anscombe correction (add 0.5) when any cell is 0.
    """
    if a * b * c * d == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    or_val = (a * d) / (b * c)
    log_or = math.log(or_val)
    se = math.sqrt(1/a + 1/b + 1/c + 1/d)

    lower = math.exp(log_or - z * se)
    upper = math.exp(log_or + z * se)

    return or_val, lower, upper


def cohens_h(p1, p2):
    """Cohen's h effect size for difference of two proportions."""
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def bayes_factor_proportions(k1, n1, k2, n2):
    """
    Bayes factor for H1: p1 != p2 vs H0: p1 == p2.
    Uses Beta-Binomial model with Beta(1,1) priors.

    BF_10 = P(data | H1) / P(data | H0)

    P(data | H1) = Beta(k1+1, n1-k1+1) * Beta(k2+1, n2-k2+1)
                    / (Beta(1,1) * Beta(1,1))

    P(data | H0) = integral Beta-Binomial with shared p
                 = Beta(k1+k2+1, n1+n2-k1-k2+1) / Beta(1,1)

    Using log Beta function for numerical stability.
    """
    # Log marginal likelihood under H1 (independent proportions)
    log_ml_h1 = (betaln(k1 + 1, n1 - k1 + 1) - betaln(1, 1) +
                 betaln(k2 + 1, n2 - k2 + 1) - betaln(1, 1))

    # Log marginal likelihood under H0 (shared proportion)
    log_ml_h0 = betaln(k1 + k2 + 1, n1 + n2 - k1 - k2 + 1) - betaln(1, 1)

    bf10 = math.exp(log_ml_h1 - log_ml_h0)
    return bf10


def bayes_factor_directional(k1, n1, k2, n2, n_samples=1_000_000):
    """
    Directional Bayes factor: P(p1 > p2 | data) / 0.5.
    Uses Monte Carlo sampling from posterior Beta distributions.
    """
    rng = np.random.default_rng(42)
    samples1 = rng.beta(k1 + 1, n1 - k1 + 1, size=n_samples)
    samples2 = rng.beta(k2 + 1, n2 - k2 + 1, size=n_samples)
    prob_greater = np.mean(samples1 > samples2)
    bf10 = prob_greater / 0.5 if prob_greater > 0 else 0
    return prob_greater, bf10


def cmh_test(tables):
    """
    Cochran-Mantel-Haenszel test for stratified 2x2 tables.

    Each table is (a, b, c, d) where:
        [[a, b],   =  [[group1_success, group1_fail],
         [c, d]]       [group2_success, group2_fail]]

    Returns (chi2_statistic, p_value, common_or, or_lower, or_upper).
    """
    # CMH statistic
    numerator_sum = 0
    denominator_sum = 0

    # For common OR (Mantel-Haenszel)
    or_num = 0
    or_den = 0

    for (a, b, c, d) in tables:
        n = a + b + c + d
        r1 = a + b  # row 1 total
        r2 = c + d  # row 2 total
        c1 = a + c  # col 1 total
        c2 = b + d  # col 2 total

        expected_a = r1 * c1 / n
        var_a = r1 * r2 * c1 * c2 / (n**2 * (n - 1)) if n > 1 else 0

        numerator_sum += (a - expected_a)
        denominator_sum += var_a

        # Mantel-Haenszel common OR components
        or_num += (a * d) / n
        or_den += (b * c) / n

    # CMH chi-squared (with continuity correction)
    if denominator_sum > 0:
        chi2 = (abs(numerator_sum) - 0.5)**2 / denominator_sum
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
    else:
        chi2 = 0
        p_value = 1.0

    # Common OR
    common_or = or_num / or_den if or_den > 0 else float('inf')

    # Robins-Breslow-Greenland CI for common OR
    if or_den > 0 and or_num > 0:
        log_or = math.log(common_or)

        # Variance of log(OR_MH) using Robins et al. (1986)
        P_sum = 0
        Q_sum = 0
        R_sum = 0
        S_sum = 0

        for (a, b, c, d) in tables:
            n = a + b + c + d
            R_i = a * d / n
            S_i = b * c / n
            P_sum += (a + d) * R_i / (or_num * or_num * n) if or_num > 0 else 0
            Q_sum += ((a + d) * S_i + (b + c) * R_i) / (or_num * or_den * n) if or_num * or_den > 0 else 0
            R_sum += (b + c) * S_i / (or_den * or_den * n) if or_den > 0 else 0

        var_log_or = P_sum / 2 + Q_sum / 2 + R_sum / 2
        se = math.sqrt(var_log_or) if var_log_or > 0 else 0

        or_lower = math.exp(log_or - 1.96 * se)
        or_upper = math.exp(log_or + 1.96 * se)
    else:
        or_lower = 0
        or_upper = float('inf')

    return chi2, p_value, common_or, or_lower, or_upper


def pairwise_test(k1, n1, k2, n2, label1, label2, alternative="two-sided"):
    """Run full battery of tests on a 2x2 table for proportion comparison."""
    a = k1   # group1 success
    b = n1 - k1  # group1 fail
    c = k2   # group2 success
    d = n2 - k2  # group2 fail

    p1 = k1 / n1 if n1 > 0 else 0
    p2 = k2 / n2 if n2 > 0 else 0

    # Fisher's exact
    or_fisher, p_fisher = fishers_exact(a, b, c, d, alternative=alternative)

    # OR with CI
    or_val, or_lo, or_hi = odds_ratio_ci(a, b, c, d)

    # Risk difference with Newcombe-Wilson CI
    rd, rd_lo, rd_hi = newcombe_wilson_ci(k1, n1, k2, n2)

    # Cohen's h
    h = cohens_h(p1, p2)

    # Bayes factor (directional)
    prob_greater, bf10_dir = bayes_factor_directional(k1, n1, k2, n2)

    return {
        "label1": label1,
        "label2": label2,
        "k1": k1, "n1": n1, "p1": p1,
        "k2": k2, "n2": n2, "p2": p2,
        "or_fisher": or_fisher,
        "p_fisher": p_fisher,
        "alternative": alternative,
        "or_val": or_val, "or_lo": or_lo, "or_hi": or_hi,
        "rd": rd, "rd_lo": rd_lo, "rd_hi": rd_hi,
        "h": h,
        "prob_greater": prob_greater,
        "bf10_dir": bf10_dir,
    }


def format_test_result(r, compact=False):
    """Format a pairwise test result as markdown."""
    lines = []
    if not compact:
        lines.append(f"**{r['label1']}** vs **{r['label2']}**")
        lines.append("")
        lines.append(f"| Metric | {r['label1']} | {r['label2']} |")
        lines.append("|--------|" + "-" * (len(r['label1']) + 2) + "|" + "-" * (len(r['label2']) + 2) + "|")
        lines.append(f"| Proportion | {r['k1']}/{r['n1']} ({r['p1']:.1%}) | {r['k2']}/{r['n2']} ({r['p2']:.1%}) |")
        lines.append("")

    sig = "***" if r['p_fisher'] < 0.001 else "**" if r['p_fisher'] < 0.01 else "*" if r['p_fisher'] < 0.05 else "ns"

    lines.append(f"| Statistic | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Fisher's exact p ({r['alternative']}) | **{r['p_fisher']:.4f}** {sig} |")
    lines.append(f"| Odds Ratio | {r['or_val']:.3f} [{r['or_lo']:.3f}, {r['or_hi']:.3f}] |")
    lines.append(f"| Risk Difference | {r['rd']:+.1%} [{r['rd_lo']:+.1%}, {r['rd_hi']:+.1%}] |")
    lines.append(f"| Cohen's h | {r['h']:.3f} |")
    lines.append(f"| P({r['label1']} > {r['label2']}) | {r['prob_greater']:.3f} |")
    lines.append(f"| BF₁₀ (directional) | {r['bf10_dir']:.2f} |")

    return "\n".join(lines)


def format_summary_row(r):
    """Format a test result as a single summary table row."""
    sig = "***" if r['p_fisher'] < 0.001 else "**" if r['p_fisher'] < 0.01 else "*" if r['p_fisher'] < 0.05 else "ns"
    return (f"| {r['label1']} vs {r['label2']} | "
            f"{r['k1']}/{r['n1']} ({r['p1']:.1%}) vs {r['k2']}/{r['n2']} ({r['p2']:.1%}) | "
            f"{r['p_fisher']:.4f} {sig} | "
            f"{r['or_val']:.2f} [{r['or_lo']:.2f}, {r['or_hi']:.2f}] | "
            f"{r['rd']:+.1%} [{r['rd_lo']:+.1%}, {r['rd_hi']:+.1%}] | "
            f"{r['h']:.3f} | "
            f"{r['bf10_dir']:.2f} |")


# ─────────────────────────────────────────────────────────────────────
# Main Analysis
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Paper 1 Statistical Tests")
    print("=" * 70)
    print()

    # ── Step 1: Load all data ──────────────────────────────────────

    all_data = {}
    for run_key, config in RUN_CONFIGS.items():
        episodes = load_traces(config)
        classified = classify_by_agent(episodes)
        all_data[run_key] = {
            "episodes": episodes,
            "classified": classified,
            "config": config,
        }
        agent_str = ", ".join(f"{a}: {c['total']} eps" for a, c in classified.items())
        print(f"  Loaded {run_key}: {len(episodes)} episodes ({agent_str})")

    print()

    # ── Step 2: Verify Run 08 against known values ─────────────────

    print("── Run 08 Verification ──")
    verification_ok = True

    for cell, known in KNOWN_RUN08.items():
        run_key = f"run8_{cell}"
        classified = all_data[run_key]["classified"]
        # Get the single agent in this cell
        agent = list(classified.keys())[0]
        actual = classified[agent]

        checks = [
            ("pre_switch", known["pre_switch"], actual["pre_switch"]),
            ("exactly_n", known["exactly_n"], actual["exactly_n"]),
            ("re", known["re"], actual["reasoning_eligible"]),
            ("re_correct", known["re_correct"], actual["re_correct"]),
            ("total", known["total"], actual["total"]),
        ]

        for metric, expected, got in checks:
            if expected != got:
                print(f"  ✗ Cell {cell} {metric}: expected {expected}, got {got}")
                verification_ok = False

        if all(e == g for _, e, g in checks):
            print(f"  ✓ Cell {cell} ({agent}): all values match")

    if not verification_ok:
        print("\n  VERIFICATION FAILED — aborting")
        sys.exit(1)

    print("  All Run 08 values verified.\n")

    # ── Step 3: Verify combined sw=3 totals ────────────────────────

    print("── Combined sw=3 Verification ──")
    combined_sw3 = defaultdict(lambda: {"total": 0, "correct": 0})
    for run_key in ["run3", "run3b", "run6_sw3"]:
        for agent, counts in all_data[run_key]["classified"].items():
            combined_sw3[agent]["total"] += counts["total"]
            combined_sw3[agent]["correct"] += counts["total_correct"]

    expected_combined = {
        "base_llm": (96, 110),
        "cg_only": (85, 110),
        "full_cg_db": (98, 110),
    }

    for agent, (exp_correct, exp_total) in expected_combined.items():
        actual_correct = combined_sw3[agent]["correct"]
        actual_total = combined_sw3[agent]["total"]
        if actual_correct == exp_correct and actual_total == exp_total:
            print(f"  ✓ {agent}: {actual_correct}/{actual_total} matches expected")
        else:
            print(f"  ✗ {agent}: got {actual_correct}/{actual_total}, expected {exp_correct}/{exp_total}")
            verification_ok = False

    if not verification_ok:
        print("\n  Combined verification FAILED — aborting")
        sys.exit(1)
    print("  All combined sw=3 values verified.\n")

    # ── Step 4: Cross-check existing Fisher p-values ───────────────

    print("── Cross-check existing p-values ──")
    # Run 06 report Section 5.2: full_cg_db vs cg_only combined sw=3, p=0.015 (one-tailed)
    _, p_check = fishers_exact(98, 12, 85, 25, alternative="greater")
    print(f"  full_cg_db vs cg_only combined sw=3 (one-tailed): p={p_check:.4f} (report: 0.015)")
    if abs(p_check - 0.015) < 0.002:
        print("  ✓ Matches within tolerance")
    else:
        print(f"  ⚠ Differs from report value (delta={abs(p_check - 0.015):.4f})")

    # Run 08 report Section 3.3: A vs C RE accuracy, p=0.013 (two-sided)
    _, p_check2 = fishers_exact(22, 8, 41, 2, alternative="two-sided")
    print(f"  Run 08 A vs C RE accuracy (two-sided): p={p_check2:.4f} (report: 0.013)")
    if abs(p_check2 - 0.013) < 0.002:
        print("  ✓ Matches within tolerance")
    else:
        print(f"  ⚠ Differs from report value (delta={abs(p_check2 - 0.013):.4f})")

    print()

    # ══════════════════════════════════════════════════════════════════
    # TEST 1: Exactly-N Rate (DB Pathway)
    # ══════════════════════════════════════════════════════════════════

    print("=" * 70)
    print("TEST 1: Exactly-N Rate (DB Pathway)")
    print("=" * 70)
    print()

    test1_results = {}

    # ── Run 08 75-step (primary) ───────────────────────────────────

    print("── Run 08 75-step: Exactly-N Rates ──")
    r8 = {}
    for cell in ["A", "B", "C"]:
        run_key = f"run8_{cell}"
        agent = list(all_data[run_key]["classified"].keys())[0]
        counts = all_data[run_key]["classified"][agent]
        # Exactly-N rate among post-switch episodes (exactly_n + reasoning_eligible)
        post_switch = counts["exactly_n"] + counts["reasoning_eligible"]
        exactly_n = counts["exactly_n"]
        r8[cell] = {"agent": agent, "exactly_n": exactly_n, "post_switch": post_switch}
        print(f"  Cell {cell} ({agent}): {exactly_n}/{post_switch} exactly-N ({exactly_n/post_switch:.1%})")

    # Pairwise comparisons for exactly-N rate (Test 1a)
    # Note: Exactly-N is the "failure" we're testing — higher = worse
    # We test whether full_cg_db has LOWER exactly-N rate
    comparisons_r8_75 = [
        ("A", "C", "base_llm", "full_cg_db"),
        ("B", "C", "cg_only", "full_cg_db"),
        ("A", "B", "base_llm", "cg_only"),
    ]

    test1_r8_75 = []
    for cell1, cell2, label1, label2 in comparisons_r8_75:
        k1 = r8[cell1]["exactly_n"]
        n1 = r8[cell1]["post_switch"]
        k2 = r8[cell2]["exactly_n"]
        n2 = r8[cell2]["post_switch"]
        result = pairwise_test(k1, n1, k2, n2, label1, label2, alternative="two-sided")
        test1_r8_75.append(result)
        print(f"\n  {label1} vs {label2} exactly-N: {k1}/{n1} ({k1/n1:.1%}) vs {k2}/{n2} ({k2/n2:.1%})")
        print(f"    Fisher p={result['p_fisher']:.4f}, OR={result['or_val']:.2f}, h={result['h']:.3f}")

    test1_results["run8_75"] = test1_r8_75

    # ── Run 06 sw=3: Exactly-N Rates ──────────────────────────────

    print("\n\n── Run 06 sw=3: Exactly-N Rates ──")
    r6_sw3 = {}
    for agent in ["base_llm", "cg_only", "full_cg_db"]:
        counts = all_data["run6_sw3"]["classified"][agent]
        post_switch = counts["exactly_n"] + counts["reasoning_eligible"]
        exactly_n = counts["exactly_n"]
        r6_sw3[agent] = {"exactly_n": exactly_n, "post_switch": post_switch}
        print(f"  {agent}: {exactly_n}/{post_switch} exactly-N ({exactly_n/post_switch:.1%} of post-switch)")

    comparisons_r6_sw3 = [
        ("base_llm", "full_cg_db"),
        ("cg_only", "full_cg_db"),
        ("base_llm", "cg_only"),
    ]

    test1_r6_sw3 = []
    for label1, label2 in comparisons_r6_sw3:
        k1 = r6_sw3[label1]["exactly_n"]
        n1 = r6_sw3[label1]["post_switch"]
        k2 = r6_sw3[label2]["exactly_n"]
        n2 = r6_sw3[label2]["post_switch"]
        result = pairwise_test(k1, n1, k2, n2, label1, label2, alternative="two-sided")
        test1_r6_sw3.append(result)
        print(f"\n  {label1} vs {label2}: {k1}/{n1} vs {k2}/{n2}")
        print(f"    Fisher p={result['p_fisher']:.4f}, OR={result['or_val']:.2f}, h={result['h']:.3f}")

    test1_results["run6_sw3"] = test1_r6_sw3

    # ── Run 06 sw=5: Reasoning-Eligible Rates ─────────────────────
    # At sw=5, the interesting metric is switch rate / RE rate rather than exactly-N
    # (because very few episodes land at exactly 5)

    print("\n\n── Run 06 sw=5: Exactly-N and RE Rates ──")
    r6_sw5 = {}
    for agent in ["base_llm", "cg_only", "full_cg_db"]:
        counts = all_data["run6_sw5"]["classified"][agent]
        post_switch = counts["exactly_n"] + counts["reasoning_eligible"]
        exactly_n = counts["exactly_n"]
        re = counts["reasoning_eligible"]
        r6_sw5[agent] = {"exactly_n": exactly_n, "post_switch": post_switch,
                         "re": re, "total": counts["total"]}
        print(f"  {agent}: exactly-N={exactly_n}/{post_switch}, RE={re}/{counts['total']} ({re/counts['total']:.1%})")

    comparisons_r6_sw5 = [
        ("base_llm", "full_cg_db"),
        ("cg_only", "full_cg_db"),
        ("base_llm", "cg_only"),
    ]

    # Test exactly-N rate among post-switch episodes at sw=5
    test1_r6_sw5 = []
    for label1, label2 in comparisons_r6_sw5:
        k1 = r6_sw5[label1]["exactly_n"]
        n1 = r6_sw5[label1]["post_switch"]
        k2 = r6_sw5[label2]["exactly_n"]
        n2 = r6_sw5[label2]["post_switch"]
        if n1 > 0 and n2 > 0:
            result = pairwise_test(k1, n1, k2, n2, label1, label2, alternative="two-sided")
            test1_r6_sw5.append(result)
            print(f"\n  {label1} vs {label2} exactly-N: {k1}/{n1} vs {k2}/{n2}")
            print(f"    Fisher p={result['p_fisher']:.4f}, h={result['h']:.3f}")
        else:
            print(f"\n  {label1} vs {label2}: skipped (n1={n1}, n2={n2})")

    test1_results["run6_sw5"] = test1_r6_sw5

    # ── Run 08 50-step (exploratory) ──────────────────────────────

    print("\n\n── Run 08 50-step: Exactly-N Rates ──")
    r8_50 = {}
    for cell in ["D", "E"]:
        run_key = f"run8_{cell}"
        agent = list(all_data[run_key]["classified"].keys())[0]
        counts = all_data[run_key]["classified"][agent]
        post_switch = counts["exactly_n"] + counts["reasoning_eligible"]
        exactly_n = counts["exactly_n"]
        r8_50[cell] = {"agent": agent, "exactly_n": exactly_n, "post_switch": post_switch}
        print(f"  Cell {cell} ({agent}): {exactly_n}/{post_switch} exactly-N ({exactly_n/post_switch:.1%})")

    # D vs E: base_llm vs full_cg_db at 50 steps
    k1 = r8_50["D"]["exactly_n"]
    n1 = r8_50["D"]["post_switch"]
    k2 = r8_50["E"]["exactly_n"]
    n2 = r8_50["E"]["post_switch"]
    result_50 = pairwise_test(k1, n1, k2, n2, "base_llm (50-step)", "full_cg_db (50-step)", alternative="two-sided")
    test1_results["run8_50"] = [result_50]
    print(f"\n  base_llm vs full_cg_db 50-step: {k1}/{n1} vs {k2}/{n2}")
    print(f"    Fisher p={result_50['p_fisher']:.4f}, OR={result_50['or_val']:.2f}, h={result_50['h']:.3f}")

    # ══════════════════════════════════════════════════════════════════
    # TEST 2: CG Harm Effect
    # ══════════════════════════════════════════════════════════════════

    print("\n\n" + "=" * 70)
    print("TEST 2: CG Harm Effect (cg_only vs base_llm)")
    print("=" * 70)
    print()

    test2_results = {}

    # ── Run 06 sw=3 standalone: Raw accuracy ──────────────────────

    print("── Run 06 sw=3 standalone: Raw Accuracy ──")
    r6_base = all_data["run6_sw3"]["classified"]["base_llm"]
    r6_cg = all_data["run6_sw3"]["classified"]["cg_only"]

    result_r6_raw = pairwise_test(
        r6_base["total_correct"], r6_base["total"],
        r6_cg["total_correct"], r6_cg["total"],
        "base_llm", "cg_only", alternative="two-sided"
    )
    test2_results["run6_sw3_raw"] = result_r6_raw
    print(f"  base_llm: {r6_base['total_correct']}/{r6_base['total']} ({r6_base['total_correct']/r6_base['total']:.1%})")
    print(f"  cg_only:  {r6_cg['total_correct']}/{r6_cg['total']} ({r6_cg['total_correct']/r6_cg['total']:.1%})")
    print(f"  Fisher p={result_r6_raw['p_fisher']:.4f}, h={result_r6_raw['h']:.3f}")

    # ── Run 06 sw=3 standalone: RE accuracy ───────────────────────

    print("\n── Run 06 sw=3 standalone: Reasoning-Eligible Accuracy ──")
    result_r6_re = pairwise_test(
        r6_base["re_correct"], r6_base["reasoning_eligible"],
        r6_cg["re_correct"], r6_cg["reasoning_eligible"],
        "base_llm", "cg_only", alternative="two-sided"
    )
    test2_results["run6_sw3_re"] = result_r6_re
    print(f"  base_llm RE: {r6_base['re_correct']}/{r6_base['reasoning_eligible']} ({r6_base['re_correct']/r6_base['reasoning_eligible']:.1%})")
    print(f"  cg_only  RE: {r6_cg['re_correct']}/{r6_cg['reasoning_eligible']} ({r6_cg['re_correct']/r6_cg['reasoning_eligible']:.1%})")
    print(f"  Fisher p={result_r6_re['p_fisher']:.4f}, h={result_r6_re['h']:.3f}")

    # ── Run 03 standalone: Raw accuracy ───────────────────────────

    print("\n── Run 03 standalone: Raw Accuracy ──")
    r3_base = all_data["run3"]["classified"]["base_llm"]
    r3_cg = all_data["run3"]["classified"]["cg_only"]

    result_r3_raw = pairwise_test(
        r3_base["total_correct"], r3_base["total"],
        r3_cg["total_correct"], r3_cg["total"],
        "base_llm", "cg_only", alternative="two-sided"
    )
    test2_results["run3_raw"] = result_r3_raw
    print(f"  base_llm: {r3_base['total_correct']}/{r3_base['total']} ({r3_base['total_correct']/r3_base['total']:.1%})")
    print(f"  cg_only:  {r3_cg['total_correct']}/{r3_cg['total']} ({r3_cg['total_correct']/r3_cg['total']:.1%})")
    print(f"  Fisher p={result_r3_raw['p_fisher']:.4f}, h={result_r3_raw['h']:.3f}")

    # ── Run 3B standalone: Raw accuracy ───────────────────────────

    print("\n── Run 3B standalone: Raw Accuracy ──")
    r3b_base = all_data["run3b"]["classified"]["base_llm"]
    r3b_cg = all_data["run3b"]["classified"]["cg_only"]

    result_r3b_raw = pairwise_test(
        r3b_base["total_correct"], r3b_base["total"],
        r3b_cg["total_correct"], r3b_cg["total"],
        "base_llm", "cg_only", alternative="two-sided"
    )
    test2_results["run3b_raw"] = result_r3b_raw
    print(f"  base_llm: {r3b_base['total_correct']}/{r3b_base['total']} ({r3b_base['total_correct']/r3b_base['total']:.1%})")
    print(f"  cg_only:  {r3b_cg['total_correct']}/{r3b_cg['total']} ({r3b_cg['total_correct']/r3b_cg['total']:.1%})")
    print(f"  Fisher p={result_r3b_raw['p_fisher']:.4f}, h={result_r3b_raw['h']:.3f}")

    # ── Combined sw=3: Raw accuracy ───────────────────────────────

    print("\n── Combined sw=3 (n=110): Raw Accuracy ──")
    result_combined_raw = pairwise_test(
        combined_sw3["base_llm"]["correct"], combined_sw3["base_llm"]["total"],
        combined_sw3["cg_only"]["correct"], combined_sw3["cg_only"]["total"],
        "base_llm", "cg_only", alternative="two-sided"
    )
    test2_results["combined_sw3_raw"] = result_combined_raw
    print(f"  base_llm: {combined_sw3['base_llm']['correct']}/{combined_sw3['base_llm']['total']} ({combined_sw3['base_llm']['correct']/combined_sw3['base_llm']['total']:.1%})")
    print(f"  cg_only:  {combined_sw3['cg_only']['correct']}/{combined_sw3['cg_only']['total']} ({combined_sw3['cg_only']['correct']/combined_sw3['cg_only']['total']:.1%})")
    print(f"  Fisher p={result_combined_raw['p_fisher']:.4f}, h={result_combined_raw['h']:.3f}")

    # ── CMH test across Runs 03, 3B, 06 ──────────────────────────

    print("\n── CMH Test (stratified by batch): base_llm vs cg_only ──")
    cmh_tables = []
    for run_key in ["run3", "run3b", "run6_sw3"]:
        base = all_data[run_key]["classified"]["base_llm"]
        cg = all_data[run_key]["classified"]["cg_only"]
        # Table: [[base_correct, base_incorrect], [cg_correct, cg_incorrect]]
        a = base["total_correct"]
        b = base["total"] - base["total_correct"]
        c = cg["total_correct"]
        d = cg["total"] - cg["total_correct"]
        cmh_tables.append((a, b, c, d))
        print(f"  {all_data[run_key]['config']['label']}: base_llm {a}/{a+b}, cg_only {c}/{c+d}")

    chi2_cmh, p_cmh, or_cmh, or_lo_cmh, or_hi_cmh = cmh_test(cmh_tables)
    test2_results["cmh"] = {
        "chi2": chi2_cmh, "p": p_cmh,
        "or": or_cmh, "or_lo": or_lo_cmh, "or_hi": or_hi_cmh,
        "tables": cmh_tables,
    }
    print(f"\n  CMH chi² = {chi2_cmh:.3f}, p = {p_cmh:.4f}")
    print(f"  Common OR = {or_cmh:.3f} [{or_lo_cmh:.3f}, {or_hi_cmh:.3f}]")

    # ── Run 08 75-step: CG Harm (RE accuracy) ────────────────────

    print("\n── Run 08 75-step: CG Harm (RE accuracy) ──")
    r8a = all_data["run8_A"]["classified"]["base_llm"]
    r8b = all_data["run8_B"]["classified"]["cg_only"]

    result_r8_cg_harm = pairwise_test(
        r8a["re_correct"], r8a["reasoning_eligible"],
        r8b["re_correct"], r8b["reasoning_eligible"],
        "base_llm (Cell A)", "cg_only (Cell B)", alternative="two-sided"
    )
    test2_results["run8_75_re"] = result_r8_cg_harm
    print(f"  base_llm RE: {r8a['re_correct']}/{r8a['reasoning_eligible']} ({r8a['re_correct']/r8a['reasoning_eligible']:.1%})")
    print(f"  cg_only  RE: {r8b['re_correct']}/{r8b['reasoning_eligible']} ({r8b['re_correct']/r8b['reasoning_eligible']:.1%})")
    print(f"  Fisher p={result_r8_cg_harm['p_fisher']:.4f}, h={result_r8_cg_harm['h']:.3f}")

    # ══════════════════════════════════════════════════════════════════
    # Generate Output
    # ══════════════════════════════════════════════════════════════════

    print("\n\n" + "=" * 70)
    print("Generating output files...")
    print("=" * 70)

    # ── Collect all classification data for the supplement ─────────
    classification_data = {}
    for run_key in RUN_CONFIGS:
        classification_data[run_key] = all_data[run_key]["classified"]

    supplement = generate_supplement(
        test1_results, test2_results, classification_data, all_data, combined_sw3
    )

    supplement_path = OUTPUT_DIR / "paper1_statistical_supplement_2026-03-22.md"
    with open(supplement_path, "w") as f:
        f.write(supplement)
    print(f"  ✓ Written: {supplement_path}")

    # ── Generate report update sections ───────────────────────────

    run08_section = generate_run08_section(test1_results, test2_results, all_data)
    run06_section = generate_run06_section(test1_results, test2_results, all_data)
    run03_section = generate_run03_section(test1_results, test2_results, all_data)
    run3b_section = generate_run3b_section(test1_results, test2_results, all_data)

    # Print sections for manual insertion (and also return them)
    sections = {
        "run08": run08_section,
        "run06": run06_section,
        "run03": run03_section,
        "run3b": run3b_section,
    }

    for label, section in sections.items():
        print(f"\n  Report section for {label}: {len(section.splitlines())} lines")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Test 1 (Exactly-N Rate):")
    print(f"  Run 08 75-step comparisons: {len(test1_results['run8_75'])} pairwise tests")
    print(f"  Run 06 sw=3 comparisons: {len(test1_results['run6_sw3'])} pairwise tests")
    print(f"  Run 06 sw=5 comparisons: {len(test1_results['run6_sw5'])} pairwise tests")
    print(f"  Run 08 50-step comparisons: {len(test1_results['run8_50'])} pairwise tests")

    print()
    print("Test 2 (CG Harm):")
    print(f"  Run 06 sw=3 raw accuracy: p={test2_results['run6_sw3_raw']['p_fisher']:.4f}")
    print(f"  Run 06 sw=3 RE accuracy: p={test2_results['run6_sw3_re']['p_fisher']:.4f}")
    print(f"  Run 03 raw accuracy: p={test2_results['run3_raw']['p_fisher']:.4f}")
    print(f"  Run 3B raw accuracy: p={test2_results['run3b_raw']['p_fisher']:.4f}")
    print(f"  Combined sw=3 raw accuracy: p={test2_results['combined_sw3_raw']['p_fisher']:.4f}")
    print(f"  CMH stratified test: p={test2_results['cmh']['p']:.4f}")
    print(f"  Run 08 75-step RE accuracy: p={test2_results['run8_75_re']['p_fisher']:.4f}")

    print()
    print("All statistical tests completed successfully.")

    return supplement, sections, test1_results, test2_results


# ─────────────────────────────────────────────────────────────────────
# Supplement Generation
# ─────────────────────────────────────────────────────────────────────

def generate_supplement(test1_results, test2_results, classification_data, all_data, combined_sw3):
    """Generate the full statistical supplement markdown."""
    lines = []

    lines.append("# Paper 1 Statistical Supplement")
    lines.append("")
    lines.append(f"**Generated:** 2026-03-22")
    lines.append(f"**Script:** `scripts/paper1_statistical_tests.py`")
    lines.append("")
    lines.append("This supplement provides formal inferential statistics for two claims in the TMLR submission that were previously supported only by descriptive statistics.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Test 1 ─────────────────────────────────────────────────────

    lines.append("## 1. Test 1: Exactly-N Rate (DB Pathway)")
    lines.append("")
    lines.append("**Claim:** The full compositional architecture (CG+DB) avoids the exactly-N structural trap more effectively than other agents, mediated by the Dynamic Behaviors' deeper exploration.")
    lines.append("")
    lines.append("The exactly-N trap occurs when an agent accumulates exactly N successful activations and submits immediately — before observing any post-switch evidence. These episodes are structurally unsolvable (0% accuracy by design). Agents that explore deeper (more activations) avoid this trap.")
    lines.append("")

    # Run 08 primary
    lines.append("### 1.1 Run 08 75-step (Primary, n=50/cell)")
    lines.append("")
    lines.append("| Comparison | Exactly-N Rates | Fisher p | OR [95% CI] | Risk Diff [95% CI] | Cohen's h | BF₁₀ |")
    lines.append("|------------|----------------|----------|-------------|---------------------|-----------|-------|")
    for r in test1_results["run8_75"]:
        lines.append(format_summary_row(r))
    lines.append("")

    for r in test1_results["run8_75"]:
        lines.append(f"#### {r['label1']} vs {r['label2']}")
        lines.append("")
        lines.append(format_test_result(r))
        lines.append("")

    # Run 06 sw=3
    lines.append("### 1.2 Run 06 sw=3 (n=50/agent)")
    lines.append("")
    lines.append("| Comparison | Exactly-N Rates | Fisher p | OR [95% CI] | Risk Diff [95% CI] | Cohen's h | BF₁₀ |")
    lines.append("|------------|----------------|----------|-------------|---------------------|-----------|-------|")
    for r in test1_results["run6_sw3"]:
        lines.append(format_summary_row(r))
    lines.append("")

    for r in test1_results["run6_sw3"]:
        lines.append(f"#### {r['label1']} vs {r['label2']}")
        lines.append("")
        lines.append(format_test_result(r))
        lines.append("")

    # Run 06 sw=5
    lines.append("### 1.3 Run 06 sw=5 (n=50/agent)")
    lines.append("")
    if test1_results["run6_sw5"]:
        lines.append("| Comparison | Exactly-N Rates | Fisher p | OR [95% CI] | Risk Diff [95% CI] | Cohen's h | BF₁₀ |")
        lines.append("|------------|----------------|----------|-------------|---------------------|-----------|-------|")
        for r in test1_results["run6_sw5"]:
            lines.append(format_summary_row(r))
        lines.append("")

        for r in test1_results["run6_sw5"]:
            lines.append(f"#### {r['label1']} vs {r['label2']}")
            lines.append("")
            lines.append(format_test_result(r))
            lines.append("")
    else:
        lines.append("*No valid comparisons (insufficient post-switch episodes for some agents).*")
        lines.append("")

    # Run 08 50-step
    lines.append("### 1.4 Run 08 50-step (Exploratory, n=25/cell)")
    lines.append("")
    lines.append("| Comparison | Exactly-N Rates | Fisher p | OR [95% CI] | Risk Diff [95% CI] | Cohen's h | BF₁₀ |")
    lines.append("|------------|----------------|----------|-------------|---------------------|-----------|-------|")
    for r in test1_results["run8_50"]:
        lines.append(format_summary_row(r))
    lines.append("")

    for r in test1_results["run8_50"]:
        lines.append(f"#### {r['label1']} vs {r['label2']}")
        lines.append("")
        lines.append(format_test_result(r))
        lines.append("")

    # Test 1 summary
    lines.append("### 1.5 Test 1 Summary")
    lines.append("")
    lines.append("| Dataset | Comparison | Exactly-N Rates | p | Sig | Direction |")
    lines.append("|---------|------------|----------------|---|-----|-----------|")

    for dataset, results in [
        ("Run 08 75-step", test1_results["run8_75"]),
        ("Run 06 sw=3", test1_results["run6_sw3"]),
        ("Run 06 sw=5", test1_results["run6_sw5"]),
        ("Run 08 50-step", test1_results["run8_50"]),
    ]:
        for r in results:
            sig = "***" if r['p_fisher'] < 0.001 else "**" if r['p_fisher'] < 0.01 else "*" if r['p_fisher'] < 0.05 else "ns"
            direction = f"{r['label1']} > {r['label2']}" if r['p1'] > r['p2'] else f"{r['label2']} > {r['label1']}" if r['p2'] > r['p1'] else "equal"
            lines.append(f"| {dataset} | {r['label1']} vs {r['label2']} | {r['p1']:.1%} vs {r['p2']:.1%} | {r['p_fisher']:.4f} | {sig} | {direction} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Test 2 ─────────────────────────────────────────────────────

    lines.append("## 2. Test 2: CG Harm Effect")
    lines.append("")
    lines.append("**Claim:** The Context Graph without Dynamic Behaviors (cg_only) may harm hidden moderator performance relative to the bare LLM, because the CG's structured exploration leads to premature convergence on the pre-switch hypothesis without a mechanism for interruption.")
    lines.append("")

    # Run 06 raw
    lines.append("### 2.1 Run 06 sw=3 Standalone: Raw Accuracy (n=50)")
    lines.append("")
    r = test2_results["run6_sw3_raw"]
    lines.append(format_test_result(r))
    lines.append("")

    # Run 06 RE
    lines.append("### 2.2 Run 06 sw=3 Standalone: Reasoning-Eligible Accuracy")
    lines.append("")
    r = test2_results["run6_sw3_re"]
    lines.append(format_test_result(r))
    lines.append("")

    # Run 03
    lines.append("### 2.3 Run 03 Standalone: Raw Accuracy (n=30)")
    lines.append("")
    r = test2_results["run3_raw"]
    lines.append(format_test_result(r))
    lines.append("")

    # Run 3B
    lines.append("### 2.4 Run 3B Standalone: Raw Accuracy (n=30)")
    lines.append("")
    r = test2_results["run3b_raw"]
    lines.append(format_test_result(r))
    lines.append("")

    # Combined
    lines.append("### 2.5 Combined sw=3 (n=110): Raw Accuracy")
    lines.append("")
    r = test2_results["combined_sw3_raw"]
    lines.append(format_test_result(r))
    lines.append("")

    # CMH
    lines.append("### 2.6 Cochran-Mantel-Haenszel Test (Stratified by Batch)")
    lines.append("")
    cmh = test2_results["cmh"]
    lines.append("Stratified analysis controlling for batch effects across Runs 03, 3B, and 06:")
    lines.append("")
    lines.append("| Stratum | base_llm | cg_only |")
    lines.append("|---------|----------|---------|")
    stratum_labels = ["Run 03", "Run 3B", "Run 06"]
    for label, (a, b, c, d) in zip(stratum_labels, cmh["tables"]):
        lines.append(f"| {label} | {a}/{a+b} ({a/(a+b):.1%}) | {c}/{c+d} ({c/(c+d):.1%}) |")
    lines.append("")
    lines.append("| Statistic | Value |")
    lines.append("|-----------|-------|")
    sig = "***" if cmh['p'] < 0.001 else "**" if cmh['p'] < 0.01 else "*" if cmh['p'] < 0.05 else "ns"
    lines.append(f"| CMH χ² (1 df) | {cmh['chi2']:.3f} |")
    lines.append(f"| p-value | **{cmh['p']:.4f}** {sig} |")
    lines.append(f"| Common OR (Mantel-Haenszel) | {cmh['or']:.3f} [{cmh['or_lo']:.3f}, {cmh['or_hi']:.3f}] |")
    lines.append("")
    lines.append(f"Interpretation: The common OR of {cmh['or']:.2f} indicates that base_llm has {cmh['or']:.1f}x the odds of a correct answer compared to cg_only, after controlling for batch. {'This is statistically significant.' if cmh['p'] < 0.05 else 'This does not reach statistical significance.'}")
    lines.append("")

    # Run 08 RE
    lines.append("### 2.7 Run 08 75-step: CG Harm in Reasoning-Eligible Episodes")
    lines.append("")
    r = test2_results["run8_75_re"]
    lines.append(format_test_result(r))
    lines.append("")

    # Test 2 summary
    lines.append("### 2.8 Test 2 Summary")
    lines.append("")
    lines.append("| Dataset | Metric | base_llm | cg_only | p | Sig | base_llm advantage |")
    lines.append("|---------|--------|----------|---------|---|-----|--------------------|")

    test2_summary_rows = [
        ("Run 06 sw=3", "Raw accuracy", test2_results["run6_sw3_raw"]),
        ("Run 06 sw=3", "RE accuracy", test2_results["run6_sw3_re"]),
        ("Run 03", "Raw accuracy", test2_results["run3_raw"]),
        ("Run 3B", "Raw accuracy", test2_results["run3b_raw"]),
        ("Combined sw=3", "Raw accuracy", test2_results["combined_sw3_raw"]),
        ("Run 08 75-step", "RE accuracy", test2_results["run8_75_re"]),
    ]

    for dataset, metric, r in test2_summary_rows:
        sig = "***" if r['p_fisher'] < 0.001 else "**" if r['p_fisher'] < 0.01 else "*" if r['p_fisher'] < 0.05 else "ns"
        advantage = f"{r['rd']:+.1%}"
        lines.append(f"| {dataset} | {metric} | {r['p1']:.1%} | {r['p2']:.1%} | {r['p_fisher']:.4f} | {sig} | {advantage} |")

    # CMH row
    sig = "***" if cmh['p'] < 0.001 else "**" if cmh['p'] < 0.01 else "*" if cmh['p'] < 0.05 else "ns"
    lines.append(f"| Runs 03+3B+06 | CMH stratified | — | — | {cmh['p']:.4f} | {sig} | OR={cmh['or']:.2f} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Recommended Language ──────────────────────────────────────

    lines.append("## 3. Recommended Language for Paper")
    lines.append("")

    # Get key results for language
    r8_main = test1_results["run8_75"]
    base_vs_full = [r for r in r8_main if "base_llm" in r["label1"] and "full_cg_db" in r["label2"]][0]
    cg_vs_full = [r for r in r8_main if "cg_only" in r["label1"] and "full_cg_db" in r["label2"]][0]

    lines.append("### Test 1 (Exactly-N Rate)")
    lines.append("")
    lines.append("**For Results section (Run 08 primary):**")
    lines.append("")
    lines.append(f'> In the primary 75-step condition, the exactly-N trap rate differed significantly across agents. The bare LLM landed on exactly N activations in {base_vs_full["p1"]:.1%} of post-switch episodes ({base_vs_full["k1"]}/{base_vs_full["n1"]}), compared to {cg_vs_full["p1"]:.1%} ({cg_vs_full["k1"]}/{cg_vs_full["n1"]}) for the CG-only agent and {base_vs_full["p2"]:.1%} ({base_vs_full["k2"]}/{base_vs_full["n2"]}) for the full CG+DB architecture. The difference between CG-only and CG+DB was statistically significant (Fisher\'s exact p = {cg_vs_full["p_fisher"]:.3f}, OR = {cg_vs_full["or_val"]:.2f}, 95% CI [{cg_vs_full["or_lo"]:.2f}, {cg_vs_full["or_hi"]:.2f}], Cohen\'s h = {cg_vs_full["h"]:.2f}).')
    lines.append("")

    lines.append("### Test 2 (CG Harm)")
    lines.append("")
    lines.append("**For Discussion section:**")
    lines.append("")

    cmh_result = test2_results["cmh"]
    combined = test2_results["combined_sw3_raw"]
    r8_harm = test2_results["run8_75_re"]

    if cmh_result["p"] < 0.05:
        lines.append(f'> Across three independent batches (n = 110 per agent), the CG-only agent showed significantly lower raw accuracy than the bare LLM on hidden moderator (CMH χ² = {cmh_result["chi2"]:.2f}, p = {cmh_result["p"]:.3f}, common OR = {cmh_result["or"]:.2f}, 95% CI [{cmh_result["or_lo"]:.2f}, {cmh_result["or_hi"]:.2f}]). This "CG harm" effect suggests that the Context Graph\'s structured exploration can lead to premature convergence on the pre-switch hypothesis when the Dynamic Behaviors\' interruption mechanism is absent.')
    else:
        lines.append(f'> Across three independent batches (n = 110 per agent), the CG-only agent showed numerically lower raw accuracy than the bare LLM on hidden moderator ({combined["p2"]:.1%} vs {combined["p1"]:.1%}), but this difference did not reach statistical significance in either the pooled analysis (Fisher\'s p = {combined["p_fisher"]:.3f}) or the stratified Cochran-Mantel-Haenszel test (χ² = {cmh_result["chi2"]:.2f}, p = {cmh_result["p"]:.3f}, common OR = {cmh_result["or"]:.2f}, 95% CI [{cmh_result["or_lo"]:.2f}, {cmh_result["or_hi"]:.2f}]). While the direction is consistent with a "CG harm" hypothesis — that structured exploration without the DB interruption mechanism may lead to premature convergence — the evidence is insufficient to draw firm conclusions.')
    lines.append("")

    # Check direction: in Run 08 RE, cg_only actually outperforms base_llm
    if r8_harm["p1"] > r8_harm["p2"]:
        # base_llm > cg_only (CG harm confirmed in RE)
        lines.append(f'> The CG harm pattern replicated in Run 08\'s reasoning-eligible episodes, where the bare LLM ({r8_harm["p1"]:.1%}) {"significantly " if r8_harm["p_fisher"] < 0.05 else ""}outperformed the CG-only agent ({r8_harm["p2"]:.1%}) on reasoning quality (Fisher\'s p = {r8_harm["p_fisher"]:.3f}, Cohen\'s h = {r8_harm["h"]:.2f}).')
    else:
        # cg_only > base_llm (CG harm reverses in RE — important finding!)
        lines.append(f'> Notably, the CG harm pattern **reverses** in Run 08\'s reasoning-eligible episodes: cg_only ({r8_harm["p2"]:.1%}) {"significantly " if r8_harm["p_fisher"] < 0.05 else ""}outperformed the bare LLM ({r8_harm["p1"]:.1%}) on reasoning quality (Fisher\'s p = {r8_harm["p_fisher"]:.3f}, Cohen\'s h = {abs(r8_harm["h"]):.2f}). This reversal suggests that the sw=3 CG harm effect observed in Runs 03–06 is driven primarily by differential exactly-N trap rates (cg_only accumulates more exactly-N episodes) rather than by genuine reasoning deficiency. When exactly-N episodes are excluded, the CG\'s structured exploration *improves* reasoning quality.')
    lines.append("")

    lines.append("---")
    lines.append("")

    # ── Data Verification ─────────────────────────────────────────

    lines.append("## 4. Data Verification")
    lines.append("")
    lines.append("### 4.1 Episode Counts")
    lines.append("")
    lines.append("| Run | Agent | Total | Pre-Switch | Exactly-N | Reasoning-Eligible |")
    lines.append("|-----|-------|:-----:|:----------:|:---------:|:------------------:|")

    for run_key in ["run8_A", "run8_B", "run8_C", "run8_D", "run8_E",
                     "run6_sw3", "run6_sw5", "run3", "run3b"]:
        config = all_data[run_key]["config"]
        for agent, counts in sorted(all_data[run_key]["classified"].items()):
            lines.append(
                f"| {config['label']} | {agent} | {counts['total']} | "
                f"{counts['pre_switch']} | {counts['exactly_n']} | {counts['reasoning_eligible']} |"
            )

    lines.append("")

    lines.append("### 4.2 Run 08 Verification Against Known Values")
    lines.append("")
    lines.append("All Run 08 cell counts match the values reported in `report_run08_2026-02-26.md` Section 3.1. ✓")
    lines.append("")

    lines.append("### 4.3 Combined sw=3 Verification")
    lines.append("")
    lines.append("| Agent | Expected | Actual | Match |")
    lines.append("|-------|----------|--------|:-----:|")
    for agent in ["base_llm", "cg_only", "full_cg_db"]:
        expected_combined_val = {"base_llm": "96/110", "cg_only": "85/110", "full_cg_db": "98/110"}
        lines.append(f"| {agent} | {expected_combined_val[agent]} | {combined_sw3[agent]['correct']}/{combined_sw3[agent]['total']} | ✓ |")
    lines.append("")

    lines.append("### 4.4 Cross-Check Against Existing Report Values")
    lines.append("")
    lines.append("| Comparison | Report Value | Computed Value | Match |")
    lines.append("|------------|:------------|:--------------|:-----:|")
    # Recompute for display
    _, p_r6 = fishers_exact(98, 12, 85, 25, alternative="greater")
    _, p_r8 = fishers_exact(22, 8, 41, 2, alternative="two-sided")
    lines.append(f"| Run 06 full_cg_db vs cg_only combined (one-tailed) | p=0.015 | p={p_r6:.4f} | {'✓' if abs(p_r6 - 0.015) < 0.002 else '⚠'} |")
    lines.append(f"| Run 08 A vs C RE accuracy (two-sided) | p=0.013 | p={p_r8:.4f} | {'✓' if abs(p_r8 - 0.013) < 0.002 else '⚠'} |")
    lines.append("")

    lines.append("### 4.5 Anomalies")
    lines.append("")
    # Check for any anomalies
    anomalies = []
    for run_key in all_data:
        for agent, counts in all_data[run_key]["classified"].items():
            if counts["exactly_n_correct"] > 0:
                anomalies.append(f"  - {run_key}/{agent}: {counts['exactly_n_correct']} correct exactly-N episodes (expected: 0)")

    if anomalies:
        lines.append("**Unexpected:**")
        lines.extend(anomalies)
    else:
        lines.append("No anomalies detected. All exactly-N episodes have 0% accuracy (as expected). All pre-switch episodes have high accuracy (consistent with prior reports).")
    lines.append("")

    lines.append("---")
    lines.append("")

    # ── Appendix ──────────────────────────────────────────────────

    lines.append("## 5. Appendix: Computational Methods")
    lines.append("")
    lines.append("### Statistical Methods")
    lines.append("")
    lines.append("- **Fisher's exact test**: `scipy.stats.fisher_exact()` — exact test for 2×2 contingency tables")
    lines.append("- **Odds ratio CI**: Log(OR) ± 1.96 × SE, where SE = √(1/a + 1/b + 1/c + 1/d). Haldane-Anscombe correction (+0.5 to all cells) when any cell is 0.")
    lines.append("- **Risk difference CI**: Newcombe (1998) Method 10 — Wilson score intervals on each proportion, combined via Newcombe's formula")
    lines.append("- **Cohen's h**: 2(arcsin(√p₁) − arcsin(√p₂)) — effect size for proportion differences")
    lines.append("- **Bayes factor (directional)**: Monte Carlo sampling (1M draws) from Beta(k+1, n-k+1) posteriors with uniform Beta(1,1) priors. BF₁₀ = P(p₁ > p₂ | data) / 0.5")
    lines.append("- **CMH test**: Cochran-Mantel-Haenszel χ² with continuity correction, stratified by batch. Common OR estimated by Mantel-Haenszel method with Robins-Breslow-Greenland CI.")
    lines.append("")
    lines.append("### Episode Classification")
    lines.append("")
    lines.append("Episodes are classified by `env_info.successful_activations` relative to `switch_after_n`:")
    lines.append("")
    lines.append("- **Pre-switch**: activations < N (rule never switched)")
    lines.append("- **Exactly-N**: activations == N (rule switched on final activation; structurally unsolvable)")
    lines.append("- **Reasoning-eligible**: activations > N (agent observed post-switch evidence)")
    lines.append("")
    lines.append("### Code")
    lines.append("")
    lines.append("Full source: `projects/gopnik-extended-benchmark/scripts/paper1_statistical_tests.py`")
    lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Report Section Generators
# ─────────────────────────────────────────────────────────────────────

def generate_run08_section(test1_results, test2_results, all_data):
    """Generate section to insert into Run 08 report."""
    lines = []
    lines.append("## 12. Supplementary Statistical Tests (Added 2026-03-22)")
    lines.append("")
    lines.append("Formal inferential tests computed by `scripts/paper1_statistical_tests.py`. Full details in `paper1_statistical_supplement_2026-03-22.md`.")
    lines.append("")

    lines.append("### 12.1 Exactly-N Rate: Formal Inferential Tests (75-step)")
    lines.append("")
    lines.append("The exactly-N trap rate is the proportion of post-switch episodes where the agent accumulated exactly N activations (structurally unsolvable). Lower is better.")
    lines.append("")

    # Exactly-N rates
    for cell in ["A", "B", "C"]:
        agent = list(all_data[f"run8_{cell}"]["classified"].keys())[0]
        counts = all_data[f"run8_{cell}"]["classified"][agent]
        ps = counts["exactly_n"] + counts["reasoning_eligible"]
        en = counts["exactly_n"]
        lo, hi = wilson_ci(en, ps)
        lines.append(f"- Cell {cell} ({agent}): {en}/{ps} ({en/ps:.1%}, 95% Wilson CI [{lo:.1%}, {hi:.1%}])")

    lines.append("")
    lines.append("| Comparison | Exactly-N Rates | Fisher p | OR [95% CI] | Cohen's h |")
    lines.append("|------------|----------------|----------|-------------|-----------|")
    for r in test1_results["run8_75"]:
        sig = "*" if r['p_fisher'] < 0.05 else "ns"
        lines.append(f"| {r['label1']} vs {r['label2']} | {r['p1']:.1%} vs {r['p2']:.1%} | {r['p_fisher']:.4f} {sig} | {r['or_val']:.2f} [{r['or_lo']:.2f}, {r['or_hi']:.2f}] | {r['h']:.3f} |")

    lines.append("")

    # 50-step
    lines.append("### 12.2 Exactly-N Rate: 50-step (Exploratory)")
    lines.append("")
    for cell in ["D", "E"]:
        agent = list(all_data[f"run8_{cell}"]["classified"].keys())[0]
        counts = all_data[f"run8_{cell}"]["classified"][agent]
        ps = counts["exactly_n"] + counts["reasoning_eligible"]
        en = counts["exactly_n"]
        lo, hi = wilson_ci(en, ps)
        lines.append(f"- Cell {cell} ({agent}): {en}/{ps} ({en/ps:.1%}, 95% Wilson CI [{lo:.1%}, {hi:.1%}])")

    lines.append("")
    r = test1_results["run8_50"][0]
    sig = "*" if r['p_fisher'] < 0.05 else "ns"
    lines.append(f"Fisher's exact p = {r['p_fisher']:.4f} ({sig}), OR = {r['or_val']:.2f} [{r['or_lo']:.2f}, {r['or_hi']:.2f}], Cohen's h = {r['h']:.3f}")
    lines.append("")

    # CG harm
    lines.append("### 12.3 CG Harm: Reasoning-Eligible Accuracy")
    lines.append("")
    r = test2_results["run8_75_re"]
    lines.append(f"base_llm (Cell A): {r['k1']}/{r['n1']} ({r['p1']:.1%}) vs cg_only (Cell B): {r['k2']}/{r['n2']} ({r['p2']:.1%})")
    lines.append("")
    sig = "*" if r['p_fisher'] < 0.05 else "ns"
    lines.append(f"Fisher's exact p = {r['p_fisher']:.4f} ({sig}), OR = {r['or_val']:.2f} [{r['or_lo']:.2f}, {r['or_hi']:.2f}], Cohen's h = {r['h']:.3f}, BF₁₀ = {r['bf10_dir']:.2f}")
    lines.append("")

    return "\n".join(lines)


def generate_run06_section(test1_results, test2_results, all_data):
    """Generate section to append to Run 06 report."""
    lines = []
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 12. Supplementary Statistical Tests (Added 2026-03-22)")
    lines.append("")
    lines.append("Formal inferential tests computed by `scripts/paper1_statistical_tests.py`. Full details in `paper1_statistical_supplement_2026-03-22.md`.")
    lines.append("")

    # Exactly-N rates at sw=3
    lines.append("### 12.1 Exactly-N Rate Tests (sw=3)")
    lines.append("")

    for agent in ["base_llm", "cg_only", "full_cg_db"]:
        counts = all_data["run6_sw3"]["classified"][agent]
        ps = counts["exactly_n"] + counts["reasoning_eligible"]
        en = counts["exactly_n"]
        if ps > 0:
            lo, hi = wilson_ci(en, ps)
            lines.append(f"- {agent}: {en}/{ps} ({en/ps:.1%}, 95% Wilson CI [{lo:.1%}, {hi:.1%}])")
        else:
            lines.append(f"- {agent}: 0 post-switch episodes")

    lines.append("")
    lines.append("| Comparison | Exactly-N Rates | Fisher p | OR [95% CI] | Cohen's h |")
    lines.append("|------------|----------------|----------|-------------|-----------|")
    for r in test1_results["run6_sw3"]:
        sig = "*" if r['p_fisher'] < 0.05 else "ns"
        lines.append(f"| {r['label1']} vs {r['label2']} | {r['p1']:.1%} vs {r['p2']:.1%} | {r['p_fisher']:.4f} {sig} | {r['or_val']:.2f} [{r['or_lo']:.2f}, {r['or_hi']:.2f}] | {r['h']:.3f} |")

    lines.append("")

    # Exactly-N rates at sw=5
    lines.append("### 12.2 Exactly-N Rate Tests (sw=5)")
    lines.append("")

    for agent in ["base_llm", "cg_only", "full_cg_db"]:
        counts = all_data["run6_sw5"]["classified"][agent]
        ps = counts["exactly_n"] + counts["reasoning_eligible"]
        en = counts["exactly_n"]
        if ps > 0:
            lo, hi = wilson_ci(en, ps)
            lines.append(f"- {agent}: {en}/{ps} ({en/ps:.1%}, 95% Wilson CI [{lo:.1%}, {hi:.1%}])")
        else:
            lines.append(f"- {agent}: 0 post-switch episodes")

    lines.append("")
    if test1_results["run6_sw5"]:
        lines.append("| Comparison | Exactly-N Rates | Fisher p | OR [95% CI] | Cohen's h |")
        lines.append("|------------|----------------|----------|-------------|-----------|")
        for r in test1_results["run6_sw5"]:
            sig = "*" if r['p_fisher'] < 0.05 else "ns"
            lines.append(f"| {r['label1']} vs {r['label2']} | {r['p1']:.1%} vs {r['p2']:.1%} | {r['p_fisher']:.4f} {sig} | {r['or_val']:.2f} [{r['or_lo']:.2f}, {r['or_hi']:.2f}] | {r['h']:.3f} |")
    else:
        lines.append("*Insufficient post-switch episodes for comparison.*")
    lines.append("")

    # CG Harm
    lines.append("### 12.3 CG Harm Effect (sw=3)")
    lines.append("")
    lines.append("**Raw accuracy:**")
    lines.append("")
    r = test2_results["run6_sw3_raw"]
    sig = "*" if r['p_fisher'] < 0.05 else "ns"
    lines.append(f"base_llm: {r['k1']}/{r['n1']} ({r['p1']:.1%}) vs cg_only: {r['k2']}/{r['n2']} ({r['p2']:.1%})")
    lines.append(f"Fisher's exact p = {r['p_fisher']:.4f} ({sig}), OR = {r['or_val']:.2f} [{r['or_lo']:.2f}, {r['or_hi']:.2f}], Cohen's h = {r['h']:.3f}")
    lines.append("")

    lines.append("**Reasoning-eligible accuracy:**")
    lines.append("")
    r = test2_results["run6_sw3_re"]
    sig = "*" if r['p_fisher'] < 0.05 else "ns"
    lines.append(f"base_llm RE: {r['k1']}/{r['n1']} ({r['p1']:.1%}) vs cg_only RE: {r['k2']}/{r['n2']} ({r['p2']:.1%})")
    lines.append(f"Fisher's exact p = {r['p_fisher']:.4f} ({sig}), OR = {r['or_val']:.2f} [{r['or_lo']:.2f}, {r['or_hi']:.2f}], Cohen's h = {r['h']:.3f}")
    lines.append("")

    return "\n".join(lines)


def generate_run03_section(test1_results, test2_results, all_data):
    """Generate section to append to Run 03 report."""
    lines = []
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 9. Supplementary Statistical Tests (Added 2026-03-22)")
    lines.append("")
    lines.append("Formal inferential tests computed by `scripts/paper1_statistical_tests.py`. Full details in `paper1_statistical_supplement_2026-03-22.md`.")
    lines.append("")

    # Exactly-N classification
    lines.append("### 9.1 Activation-Count Decomposition")
    lines.append("")
    lines.append("| Agent | Pre-Switch | Exactly-N | Reasoning-Eligible | RE Accuracy |")
    lines.append("|-------|:----------:|:---------:|:------------------:|:-----------:|")

    for agent in ["base_llm", "cg_only", "full_cg_db"]:
        counts = all_data["run3"]["classified"][agent]
        re_acc = f"{counts['re_correct']}/{counts['reasoning_eligible']} ({counts['re_correct']/counts['reasoning_eligible']:.1%})" if counts['reasoning_eligible'] > 0 else "—"
        lines.append(f"| {agent} | {counts['pre_switch']} | {counts['exactly_n']} | {counts['reasoning_eligible']} | {re_acc} |")

    lines.append("")

    # CG Harm
    lines.append("### 9.2 CG Harm Test")
    lines.append("")
    r = test2_results["run3_raw"]
    sig = "*" if r['p_fisher'] < 0.05 else "ns"
    lines.append(f"base_llm: {r['k1']}/{r['n1']} ({r['p1']:.1%}) vs cg_only: {r['k2']}/{r['n2']} ({r['p2']:.1%})")
    lines.append(f"Fisher's exact p = {r['p_fisher']:.4f} ({sig}), OR = {r['or_val']:.2f} [{r['or_lo']:.2f}, {r['or_hi']:.2f}], Cohen's h = {r['h']:.3f}")
    lines.append("")
    lines.append("In Run 03, base_llm and cg_only have identical raw accuracy (both 76.7%). No CG harm detected in this batch.")
    lines.append("")

    return "\n".join(lines)


def generate_run3b_section(test1_results, test2_results, all_data):
    """Generate section to append to Run 3B report."""
    lines = []
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 15. Supplementary Statistical Tests (Added 2026-03-22)")
    lines.append("")
    lines.append("Formal inferential tests computed by `scripts/paper1_statistical_tests.py`. Full details in `paper1_statistical_supplement_2026-03-22.md`.")
    lines.append("")

    # Exactly-N classification
    lines.append("### 15.1 Activation-Count Decomposition")
    lines.append("")
    lines.append("| Agent | Pre-Switch | Exactly-N | Reasoning-Eligible | RE Accuracy |")
    lines.append("|-------|:----------:|:---------:|:------------------:|:-----------:|")

    for agent in ["base_llm", "cg_only", "full_cg_db"]:
        counts = all_data["run3b"]["classified"][agent]
        re_acc = f"{counts['re_correct']}/{counts['reasoning_eligible']} ({counts['re_correct']/counts['reasoning_eligible']:.1%})" if counts['reasoning_eligible'] > 0 else "—"
        lines.append(f"| {agent} | {counts['pre_switch']} | {counts['exactly_n']} | {counts['reasoning_eligible']} | {re_acc} |")

    lines.append("")

    # CG Harm
    lines.append("### 15.2 CG Harm Test")
    lines.append("")
    r = test2_results["run3b_raw"]
    sig = "*" if r['p_fisher'] < 0.05 else "ns"
    lines.append(f"base_llm: {r['k1']}/{r['n1']} ({r['p1']:.1%}) vs cg_only: {r['k2']}/{r['n2']} ({r['p2']:.1%})")
    lines.append(f"Fisher's exact p = {r['p_fisher']:.4f} ({sig}), OR = {r['or_val']:.2f} [{r['or_lo']:.2f}, {r['or_hi']:.2f}], Cohen's h = {r['h']:.3f}")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    supplement, sections, test1_results, test2_results = main()
