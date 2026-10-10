# Exactly-N Denominator Revision: Run 08

**Date:** 2026-04-01
**Scope:** Recompute Run 08 exactly-N statistics using all-episodes denominators
**Script:** Inline recomputation from `scripts/paper1_statistical_tests.py` trace-loading logic
**Data source:** `data/traces_run8/` (cells A-E, unchanged)

---

## 1. Motivation

The Paper 1 draft was updated to use **all-episodes (n per cell)** as the denominator for exactly-N rates in Tables 5 and B4 (covering Runs 03, 04, 06). Run 08 exactly-N rates still used **switched-episodes denominators** (i.e., episodes where successful_activations >= N), creating an inconsistency across the paper.

The switched-episodes denominator answers "of episodes that reached the switch point, what fraction landed on exactly N?" The all-episodes denominator answers "of all episodes in the cell, what fraction landed on exactly N?" The latter is more interpretable for cross-agent comparison because it is not conditioned on a quantity (switch rate) that itself varies by agent.

## 2. Numerator Verification

All numerators were verified against raw trace files by loading every JSON trace in `data/traces_run8/`, extracting `env_info.successful_activations`, and classifying episodes by the standard three-way decomposition (pre-switch / exactly-N / reasoning-eligible) with switch_after_n = 3.

| Cell | Agent | Total | Pre-Switch | Exactly-N | RE | Numerator Match |
|------|-------|:-----:|:----------:|:---------:|:--:|:---------------:|
| A | base_llm | 50 | 13 | 7 | 30 | verified |
| B | cg_only | 50 | 3 | 14 | 33 | verified |
| C | full_cg_db | 50 | 4 | 3 | 43 | verified |
| D | base_llm | 25 | 5 | 5 | 15 | verified |
| E | full_cg_db | 25 | 1 | 4 | 20 | verified |

All five numerators match `KNOWN_RUN08` in `scripts/paper1_statistical_tests.py` and the values reported in `report_run08_2026-02-26.md` Section 3.1.

## 3. Recomputed Exactly-N Rates (75-Step Cells)

| Agent | Old Denominator | Old Rate | New Denominator | New Rate |
|-------|:-:|:-:|:-:|:-:|
| base_llm (Cell A) | 37 (switched) | 18.9% (7/37) | 50 (all) | 14.0% (7/50) |
| cg_only (Cell B) | 47 (switched) | 29.8% (14/47) | 50 (all) | 28.0% (14/50) |
| full_cg_db (Cell C) | 46 (switched) | 6.5% (3/46) | 50 (all) | 6.0% (3/50) |

The cg_only and full_cg_db rates change minimally because almost all their episodes reached the switch point (47/50 and 46/50 respectively). The base_llm rate drops more (18.9% to 14.0%) because 13 of its 50 episodes never switched.

## 4. Recomputed Pairwise Comparisons (75-Step)

### 4.1 cg_only vs full_cg_db

| Statistic | Old (switched denom) | New (all-episodes denom) |
|-----------|:----:|:----:|
| Rates | 14/47 (29.8%) vs 3/46 (6.5%) | 14/50 (28.0%) vs 3/50 (6.0%) |
| Fisher's exact p (one-tailed) | 0.006 | 0.003 |
| Odds Ratio [95% CI] | 6.08 [1.61, 22.92] | 6.09 [1.63, 22.82] |
| Cohen's h | 0.64 | 0.62 |

The primary comparison **strengthens** under the new denominator (p = 0.003 vs 0.006). The OR is essentially unchanged. This occurs because both denominators grew by approximately the same small amount (3 and 4 pre-switch episodes respectively), preserving the ratio.

### 4.2 base_llm vs full_cg_db

| Statistic | Old (switched denom) | New (all-episodes denom) |
|-----------|:----:|:----:|
| Rates | 7/37 (18.9%) vs 3/46 (6.5%) | 7/50 (14.0%) vs 3/50 (6.0%) |
| Fisher's exact p (one-tailed) | 0.10 | 0.159 |
| Cohen's h | (not stated) | 0.27 |

This comparison **weakens** because base_llm had 13 pre-switch episodes (vs 4 for full_cg_db), so the denominator inflation is asymmetric. The base_llm rate drops from 18.9% to 14.0% while full_cg_db barely moves. This was already non-significant and remains so.

## 5. Recomputed 50-Step Cell (Appendix B.1)

| Statistic | Old | New |
|-----------|:---:|:---:|
| full_cg_db (Cell E) exactly-N rate | 16.7% (4/24) | 16.0% (4/25) |

Minimal change: only 1 pre-switch episode was added to the denominator.

## 6. Bar Chart Segment Percentages

The draft references cg_only as "28% of its bar" and full_cg_db as "6% of its bar." Under the all-episodes denominator:

- cg_only: 14/50 = 28% (unchanged)
- full_cg_db: 3/50 = 6% (unchanged)

These values happen to round to the same integers.

## 7. Find-and-Replace Table

| Location in draft | Old value | New value |
|---|---|---|
| Section 5.3, base_llm exactly-N rate | 18.9% (7/37) | 14.0% (7/50) |
| Section 5.3, cg_only exactly-N rate | 29.8% (14/47) | 28.0% (14/50) |
| Section 5.3, full_cg_db exactly-N rate | 6.5% (3/46) | 6.0% (3/50) |
| Section 5.3, Fisher's p cg_only vs full_cg_db (one-tailed) | p=0.006 | p=0.003 |
| Section 5.3, OR cg_only vs full_cg_db | OR=6.08 [1.61, 22.92] | OR=6.09 [1.63, 22.82] |
| Section 5.3, Cohen's h cg_only vs full_cg_db | h=0.64 | h=0.62 |
| Section 5.3, Fisher's p base_llm vs full_cg_db (one-tailed) | p=0.10 | p=0.159 |
| Table 6, exactly-N rate progression | 29.8% to 6.5% | 28.0% to 6.0% |
| Section 6.1, "29.8% to 6.5% (p=0.006)" | 29.8% to 6.5% (p=0.006) | 28.0% to 6.0% (p=0.003) |
| Appendix B.1, full_cg_db 50-step exactly-N | 16.7% (4/24) | 16.0% (4/25) |
| Section 5.3, "28% of its bar" (cg_only) | 28% | 28% (no change) |
| Section 5.3, "6%" of its bar (full_cg_db) | 6% | 6% (no change) |

## 8. Impact Assessment

- No qualitative conclusions change. The cg_only vs full_cg_db exactly-N difference remains significant and medium-to-large (Cohen's h = 0.62).
- The base_llm vs full_cg_db comparison was already non-significant and stays non-significant.
- The revision makes Run 08 consistent with the denominator convention used in Tables 5 and B4 for Runs 03, 04, and 06.
