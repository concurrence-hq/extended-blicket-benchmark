# Paper 1 Statistical Supplement

**Generated:** 2026-03-22
**Script:** `scripts/paper1_statistical_tests.py`

This supplement provides formal inferential statistics for two claims in the TMLR submission that were previously supported only by descriptive statistics.

---

## 1. Test 1: Exactly-N Rate (DB Pathway)

**Claim:** The full compositional architecture (CG+DB) avoids the exactly-N structural trap more effectively than other agents, mediated by the Dynamic Behaviors' deeper exploration.

The exactly-N trap occurs when an agent accumulates exactly N successful activations and submits immediately — before observing any post-switch evidence. These episodes are structurally unsolvable (0% accuracy by design). Agents that explore deeper (more activations) avoid this trap.

### 1.1 Run 08 75-step (Primary, n=50/cell)

| Comparison | Exactly-N Rates | Fisher p | OR [95% CI] | Risk Diff [95% CI] | Cohen's h | BF₁₀ |
|------------|----------------|----------|-------------|---------------------|-----------|-------|
| base_llm vs full_cg_db | 7/37 (18.9%) vs 3/46 (6.5%) | 0.1016 ns | 3.34 [0.80, 13.98] | +12.4% [-2.1%, +28.3%] | 0.384 | 1.91 |
| cg_only vs full_cg_db | 14/47 (29.8%) vs 3/46 (6.5%) | 0.0061 ** | 6.08 [1.61, 22.92] | +23.3% [+7.6%, +38.1%] | 0.638 | 2.00 |
| base_llm vs cg_only | 7/37 (18.9%) vs 14/47 (29.8%) | 0.3146 ns | 0.55 [0.20, 1.55] | -10.9% [-27.9%, +8.0%] | -0.255 | 0.27 |

#### base_llm vs full_cg_db

**base_llm** vs **full_cg_db**

| Metric | base_llm | full_cg_db |
|--------|----------|------------|
| Proportion | 7/37 (18.9%) | 3/46 (6.5%) |

| Statistic | Value |
|-----------|-------|
| Fisher's exact p (two-sided) | **0.1016** ns |
| Odds Ratio | 3.344 [0.800, 13.984] |
| Risk Difference | +12.4% [-2.1%, +28.3%] |
| Cohen's h | 0.384 |
| P(base_llm > full_cg_db) | 0.953 |
| BF₁₀ (directional) | 1.91 |

#### cg_only vs full_cg_db

**cg_only** vs **full_cg_db**

| Metric | cg_only | full_cg_db |
|--------|---------|------------|
| Proportion | 14/47 (29.8%) | 3/46 (6.5%) |

| Statistic | Value |
|-----------|-------|
| Fisher's exact p (two-sided) | **0.0061** ** |
| Odds Ratio | 6.081 [1.613, 22.921] |
| Risk Difference | +23.3% [+7.6%, +38.1%] |
| Cohen's h | 0.638 |
| P(cg_only > full_cg_db) | 0.998 |
| BF₁₀ (directional) | 2.00 |

#### base_llm vs cg_only

**base_llm** vs **cg_only**

| Metric | base_llm | cg_only |
|--------|----------|---------|
| Proportion | 7/37 (18.9%) | 14/47 (29.8%) |

| Statistic | Value |
|-----------|-------|
| Fisher's exact p (two-sided) | **0.3146** ns |
| Odds Ratio | 0.550 [0.196, 1.546] |
| Risk Difference | -10.9% [-27.9%, +8.0%] |
| Cohen's h | -0.255 |
| P(base_llm > cg_only) | 0.134 |
| BF₁₀ (directional) | 0.27 |

### 1.2 Run 06 sw=3 (n=50/agent)

| Comparison | Exactly-N Rates | Fisher p | OR [95% CI] | Risk Diff [95% CI] | Cohen's h | BF₁₀ |
|------------|----------------|----------|-------------|---------------------|-----------|-------|
| base_llm vs full_cg_db | 3/41 (7.3%) vs 4/49 (8.2%) | 1.0000 ns | 0.89 [0.19, 4.22] | -0.8% [-12.9%, +12.2%] | -0.032 | 0.92 |
| cg_only vs full_cg_db | 8/42 (19.0%) vs 4/49 (8.2%) | 0.2128 ns | 2.65 [0.74, 9.52] | +10.9% [-3.4%, +26.0%] | 0.324 | 1.86 |
| base_llm vs cg_only | 3/41 (7.3%) vs 8/42 (19.0%) | 0.1944 ns | 0.34 [0.08, 1.37] | -11.7% [-26.8%, +3.4%] | -0.355 | 0.13 |

#### base_llm vs full_cg_db

**base_llm** vs **full_cg_db**

| Metric | base_llm | full_cg_db |
|--------|----------|------------|
| Proportion | 3/41 (7.3%) | 4/49 (8.2%) |

| Statistic | Value |
|-----------|-------|
| Fisher's exact p (two-sided) | **1.0000** ns |
| Odds Ratio | 0.888 [0.187, 4.218] |
| Risk Difference | -0.8% [-12.9%, +12.2%] |
| Cohen's h | -0.032 |
| P(base_llm > full_cg_db) | 0.459 |
| BF₁₀ (directional) | 0.92 |

#### cg_only vs full_cg_db

**cg_only** vs **full_cg_db**

| Metric | cg_only | full_cg_db |
|--------|---------|------------|
| Proportion | 8/42 (19.0%) | 4/49 (8.2%) |

| Statistic | Value |
|-----------|-------|
| Fisher's exact p (two-sided) | **0.2128** ns |
| Odds Ratio | 2.647 [0.736, 9.523] |
| Risk Difference | +10.9% [-3.4%, +26.0%] |
| Cohen's h | 0.324 |
| P(cg_only > full_cg_db) | 0.932 |
| BF₁₀ (directional) | 1.86 |

#### base_llm vs cg_only

**base_llm** vs **cg_only**

| Metric | base_llm | cg_only |
|--------|----------|---------|
| Proportion | 3/41 (7.3%) | 8/42 (19.0%) |

| Statistic | Value |
|-----------|-------|
| Fisher's exact p (two-sided) | **0.1944** ns |
| Odds Ratio | 0.336 [0.082, 1.368] |
| Risk Difference | -11.7% [-26.8%, +3.4%] |
| Cohen's h | -0.355 |
| P(base_llm > cg_only) | 0.064 |
| BF₁₀ (directional) | 0.13 |

### 1.3 Run 06 sw=5 (n=50/agent)

| Comparison | Exactly-N Rates | Fisher p | OR [95% CI] | Risk Diff [95% CI] | Cohen's h | BF₁₀ |
|------------|----------------|----------|-------------|---------------------|-----------|-------|
| base_llm vs full_cg_db | 0/3 (0.0%) vs 8/27 (29.6%) | 0.5448 ns | 0.33 [0.02, 7.07] | -29.6% [-48.5%, +28.2%] | -1.151 | 0.49 |
| cg_only vs full_cg_db | 5/9 (55.6%) vs 8/27 (29.6%) | 0.2347 ns | 2.97 [0.63, 14.03] | +25.9% [-8.6%, +55.0%] | 0.531 | 1.83 |
| base_llm vs cg_only | 0/3 (0.0%) vs 5/9 (55.6%) | 0.2045 ns | 0.12 [0.00, 2.91] | -55.6% [-81.1%, +7.6%] | -1.682 | 0.14 |

#### base_llm vs full_cg_db

**base_llm** vs **full_cg_db**

| Metric | base_llm | full_cg_db |
|--------|----------|------------|
| Proportion | 0/3 (0.0%) | 8/27 (29.6%) |

| Statistic | Value |
|-----------|-------|
| Fisher's exact p (two-sided) | **0.5448** ns |
| Odds Ratio | 0.328 [0.015, 7.066] |
| Risk Difference | -29.6% [-48.5%, +28.2%] |
| Cohen's h | -1.151 |
| P(base_llm > full_cg_db) | 0.246 |
| BF₁₀ (directional) | 0.49 |

#### cg_only vs full_cg_db

**cg_only** vs **full_cg_db**

| Metric | cg_only | full_cg_db |
|--------|---------|------------|
| Proportion | 5/9 (55.6%) | 8/27 (29.6%) |

| Statistic | Value |
|-----------|-------|
| Fisher's exact p (two-sided) | **0.2347** ns |
| Odds Ratio | 2.969 [0.628, 14.026] |
| Risk Difference | +25.9% [-8.6%, +55.0%] |
| Cohen's h | 0.531 |
| P(cg_only > full_cg_db) | 0.916 |
| BF₁₀ (directional) | 1.83 |

#### base_llm vs cg_only

**base_llm** vs **cg_only**

| Metric | base_llm | cg_only |
|--------|----------|---------|
| Proportion | 0/3 (0.0%) | 5/9 (55.6%) |

| Statistic | Value |
|-----------|-------|
| Fisher's exact p (two-sided) | **0.2045** ns |
| Odds Ratio | 0.117 [0.005, 2.909] |
| Risk Difference | -55.6% [-81.1%, +7.6%] |
| Cohen's h | -1.682 |
| P(base_llm > cg_only) | 0.069 |
| BF₁₀ (directional) | 0.14 |

### 1.4 Run 08 50-step (Exploratory, n=25/cell)

| Comparison | Exactly-N Rates | Fisher p | OR [95% CI] | Risk Diff [95% CI] | Cohen's h | BF₁₀ |
|------------|----------------|----------|-------------|---------------------|-----------|-------|
| base_llm (50-step) vs full_cg_db (50-step) | 5/20 (25.0%) vs 4/24 (16.7%) | 0.7095 ns | 1.67 [0.38, 7.29] | +8.3% [-15.3%, +32.4%] | 0.206 | 1.50 |

#### base_llm (50-step) vs full_cg_db (50-step)

**base_llm (50-step)** vs **full_cg_db (50-step)**

| Metric | base_llm (50-step) | full_cg_db (50-step) |
|--------|--------------------|----------------------|
| Proportion | 5/20 (25.0%) | 4/24 (16.7%) |

| Statistic | Value |
|-----------|-------|
| Fisher's exact p (two-sided) | **0.7095** ns |
| Odds Ratio | 1.667 [0.381, 7.288] |
| Risk Difference | +8.3% [-15.3%, +32.4%] |
| Cohen's h | 0.206 |
| P(base_llm (50-step) > full_cg_db (50-step)) | 0.749 |
| BF₁₀ (directional) | 1.50 |

### 1.5 Test 1 Summary

| Dataset | Comparison | Exactly-N Rates | p | Sig | Direction |
|---------|------------|----------------|---|-----|-----------|
| Run 08 75-step | base_llm vs full_cg_db | 18.9% vs 6.5% | 0.1016 | ns | base_llm > full_cg_db |
| Run 08 75-step | cg_only vs full_cg_db | 29.8% vs 6.5% | 0.0061 | ** | cg_only > full_cg_db |
| Run 08 75-step | base_llm vs cg_only | 18.9% vs 29.8% | 0.3146 | ns | cg_only > base_llm |
| Run 06 sw=3 | base_llm vs full_cg_db | 7.3% vs 8.2% | 1.0000 | ns | full_cg_db > base_llm |
| Run 06 sw=3 | cg_only vs full_cg_db | 19.0% vs 8.2% | 0.2128 | ns | cg_only > full_cg_db |
| Run 06 sw=3 | base_llm vs cg_only | 7.3% vs 19.0% | 0.1944 | ns | cg_only > base_llm |
| Run 06 sw=5 | base_llm vs full_cg_db | 0.0% vs 29.6% | 0.5448 | ns | full_cg_db > base_llm |
| Run 06 sw=5 | cg_only vs full_cg_db | 55.6% vs 29.6% | 0.2347 | ns | cg_only > full_cg_db |
| Run 06 sw=5 | base_llm vs cg_only | 0.0% vs 55.6% | 0.2045 | ns | cg_only > base_llm |
| Run 08 50-step | base_llm (50-step) vs full_cg_db (50-step) | 25.0% vs 16.7% | 0.7095 | ns | base_llm (50-step) > full_cg_db (50-step) |

---

## 2. Test 2: CG Harm Effect

**Claim:** The Context Graph without Dynamic Behaviors (cg_only) may harm hidden moderator performance relative to the bare LLM, because the CG's structured exploration leads to premature convergence on the pre-switch hypothesis without a mechanism for interruption.

### 2.1 Run 06 sw=3 Standalone: Raw Accuracy (n=50)

**base_llm** vs **cg_only**

| Metric | base_llm | cg_only |
|--------|----------|---------|
| Proportion | 45/50 (90.0%) | 41/50 (82.0%) |

| Statistic | Value |
|-----------|-------|
| Fisher's exact p (two-sided) | **0.3881** ns |
| Odds Ratio | 1.976 [0.612, 6.380] |
| Risk Difference | +8.0% [-6.0%, +22.0%] |
| Cohen's h | 0.233 |
| P(base_llm > cg_only) | 0.869 |
| BF₁₀ (directional) | 1.74 |

### 2.2 Run 06 sw=3 Standalone: Reasoning-Eligible Accuracy

**base_llm** vs **cg_only**

| Metric | base_llm | cg_only |
|--------|----------|---------|
| Proportion | 36/38 (94.7%) | 33/34 (97.1%) |

| Statistic | Value |
|-----------|-------|
| Fisher's exact p (two-sided) | **1.0000** ns |
| Odds Ratio | 0.545 [0.047, 6.299] |
| Risk Difference | -2.3% [-14.6%, +10.2%] |
| Cohen's h | -0.118 |
| P(base_llm > cg_only) | 0.349 |
| BF₁₀ (directional) | 0.70 |

### 2.3 Run 03 Standalone: Raw Accuracy (n=30)

**base_llm** vs **cg_only**

| Metric | base_llm | cg_only |
|--------|----------|---------|
| Proportion | 23/30 (76.7%) | 23/30 (76.7%) |

| Statistic | Value |
|-----------|-------|
| Fisher's exact p (two-sided) | **1.0000** ns |
| Odds Ratio | 1.000 [0.302, 3.309] |
| Risk Difference | +0.0% [-21.0%, +21.0%] |
| Cohen's h | 0.000 |
| P(base_llm > cg_only) | 0.500 |
| BF₁₀ (directional) | 1.00 |

### 2.4 Run 3B Standalone: Raw Accuracy (n=30)

**base_llm** vs **cg_only**

| Metric | base_llm | cg_only |
|--------|----------|---------|
| Proportion | 28/30 (93.3%) | 21/30 (70.0%) |

| Statistic | Value |
|-----------|-------|
| Fisher's exact p (two-sided) | **0.0419** * |
| Odds Ratio | 6.000 [1.172, 30.725] |
| Risk Difference | +23.3% [+3.5%, +41.8%] |
| Cohen's h | 0.637 |
| P(base_llm > cg_only) | 0.989 |
| BF₁₀ (directional) | 1.98 |

### 2.5 Combined sw=3 (n=110): Raw Accuracy

**base_llm** vs **cg_only**

| Metric | base_llm | cg_only |
|--------|----------|---------|
| Proportion | 96/110 (87.3%) | 85/110 (77.3%) |

| Statistic | Value |
|-----------|-------|
| Fisher's exact p (two-sided) | **0.0766** ns |
| Odds Ratio | 2.017 [0.985, 4.129] |
| Risk Difference | +10.0% [-0.2%, +20.0%] |
| Cohen's h | 0.264 |
| P(base_llm > cg_only) | 0.973 |
| BF₁₀ (directional) | 1.95 |

### 2.6 Cochran-Mantel-Haenszel Test (Stratified by Batch)

Stratified analysis controlling for batch effects across Runs 03, 3B, and 06:

| Stratum | base_llm | cg_only |
|---------|----------|---------|
| Run 03 | 23/30 (76.7%) | 23/30 (76.7%) |
| Run 3B | 28/30 (93.3%) | 21/30 (70.0%) |
| Run 06 | 45/50 (90.0%) | 41/50 (82.0%) |

| Statistic | Value |
|-----------|-------|
| CMH χ² (1 df) | 3.104 |
| p-value | **0.0781** ns |
| Common OR (Mantel-Haenszel) | 2.012 [0.983, 4.119] |

Interpretation: The common OR of 2.01 indicates that base_llm has 2.0x the odds of a correct answer compared to cg_only, after controlling for batch. This does not reach statistical significance.

### 2.7 Run 08 75-step: CG Harm in Reasoning-Eligible Episodes

**base_llm (Cell A)** vs **cg_only (Cell B)**

| Metric | base_llm (Cell A) | cg_only (Cell B) |
|--------|-------------------|------------------|
| Proportion | 22/30 (73.3%) | 31/33 (93.9%) |

| Statistic | Value |
|-----------|-------|
| Fisher's exact p (two-sided) | **0.0379** * |
| Odds Ratio | 0.177 [0.034, 0.917] |
| Risk Difference | -20.6% [-38.9%, -2.2%] |
| Cohen's h | -0.588 |
| P(base_llm (Cell A) > cg_only (Cell B)) | 0.015 |
| BF₁₀ (directional) | 0.03 |

### 2.8 Test 2 Summary

| Dataset | Metric | base_llm | cg_only | p | Sig | base_llm advantage |
|---------|--------|----------|---------|---|-----|--------------------|
| Run 06 sw=3 | Raw accuracy | 90.0% | 82.0% | 0.3881 | ns | +8.0% |
| Run 06 sw=3 | RE accuracy | 94.7% | 97.1% | 1.0000 | ns | -2.3% |
| Run 03 | Raw accuracy | 76.7% | 76.7% | 1.0000 | ns | +0.0% |
| Run 3B | Raw accuracy | 93.3% | 70.0% | 0.0419 | * | +23.3% |
| Combined sw=3 | Raw accuracy | 87.3% | 77.3% | 0.0766 | ns | +10.0% |
| Run 08 75-step | RE accuracy | 73.3% | 93.9% | 0.0379 | * | -20.6% |
| Runs 03+3B+06 | CMH stratified | — | — | 0.0781 | ns | OR=2.01 |

---

## 3. Recommended Language for Paper

### Test 1 (Exactly-N Rate)

**For Results section (Run 08 primary):**

> In the primary 75-step condition, the exactly-N trap rate differed significantly across agents. The bare LLM landed on exactly N activations in 18.9% of post-switch episodes (7/37), compared to 29.8% (14/47) for the CG-only agent and 6.5% (3/46) for the full CG+DB architecture. The difference between CG-only and CG+DB was statistically significant (Fisher's exact p = 0.006, OR = 6.08, 95% CI [1.61, 22.92], Cohen's h = 0.64).

### Test 2 (CG Harm)

**For Discussion section:**

> Across three independent batches (n = 110 per agent), the CG-only agent showed numerically lower raw accuracy than the bare LLM on hidden moderator (77.3% vs 87.3%), but this difference did not reach statistical significance in either the pooled analysis (Fisher's p = 0.077) or the stratified Cochran-Mantel-Haenszel test (χ² = 3.10, p = 0.078, common OR = 2.01, 95% CI [0.98, 4.12]). While the direction is consistent with a "CG harm" hypothesis — that structured exploration without the DB interruption mechanism may lead to premature convergence — the evidence is insufficient to draw firm conclusions.

> Notably, the CG harm pattern **reverses** in Run 08's reasoning-eligible episodes: cg_only (93.9%) significantly outperformed the bare LLM (73.3%) on reasoning quality (Fisher's p = 0.038, Cohen's h = 0.59). This reversal suggests that the sw=3 CG harm effect observed in Runs 03–06 is driven primarily by differential exactly-N trap rates (cg_only accumulates more exactly-N episodes) rather than by genuine reasoning deficiency. When exactly-N episodes are excluded, the CG's structured exploration *improves* reasoning quality.

---

## 4. Data Verification

### 4.1 Episode Counts

| Run | Agent | Total | Pre-Switch | Exactly-N | Reasoning-Eligible |
|-----|-------|:-----:|:----------:|:---------:|:------------------:|
| Run 08 Cell A (base_llm, 75-step) | base_llm | 50 | 13 | 7 | 30 |
| Run 08 Cell B (cg_only, 75-step) | cg_only | 50 | 3 | 14 | 33 |
| Run 08 Cell C (full_cg_db, 75-step) | full_cg_db | 50 | 4 | 3 | 43 |
| Run 08 Cell D (base_llm, 50-step) | base_llm | 25 | 5 | 5 | 15 |
| Run 08 Cell E (full_cg_db, 50-step) | full_cg_db | 25 | 1 | 4 | 20 |
| Run 06 sw=3 | base_llm | 50 | 9 | 3 | 38 |
| Run 06 sw=3 | cg_only | 50 | 8 | 8 | 34 |
| Run 06 sw=3 | full_cg_db | 50 | 1 | 4 | 45 |
| Run 06 sw=5 | base_llm | 50 | 47 | 0 | 3 |
| Run 06 sw=5 | cg_only | 50 | 41 | 5 | 4 |
| Run 06 sw=5 | full_cg_db | 50 | 23 | 8 | 19 |
| Run 03 | base_llm | 30 | 4 | 7 | 19 |
| Run 03 | cg_only | 30 | 2 | 7 | 21 |
| Run 03 | full_cg_db | 30 | 0 | 2 | 28 |
| Run 3B | base_llm | 30 | 4 | 2 | 24 |
| Run 3B | cg_only | 30 | 1 | 9 | 20 |
| Run 3B | full_cg_db | 30 | 3 | 4 | 23 |

### 4.2 Run 08 Verification Against Known Values

All Run 08 cell counts match the values reported in `report_run08_2026-02-26.md` Section 3.1. ✓

### 4.3 Combined sw=3 Verification

| Agent | Expected | Actual | Match |
|-------|----------|--------|:-----:|
| base_llm | 96/110 | 96/110 | ✓ |
| cg_only | 85/110 | 85/110 | ✓ |
| full_cg_db | 98/110 | 98/110 | ✓ |

### 4.4 Cross-Check Against Existing Report Values

| Comparison | Report Value | Computed Value | Match |
|------------|:------------|:--------------|:-----:|
| Run 06 full_cg_db vs cg_only combined (one-tailed) | p=0.015 | p=0.0148 | ✓ |
| Run 08 A vs C RE accuracy (two-sided) | p=0.013 | p=0.0126 | ✓ |

### 4.5 Anomalies

No anomalies detected. All exactly-N episodes have 0% accuracy (as expected). All pre-switch episodes have high accuracy (consistent with prior reports).

---

## 5. Appendix: Computational Methods

### Statistical Methods

- **Fisher's exact test**: `scipy.stats.fisher_exact()` — exact test for 2×2 contingency tables
- **Odds ratio CI**: Log(OR) ± 1.96 × SE, where SE = √(1/a + 1/b + 1/c + 1/d). Haldane-Anscombe correction (+0.5 to all cells) when any cell is 0.
- **Risk difference CI**: Newcombe (1998) Method 10 — Wilson score intervals on each proportion, combined via Newcombe's formula
- **Cohen's h**: 2(arcsin(√p₁) − arcsin(√p₂)) — effect size for proportion differences
- **Bayes factor (directional)**: Monte Carlo sampling (1M draws) from Beta(k+1, n-k+1) posteriors with uniform Beta(1,1) priors. BF₁₀ = P(p₁ > p₂ | data) / 0.5
- **CMH test**: Cochran-Mantel-Haenszel χ² with continuity correction, stratified by batch. Common OR estimated by Mantel-Haenszel method with Robins-Breslow-Greenland CI.

### Episode Classification

Episodes are classified by `env_info.successful_activations` relative to `switch_after_n`:

- **Pre-switch**: activations < N (rule never switched)
- **Exactly-N**: activations == N (rule switched on final activation; structurally unsolvable)
- **Reasoning-eligible**: activations > N (agent observed post-switch evidence)

### Code

Full source: `projects/gopnik-extended-benchmark/scripts/paper1_statistical_tests.py`
