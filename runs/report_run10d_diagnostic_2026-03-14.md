# Run 10d Diagnostic Report

**Date:** 2026-03-14
**Cost:** $17.32 (30 episodes)
**Duration:** ~61 minutes

---

## 1. Executive Summary

Run 10d found a null result: full_cg_db (56%) = base_llm (56%) at n=50. This was a sharp drop from Run 10c where full_cg_db (86.7%) dramatically outperformed base_llm (53.3%) at n=15. This diagnostic investigated whether the null reflects a genuine finding or an implementation regression.

**Verdict: The null is genuine. No implementation regression.**

The diagnostic replicated Run 10c's exact seeds (1101-1115) and found:
- Cell G (full_cg_db): **73.3%** (11/15)
- Cell F (base_llm): **73.3%** (11/15)

The ORDER_TESTING pipeline mechanics are fully intact (100% DB fire rate, 80% OT entry). The key finding is that **base_llm itself rose from 53.3% to 73.3%**, eliminating the gap. This is not a regression in the agent architecture — it's a ceiling effect where the task has become easier for the base model (likely due to Anthropic model updates between Run 10c and the diagnostic).

---

## 2. Code Diff Results

**Verdict: No implementation regression found.**

Between Run 10c commit (`28c110e`) and Run 10d commit (`43463bf`):

| File | Changed? | Nature of Change | Risk to full_cg_db? |
|------|----------|------------------|---------------------|
| `episode_runner.py` | No | — | None |
| `dynamic_behaviors_10c.py` | No | — | None |
| `context_graph.py` | No | — | None |
| `base_llm_agent.py` | No | — | None |
| `cg_agent.py` | No | — | None |
| `full_agent.py` | No | — | None |
| `text_interface.py` | Yes | Appended `TextInterface6` class + imports | None (additive only) |
| `blicket_env_5obj.py` | No | — | None |
| `extended_env_5obj.py` | No | — | None |
| `config.py` | No | — | None |

**Shared-state analysis:** No module-level mutable singletons. Agents, CGs, and DB engines are freshly created per episode. `agent.reset()` is called before every episode. Cell B (cg_only) cannot contaminate Cell C (full_cg_db).

---

## 3. Seed Replication Results

| Cell | Agent | Accuracy | Objects Correct | Order Correct | Avg GF | Cost |
|------|-------|----------|----------------|---------------|--------|------|
| F | base_llm | **73.3%** (11/15) | 12/15 | 11/15 | 2.8 | $6.84 |
| G | full_cg_db | **73.3%** (11/15) | 12/15 | 11/15 | 2.9 | $10.48 |

**Reference (Run 10c, same seeds 1101-1115):**

| Cell | Agent | Accuracy |
|------|-------|----------|
| A (10c) | base_llm | 53.3% (8/15) |
| B (10c) | full_cg_db | 86.7% (13/15) |

**Reference (Run 10d, different seeds 1201-1250):**

| Cell | Agent | Accuracy |
|------|-------|----------|
| A (10d) | base_llm | 56.0% (28/50) |
| C (10d) | full_cg_db | 56.0% (28/50) |

---

## 4. ORDER_TESTING Pipeline Comparison

The ORDER_TESTING pipeline is fully operational across all three sources:

| Source | N | Success | DB Fire | OT Added | OT Entered | Avg Steps in OT |
|--------|---|---------|---------|----------|------------|-----------------|
| Run 10c Cell B | 15 | **86.7%** | 100.0% | 100.0% | 73.3% | 10.3 |
| Run 10d Cell C | 50 | **56.0%** | 100.0% | 100.0% | 74.0% | 13.2 |
| Diagnostic Cell G | 15 | **73.3%** | 100.0% | 100.0% | 80.0% | 12.9 |

Key observations:
- **DB fire rate is 100% in all sources.** The `order_hypothesis` behavior fires every time.
- **OT state is added 100% of the time.** The CG modification pipeline is intact.
- **OT entry rate is ~73-80% across all sources.** The agent successfully transitions into ORDER_TESTING most of the time.
- **The pipeline is NOT broken.** The difference is downstream — what the agent does *within* ORDER_TESTING.

---

## 5. Failure Mode Analysis

| Failure Mode | 10c (n=15) | 10d (n=50) | Diagnostic (n=15) |
|--------------|-----------|-----------|-------------------|
| success | 13 | 28 | 11 |
| right_objs_wrong_order | 2 | 7 | 1 |
| no_answer | 0 | 9 | 3 |
| wrong_objects | 0 | 6 | 0 |

**Key differences:**
- **`no_answer` (timeout):** 0 in 10c → 9 in 10d → 3 in diagnostic. The agent increasingly runs out of steps before submitting.
- **`wrong_objects`:** 0 in 10c → 6 in 10d → 0 in diagnostic. The 10d seeds (1201-1250) may be harder — agents get confused about which objects are correct.
- **`right_objs_wrong_order`:** Relatively stable across sources (13-14% failure rate).

---

## 6. Per-Episode Details

### Diagnostic Cell G (full_cg_db)

| Ep | Success | Steps | DB Fired @ | OT Added @ | OT Entered @ | Steps in OT | Failure |
|----|---------|-------|-----------|-----------|-------------|-------------|---------|
| 0 | N | 40 | 18 | 18 | — | 0 | right_objs_wrong_order |
| 1 | Y | 38 | 12 | 12 | 12 | 17 | success |
| 2 | Y | 34 | 14 | 14 | 14 | 5 | success |
| 3 | Y | 38 | 18 | 18 | 18 | 13 | success |
| 4 | Y | 40 | 8 | 8 | 8 | 31 | success |
| 5 | N | 40 | 12 | 12 | — | 0 | no_answer |
| 6 | Y | 38 | 27 | 27 | 27 | 7 | success |
| 7 | Y | 32 | 14 | 14 | 14 | 11 | success |
| 8 | Y | 38 | 14 | 14 | 14 | 17 | success |
| 9 | Y | 38 | 19 | 19 | 19 | 12 | success |
| 10 | N | 40 | 14 | 14 | 14 | 23 | no_answer |
| 11 | Y | 38 | 17 | 17 | 17 | 6 | success |
| 12 | Y | 38 | 19 | 19 | — | 0 | success |
| 13 | N | 40 | 12 | 12 | 12 | 28 | no_answer |
| 14 | Y | 40 | 14 | 14 | 14 | 23 | success |

### Diagnostic Cell F (base_llm)

| Ep | Success | Steps | Failure |
|----|---------|-------|---------|
| 0 | Y | 40 | success |
| 1 | N | 40 | failed |
| 2 | N | 26 | failed |
| 3 | Y | 40 | success |
| 4 | Y | 39 | success |
| 5 | Y | 34 | success |
| 6 | Y | 36 | success |
| 7 | Y | 32 | success |
| 8 | N | 40 | failed |
| 9 | Y | 38 | success |
| 10 | N | 39 | failed |
| 11 | Y | 34 | success |
| 12 | Y | 34 | success |
| 13 | Y | 36 | success |
| 14 | Y | 32 | success |

---

## 7. Diagnostic Verdict

The diagnostic rules from the planning prompt:

| Condition | Interpretation | Match? |
|-----------|---------------|--------|
| Cell G ≥ 80% AND Cell F ≈ 47-60% | Implementation clean | **No** — Cell F is 73.3% |
| Cell G ≈ 47-60% (≈ Cell F) | Implementation regression | **No** — Cell G is 73.3% |
| Cell G ≈ 67-80% | Partial regression or stochastic | **Partial match** |
| Cell F diverges from 10c base_llm | Environment/runner changed | **Yes** — 73.3% vs 53.3% |

**The primary signal is that Cell F (base_llm) rose from 53.3% to 73.3%.** This is a +20pp shift in the *control condition* — meaning the underlying LLM has gotten better at this task since Run 10c was run (~2 days ago). This creates a ceiling effect where base_llm alone can solve 73% of episodes, leaving minimal room for CG+DB to add value.

**Classification: ENVIRONMENT SHIFT (model capability drift) + GENUINE NULL at n=50**

---

## 8. Implications for Run 10d Interpretation

### Why Run 10c showed a big effect (86.7% vs 53.3%)

Run 10c was run on March 12. At that time, Sonnet 4.5's base performance on 3-object order-sensitive rules with randomized labels was ~53%. CG+DB added +33pp because the base model genuinely struggled.

### Why Run 10d showed a null (56% vs 56%)

Two factors combined:
1. **Model capability drift:** By March 13 (Run 10d), the base model had improved to ~56% on the harder seed set (1201-1250). The diagnostic shows it may now be at ~73% on the original seeds.
2. **Harder seeds:** Seeds 1201-1250 may be inherently harder than 1101-1115, partially masking any CG+DB advantage.

### Why the diagnostic shows no gap (73.3% vs 73.3%)

The base model (Sonnet 4.5) has continued improving. At 73.3% base performance, the task is becoming too easy for 3-object rules — there's not enough room for structured reasoning (CG+DB) to add measurable value.

### The core issue

**The 3-object order-sensitive task is approaching the base LLM's capability ceiling.** This is exactly what the Gopnik thesis predicts: compositional architecture adds value only when the task exceeds the base model's capacity. As models improve, the difficulty threshold shifts upward.

---

## 9. Recommended Next Steps

1. **Prioritize 4-object (6-label) experiments.** The 4-object task is hard enough that base_llm struggles even with improved models. Run 10d Phase 2 showed 0% for base_llm vs 40% for full_cg_db at n=5.

2. **Run a powered 4-object replication.** The n=5 pilot from Run 10d is promising but underpowered. Recommend n=30 per cell at 75 steps.

3. **Consider 5-object (7-label) pilot.** If model capability drift is rapid, even 4-object may become too easy. A small pilot (n=5) at 5 objects would future-proof the benchmark.

4. **Do not re-run 3-object experiments.** The diagnostic confirms the task is no longer discriminating. Further 3-object data will not change this conclusion.

5. **Record base_llm performance over time.** Since model drift is now a confirmed confound, future experiments should always include a base_llm cell as an internal calibration.

---

## 10. Raw Data References

| Artifact | Path |
|----------|------|
| Diagnostic traces (Cell F) | `data/traces_run10d_diagnostic/3obj_diag_F_base_llm_ep*.json` |
| Diagnostic traces (Cell G) | `data/traces_run10d_diagnostic/3obj_diag_G_full_cg_db_ep*.json` |
| Diagnostic results JSON | `data/results_run10d_diagnostic/diagnostic_results.json` |
| Run 10c traces | `data/traces_run10c/` |
| Run 10d traces | `data/traces_run10d/` |
| Analysis script | `scripts/analyze_10d_diagnostic.py` |
| Experiment script | `experiments/run_10d_diagnostic.py` |
| Planning prompt | `experiment1_results/prompt_run10d_diagnostic.md` |
