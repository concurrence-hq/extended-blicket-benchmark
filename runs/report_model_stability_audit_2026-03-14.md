# Retrospective Model Stability Audit

**Date:** 2026-03-14
**Scope:** Extended Blicket Benchmark, Runs 01–10d-diagnostic (Feb 13 – Mar 14, 2026)
**Motivation:** Run 10d diagnostic found base_llm performance on identical seeds shifted by +20pp between Run 10c (March 12) and the diagnostic (March 14)

---

## 1. Model String Verification

### 1.1 Trace Evidence

Every trace file across all runs records the action model in `agent_trace.model`. All traces examined show the same value.

| Run | Date | N Traces | Action Model (from traces) | Trigger Model | Match? |
|-----|------|---------|---------------------------|---------------|--------|
| 01 | Feb 13 | 0 (no structured traces) | N/A | N/A | — |
| 02 | Feb 16 | ~120 | `claude-sonnet-4-5-20250929` | (Haiku used, string not recorded) | Y |
| 03 | Feb 17–18 | ~120 | `claude-sonnet-4-5-20250929` | (Haiku used, string not recorded) | Y |
| 03b | Feb 18 | ~30 | `claude-sonnet-4-5-20250929` | (Haiku used, string not recorded) | Y |
| 04 | Feb 19 | ~45 | `claude-sonnet-4-5-20250929` | (Haiku used, string not recorded) | Y |
| 05 | Feb 20 | — | (no separate trace dir found) | — | — |
| 06 | Feb 24 | ~100+ | `claude-sonnet-4-5-20250929` | (Haiku used, string not recorded) | Y |
| 07 | Feb 25 | ~120 | `claude-sonnet-4-5-20250929` | (Haiku used, string not recorded) | Y |
| 08 | Feb 26 | ~150+ | `claude-sonnet-4-5-20250929` | (Haiku used, string not recorded) | Y |
| 09 | Mar 10 | 60 | `claude-sonnet-4-5-20250929` | (Haiku used, string not recorded) | Y |
| 10 | Mar 11 | 60 | `claude-sonnet-4-5-20250929` | (Haiku used, string not recorded) | Y |
| 10b | Mar 11 | 60 | `claude-sonnet-4-5-20250929` | (Haiku used, string not recorded) | Y |
| 10c | Mar 12 | 60 | `claude-sonnet-4-5-20250929` | (Haiku used, string not recorded) | Y |
| 10d | Mar 13–14 | 160 | `claude-sonnet-4-5-20250929` | (Haiku used, string not recorded) | Y |
| 10d-diag | Mar 14 | 30 | `claude-sonnet-4-5-20250929` | (Haiku used, string not recorded) | Y |

### 1.2 Code Evidence

`experiments/config.py` defines defaults:
- `action_model: str = "claude-sonnet-4-5-20250929"`
- `trigger_model: str = "claude-haiku-4-5-20251001"`

No experiment script overrides these values. Git history shows no commits that modified the model strings.

### 1.3 Trigger Model Gap

The trigger model string (`claude-haiku-4-5-20251001`) is NOT recorded in trace files — only the action model is stored. The `db_trigger_tokens` field confirms Haiku was called (nonzero trigger token counts for full_cg_db episodes), but the actual response model is not logged.

**Finding: No model string discrepancy detected. All runs used `claude-sonnet-4-5-20250929` for actions. Trigger model string not recorded in traces (gap).**

---

## 2. Sampling Parameter Verification

### 2.1 Code Analysis

| Parameter | Value | Source | Consistent Across Runs? |
|-----------|-------|--------|------------------------|
| temperature | **Not set** (API default) | Not in any agent/config file | N/A — relies on server default |
| top_p | **Not set** (API default) | Not in any agent/config file | N/A — relies on server default |
| top_k | **Not set** (API default) | Not in any agent/config file | N/A — relies on server default |
| max_tokens (action) | **1024** | `base_llm_agent.py` line 165 | Yes — unchanged in git history |
| max_tokens (trigger) | **10** | `dynamic_behaviors.py` line 271 | Yes — unchanged in git history |

### 2.2 Risk Assessment

**HIGH RISK:** Temperature, top_p, and top_k are not explicitly set. This means:
- Anthropic can change server-side defaults without any code or config change being visible
- There is no way to verify from traces what sampling parameters were actually used
- If Anthropic changed the default temperature for `claude-sonnet-4-5-20250929` between Feb 13 and Mar 14, this would be invisible to us

**Recommendation:** Pin `temperature=1.0` (or whatever the current default is) explicitly in `base_llm_agent.py` to prevent silent drift.

---

## 3. Base_llm Performance Trajectory

### 3.1 Hidden Moderator Conditions (Runs 02–08)

These runs used the hidden moderator paradigm on 5 objects. The comparable condition is `hidden_moderator` with switch=3 (sw3).

| Run | Date | Condition | N | base_llm Accuracy | 95% CI |
|-----|------|-----------|---|:-----------------:|--------|
| 02 | Feb 16 | hidden_moderator | 30 | 100.0% | — |
| 03 | Feb 17 | hidden_moderator | 30 | 76.7% | ±15.1pp |
| 03b | Feb 18 | hidden_moderator | 30 | 93.3% | ±8.9pp |
| 06 | Feb 24 | hidden_moderator_sw3 | 50 | 90.0% | ±8.3pp |
| 08 | Feb 26 | hidden_moderator_conj_CDE_s75 | 50 | 70.0% | ±12.7pp |

**Note:** These conditions are NOT directly comparable — Run 02 used simple {A,B} conjunctive, while Run 08 used harder triple conjunction {C,D,E}. The declining accuracy primarily reflects increasing task difficulty, not model drift.

**Assessment:** Variation is explained by condition changes, not model drift.

### 3.2 Order-Sensitive Conditions (Runs 09–10d-diagnostic)

These are the runs most relevant to the drift question. All use order-sensitive rules with 5 objects.

| Run | Date | Condition | Step Budget | Feedback | Labels | N | base_llm Acc | 95% CI |
|-----|------|-----------|:-----------:|----------|--------|---|:------------:|--------|
| 02 | Feb 16 | order_sensitive | 50 | None | Fixed {A,B} | 30 | 100.0% | — |
| 03 | Feb 17 | order_sensitive | 50 | None | Fixed {A,B} | 30 | 100.0% | — |
| 09 | Mar 10 | 3obj_order_A | 20 | None | Fixed {A,B,C} | 15 | 6.7% | ±12.6pp |
| 10-A | Mar 11 | 3obj_order_pf_A | 20 | Partial | Fixed {A,B,C} | 15 | 6.7% | ±12.6pp |
| 10-C | Mar 11 | 3obj_order_pf_C | 40 | Partial | Fixed {A,B,C} | 15 | 86.7% | ±17.2pp |
| 10b | Mar 11 | 3obj_order_gf_A | 40 | Gradual | Fixed {A,B,C} | 15 | 53.3% | ±25.2pp |
| **10c** | **Mar 12** | **3obj_order_rand_A** | **40** | **Gradual** | **Randomized** | **15** | **53.3%** | **±25.2pp** |
| **10d** | **Mar 13** | **3obj_order_rand_A** | **40** | **Gradual** | **Randomized** | **50** | **56.0%** | **±13.8pp** |
| **10d-diag** | **Mar 14** | **3obj_diag_F** | **40** | **Gradual** | **Randomized** | **15** | **73.3%** | **±22.4pp** |

**Directly comparable conditions** (bolded): Runs 10c, 10d, and 10d-diagnostic all share:
- 3 objects, randomized labels, 40 steps, gradual feedback, same episode runner code

### 3.3 Drift Analysis

**Comparable 3-object order-sensitive (randomized labels, 40 steps, gradual feedback):**

| Run | Date | N | base_llm Accuracy |
|-----|------|---|:-----------------:|
| 10c | Mar 12 | 15 | 53.3% |
| 10d | Mar 13 | 50 | 56.0% |
| 10d-diag | Mar 14 | 15 | 73.3% |

- **Range:** 53.3% – 73.3% (spread: 20.0pp)
- **Expected 95% CI at n=15:** ±25.2pp (at p=0.53)
- **Expected 95% CI at n=50:** ±13.8pp (at p=0.56)

The 10c→10d-diagnostic shift (+20pp) is within the 95% CI for n=15 samples (±25pp), so it is **technically consistent with sampling noise**. However:

1. The shift is monotonically upward (53→56→73)
2. The 10d-diagnostic used the SAME seeds as 10c (1101-1115), so seed difficulty is controlled
3. The n=50 Run 10d (56.0%) is more reliable and sits between the two n=15 estimates

**Also notable:** Runs 02–03 (Feb 16–17) showed 100% base_llm accuracy on 2-object order-sensitive at 50 steps. This extremely high performance preceded the major condition change (2→3 objects, 50→20/40 steps, fixed→randomized labels), making it uninformative for drift detection.

### 3.4 Verdict on Trajectory

**Inconclusive.** The +20pp shift from 10c to 10d-diagnostic is within sampling noise for n=15, but the monotonic upward trend and identical seed control are suggestive. With only 3 data points and small samples, we cannot definitively distinguish model drift from sampling variation.

---

## 4. Configuration Consistency Check

### 4.1 Git History of Critical Files

| Commit | Date | Files Changed | Description | Runs Affected | Risk |
|--------|------|--------------|-------------|---------------|------|
| `00d09c3` | Feb 24 | All files | Reorganization — moved from pilot-blicket/ to standalone project | Baseline for all | None (structural) |
| `2e89208` | Mar 10 | `base_llm_agent.py`, `episode_runner.py`, `blicket_env.py`, `extended_env_5obj.py`, `text_interface.py` | Run 09: Added order-sensitive support — 3-tuple answer parsing, order_correct tracking, TextInterface5 | Run 09+ | **Med** — changed answer parsing for ALL agents |
| `b87c779` | Mar 11 | `config.py`, `episode_runner.py` | Run 10: Added PartialFeedbackConfig and partial feedback delivery in episode_runner | Run 10+ | Low — additive, gated behind config flag |
| `64189da` | Mar 11 | `config.py`, `episode_runner.py` | Run 10b: Added GradualFeedbackConfig and gradual feedback delivery | Run 10b+ | Low — additive, gated behind config flag |
| `28c110e` | Mar 12 | `dynamic_behaviors_10c.py` (NEW), `full_agent.py` | Run 10c: Revised ORDER_TESTING DB, FullAgent accepts `behaviors` param | Run 10c+ | **Med** — new behavior set, but only affects full_cg_db cells |
| `43463bf` | Mar 14 | `text_interface.py`, `blicket_env_6obj.py` (NEW), `extended_env_6obj.py` (NEW) | Run 10d: Added 6-object env + TextInterface6 | Run 10d Phase 2 only | None — new classes, no existing code changed |

### 4.2 Key Findings

1. **No changes to base_llm_agent.py since Run 09.** The agent code that drives base_llm cells has been stable since March 10.

2. **No changes to episode_runner.py since Run 10b.** The feedback mechanism and episode flow have been stable since March 11.

3. **No changes to config.py since Run 10b.** Model strings, cost limits, and thresholds unchanged.

4. **dynamic_behaviors_10c.py is new for Run 10c** but only affects full_cg_db cells. base_llm cells never interact with this code.

5. **System prompts (base_llm) unchanged.** The system prompt rendered by `TextInterface5` has been stable since Run 09.

6. **DB thresholds unchanged.** `fire_threshold=7.0` (order_hypothesis), stagnation threshold in `dynamic_behaviors.py` — no changes in git history since reorganization.

### 4.3 Configuration Risk Assessment

**No configuration changes explain the base_llm drift.** All code that could affect base_llm behavior (agent code, episode runner, environment, text interface) has been frozen since Run 10b (March 11). The +20pp shift from 10c to 10d-diagnostic occurred with zero code changes to any file in the base_llm path.

---

## 5. API Response Metadata Check

### 5.1 Metadata Availability

| Field | Present in Traces? |
|-------|--------------------|
| `agent_trace.model` | **Yes** — records the requested model string |
| `system_fingerprint` | No |
| `x-model-version` | No |
| `server_timing` | No |
| `api_version` | No |
| `stop_reason` | No |
| `stop_sequence` | No |
| `model_version` | No |

### 5.2 What's Stored vs. What's Not

The trace files store:
- Requested model string (`agent_trace.model`)
- Token usage (`agent_trace.token_usage`)
- Full conversation history
- Raw LLM responses

The trace files do **not** store:
- Response headers (where `x-model-version` would appear)
- The `model` field from the API response (which could differ from the request if Anthropic aliased versions)
- Any fingerprint or versioning metadata

### 5.3 Gap Analysis

**CRITICAL GAP:** The Anthropic API returns a `model` field in the response that reflects the actual model that served the request. This is NOT captured in our traces. If Anthropic silently updated `claude-sonnet-4-5-20250929` to point to a different underlying model checkpoint, we would have no record of it.

**Recommendation:** Capture `response.model` and `response.id` from API responses in future traces.

---

## 6. Overall Verdict

**Possible drift: suggestive evidence, but inconclusive.**

### Evidence For Drift
1. base_llm accuracy on identical seeds rose from 53.3% (Mar 12) to 73.3% (Mar 14) — a +20pp shift with zero code changes
2. The shift is monotonically upward across 3 consecutive measurements (53→56→73)
3. Seeds were controlled (1101-1115 in both 10c and 10d-diagnostic)
4. Sampling parameters rely on API defaults — invisible server-side changes are possible
5. API response metadata is not logged — no way to detect model version aliasing

### Evidence Against Drift
1. The +20pp shift is within the 95% CI for n=15 binomial samples (±25pp at p=0.53)
2. Only 3 data points in the comparable condition — insufficient for trend detection
3. The n=50 measurement (10d, 56.0%) is between the two n=15 estimates, consistent with the true rate being ~55-60%
4. Model string `claude-sonnet-4-5-20250929` is consistently recorded across all traces
5. No code changes to any file in the base_llm execution path since March 11

### Assessment

The observed variation is **consistent with sampling noise** (barely), but the controlled-seed comparison is concerning. If the true base_llm rate shifted from ~53% to ~73% in 48 hours with no code changes, the most likely explanation is a server-side model update that was invisible to our instrumentation.

**Cross-run comparison reliability:**
- Runs within the same day (e.g., 10 and 10b on March 11): **reliable**
- Runs 1-2 days apart (e.g., 10c and 10d): **probably reliable** (10d's n=50 estimate of 56% is close to 10c's 53%)
- Runs >2 days apart with small samples: **use caution** — may be confounded by model drift

---

## 7. Recommendations

### Immediate (before next experiment)

1. **Pin sampling parameters.** Add `temperature=1.0` (or current default) explicitly to all API calls in `base_llm_agent.py`, `cg_agent.py`, and `full_agent.py`. This prevents silent changes to server-side defaults.

2. **Log API response metadata.** Capture `response.model` and `response.id` from every API call and store in trace files. This detects model aliasing.

3. **Always include a base_llm calibration cell.** Every future run should include a base_llm cell on a fixed seed set (e.g., seeds 1101-1115, 40 steps, 3-obj randomized) as an internal calibration. If the calibration cell deviates from the historical baseline (~55%), model drift is confirmed.

### For current data interpretation

4. **Treat within-run comparisons as reliable.** Cells within the same run share identical model behavior. The key comparison (base_llm vs full_cg_db within Run 10d) is valid regardless of drift.

5. **Treat cross-run trends with caution.** The apparent decline from 10c (86.7% full_cg_db) to 10d (56.0% full_cg_db) cannot be cleanly attributed to seed difficulty vs. model drift without controlling for both simultaneously.

6. **Focus on within-run effect sizes.** For the paper, report effect sizes from individual runs (especially Run 10d with n=50) rather than cross-run trends.

### For future benchmark design

7. **Use larger samples.** n=15 is too small for drift detection (±25pp CI). n=50 reduces CI to ±14pp.

8. **Run calibration cells across days.** A small (n=5) daily calibration run on fixed seeds would detect drift early.

9. **Consider model pinning.** Contact Anthropic about whether `claude-sonnet-4-5-20250929` is guaranteed to be stable, or whether it may be silently updated. If unstable, request a fixed checkpoint identifier.

---

## Appendix A: Analysis Script

`scripts/model_stability_audit.py` — automated extraction of model strings, sampling parameters, performance trajectory, and API metadata across all runs.

## Appendix B: Planning Prompt

The planning prompt that defined this audit's scope and methodology (not included in this repository).
