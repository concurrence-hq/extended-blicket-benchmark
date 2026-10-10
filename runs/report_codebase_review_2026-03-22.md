# Pre-Publication Codebase Review — Paper 1 Correctness Audit

**Date:** 2026-03-22
**Reviewer:** Claude Opus 4.6 (automated)
**Scope:** `projects/gopnik-extended-benchmark/` — all files relevant to TMLR Paper 1
**Files examined:** 35 source files (~12,000 lines), 10 test files (~3,400 lines)

---

## A. Verification Matrix

### Claim 1: Clean Three-Agent Ablation

**Summary:** The three agent types (base_llm, cg_only, full_cg_db) form a clean ablation chain where each adds exactly one component.

**Files examined:** `agents/base_llm_agent.py`, `agents/cg_agent.py`, `agents/full_agent.py`, `experiments/config.py`

**Verdict: VERIFIED**

**Evidence:**
- `BaseLLMAgent` (base_llm): Task prompt only. System prompt = `text_interface.render_task_description()`. No CG, no DB. Agent type string: `"base_llm"` (line 195).
- `CGAgent` (cg_only) extends `BaseLLMAgent`: Adds CG injection to system prompt via `render_for_prompt()` (line 90). Processes `TRANSITION:` responses. Agent type string: `"cg_only"` (line 208).
- `FullAgent` (full_cg_db) extends `CGAgent`: Adds `DynamicBehaviorEngine`. Evaluates DB triggers before delegating to `super().get_action()` (line 124). Agent type string: `"full_cg_db"` (line 132).

The inheritance chain is clean: `BaseLLMAgent → CGAgent → FullAgent`. Each level adds exactly one component. No leakage — `BaseLLMAgent` has no CG/DB code paths.

**Agent type strings are consistent** across the codebase:
- `config.py:AgentType` enum: `BASE_LLM = "base_llm"`, `CG_ONLY = "cg_only"`, `FULL_CG_DB = "full_cg_db"`
- `get_trace_data()` in each agent class returns the matching string
- `paper1_statistical_tests.py:RUN_CONFIGS` uses these exact strings in the `agents` lists

---

### Claim 2: Hidden Moderator Switch Correctness

**Summary:** The hidden moderator switches the causal rule after exactly N successful activations.

**Files examined:** `environment/extended_env.py`, `environment/extended_env_5obj.py`

**Verdict: VERIFIED**

**Evidence:**
- `_successful_activations` counter (initialized to 0, reset on `reset()`)
- Increment logic (`extended_env_5obj.py:111-112`): `if obs.detector_active and obs.detector_changed` — counts transitions TO active, not sustained active states. This is correct for "successful activation" semantics.
- Switch trigger (`extended_env_5obj.py:113-119`): `if not self._rule_switched and self._successful_activations >= self.moderator_config.switch_after_n` — fires exactly when count reaches threshold.
- Post-switch recomputation (`lines 121-131`): After rule swap, recomputes activation under new rule and patches the observation if state changed. Correct.
- `get_info()` exposes `successful_activations` and `rule_switched` for trace logging.

**Switch threshold consistency (Chain 5):** Run scripts use `SWITCH_AFTER_N = 3` (runs 3b, 7, 8) or variable (run 6: 3 and 5). `paper1_statistical_tests.py:RUN_CONFIGS` specifies matching `switch_after_n` values for each run. Verified consistent.

**One subtlety:** The switch fires at `>=` not `==`. This means if somehow two activations happened simultaneously (impossible given step-by-step execution), it would still work. The `>=` is defensive and correct.

---

### Claim 3: Reasoning-Eligible Classification

**Summary:** Episodes are classified as pre_switch/exactly_N/reasoning_eligible based on activation count relative to switch_after_n.

**Files examined:** `environment/extended_env_5obj.py`, `experiments/episode_runner.py`, `scripts/paper1_statistical_tests.py`

**Verdict: VERIFIED**

**Evidence (Chain 1 — RE Classification Pipeline):**

1. **Source:** `ExtendedBlicketEnv5._successful_activations` (counts `detector_active and detector_changed` transitions)
2. **Exposure:** `get_info()` returns `{"successful_activations": self._successful_activations, ...}` (line 212)
3. **Trace storage:** `episode_runner._save_trace()` includes `env_info=env.get_info()` (line 393)
4. **Classification:** `paper1_statistical_tests.load_traces()` reads `env_info["successful_activations"]` (line 148) and classifies:
   - `< switch_n` → `"pre_switch"` (line 154)
   - `== switch_n` → `"exactly_n"` (line 156)
   - `> switch_n` → `"reasoning_eligible"` (line 158)

The boundary conditions use strict `<`, `==`, `>` — no off-by-one. The chain is clean from environment state through JSON trace to statistical analysis.

**Verification anchor:** `KNOWN_RUN08` values (lines 114-120) are cross-checked against trace data at runtime (lines 494-520). The script aborts if any mismatch is found.

---

### Claim 4: DB Firing is Recorded Faithfully

**Summary:** Dynamic Behavior activations are logged in traces and available for analysis.

**Files examined:** `agents/full_agent.py`, `agents/dynamic_behaviors.py`, `experiments/episode_runner.py`, `analysis/metrics.py`

**Verdict: VERIFIED**

**Evidence (Chain 2 — DB Activation Recording):**

1. **Engine state:** `DynamicBehaviorEngine._fired` (set of behavior names), `._score_log` (list of score dicts)
2. **Agent logging:** `FullAgent._db_activation_log` appends `{"step": step, "notification": ..., "fired_behaviors": list(self.db_engine.fired_behaviors)}` on each notification (lines 114-118)
3. **Trace export:** `get_trace_data()` includes:
   - `"db_activation_log"`: list of per-step firing events (line 133)
   - `"db_fired_behaviors"`: final set of all fired behavior names (line 134)
   - `"db_trigger_tokens"`: total trigger evaluation tokens (line 135)
   - `"db_score_log"`: complete score history (line 136)
4. **Analysis consumption:** `metrics.py` reads `db_activation_log` (line 84) and `db_fired_behaviors` (line 86) using matching key names.

Key names are consistent across all layers: `"db_activation_log"` and `"db_fired_behaviors"`.

---

### Claim 5: Fisher's Exact Test Implementation

**Summary:** Statistical tests use standard Fisher's exact test via scipy.

**Files examined:** `scripts/paper1_statistical_tests.py`

**Verdict: VERIFIED**

**Evidence:**
- `fishers_exact()` (line 235) wraps `scipy.stats.fisher_exact()` — standard implementation.
- 2×2 table construction is correct: `[[success_group1, fail_group1], [success_group2, fail_group2]]`.
- Both two-sided and one-sided alternatives are used appropriately.
- Cross-check against existing report values (lines 554-569) confirms reproducibility within tolerance (delta < 0.002).

---

### Claim 6: CMH Test Implementation

**Summary:** Cochran-Mantel-Haenszel test stratifies across batches to control for batch effects.

**Files examined:** `scripts/paper1_statistical_tests.py`

**Verdict: VERIFIED**

**Evidence:**
- `cmh_test()` (lines 309-382) implements the standard CMH statistic with continuity correction.
- Variance formula uses the hypergeometric variance `r1 * r2 * c1 * c2 / (n² * (n-1))` — correct.
- Common OR uses Mantel-Haenszel estimator (`∑(ad/n) / ∑(bc/n)`) — standard.
- CI uses Robins-Breslow-Greenland method — appropriate for small-sample situations.
- Strata are correct: one table per run (03, 3B, 06) for base_llm vs cg_only (lines 799-808).

---

### Claim 7: Bayes Factor Implementation

**Summary:** Directional Bayes factors use Monte Carlo sampling from Beta posteriors.

**Files examined:** `scripts/paper1_statistical_tests.py`

**Verdict: VERIFIED**

**Evidence:**
- `bayes_factor_directional()` (lines 296-306) samples from Beta(k+1, n-k+1) posteriors — correct uniform prior.
- 1M samples with fixed seed (42) for reproducibility.
- BF₁₀ = P(p₁ > p₂ | data) / 0.5 — correct directional formulation.
- `bayes_factor_proportions()` (lines 270-293) computes two-sided BF using Beta-Binomial marginal likelihoods — standard.

---

### Claim 8: Order-Sensitive and Stochastic Conditions

**Summary:** The environment correctly implements order-sensitive, stochastic, and combined conditions.

**Files examined:** `environment/extended_env.py`, `environment/extended_env_5obj.py`, `environment/blicket_env.py`

**Verdict: VERIFIED (for Paper 1 scope)**

**Evidence:**
- ORDER_SENSITIVE: `_check_order()` extracts the subsequence of required objects from `_placement_order` — correct subsequence check (not contiguous substring).
- STOCHASTIC: Uses `_rng.random() < activation_probability` with seeded RNG — correct.
- ORDER_STOCHASTIC (combined): Checks order first, then stochastic — correct composition.
- HIDDEN_MODERATOR: Delegates to `base_satisfied` (base rule) — correct since switching handles the rule change.
- Placement order tracking: Correctly maintains `_placement_order` list, removes on object removal (lines 98-105).

**Note:** Paper 1 focuses exclusively on HIDDEN_MODERATOR condition. ORDER_SENSITIVE/STOCHASTIC are used in Run 10+ (Paper 2 scope) and are tested but not central to this audit.

---

## B. Code Smell Audit

### CS1: Config Flag Gating (Run Script Correctness)

**What checked:** Do run scripts correctly gate agent types, DB versions, and CG versions?

**What found:**
- Paper 1 run scripts (3b, 4, 6, 7, 8) use the **default** `create_blicket_behaviors()` (base DB version). No versioned DB imports. This is correct — Paper 1's hidden moderator experiments predate the DB revisions (10c, 11b, 11c, 11e).
- CG version is **not explicitly specified** in any Paper 1 run script — all use the default `v1` CG via `CGAgent`/`FullAgent` constructors. Correct for Paper 1 scope.
- Agent type routing in `create_agent()` functions is clean switch-on-enum in all 5 scripts.

**Issues:** None.

---

### CS2: check_answer Edge Cases

**What checked:** Does `check_answer()` handle all rule types correctly, including order validation?

**What found:**
- Base `BlicketEnv5.check_answer()` (line 174): Checks `rule_type` and `required_objects` equality. Does NOT validate order — by design (order is handled in `ExtendedBlicketEnv5`).
- `ExtendedBlicketEnv5.check_answer()` (line 169): Calls `super()`, then adds `order_correct` for ORDER_SENSITIVE and ORDER_STOCHASTIC conditions. Overrides `correct` to require order correctness.
- HIDDEN_MODERATOR: Uses base `check_answer()` — checks against `self.causal_rule` which has already been switched if applicable. This means the answer is checked against the **current** (post-switch) rule, which is correct behavior.

**Issues:** None for Paper 1 scope. The `check_answer` for HIDDEN_MODERATOR checks against whichever rule is currently active. If the agent answers before the switch, the pre-switch rule is used. If after, the post-switch rule. This is semantically correct — the "right answer" changes when the rule changes.

---

### CS3: Activation Counting Consistency

**What checked:** Is the activation count in `_successful_activations` consistent with what the agent observes?

**What found:**
- The counter increments when `obs.detector_active and obs.detector_changed` (line 111 in extended_env_5obj.py).
- This means: only fresh activations are counted (detector transitioning from OFF→ON), not sustained ON states.
- The agent sees the same activation via the observation message ("The detector is now ACTIVE").
- The step trace records `detector_active` and `detector_changed` per step.

**Issues:** None. The counting is consistent across all three views (environment state, agent observation, trace logging).

---

### CS4: Trace Completeness

**What checked:** Does `_save_trace()` capture all data needed for post-hoc analysis?

**What found:**
- `_save_trace()` (episode_runner.py:400-468) saves:
  - `result`: All episode-level metrics (success, steps, tokens, parse failures, feedback events)
  - `steps`: Per-step trace (action, objects, detector state, CG state, DB notifications, feedback)
  - `agent_trace`: Full agent data (conversation, raw responses, token usage, CG final state, transition log, DB logs, instrumentation)
  - `env_info`: Final environment state (rule type, objects, conditions, activations, switch status)

**Issues:** None. The trace is comprehensive. The `env_info` snapshot is taken at episode end, which means `successful_activations` reflects the final count.

---

### CS5: Seed Handling

**What checked:** Are seeds propagated correctly and are they independent across runs?

**What found (Chain 4 — Seed Propagation):**
1. Each run script has a unique `BASE_SEED`: 251 (3b), 307 (4), 401 (6), 501 (7), 601 (8)
2. `ExperimentConfig.get_episode_seed()` uses `hash((condition, agent, episode, base_seed))` — deterministic per (run × condition × agent × episode)
3. Seeds are passed to `ExtendedBlicketEnv5(rng_seed=seed)` which creates `np.random.default_rng(rng_seed)` — proper NumPy RNG.
4. Seeds are NOT passed to the LLM — API calls use `temperature=1.0` without a seed parameter. This is correct for Paper 1: stochastic LLM behavior is the experiment, environment stochasticity needs to be controlled.

**Issues:** None. Seeds are independent across runs and deterministic within runs.

---

### CS6: DB Version Routing

**What checked:** Do run scripts import and use the correct DB version?

**What found:**
- Paper 1 runs (3b, 4, 6, 7, 8): Use default `FullAgent()` which calls `create_blicket_behaviors()` (base version, 4 behaviors). Correct.
- Run 10c-10g: Import `create_blicket_behaviors_10c()`. Correct.
- Run 11a: Import `create_blicket_behaviors_10c()`. Correct.
- Run 11b: Import `create_blicket_behaviors_11b()`. Correct.
- Run 11c: Import `create_blicket_behaviors_11c()`. Correct.
- Run 11e: Import `create_blicket_behaviors_11e()`. Correct.

The inheritance chain is: base → 10c (revised ORDER_TESTING) → 11b (revised stochasticity reachability) → 11c (lowered threshold) → 11e (lean 3-behavior). Each version correctly inherits from its predecessor.

**Issues:** None.

---

### CS7: Prompt Contamination

**What checked:** Does any agent type inadvertently receive CG/DB prompts or system messages intended for other agent types?

**What found:**
- `BaseLLMAgent._build_system_prompt()` returns only `self.text_interface.render_task_description()`. No CG context.
- `CGAgent._build_system_prompt()` appends CG context and response format instructions (`CG_STATE:`, `TRANSITION:`, etc.)
- `FullAgent` does not override `_build_system_prompt()` — inherits from `CGAgent`. DB notifications are injected into the conversation history (as `user` messages), not the system prompt.
- The `[SYSTEM]` prefix on DB notifications is a convention, not a system-level injection. These appear as user messages in the conversation.

**Issues:** None. The separation is clean.

**One observation (not a bug):** DB notification messages are injected as `{"role": "user", "content": notification}` (full_agent.py:113). This means the LLM sees them as user messages, not system messages. This is intentional — system prompt is rebuilt each turn with current CG state, and DB notifications become part of conversation history.

---

## C. Test Coverage Assessment

### Coverage Matrix

| Layer | Files | Tests | Approx Coverage | Priority Gaps |
|-------|-------|-------|-----------------|---------------|
| Environment (3-obj) | `test_environment.py` | 40 tests | ~95% | — |
| Environment (5-obj) | `test_environment_5obj.py` | 49 tests | ~95% | — |
| Text Interface (3-obj) | `test_text_interface.py` | 13 tests | ~85% | — |
| Text Interface (5-obj) | `test_text_interface_5obj.py` | 22 tests | ~85% | `stochastic` rule type parsing gap |
| Context Graph | `test_context_graph.py` | 48 tests | ~90% | Desync fix not tested |
| Dynamic Behaviors (11b) | `test_dynamic_behaviors_11b.py` | 17 tests | ~70% | No agent integration |
| Dynamic Behaviors (11c) | `test_dynamic_behaviors_11c.py` | 13 tests | ~70% | No agent integration |
| Competitive Inhibition | `test_competitive_inhibition.py` | 11 tests | ~85% | — |
| Partial Feedback | `test_partial_feedback.py` | 18 tests | ~95% | — |
| Gradual Feedback | `test_gradual_feedback.py` | 25 tests | ~95% | — |
| **Agent Layer** | **None** | **0 tests** | **0%** | **CRITICAL** |
| **Episode Runner** | **None** | **0 tests** | **0%** | **CRITICAL** |
| **Analysis Layer** | **None** | **0 tests** | **0%** | Moderate |
| **Run Scripts** | **None** | **0 tests** | **0%** | Low |

### Gap Rankings

**Critical (affects claim correctness if broken):**

1. **No agent integration tests.** `CGAgent._process_cg_transitions()`, `_pending_desync_correction`, `FullAgent.get_action()` with DB evaluation — none tested. The desync fix (blocking invalid CG transitions) was identified as a confound in Run 10g, meaning it has material impact on results. No unit test validates it fires correctly.

2. **No episode runner tests.** The `run_episode()` function is the integration point for all components. The answer-checking flow (CHECK action → `get_answer()` → `check_answer()`) is untested end-to-end.

**Moderate (confusion or maintenance risk):**

3. **No analysis/metrics tests.** `compute_episode_metrics()`, `compute_aggregate_metrics()` — metric computation is untested. If trace format changes, metrics silently break.

4. **`stochastic` parsing gap in text_interface.py.** The answer prompt offers "stochastic" as a standalone option (line 315), but the parser has no `CausalRuleType.STOCHASTIC` enum value. If an agent answers `RULE_TYPE: stochastic`, parsing falls through to `pass` (line 421) and the answer is treated as unparseable. This doesn't affect Paper 1 (hidden moderator only), but would affect Paper 2 (order_stochastic conditions).

**Low (cleanup):**

5. **Test files for base DB and 10c not present.** Only 11b and 11c have test files. The base `dynamic_behaviors.py` and `dynamic_behaviors_10c.py` are tested only indirectly through inheritance in 11b/11c tests.

### Stale Tests

No tests appear stale — all test current behavior. However, tests for v3_static_linear/v3_static_branch CG variants exist but these variants are not used in any run script (they were experimental designs that were not deployed).

---

## D. Dead Code & Stale Paths

### Dead/Unused Code

| Item | Location | Status | Risk |
|------|----------|--------|------|
| `BlicketEnv` (3-object base) | `environment/blicket_env.py` | Imported by extended_env.py but not used directly in Paper 1 runs | None — serves as inheritance base |
| `TextInterface` (3-object) | `environment/text_interface.py` | Not used in Paper 1 (all runs use 5-object) | None — used in early runs |
| `TextInterface6` (6-object) | `environment/text_interface.py:491` | Only imported in validate scripts | Low — not shipped in paper |
| `blicket_env_6obj` module | `environment/blicket_env_6obj.py` | Only referenced by text_interface.py and validate scripts | Low |
| v3_static_linear CG | `context_graph.py:705` | Has tests but no run script uses it | None — experimental artifact |
| v3_static_branch CG | `context_graph.py:869` | Has tests but no run script uses it | None — experimental artifact |
| v4 CG | `context_graph.py:1055` | Used only in Run 11e (Paper 2 scope) | None |
| `ExperimentConfig` defaults | `config.py:51-57` | Default blickets/order/probability — overridden by every run script | None |
| `pilot_validation.py` | `experiments/pilot_validation.py` | Standalone validation script, not part of main run pipeline | None |

### Superseded Files

| File | Superseded By | Notes |
|------|---------------|-------|
| `dynamic_behaviors.py` (base) | 10c, 11b, 11c, 11e versions | Still used by Paper 1 run scripts (correct for that scope) |
| Run 3 script | Run 3b script (replication) | Run 3 data still included in analysis |

### Unused Config Flags

| Flag | Location | Used? |
|------|----------|-------|
| `order_testing_max_steps` | `CGAgent.__init__` | Only set in v3_static variants (not used in Paper 1) |
| `reliability_testing_max_steps` | `CGAgent.__init__` | Only set in v3_static variants (not used in Paper 1) |
| `CostTracker.max_total` default (500.0) | `config.py:100` | Always overridden by run scripts |
| `max_cost_per_episode` default (2.0) | `config.py:47` | Used as-is in most run scripts |

---

## E. Recommendations

### Critical (Affects Paper Claims — Fix Before Submission)

None identified. All 8 paper claims verified. All cross-file verification chains are clean.

### Moderate (Confusion Risk — Fix If Time Allows)

1. **Add agent integration tests.** Priority: `CGAgent._process_cg_transitions()` and the desync fix (`_pending_desync_correction`). The desync fix was identified as a confound in Run 10g (187 fires on v1 CG when expected 0). A unit test would prevent regression.

2. **Add episode runner integration test.** A minimal test with a mock LLM client that validates the CHECK → answer → check_answer flow would catch any future breakage of the critical path.

3. **Fix `stochastic` parsing gap in TextInterface5.** The answer prompt offers "stochastic" as an option but the parser doesn't handle it. For Paper 2, add `CausalRuleType.STOCHASTIC` to the enum or map standalone "stochastic" to the appropriate type. Not needed for Paper 1.

4. **Document the `successful_activations` counting semantics.** The counter increments on `detector_active and detector_changed` (OFF→ON transitions only). This is correct but non-obvious. A docstring or comment would prevent future misunderstanding.

### Minor (Cleanup — No Urgency)

5. **Remove v3_static_linear/v3_static_branch CG variants** if they will never be used. They add ~300 lines of code and tests for experimental designs that were abandoned.

6. **Consider testing the analysis layer.** `paper1_statistical_tests.py` includes its own verification logic (KNOWN_RUN08 checks), which is excellent. But `analysis/metrics.py` and `analysis/visualize.py` have zero tests. Low priority since they're consumed manually, not in automated pipelines.

7. **Dead scripts in `scripts/` directory.** Three untracked files (`analyze_run10b_comprehensive.py`, `analyze_run10b_traces.py`, `parse_db_fires_from_log.py`) appear to be one-off analysis scripts. Consider either committing with a note or removing.

---

## Appendix: Cross-File Verification Chains (Full Trace)

### Chain 1: RE Classification Pipeline
```
ExtendedBlicketEnv5._successful_activations  [env counter, incremented on detector_active && detector_changed]
  → env.get_info()["successful_activations"]  [exposed in dict]
  → episode_runner._save_trace(env_info=env.get_info())  [stored in JSON]
  → paper1_statistical_tests.load_traces() → env_info["successful_activations"]  [read from JSON]
  → if < switch_n: "pre_switch" | == switch_n: "exactly_n" | > switch_n: "reasoning_eligible"
```
**Status:** Clean. Boundary conditions correct. Self-verification via KNOWN_RUN08.

### Chain 2: DB Activation Recording
```
DynamicBehaviorEngine._fired (set[str])
  → FullAgent._db_activation_log.append({"step": step, "notification": ..., "fired_behaviors": list(self.db_engine.fired_behaviors)})
  → get_trace_data() → {"db_activation_log": [...], "db_fired_behaviors": [...]}
  → _save_trace() → JSON → agent_trace → metrics.py reads "db_activation_log", "db_fired_behaviors"
```
**Status:** Clean. Key names match across all layers.

### Chain 3: Agent Type Strings
```
base_llm_agent.get_trace_data()["agent_type"] = "base_llm"
cg_agent.get_trace_data()["agent_type"] = "cg_only"
full_agent.get_trace_data()["agent_type"] = "full_cg_db"
  → episode_runner.result.agent_type → _save_trace()
  → paper1_statistical_tests.RUN_CONFIGS["agents"] = ["base_llm", "cg_only", "full_cg_db"]
```
**Status:** Clean. Strings match exactly.

### Chain 4: Seed Propagation
```
run_script.BASE_SEED (unique per run: 251/307/401/501/601)
  → config.get_episode_seed(condition, agent_type.value, ep) = hash((..., BASE_SEED)) % 2^31
  → ExtendedBlicketEnv5(rng_seed=seed)
  → np.random.default_rng(rng_seed)
```
**Status:** Clean. Seeds are deterministic per-episode and independent across runs.

### Chain 5: Switch Threshold Consistency
```
run_script.SWITCH_AFTER_N (or per-cell switch_n)
  → HiddenModeratorConfig(switch_after_n=...)
  → ExtendedBlicketEnv5.step(): fires when _successful_activations >= switch_after_n
  → paper1_statistical_tests.RUN_CONFIGS["switch_after_n"] = matching value
```
**Status:** Clean. All matching.

---

## Summary

**The codebase is sound for Paper 1 submission.** All 8 paper claims verified. All 7 code smells audited clean. No critical issues found. The main recommendation is to add agent integration tests before Paper 2 work, particularly around the desync fix and the `stochastic` parsing gap — neither affects Paper 1 but both are relevant to the broader research program.
