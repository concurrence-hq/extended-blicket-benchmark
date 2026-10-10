# Run 11b Codebase Changes

**Date:** 2026-03-16
**Branch:** `gopnik-extended-benchmark`
**Predecessor:** Run 11a (report: `experiment1_results/report_run11a_2026-03-15.md`)

---

## 1. Problems Addressed

Three problems identified in Run 11a motivate these changes.

### P1: RELIABILITY_TESTING unreachable (Run 11a Section 6.3)

`stochasticity_hypothesis` fired in 15/15 episodes, but RELIABILITY_TESTING was entered in **0/15**. Root cause: the base behavior uses `connect_from=["DIMENSION_DISCOVERY", "COMBINATION_TESTING"]`, but when it fires at step ~22, the agent is typically in ORDER_TESTING, VERIFICATION, or HYPOTHESIS_EVALUATION. No transition edge exists from those states to RELIABILITY_TESTING. The DB fires, the notification is injected, but the agent has no valid path to reach the new state.

### P2: CG under-engagement (Run 11a Section 6.4)

8/15 episodes (53%) never left INITIAL_EXPLORATION. The agent tested all 5 individual objects in 5 steps, then continued testing combinations within INITIAL_EXPLORATION without transitioning to COMBINATION_TESTING. The CG state machine offers structured exploration phases, but the agent doesn't use them because there is no forcing mechanism. The "explore → combine → hypothesize → verify" progression only happens when the agent voluntarily transitions.

### P3: `db_notifications` unpopulated in traces (Run 11a Appendix A)

The step-level trace array (`steps[].db_notifications`) was always `[]` despite DB firing. The actual data lived in `agent_trace.db_activation_log` and `agent_trace.db_score_log`, but the per-step field was never wired up. This caused the initial Run 11a report to miss the competitive inhibition analysis entirely.

---

## 2. Changes

### Change D: `agents/dynamic_behaviors_11b.py` (new file, ~120 lines)

**Fixes:** P1 (RELIABILITY_TESTING unreachable)

`create_blicket_behaviors_11b()` inherits from `create_blicket_behaviors_10c()` and replaces only `stochasticity_hypothesis`. All other behaviors (including 10c's revised `order_hypothesis`) pass through unchanged.

| Property | Base (10c) | Revised (11b) |
|----------|-----------|---------------|
| `connect_from` | `DIMENSION_DISCOVERY`, `COMBINATION_TESTING` | + `VERIFICATION`, `HYPOTHESIS_EVALUATION`, `ORDER_TESTING` (5 total) |
| RELIABILITY_TESTING exits | `HYPOTHESIS_EVALUATION` only | + `VERIFICATION` (primary exit), `HYPOTHESIS_EVALUATION` (escape) |
| Guidelines | 4 generic items | 7 items: explicit repeat protocol, probability formula, wrong-candidate escape |
| Notification text | Passive ("is available") | Directive ("Transition to RELIABILITY_TESTING to repeat...") |

**Expected effect:** When `stochasticity_hypothesis` fires at step ~22, the agent now has a valid transition to RELIABILITY_TESTING from whatever state it occupies. The two exit transitions prevent the agent from getting trapped.

### Change C: `db_notifications` in traces (~12 lines across 2 files)

**Fixes:** P3 (unpopulated trace field)

| File | Change |
|------|--------|
| `agents/full_agent.py` | Added `_last_step_notifications: list[str]` field, populated after each DB evaluation, cleared at start of each `get_action()`. Added `get_last_step_notifications()` getter. |
| `experiments/episode_runner.py` | Before creating `StepTrace`, calls `agent.get_last_step_notifications()` via `hasattr` guard. Passes result as `db_notifications` to `StepTrace`. |

**Purely observational.** No new LLM calls, no behavior change. `hasattr` guard ensures `BaseLLMAgent` and `CGAgent` produce empty lists as before. Future traces will show which steps had DB notifications directly in the step-level data.

### Change B: INITIAL_EXPLORATION backstop (~55 lines across 2 files)

**Fixes:** P2 (CG under-engagement)

| File | Change |
|------|--------|
| `agents/context_graph.py` | Added 3 fields: `steps_in_current_state: int`, `initial_exploration_max_steps: int | None`, `_backstop_fired: bool` (private). Added `advance_step()` method. `transition_to()` now resets step counter. |
| `agents/cg_agent.py` | Calls `advance_step()` at start of `get_action()`. If backstop fires: logs to `modification_log`, injects notification, records in `_transition_log` with `"backstop": True`. |

**Mechanism:** When `steps_in_current_state >= initial_exploration_max_steps` and the agent is in INITIAL_EXPLORATION, the backstop forces a transition to COMBINATION_TESTING and injects a system notification. The backstop fires at most once per episode (`_backstop_fired` flag).

**Backward compatible:** Default `initial_exploration_max_steps=None` means the backstop never fires. `create_blicket_cg()` is unchanged. Old traces without the new fields load via Pydantic defaults. The backstop is only active when explicitly configured.

**Call chain:** `FullAgent.get_action()` → DB evaluation (sees true INITIAL_EXPLORATION state) → `super().get_action()` → `CGAgent.get_action()` → `advance_step()` → backstop fires → LLM sees post-backstop state. DB evaluation and backstop are separated: DBs see the real state before any forced transition.

---

## 3. Files Summary

| File | Action | Lines Changed |
|------|--------|:------------:|
| `agents/dynamic_behaviors_11b.py` | **Created** | ~120 |
| `agents/full_agent.py` | Modified | +12 |
| `experiments/episode_runner.py` | Modified | +5 |
| `agents/context_graph.py` | Modified | +35 |
| `agents/cg_agent.py` | Modified | +20 |
| `tests/test_dynamic_behaviors_11b.py` | **Created** | ~170 |
| `tests/test_context_graph.py` | Modified | +75 |

**Not modified:** `dynamic_behaviors.py`, `dynamic_behaviors_10c.py`, `base_llm_agent.py`.

---

## 4. Test Results

233 tests pass (209 existing + 24 new). 0 failures.

| Test File | New Tests | Coverage |
|-----------|:---------:|---------|
| `test_dynamic_behaviors_11b.py` | 16 | Structure, reachability from all 5 states, exit transitions, base-not-modified regression |
| `test_context_graph.py` | 8 | Backstop disabled by default, fires at threshold, fires only once, only affects INITIAL_EXPLORATION, step counter reset, serialization roundtrip, old data loads with defaults |

---

## 5. Expected Run 11b Configuration

These changes are activated by two configuration choices at run time:

| Parameter | Value | Effect |
|-----------|-------|--------|
| `behaviors=create_blicket_behaviors_11b()` | Revised stochasticity_hypothesis | P1 fix active |
| `initial_exploration_max_steps=15` | Backstop at 15 steps | P2 fix active |

P3 fix (`db_notifications` wiring) is always active — no configuration needed.

**Backward compatibility check:** Running with `create_blicket_behaviors_10c()` and `initial_exploration_max_steps=None` reproduces the exact Run 11a behavior.

---

## 6. What to Watch in Run 11b Results

| Metric | Run 11a Baseline | Expected Direction | Why |
|--------|:----------------:|:------------------:|-----|
| RELIABILITY_TESTING entry rate | 0/15 (0%) | > 0% | P1 fix: transition edges now exist from all states |
| Episodes leaving INITIAL_EXPLORATION | 7/15 (47%) | Higher | P2 fix: backstop forces exit at step 15 |
| `steps[].db_notifications` populated | Never | On steps where DBs fire | P3 fix: wiring added |
| full_cg_db accuracy | 10/15 (67%) | No prediction | More CG engagement could help or hurt |
| CG traversal vs accuracy paradox | Multi-state = 43%, single-state = 88% | Narrower gap | If backstop forces all episodes into multi-state, the confound is removed |

**Key question:** Does forcing CG engagement (backstop) improve accuracy, or does it confirm the Run 11a paradox that more CG states = worse outcomes? If the paradox was purely confounded by episode difficulty (as hypothesized in Section 6.4 of the Run 11a report), forcing all episodes into multi-state traversal should not hurt accuracy. If the paradox reflects genuine CG overhead, accuracy may drop.
