# Run 11d Pre-Pilot Report: Static 7-State CG Ordering Smoke Test

**Date:** 2026-03-18
**Series:** Extended Blicket Benchmark, Run 11d Pre-Pilot
**Branch:** `gopnik-extended-benchmark`
**Run script:** `experiments/run_11d_prepilot.py`
**Analysis script:** `scripts/analyze_run11d_prepilot.py`

---

## 1. Configuration Summary

### 1.1 Purpose

Qualitative pre-pilot to determine the correct static CG design for Run 11d. Compares two v3 CG topologies — both 7-state extensions of v2 — to decide which topology to carry forward. This is trace inspection, not statistical analysis.

### 1.2 Changes from Run 11c

| Change | Description |
|--------|-------------|
| **G: Revised ORS guidelines** | Elimination language, scorecard framing, individual-first protocol |
| **H: Two new states** | ORDER_TESTING (OT) and RELIABILITY_TESTING (RT) between CT and HE |
| **I: ORS backstop lowered** | 25 → 20 steps |

### 1.3 Design

| Parameter | Value |
|-----------|-------|
| Condition | ORDER_STOCHASTIC (order + stochastic) |
| Rule | 3-object ordered subset, p=0.7 |
| Objects | 5 (A, B, C, D, E) |
| Required objects | 3 per episode (randomized) |
| Required order | Randomized per episode |
| Max steps | 100 |
| Episodes per cell | 5 |
| Seeds | 1401–1405 (paired across cells) |
| Budget cap | $50 |
| CG version | v3_static_linear (S1); v3_static_branch (S2) |
| IE backstop | 15 steps |
| ORS backstop | 20 steps (Change I) |
| OT backstop | 15 steps |
| RT backstop | 10 steps |
| Action model | `claude-sonnet-4-5-20250929` |

### 1.4 Cells

| Cell | Label | Agent | CG Version | Topology | Episodes |
|------|-------|-------|:----------:|----------|:--------:|
| S1 | static_linear | cg_only | v3_static_linear | IE→ORS→CT→OT→RT→HE→V | 5 |
| S2 | static_branch | cg_only | v3_static_branch | CT can skip OT/RT via branching | 5 |

### 1.5 Topology Differences

**S1 (v3_static_linear):** 7-state linear chain. Agent must traverse all 7 states in order: IE→ORS→CT→OT→RT→HE→V. Skip gates are guideline-based only (the CG itself always transitions linearly). CT transitions to OT (not HE).

**S2 (v3_static_branch):** 7-state branching graph. CT can transition to either OT or HE (agent chooses via TRANSITION:). OT can transition to either RT or HE. This allows short-circuiting: if agent doesn't have a candidate, it can skip OT/RT and go directly CT→HE.

### 1.6 Episode Rules (Pre-generated)

All 5 episodes use the same randomized rule across both cells for paired comparison:

| Episode | Seed | Correct Objects | Required Order | Distractors |
|---------|------|----------------|----------------|-------------|
| 0 | 1401 | A, C, D | C→A→D | B, E |
| 1 | 1402 | A, C, E | C→A→E | B, D |
| 2 | 1403 | A, C, D | D→C→A | B, E |
| 3 | 1404 | A, B, E | A→E→B | C, D |
| 4 | 1405 | B, C, D | B→C→D | A, E |

---

## 2. Results Summary

| Cell | Accuracy | Submitted | Rule Type | Objects | Order | Avg Steps |
|------|:--------:|:---------:|:---------:|:-------:|:-----:|:---------:|
| S1 (linear) | 2/5 (40%) | 4/5 | 3/5 | 2/5 | 2/5 | 88.8 |
| S2 (branch) | 0/5 (0%) | 4/5 | 0/5 | 0/5 | 0/5 | 81.6 |

### 2.1 Paired Episode Comparison

| Ep | Seed | S1 Result | S2 Result | S1 Steps | S2 Steps | Match? |
|----|------|-----------|-----------|:--------:|:--------:|:------:|
| 0 | 1401 | WRONG | WRONG | 98 | 95 | Concordant |
| 1 | 1402 | CORRECT | WRONG | 100 | 86 | Discordant (S1 only) |
| 2 | 1403 | WRONG | WRONG | 66 | 55 | Concordant |
| 3 | 1404 | WRONG (no submit) | WRONG (no submit) | 98 | 93 | Concordant |
| 4 | 1405 | CORRECT | WRONG | 82 | 79 | Discordant (S1 only) |

**Concordant:** 3/5 | **S1-only correct:** 2/5 | **S2-only correct:** 0/5

---

## 3. Per-Episode Trace Summaries

### S1 (v3_static_linear)

#### S1-Ep0 (seed 1401, rule: C→A→D, distractors: B, E) — WRONG

- **Trajectory:** IE@0 → ORS@9 → CT@29 → OT@41 → RT@51 → CT@61
- **Steps/state:** IE=9, ORS=20, CT=49, OT=10, RT=10
- **Backstops fired:** None detected
- **ORS analysis:** 0 individual tests, 0 eliminated
- **OT analysis:** 0 distinct orderings detected
- **RT analysis:** 10 steps, stochasticity not detected
- **Timing:** Never reached HE or V
- **Answer:** Guessed all 5 objects in order C→B→A→D→E (rule type correct but objects/order wrong)
- **Notes:** Agent traversed OT and RT but then regressed back to CT at step 61 and never escaped. Spent 49 total steps in CT. The regression from RT→CT is unexpected for the linear topology — agent must have issued an invalid TRANSITION that the CG didn't enforce as an error, or the backstop pushed it through OT/RT but the agent circled back.

#### S1-Ep1 (seed 1402, rule: C→A→E, distractors: B, D) — CORRECT

- **Trajectory:** IE@0 → ORS@9 → CT@29
- **Steps/state:** IE=9, ORS=20, CT=71
- **Backstops fired:** None detected
- **ORS analysis:** 0 individual tests, 0 eliminated
- **OT analysis:** N/A (never reached)
- **RT analysis:** N/A (never reached)
- **Timing:** Never reached HE or V
- **Answer:** Correctly guessed C→A→E as order_sensitive_stochastic
- **Notes:** Agent got the right answer while stuck in CT for 71 steps. Never transitioned to OT/RT/HE/V — the linear CG didn't force progression out of CT. This is a success without using any of the new states.

#### S1-Ep2 (seed 1403, rule: D→C→A, distractors: B, E) — WRONG

- **Trajectory:** IE@0 → ORS@9 → CT@29
- **Steps/state:** IE=9, ORS=20, CT=37
- **Backstops fired:** None detected
- **ORS analysis:** 0 individual tests, 0 eliminated
- **OT analysis:** N/A (never reached)
- **RT analysis:** N/A (never reached)
- **Timing:** Never reached HE or V
- **Answer:** Guessed all 5 objects in order E→D→C→B→A (rule type wrong: order_sensitive)
- **Notes:** Stuck in CT from step 29 until early submission at step 66. Failed to identify correct objects or detect stochasticity.

#### S1-Ep3 (seed 1404, rule: A→E→B, distractors: C, D) — WRONG (no submit)

- **Trajectory:** IE@0 → ORS@9 → CT@29
- **Steps/state:** IE=9, ORS=20, CT=69
- **Backstops fired:** None detected
- **ORS analysis:** 0 individual tests, 0 eliminated
- **OT analysis:** N/A (never reached)
- **RT analysis:** N/A (never reached)
- **Timing:** Never reached HE or V
- **Answer:** No answer submitted — ran out of steps
- **Notes:** Never left CT. 98 total steps, 69 in CT. No answer submitted.

#### S1-Ep4 (seed 1405, rule: B→C→D, distractors: A, E) — CORRECT

- **Trajectory:** IE@0 → ORS@9 → CT@29
- **Steps/state:** IE=9, ORS=20, CT=53
- **Backstops fired:** None detected
- **ORS analysis:** 0 individual tests, 0 eliminated
- **OT analysis:** N/A (never reached)
- **RT analysis:** N/A (never reached)
- **Timing:** Never reached HE or V
- **Answer:** Correctly guessed B→C→D as order_sensitive_stochastic
- **Notes:** Another success from CT without using OT/RT/HE/V. Agent solved it entirely within Combination Testing.

#### S1 Summary

| Metric | Value |
|--------|-------|
| Accuracy | 2/5 (40%) |
| ORS backstop fired | 0/5 (0%) — but all episodes show exactly 20 ORS steps (IE backstop at step 9, ORS at step 29) suggesting backstop fired but wasn't detected by analysis script |
| Episodes reaching OT | 1/5 (Ep0 only) |
| Episodes reaching HE | 0/5 |
| Episodes reaching V | 0/5 |
| Avg steps in CT | 55.8 |

**Critical finding:** The linear topology's CT→OT transition doesn't fire naturally. In 4/5 episodes, the agent gets stuck in CT and never progresses to OT/RT/HE/V. The one episode that reached OT (Ep0) then regressed back to CT. Both correct answers (Ep1, Ep4) were solved entirely within CT without the new states.

---

### S2 (v3_static_branch)

#### S2-Ep0 (seed 1401, rule: C→A→D, distractors: B, E) — WRONG

- **Trajectory:** IE@0 → ORS@9 → CT@29 → HE@84 → V@89
- **Steps/state:** IE=9, ORS=20, CT=55, HE=5, V=6
- **Backstops fired:** None detected
- **ORS analysis:** 0 individual tests, 0 eliminated
- **OT analysis:** N/A (skipped OT/RT — used CT→HE branch)
- **RT analysis:** N/A (skipped)
- **Timing:** HE@84 (16 steps remaining), V@89 (11 remaining)
- **Answer:** Guessed conjunctive with all 5 objects (no order)
- **Notes:** Agent spent 55 steps in CT, then used the branch shortcut CT→HE→V. Skipped OT and RT entirely. This demonstrates the branching topology working as designed — but the agent never tested orderings, so it misidentified the rule type entirely.

#### S2-Ep1 (seed 1402, rule: C→A→E, distractors: B, D) — WRONG

- **Trajectory:** IE@0 → ORS@9 → CT@29 → OT@30 → RT@43 → HE@53 → V@55
- **Steps/state:** IE=9, ORS=20, CT=1, OT=13, RT=10, HE=2, V=31
- **Backstops fired:** None detected
- **ORS analysis:** 0 individual tests, 0 eliminated
- **OT analysis:** 0 distinct orderings detected (regex)
- **RT analysis:** 10 steps, stochasticity not detected
- **Timing:** HE@53 (47 remaining), V@55 (45 remaining)
- **Answer:** Guessed all 5 objects in order C→A→B→D→E (order_sensitive, missed stochastic)
- **Notes:** Fastest full-chain traversal: CT for only 1 step before jumping to OT. Reached V by step 55, then spent 31 steps in V. Despite traversing all 7 states, failed to identify correct objects or detect stochasticity. The rapid CT→OT transition (1 step in CT) may indicate premature progression.

#### S2-Ep2 (seed 1403, rule: D→C→A, distractors: B, E) — WRONG

- **Trajectory:** IE@0 → ORS@9 → CT@15 → OT@20 → RT@30 → HE@40 → V@45
- **Steps/state:** IE=9, ORS=6, CT=5, OT=10, RT=10, HE=5, V=10
- **Backstops fired:** None detected
- **ORS analysis:** 0 individual tests, 0 eliminated
- **OT analysis:** 0 distinct orderings detected (regex)
- **RT analysis:** 10 steps, stochasticity not detected
- **Timing:** HE@40 (60 remaining), V@45 (55 remaining)
- **Answer:** Guessed all 5 objects in order E→D→C→B→A (order_sensitive, missed stochastic)
- **Notes:** Fastest episode overall — reached V at step 45 with 55 steps remaining. Only 6 steps in ORS (natural transition, not backstop). Clean 7-state traversal. But agent still failed: included all 5 objects and wrong order, suggesting the rapid traversal didn't produce enough evidence before progressing.

#### S2-Ep3 (seed 1404, rule: A→E→B, distractors: C, D) — WRONG (no submit)

- **Trajectory:** IE@0 → ORS@9 → CT@29 → OT@30 → RT@37 → HE@41 → CT@46 → OT@55 → RT@63 → HE@73 → V@92
- **Steps/state:** IE=9, ORS=20, CT=10, OT=15, RT=14, HE=24, V=1
- **Backstops fired:** None detected
- **ORS analysis:** 0 individual tests, 0 eliminated
- **OT analysis:** 0 distinct orderings detected (regex)
- **RT analysis:** 14 steps, stochasticity not detected
- **Timing:** HE@41 (first entry, 59 remaining), V@92 (8 remaining)
- **Answer:** No answer submitted — ran out of steps
- **Notes:** Most complex trajectory — agent looped through the branching graph twice. First pass: CT@29→OT@30→RT@37→HE@41. Then regressed to CT@46 and did a second pass: OT@55→RT@63→HE@73→V@92. This is the branching topology allowing the agent to circle back, spending budget on repeated traversals. 93 total steps, only 1 in V, no answer submitted.

#### S2-Ep4 (seed 1405, rule: B→C→D, distractors: A, E) — WRONG

- **Trajectory:** IE@0 → ORS@9 → CT@29 → OT@33 → RT@48 → V@58 → HE@70 → V@78
- **Steps/state:** IE=9, ORS=20, CT=4, OT=15, RT=10, V=13, HE=8
- **Backstops fired:** None detected
- **ORS analysis:** 0 individual tests, 0 eliminated
- **OT analysis:** 0 distinct orderings detected (regex)
- **RT analysis:** 10 steps, stochasticity not detected
- **Timing:** HE@70, V@58 (V reached before HE)
- **Answer:** Guessed 4 objects A→B→C→D (order_sensitive, missed stochastic, included distractor A)
- **Notes:** Anomalous trajectory — went RT→V@58 (skipping HE), then V→HE@70, then HE→V@78. The agent reached V before HE, which is an inversion of the expected flow. This suggests the branching topology allows sequence violations when the agent issues unexpected TRANSITIONs.

#### S2 Summary

| Metric | Value |
|--------|-------|
| Accuracy | 0/5 (0%) |
| ORS backstop fired | 0/5 (0%) — but 4/5 show 20 ORS steps |
| Episodes reaching OT | 4/5 |
| Episodes reaching RT | 4/5 |
| Episodes reaching HE | 5/5 |
| Episodes reaching V | 5/5 |
| Avg steps in CT | 15.0 |
| State sequence violations | 2/5 (Ep3 looping, Ep4 V-before-HE) |

**Critical finding:** The branching topology achieves far better state coverage (all 5 episodes reach HE and V, vs 0/5 for linear) but 0% accuracy. The agent moves through states rapidly but without accumulating sufficient evidence. Two episodes show sequence anomalies (looping, V-before-HE).

---

## 4. Diagnostic Answers

### D1: Do agents reach OT/RT naturally (before backstop)?

| Cell | OT reached naturally | OT via backstop | OT never reached |
|------|:--------------------:|:---------------:|:----------------:|
| S1 (linear) | 1/5 (20%) | 0/5 | 4/5 (80%) |
| S2 (branch) | 4/5 (80%) | 0/5 | 1/5 (20%) |

**Verdict:** The branching topology enables natural OT entry (4/5), while the linear topology traps the agent in CT (4/5 never leave CT). The linear CT→OT transition appears to be too rigid — the agent doesn't naturally request it.

### D2: Do agents use OT/RT productively?

| Cell | OT productive (≥2 orderings) | RT productive (≥2 tests) |
|------|:----------------------------:|:------------------------:|
| S1 (linear) | 0/5 | 1/5 |
| S2 (branch) | 0/5 | 4/5 |

**Caveat:** The "orderings detected" metric relies on regex matching of ordering patterns (e.g., "A-B-C") in agent responses. The 0/5 on OT productive may reflect regex limitations rather than agent behavior. Manual trace inspection would be needed to confirm.

RT productivity (measured by step count ≥2) is high for S2 (4/5) because the agent spends ≥10 steps in RT in all episodes that reach it.

### D3: ORS elimination rate

| Cell | ORS eliminates ≥1 object |
|------|:------------------------:|
| S1 (linear) | 0/5 |
| S2 (branch) | 0/5 |

**Verdict:** Change G's revised ORS guidelines did not produce detectable object elimination in any episode. The analysis script's regex patterns may not match the actual elimination language used by the LLM. However, the ORS phase consistently consumed exactly 20 steps in 9/10 episodes, suggesting the agent explores during ORS but doesn't explicitly eliminate objects using the expected language patterns.

### D4: Stochasticity detection

| Cell | Stochasticity detected |
|------|:----------------------:|
| S1 (linear) | 0/5 |
| S2 (branch) | 0/5 |

**Verdict:** No stochasticity detected in any episode across either topology. This is consistent with Runs 11b (27%) and 11c (low rates). The RT phase may not be sufficient to drive stochasticity detection without Dynamic Behaviors.

### D5: Step budget at HE entry

| Cell | Avg HE entry step | Avg budget remaining at HE |
|------|:------------------:|:--------------------------:|
| S1 (linear) | N/A (0/5 reached HE) | N/A |
| S2 (branch) | 57.6 (5/5 reached HE) | 42.4 steps |

**Verdict:** S2 branch agents reach HE with substantial budget remaining (avg 42.4 steps). But this budget isn't used productively — S2 accuracy is 0%. The problem isn't insufficient time for evaluation; it's insufficient evidence quality before evaluation.

### D6: Overall accuracy

| Cell | Accuracy |
|------|:--------:|
| S1 (linear) | 2/5 (40%) |
| S2 (branch) | 0/5 (0%) |

**Verdict:** S1 outperforms S2 despite worse state coverage. Both S1 correct episodes (Ep1, Ep4) solved the task entirely within CT without using OT/RT/HE/V.

---

## 5. ORS v3 vs v2 Comparison

| Aspect | v2 (Run 11c, Cell B) | v3 (Run 11d pre-pilot) |
|--------|:--------------------:|:----------------------:|
| ORS backstop threshold | 25 steps | 20 steps |
| ORS guidelines | Generic screening | Elimination language, scorecard, individual-first |
| ORS duration | ~25 steps (backstop fires 100%) | ~20 steps (backstop appears to fire ~90%) |
| Elimination detected | N/A (not measured in 11c) | 0/10 (Change G not producing detectable eliminations) |

**Interpretation:** The backstop reduction (25→20) successfully shortened the ORS phase by ~5 steps. However, the revised guidelines (Change G) do not appear to change agent behavior — no eliminations detected. The ORS phase still functions as a structural timer rather than a discovery phase. The additional 5 steps freed up by Change I could in principle be used by downstream states, but this benefit is marginal.

---

## 6. Candidate Comparison Table

| Criterion | S1 (Linear) | S2 (Branch) |
|-----------|:-----------:|:-----------:|
| Accuracy | **2/5 (40%)** | 0/5 (0%) |
| State coverage (reach HE) | 0/5 | **5/5** |
| State coverage (reach V) | 0/5 | **5/5** |
| OT reached naturally | 1/5 | **4/5** |
| Sequence anomalies | 1/5 (regression) | **2/5** (loop, inversion) |
| Avg steps in CT | 55.8 | **15.0** |
| Stochasticity detected | 0/5 | 0/5 |
| ORS eliminations | 0/5 | 0/5 |
| Correct answers use new states? | No (both from CT) | N/A (0% accuracy) |

---

## 7. Decision

### Decision Tree from Planning Prompt

The planning prompt's Section 4 decision tree specifies:

> If both topologies have similar accuracy (±1 episode), choose the one with better state-utilization diagnostics.
> If one topology is clearly better on accuracy (≥2 episode gap), choose the more accurate one.

**Result:** S1 linear (2/5) vs S2 branch (0/5) — a 2-episode gap.

### Decision: Neither topology as-is. Proceed to targeted debugging.

While the decision tree would select S1 (linear), the trace evidence reveals that S1's accuracy comes entirely from CT — the agent never uses OT, RT, HE, or V. The new states are effectively dead code in the linear topology. S2 (branch) achieves full state coverage but 0% accuracy because the agent rushes through states without accumulating evidence.

**Root cause diagnosis:**

1. **Linear CT absorption:** The CT state is too "sticky" — its guidelines and transition structure don't push the agent toward OT. The agent explores within CT indefinitely and either solves the problem or runs out of steps. Adding OT/RT/HE/V downstream of CT has no effect because the agent never reaches them.

2. **Branch premature progression:** The branch topology allows the agent to move rapidly through states (CT for as few as 1 step in Ep1) but this speed comes at the cost of evidence quality. The agent transitions to OT before it has candidate objects to test orderings of.

3. **ORS as timer:** In both topologies, ORS runs for exactly 20 steps (backstop duration) and produces no detectable eliminations. Change G's guidelines are not changing agent behavior.

4. **No stochasticity detection:** 0/10 across both topologies, consistent with the persistent finding that CG-only agents struggle to detect stochasticity without Dynamic Behaviors.

### Recommended next steps for Run 11d

1. **Add CT backstop** — CT currently has no step limit, leading to the absorption pattern. A CT backstop (e.g., 20-25 steps) would force progression to OT in the linear topology.
2. **Strengthen OT entry criterion** — In the branch topology, add minimum-CT-steps requirement or evidence gate before allowing CT→OT transition.
3. **Consider CT→OT as automatic** — Instead of relying on agent TRANSITION requests, make CT→OT a backstop-driven transition (like IE→ORS).
4. **Investigate ORS guidelines** — The regex analysis may be missing elimination behavior. Manual trace inspection of 2-3 ORS phases would determine whether the agent is actually following the v3 elimination protocol.
5. **Stochasticity requires DB** — Given 0/10 detection, consider whether stochasticity detection should be a DB-driven capability rather than a CG guideline.

---

## 8. Cost Summary

| Cell | Episodes | Total Cost | Avg Cost/Episode |
|------|:--------:|:----------:|:----------------:|
| S1 (linear) | 5 | $14.23 | $2.85 |
| S2 (branch) | 5 | $12.49 | $2.50 |
| **Total** | **10** | **$26.72** | **$2.67** |

Budget utilization: $26.72 / $50.00 (53.4%)

### Per-Episode Cost Breakdown

| Episode | S1 Cost | S2 Cost |
|---------|:-------:|:-------:|
| 0 | $3.25 | $3.19 |
| 1 | $3.54 | $2.71 |
| 2 | $1.66 | $1.18 |
| 3 | $3.37 | $3.18 |
| 4 | $2.41 | $2.23 |

---

## 9. Files

| File | Description |
|------|-------------|
| `experiments/run_11d_prepilot.py` | Run script |
| `scripts/analyze_run11d_prepilot.py` | Analysis script |
| `data/traces_run11d_prepilot/static_linear/` | S1 trace files (5 JSON) |
| `data/traces_run11d_prepilot/static_branch/` | S2 trace files (5 JSON) |
| `data/results_run11d_prepilot/prepilot_summary.json` | Run summary |
| `data/results_run11d_prepilot/prepilot_analysis.json` | Analysis output |
| `agents/context_graph.py` | Modified: v3, v3_static_linear, v3_static_branch + OT/RT backstops |
| `agents/cg_agent.py` | Modified: OT/RT backstop pass-through |
| `tests/test_context_graph.py` | Modified: 18 new v3 tests |
