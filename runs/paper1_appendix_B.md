# Appendix B: Supplementary Tables

---

## Table B1: Runway Compression Decomposition (Run 08)

Run 08 tested three agent types at 75 max steps (Cells A-C, n=50 each) and two agents at 50 max steps (Cells D-E, n=25 each) on the triple conjunction {C, D, E} post-switch rule with 5 objects.

### B1a: Three-Way Episode Classification

| Cell | Agent | Max Steps | n | Pre-Switch | Exactly-N | Reasoning-Eligible | Raw Accuracy |
|------|-------|:---------:|:-:|:----------:|:---------:|:------------------:|:------------:|
| A | base_llm | 75 | 50 | 13 (100%) | 7 (0%) | 30 (73.3%) | 70.0% |
| B | cg_only | 75 | 50 | 3 (100%) | 14 (0%) | 33 (93.9%) | 68.0% |
| C | full_cg_db | 75 | 50 | 4 (100%) | 3 (0%) | 43 (95.3%) | 90.0% |
| D | base_llm | 50 | 25 | 5 (100%) | 5 (0%) | 15 (80.0%) | 68.0% |
| E | full_cg_db | 50 | 25 | 1 (0%) | 4 (0%) | 20 (90.0%) | 72.0% |

**Source:** report_run08_2026-02-26.md, Sections 2-3.

**Notes:**
- Pre-switch episodes are 100% correct across all cells except Cell E (1 episode, a parse failure at 9 steps).
- Exactly-N episodes are universally 0% correct -- structurally unsolvable.
- Reasoning-eligible (RE) accuracy is the primary metric.

### B1b: Exactly-N Rate Among Post-Switch Episodes

| Cell | Agent | Max Steps | Switched | Exactly-N | Exactly-N Rate |
|------|-------|:---------:|:--------:|:---------:|:--------------:|
| A | base_llm | 75 | 37 | 7 | 18.9% |
| B | cg_only | 75 | 47 | 14 | 29.8% |
| C | full_cg_db | 75 | 46 | 3 | 6.5% |
| D | base_llm | 50 | 20 | 5 | 25.0% |
| E | full_cg_db | 50 | 24 | 4 | 16.7% |

**Source:** report_run08_2026-02-26.md, Section 3.4.

### B1c: Runway Compression Comparison (75-step vs. 50-step)

| Agent | Metric | 75-step | 50-step | Fisher's p |
|-------|--------|:-------:|:-------:|:----------:|
| base_llm | RE accuracy | 73.3% (22/30) | 80.0% (12/15) | 0.726 |
| base_llm | Raw accuracy | 70.0% | 68.0% | -- |
| base_llm | Mean steps | 25.7 | 26.2 | -- |
| base_llm | Max steps used | 48 | 41 | -- |
| full_cg_db | RE accuracy | 95.3% (41/43) | 90.0% (18/20) | 0.586 |
| full_cg_db | Raw accuracy | 90.0% | 72.0% | -- |
| full_cg_db | Exactly-N rate | 6.5% (3/46) | 16.7% (4/24) | -- |
| full_cg_db | Mean steps | 52.3 | 39.8 | -- |
| full_cg_db | Episodes > 50 steps | 56% | N/A | -- |

**Source:** report_run08_2026-02-26.md, Section 7.

### Verification Notes (Table B1)

| Claim | Expected | Verified | Status |
|-------|----------|----------|--------|
| base_llm RE: 75 vs 50 | 73.3% vs 80.0%, p=0.726 | 22/30 vs 12/15, Fisher p=0.726 | CONFIRMED |
| full_cg_db RE: 75 vs 50 | 95.3% vs 90.0%, p=0.586 | 41/43 vs 18/20, Fisher p=0.586 | CONFIRMED |
| full_cg_db raw accuracy: 75 vs 50 | 90.0% vs 72.0% | 45/50 vs 18/25 | CONFIRMED |
| full_cg_db exactly-N rate: 75 vs 50 | 6.5% vs 16.7% | 3/46 vs 4/24 | CONFIRMED |

---

## Table B2: DB4 Rule-Change Hypothesis (RC) Mechanism by Run

This table tracks the `rule_change_hypothesis` dynamic behavior across all runs where full_cg_db faced the hidden moderator condition. All rows report full_cg_db data only.

### B2a: RC Detection Performance

| Run | Config | Switched | Post-Switch Correct | RC Firings | TP | FP | FN | Sensitivity | PPV |
|-----|--------|:--------:|:-------------------:|:----------:|:--:|:--:|:--:|:-----------:|:---:|
| Run 02 | 3-obj, sw=3 | 24 | 19 | 20 | 19 | 1 | 0 | 100% (19/19) | **95.0% (19/20)** |
| Run 03 | 5-obj, sw=3 | 30 | 28 | 28 | 28 | 0 | 0 | 100% (28/28) | 100% (28/28) |
| Run 3B | 5-obj, sw=3 | 27 | 23 | 23 | 23 | 0 | 0 | 100% (23/23) | 100% (23/23) |
| Run 06 sw=3 | 5-obj, sw=3 | 49 | 43 | 44 | 43 | 1 | 0 | 100% (43/43) | 97.7% (43/44) |
| Run 06 sw=5 | 5-obj, sw=5 | 27 | 19 | 21 | 19 | 2 | 0 | 100% (19/19) | 90.5% (19/21) |
| **Total** | | **157** | **132** | **136** | **132** | **4** | **0** | **100%** | **97.1% (132/136)** |

**Column definitions:**
- **Switched**: Episodes where the rule actually switched (full_cg_db only).
- **Post-Switch Correct**: Episodes that switched AND the agent answered correctly.
- **RC Firings**: Total RC firings among switched episodes.
- **TP**: RC fired AND episode was post-switch correct.
- **FP**: RC fired AND episode was post-switch incorrect.
- **FN**: RC did NOT fire AND episode was post-switch correct.
- **Sensitivity**: TP / (TP + FN). Answers: "When the agent gets a switched episode correct, did RC fire?"
- **PPV**: TP / (TP + FP). Answers: "When RC fires in a switched episode, is the agent correct?"

### Source Data Extraction

**Run 02** (report_run02_2026-02-16.md, Section 5.3):
- 24 switched episodes (80% of 30).
- Post-switch correct: 19/24.
- RC fired in 19/19 correct post-switch episodes (TP=19).
- RC fired in 1/5 incorrect post-switch episodes (FP=1).
- RC did not fire in 0/19 correct post-switch episodes (FN=0).
- RC firings among switched: 19 + 1 = 20.

**Run 03** (report_run03_2026-02-18.md, Section 4.5):
- 30 switched episodes (100% of 30).
- Post-switch correct: 28/30.
- RC fired in 28/28 correct episodes (TP=28).
- RC fired in 0/2 incorrect episodes (FP=0).
- FN=0. RC firings among switched: 28.

**Run 3B** (report_run04_2026-02-19.md, Sections 9.1 and 8):
- 27 switched episodes (90% of 30).
- Post-switch correct: 23/27.
- From Section 9.1 batch breakdown: Run 3B standalone has 23/23 RC fired -> correct (TP=23).
- All 4 incorrect Run 3B episodes had RC not fired (Section 9.1: "In all 4 incorrect episodes, `rule_change_hypothesis` did not fire").
- FP=0, FN=0. RC firings among switched: 23.

**Run 06 sw=3** (report_run06_2026-02-24.md, Section 7.1):
- 49 switched episodes (98% of 50).
- Post-switch correct: 43.
- RC fired in 44/49 switched episodes.
- PPV = 43/44 (97.7%): TP=43, FP=1.
- Sensitivity = 43/43 (100%): FN=0.
- RC firings among switched: 44.

**Run 06 sw=5** (report_run06_2026-02-24.md, Section 7.1):
- 27 switched episodes (54% of 50).
- Post-switch correct: 19.
- RC fired in 21/27 switched episodes.
- RC among incorrect switched = 2/8: FP=2.
- Sensitivity = 19/19 (100%): TP=19, FN=0.
- RC firings among switched: 21.
- Note: 1 additional RC false alarm in a non-switched episode (excluded from this table).

**Firings total:** 20 + 28 + 23 + 44 + 21 = **136**. CONFIRMED.

### Verification Notes (Table B2)

| Claim | Expected | Verified | Status |
|-------|----------|----------|--------|
| Run 02 PPV | 100% (19/19) | **95.0% (19/20)** | DISCREPANCY |
| Run 02 Sensitivity | 100% | 100% (19/19) | CONFIRMED |
| Run 03 PPV | 100% (28/28) | 100% (28/28) | CONFIRMED |
| Run 03 Sensitivity | 100% | 100% (28/28) | CONFIRMED |
| Run 3B PPV | 100% (23/23) | 100% (23/23) | CONFIRMED |
| Run 3B Sensitivity (post-switch correct) | 100% | 100% (23/23) | CONFIRMED |
| Run 3B Sensitivity per report | 94.4% (51/54 combined) | Uses all correct episodes (incl. pre-switch) as denominator | NOTE |
| Run 06 sw=3 PPV | 97.7% (43/44) | 97.7% (43/44) | CONFIRMED |
| Run 06 sw=3 Sensitivity | 100% | 100% (43/43) | CONFIRMED |
| Run 06 sw=5 PPV | 90.9% (20/22) | 90.5% (19/21) among switched; 90.9% (20/22) including non-switched | CLARIFICATION NEEDED |
| Run 06 sw=5 Sensitivity | 100% | 100% (19/19) | CONFIRMED |
| Total firings | 136 | 136 | CONFIRMED |

### Discrepancy Details

**Run 02 PPV:** The task expected PPV = 100% (19/19), implying RC fired exactly 19 times, all in correct episodes. The source report (Section 5.3) shows RC fired in 19/19 correct post-switch episodes **and** 1/5 incorrect post-switch episodes, yielding 20 total RC firings among switched episodes and PPV = 19/20 = 95.0%. The 19/19 figure reflects only the TP count, not the full PPV calculation.

**Run 06 sw=5 PPV:** The report states PPV = 20/22 (90.9%), but this denominator (22) includes 1 RC firing in a non-switched episode. Among the 21 RC firings in switched episodes only, PPV = 19/21 = 90.5%. The table above uses the switched-only framing (21 firings, PPV 90.5%) for consistency with the other rows. If non-switched false alarms are included in the PPV calculation, the figure becomes 20/22 = 90.9% as reported.

---

## Table B3: Error Classification (Run 08, 75-Step Cells)

### B3a: Error Summary

| Cell | Agent | Total Errors | Exactly-N | RE Errors | Pre-Switch Errors |
|------|-------|:------------:|:---------:|:---------:|:-----------------:|
| A | base_llm | 15 | 7 | 8 | 0 |
| B | cg_only | 16 | 14 | 2 | 0 |
| C | full_cg_db | 5 | 3 | 2 | 0 |
| **Total (75-step)** | | **36** | **24** | **12** | **0** |

**Source:** report_run08_2026-02-26.md, Section 5.1.

### B3b: Exactly-N Errors (Structural)

| Cell | Agent | Exactly-N Errors | All Submitted {A,B}? |
|------|-------|:----------------:|:--------------------:|
| A | base_llm | 7 | Yes (7/7) |
| B | cg_only | 14 | Yes (14/14) |
| C | full_cg_db | 3 | Yes (3/3) |
| D | base_llm (50-step) | 5 | Yes (5/5) |
| E | full_cg_db (50-step) | 4 | Yes (4/4) |
| **Total (all cells)** | | **33** | **Yes (33/33)** |

All 33 exactly-N episodes across all cells submitted {A, B} (the pre-switch rule). Zero exceptions. These episodes triggered the switch on the agent's final activation and submitted before observing any post-switch evidence. The error is structural, not a reasoning failure.

**Source:** report_run08_2026-02-26.md, Section 5.2 and Appendix C.

### B3c: Reasoning-Eligible Errors (75-Step Cells)

| Cell | Agent | RE Errors | RE Total | RE Error Rate | Error Breakdown |
|------|-------|:---------:|:--------:|:-------------:|-----------------|
| A | base_llm | 8 | 30 | 26.7% | 8x over-inclusion (guessed all five {A,B,C,D,E}) |
| B | cg_only | 2 | 33 | 6.1% | 1x over-inclusion (guessed all five), 1x timeout |
| C | full_cg_db | 2 | 43 | 4.7% | 1x over-inclusion (guessed all five), 1x superset {A,C,D,E} |

**Source:** report_run08_2026-02-26.md, Section 5.3.

### B3d: RE Error Classification by Type

| Error Type | Cell A (base_llm) | Cell B (cg_only) | Cell C (full_cg_db) | Total |
|------------|:------------------:|:-----------------:|:-------------------:|:-----:|
| Over-inclusion (guessed all 5) | 8 | 1 | 1 | 10 |
| Superset (proper subset error) | 0 | 0 | 1 | 1 |
| Timeout | 0 | 1 | 0 | 1 |
| Wrong type (e.g., disjunctive) | 0 | 0 | 0 | 0 |
| **Total RE errors** | **8** | **2** | **2** | **12** |

### B3e: Extended RE Error Data (Including 50-Step Cells)

| Cell | Agent | RE Errors | Error Breakdown |
|------|-------|:---------:|-----------------|
| A | base_llm (75) | 8 | 8x over-inclusion |
| D | base_llm (50) | 3 | 3x over-inclusion |
| **base_llm total** | | **11** | **11x over-inclusion (100%)** |
| B | cg_only (75) | 2 | 1x over-inclusion, 1x timeout |
| C | full_cg_db (75) | 2 | 1x over-inclusion, 1x superset |
| E | full_cg_db (50) | 2 | 1x over-inclusion, 1x wrong type (disjunctive) |
| **Scaffolded total** | | **6** | 3x over-inclusion, 1x superset, 1x timeout, 1x wrong type |

**Source:** report_run08_2026-02-26.md, Section 5.3.

### Verification Notes (Table B3)

| Claim | Expected | Verified | Status |
|-------|----------|----------|--------|
| All 33 {A,B} submissions = exactly-N | 33/33 | 33/33 (Sections 5.2 and Appendix C) | CONFIRMED |
| 11/11 base_llm RE errors = over-inclusion | 11/11 | 8 (Cell A) + 3 (Cell D) = 11, all "guessed all five" | CONFIRMED |
| 4 scaffolded-agent RE errors (75-step) | 4 total: 2 over-inclusion, 1 superset, 1 timeout | Cell B: 1 over-inclusion + 1 timeout; Cell C: 1 over-inclusion + 1 superset | CONFIRMED |
| Zero RE episodes answered {A,B} | 0 | Section 5.4: "Zero reasoning-eligible episodes across any cell answered {A,B}" | CONFIRMED |

---

## Summary of All Discrepancies

| Table | Item | Expected | Found | Severity |
|-------|------|----------|-------|----------|
| B2 | Run 02 PPV | 100% (19/19) | 95.0% (19/20) -- RC fired in 1 incorrect post-switch episode | MATERIAL -- source report Section 5.3 clearly shows RC fired in 1/5 wrong disjunctive episodes |
| B2 | Run 06 sw=5 PPV | 90.9% (20/22) | 90.5% (19/21) among switched episodes only; 90.9% (20/22) if non-switched false alarm included | FRAMING -- depends on whether non-switched false alarms are included in PPV denominator |

All other expected values were confirmed against source reports.
