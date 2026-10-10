# The Pilot-Blicket Experiment: Setup and Design

This document explains the pilot-blicket experimental setup for someone unfamiliar with the project. It bridges the gap between the code (which shows *what*) and the theory (which shows *why*).

For the companion document on agent internals, see [`agents/agent_explanations.md`](../agents/agent_explanations.md).

---

## 1. Introduction — Why Blickets?

The blicket detector is a paradigm from developmental psychology. A child sees a machine — the "detector" — that glows when certain objects ("blickets") are placed on it. The child must figure out which objects are blickets and what rule governs the detector. This setup, formalized by Kosoy, Liu, Collins et al. (2022) at the First Conference on Causal Learning and Reasoning (CLeaR), tests *causal induction through active experimentation*.

The key finding from child development: 4-year-olds succeed where reinforcement learning agents fail. Children don't just learn "A and B are blickets." They form **overhypotheses** — abstract beliefs about *what kinds of causal rules are possible*. Is the rule conjunctive or disjunctive? Could it be order-dependent? Probabilistic? These overhypotheses constrain the hypothesis space and, critically, *transfer across tasks*. A child who has learned that order can matter will check for order effects in a new task, even before seeing evidence.

This connects directly to **coordinate system revision**. When a learner's causal model breaks, they don't just update probabilities over existing variables — they discover that *new dimensions are causally relevant*. Placement order. Stochasticity. Rule instability. This is changing the coordinate system itself: realizing the relevant structure lives in dimensions you weren't tracking.

Why does this matter for AI? Current LLM training regimes fix an embedding space through next-token prediction. Fine-tuning, RLHF, and in-context learning optimize *within* this fixed coordinate system — adjusting which directions are weighted, not what directions exist. This is **micro-design**: optimizing within fixed dimensions for logarithmic improvements. The composition thesis distinguishes micro-design from **macro-design** — changing the coordinate system itself by discovering new variables, yielding step-function capability jumps.

The experiment's claim: a compositional architecture (Context Graph + Dynamic Behaviors) enables coordinate system revision in a controlled causal learning task. Neither component alone suffices — Context Graphs without Dynamic Behaviors are static; Dynamic Behaviors require Context Graphs to modify. Together, we hypothesize they produce genuine macro-design capability.

**Primary reference:**
> Kosoy, E., Liu, A., Collins, J., Chan, D., Hamrick, J.B., Ke, N.R., Huang, S., Kaufmann, B., Canny, J., & Gopnik, A. (2022). Learning Causal Overhypotheses through Exploration in Children and Computational Models. *Proceedings of the First Conference on Causal Learning and Reasoning (CLeaR)*, PMLR 177:390-406. [https://proceedings.mlr.press/v177/kosoy22a.html](https://proceedings.mlr.press/v177/kosoy22a.html)

---

## 2. The Blicket Detector Environment

### Basic Setup

The environment consists of:
- **Objects**: A set of named objects (A, B, C in the 3-object variant; A through E in the 5-object variant)
- **A detector**: a device that can be ACTIVE (glowing) or INACTIVE (not glowing)
- **A hidden causal rule**: determines when the detector activates based on which objects are present

### Actions

The agent can take three types of actions:
- **Place**: put an object on the detector (e.g., "Place object A on detector")
- **Remove**: take an object off the detector (e.g., "Remove object A from detector")
- **Check/Submit**: submit an answer identifying the rule type and which objects are blickets

### The Agent's Goal

Identify two things:
1. **Which objects are blickets** (the required objects for the rule)
2. **The rule type** — conjunctive (ALL required objects must be present) or disjunctive (ANY required object suffices)

The answer format is:
```
RULE_TYPE: conjunctive OR disjunctive
BLICKETS: <comma-separated list, e.g. A, B>
```

### How an Interaction Cycle Works

```
Agent sees:                    Agent responds:         Environment returns:
┌─────────────────────┐
│ Objects on detector: │        REASONING: I should    "You placed A on the
│   (none)            │        test A first to see     detector. The detector
│ Detector: INACTIVE  │  ───►  if it's a blicket.  ──► remains INACTIVE."
│ Steps taken: 0/50   │        ACTION: 1
│                     │
│ Available actions:  │
│  1. Place object A  │
│  2. Place object B  │
│  3. Place object C  │
│  4. Submit answer   │
└─────────────────────┘
```

The environment renders numbered action lists. The agent responds with reasoning and an action number. The environment returns a natural-language observation describing what happened: which object was placed or removed, and whether the detector is now active or inactive (and whether that changed).

Detector status messages:
- "The detector is now ACTIVE (glowing)."
- "The detector is now INACTIVE (not glowing)."
- "The detector remains ACTIVE (glowing)."
- "The detector remains INACTIVE (not glowing)."

The agent has a step budget (50 steps for 3 objects, 75 for 5 objects). If it doesn't submit an answer within that budget, the episode ends with no answer recorded.

**Source files:** `environment/blicket_env.py`, `environment/text_interface.py`

---

## 3. The Five Conditions

This is the heart of the experiment. Each condition uses the same blicket detector interface but hides a different causal rule, probing a specific dimension of coordinate system revision.

### 3a. Standard Conditions — The Initial Coordinate System

These two conditions test basic causal induction within a familiar framework. They establish whether the agent can reason about causality *at all*, before asking whether it can revise its framework.

#### Conjunctive (blickets = {A, B})

ALL required objects must be present simultaneously for activation. Placing A alone → INACTIVE. Placing B alone → INACTIVE. Placing A and B together → ACTIVE. The agent that understands "and" logic solves this.

The coordinate system is straightforward: **object identity → rule type → answer**. No hidden dimensions.

#### Disjunctive (blickets = {A, B})

ANY single required object is sufficient for activation. Placing A alone → ACTIVE. Placing B alone → ACTIVE. Placing C alone → INACTIVE. The opposite logic from conjunctive, but the same coordinate system: object identity is the only relevant dimension.

#### Why Both Exist

They establish a ceiling. If an agent fails on these, it can't do basic causal induction at all — and there's no point testing coordinate system revision. In practice, all three agent architectures hit 100% on both conditions across every run. The LLM is a capable reasoner when the coordinate system is known.

### 3b. Order-Sensitive — When Sequence Matters

**Rule:** Conjunctive {A, B} AND objects must be placed in order [A, then B].

This condition looks like standard conjunctive at first — you need both A and B on the detector. But placement order matters. Place A first, then B → ACTIVE. Place B first, then A → INACTIVE. Same objects on the detector, different result.

**What this probes:** The agent must discover that *placement sequence* is a causally relevant variable — a dimension entirely absent from the conjunctive/disjunctive framework. The overhypothesis "rules depend only on which objects are present" must be revised to "rules may also depend on the order objects are placed."

**Implementation detail:** The environment maintains a `_placement_order` list that tracks the sequence of additions. When an object is removed, all its occurrences are also removed from the order history. The `_check_order()` method extracts the subsequence of required objects from the current placement order and checks whether it matches the required order exactly.

**Concrete walkthrough:**
```
Step 1: Place A → INACTIVE       (A alone isn't enough — looks conjunctive)
Step 2: Place B → ACTIVE          (A+B works — standard conjunctive?)
Step 3: Remove A, Remove B
Step 4: Place B → INACTIVE        (B alone doesn't work)
Step 5: Place A → INACTIVE        (B+A doesn't work! Same objects, different result!)
        ↑ THIS is the coordinate system violation
```

At Step 5, the agent has both A and B on the detector — the same configuration that worked at Step 2 — but the detector stays inactive. The only difference is the *order* of placement. The agent must revise its model to include sequence as a causal dimension.

### 3c. Stochastic — When Rules Are Probabilistic

**Rule:** Conjunctive {A, B} with 70% activation probability.

When both blickets are present, the detector activates approximately 70% of the time. When the rule is not satisfied (wrong objects), the detector never activates (0%). The environment uses a seeded numpy RNG (`np.random.default_rng`) for reproducibility.

**What this probes:** The agent must discover that *activation is probabilistic*, not deterministic. The overhypothesis "the same configuration always gives the same result" must be revised to "the same configuration gives different results with some probability."

**Why 70%:** High enough that the pattern is detectable — the agent will see activation more often than not with the correct objects. But low enough to create genuine ambiguity — 30% of the time, the correct configuration produces no activation. At p=1.0, it's just conjunctive. At p=0.5, it's nearly indistinguishable from noise.

**Concrete walkthrough:**
```
Step 1:  Place A+B → ACTIVE       (looks conjunctive)
Step 2:  Clear, re-place A+B → ACTIVE
Step 3:  Clear, re-place A+B → INACTIVE!   ← same objects, different result
Step 4:  Clear, re-place A+B → ACTIVE
Step 5:  Clear, re-place A+B → INACTIVE    ← inconsistency persists
         ↑ Must revise from "deterministic" to "probabilistic"
```

The agent faces a dilemma: is A+B the wrong combination (and the earlier activations were flukes)? Or is the rule probabilistic? Resolving this requires repeated testing and statistical reasoning — a kind of reasoning that turns out to be difficult for all three agent architectures.

### 3d. Hidden Moderator — When Rules Change

**Rule:** Conjunctive {A, B}, but after 3 successful activations, switches permanently to Disjunctive {C}.

The first phase behaves like normal conjunctive: place A and B together and the detector activates. But the environment silently counts successful activations. After the 3rd activation, the rule switches: now only C activates the detector (as a disjunct), and A+B no longer works.

**What this probes:** The agent must discover that *the rule itself has changed*. The overhypothesis "there is one stable rule governing the detector" must be revised to "the rule can change based on some hidden condition." This is the most demanding revision — the agent must not only detect the change but re-learn the rule from scratch.

**Why `switch_after=3`:** Enough activations to establish confidence in the initial rule (the agent believes it has solved the task), but few enough that the switch happens before the agent submits an answer (the agent encounters the contradiction while still exploring).

**Concrete walkthrough:**
```
Steps 1-6:  Test objects, find A+B works → form "conjunctive, A and B" hypothesis
Step 7:     A+B → ACTIVE  (activation count: 1)
Step 8:     Clear, A+B → ACTIVE (count: 2)
Step 9:     Clear, A+B → ACTIVE (count: 3) → RULE SWITCHES SILENTLY
Step 10:    Clear, A+B → INACTIVE!   ← WHAT? A+B no longer works!
Step 11:    Place C → ACTIVE          ← C alone works now?!
            ↑ Must revise entire causal model
```

**Implementation detail:** The switch happens when `_successful_activations >= switch_after_n`. The environment replaces `self.causal_rule` with `self.hidden_moderator_config.alternate_rule`, sets `_rule_switched = True`, and recomputes activation under the new rule. If the current objects on the detector don't satisfy the new rule, the observation immediately reflects the deactivation.

### 3e. The Coordinate System Revision Hierarchy

Each condition maps to a specific type of revision in the agent's causal model:

| Condition | Initial Coordinate System | Required Revision | Gopnik Parallel |
|-----------|--------------------------|-------------------|-----------------|
| Conjunctive | Object identity → rule type | None (baseline) | Basic causal induction |
| Disjunctive | Object identity → rule type | None (baseline) | Basic causal induction |
| Order-sensitive | Object identity only | + Placement sequence dimension | Discovering temporal causality |
| Stochastic | Deterministic rules | + Probability dimension | Discovering stochastic causality |
| Hidden moderator | Stable rules | + Rule instability dimension | Theory revision after contradicting evidence |

The conditions form a hierarchy of revision difficulty:
- **Conjunctive/Disjunctive**: No revision needed. The initial coordinate system (which objects?) is sufficient.
- **Order-sensitive**: One new dimension (sequence). The agent must discover that *when* objects are placed matters, not just *which* objects.
- **Stochastic**: One new dimension (probability). The agent must discover that identical configurations can yield different results.
- **Hidden moderator**: One new dimension (rule stability) plus complete re-learning. The agent must discover that the rule itself is not fixed, then solve a new rule from scratch.

---

## 4. The Experimental Design Matrix

### Design Structure

The experiment follows a **5 conditions × 3 agents × 30 episodes** factorial design:

|  | base_llm | cg_only | full_cg_db |
|---|---|---|---|
| conjunctive | 30 episodes | 30 episodes | 30 episodes |
| disjunctive | 30 episodes | 30 episodes | 30 episodes |
| order_sensitive | 30 episodes | 30 episodes | 30 episodes |
| stochastic | 30 episodes | 30 episodes | 30 episodes |
| hidden_moderator | 30 episodes | 30 episodes | 30 episodes |

**Total: 450 episodes per run.**

### Why 30 Episodes Per Cell

30 episodes provides sufficient statistical power for accuracy comparisons between agents within a condition. With binary outcomes (correct/incorrect), 30 trials give a standard error of ~9% at 50% accuracy and ~5% at 80% accuracy — enough to detect meaningful differences between architectures.

### The Standard/Extended Split

Conditions are split into two phases:
- **Standard** (conjunctive + disjunctive): 180 episodes. Run first as a sanity check. All agents should hit 100%. If they don't, something is broken.
- **Extended** (order-sensitive + stochastic + hidden moderator): 270 episodes. The actual experiment. These conditions require coordinate system revision.

### Parameters

Configuration values from `experiments/config.py`:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `episodes_per_condition` | 30 | Statistical power for accuracy comparisons |
| `max_steps_per_episode` | 50 (3-obj) / 75 (5-obj) | Budget for exploration before forced answer |
| `max_retries_per_action` | 3 | Parse failure recovery |
| `action_model` | claude-sonnet-4-5-20250929 | Capable reasoning for causal induction |
| `trigger_model` | claude-haiku-4-5-20251001 | Cost-efficient DB trigger evaluation |
| `max_cost_per_episode` | $2.00 | Per-episode safety cap |
| `max_total_cost` | $500.00 | Total budget constraint |
| `conjunctive_blickets` | {A, B} | Two required objects |
| `disjunctive_blickets` | {A, B} | Same objects, different logic |
| `order_required_order` | [A, B] | A must precede B |
| `stochastic_probability` | 0.7 | Detectable but ambiguous |
| `moderator_switch_after` | 3 | Enough to establish, then break confidence |
| `base_seed` | 42 | Reproducibility |

### Reproducibility

Each episode gets a deterministic seed derived from the condition, agent, episode number, and base seed via `hash((condition, agent, episode, base_seed)) % 2^31`. This ensures that the same episode produces the same stochastic outcomes across runs, enabling direct comparison.

### Cost Model

Token costs tracked per episode using published model pricing:
- **Sonnet 4.5**: $3.00/M input, $15.00/M output
- **Haiku 4.5**: $0.80/M input, $4.00/M output

**Source file:** `experiments/config.py`

---

## 5. What Changed Across Runs

The experiment ran three times, each iteration addressing failures identified in the previous run. This progression is itself a core finding: it demonstrates which aspects of the architecture matter and how they interact.

### 5a. Run 1 (2026-02-13) — Initial Baseline

**Configuration:** 3 objects (A, B, C), 50-step budget, 450 episodes
**Cost:** $76.35

**Results (extended conditions only — standard was 100% across all cells):**

| Condition | base_llm | cg_only | full_cg_db |
|-----------|----------|---------|------------|
| order_sensitive | 93.3% | 0.0% | 0.0% |
| stochastic | 80.0% | 60.0% | 80.0% |
| hidden_moderator | 90.0% | 83.3% | 86.7% |
| **Extended overall** | **87.8%** | **47.8%** | **55.6%** |

**The paradox:** The compositional architecture *underperformed* the bare LLM. The system designed to enable coordinate system revision was worse at every condition (overall) than an LLM with just a task prompt.

**Three root causes identified:**

1. **Environment bug (order-sensitive):** The `_placement_order` list was append-only — removing objects from the detector didn't clear them from the order history. CG agents' systematic "clear between tests" strategy (remove all, then re-place) permanently corrupted the order tracking. Place A, Place B, Remove A, Remove B left the order as [A, B, A, B, ...], making it impossible for the order check to pass. base_llm succeeded accidentally — it tended to leave objects on the detector and just add more, avoiding the remove-then-retest pattern.

2. **Reactive-only DB triggers:** Dynamic Behaviors required a `had_unexpected_result` flag to fire. In order-sensitive episodes, CG agents never observed activation (due to the environment bug), so no "unexpected result" existed to trigger dimension discovery. The DBs that were supposed to detect order-dependence literally could not fire.

3. **Prescriptive CG guidelines:** The Context Graph specified fixed procedures: "Test each object A, B, C individually," "Test pairs: AB, AC, BC." These step-by-step instructions locked agents into rigid strategies that couldn't adapt when the environment behaved unexpectedly.

### 5b. Run 2 (2026-02-16) — Bug Fixes and Architecture Redesign

**Configuration:** 3 objects, 50-step budget, 300 episodes (30 standard smoke test + 270 extended)
**Cost:** $90.13

**Changes and motivations:**

| Change | Motivation |
|--------|------------|
| Fix `_placement_order` to remove objects when removed | Environment bug made order-sensitive unfair |
| Redesign CG: prescriptive → objective-oriented guidelines | Fixed strategies can't adapt to unknown conditions |
| Add stagnation-mode DB trigger | Reactive-only triggers couldn't fire without activations |
| Graded 0-10 trigger evaluation (was binary) | Binary too coarse for nuanced evidence detection |
| Remove `requires_hypothesis_failure` gate from DB4 | Gate prevented firing on hidden moderator evidence |

**Results (extended conditions):**

| Condition | base_llm | cg_only | full_cg_db |
|-----------|----------|---------|------------|
| order_sensitive | 100.0% | 96.7% | 86.7% |
| stochastic | 73.3% | 70.0% | 76.7% |
| hidden_moderator | 100.0% | 73.3% | 83.3% |
| **Extended overall** | **91.1%** | **80.0%** | **82.2%** |

**What improved:**
- Order-sensitive went from 0% → 96.7% (cg_only), 0% → 86.7% (full_cg_db) — the environment bug fix validated the architecture's capability.
- Hidden moderator post-switch accuracy revealed an important distinction: not all episodes actually trigger the rule switch (the agent may submit before 3 activations). In Run 2, full_cg_db triggered the switch in 24 of 30 episodes (80%) — much higher than base_llm (1 of 30, 3%) because CG agents explore more systematically. Among those 24 post-switch episodes, full_cg_db got 19 correct (79.2%). The `rule_change_hypothesis` DB fired with 100% precision — it fired in all 19 correct post-switch episodes and only 1 of 5 wrong ones.
- The pre/post-switch split reveals the real test:

| Agent | Episodes switched | Pre-switch accuracy | Post-switch accuracy | Overall |
|-------|-------------------|--------------------|--------------------|---------|
| base_llm | 1/30 (3%) | 29/29 (100%) | 1/1 (100%) | 100.0% |
| cg_only | 11/30 (37%) | 19/19 (100%) | 3/11 (27.3%) | 73.3% |
| full_cg_db | 24/30 (80%) | 6/6 (100%) | 19/24 (79.2%) | 83.3% |

  base_llm's 100% is misleading — it almost never encounters the hard part. It answers before triggering 3 activations. cg_only encounters the switch but mostly fails to recover. full_cg_db encounters the switch at the highest rate *and* recovers most often.

**What didn't improve:**
- full_cg_db *underperformed* cg_only on order-sensitive (86.7% vs 96.7%). Why? The `rule_change_hypothesis` DB was firing as a false positive on order-sensitive episodes. When an agent with the correct conjunctive+order frame got pushed into a MODERATOR_DETECTION state by a false DB fire, it started looking for rule changes that didn't exist — and lost its correct hypothesis.

**Key insight:** DBs provide both signal and noise. The noise (false positives on wrong conditions) can cancel the signal (true positives on correct conditions). Run 3 must address trigger selectivity.

### 5c. Run 3 (2026-02-17/18) — Scaling and Competitive Inhibition

**Configuration:** 5 objects (A-E), 75-step budget, 330 episodes (30 standard + 270 extended + Phase A validation)
**Cost:** ~$300

**Changes and motivations:**

| Change | Motivation |
|--------|------------|
| Raise diagnostic DB thresholds: 6.0 → 7.0 | Filter marginal trigger scores |
| Add competitive inhibition (winner-take-most) | Prevent false-positive DB interference |
| Scale to 5 objects (A-E) with 75-step budget | Test architecture in larger search space |
| Phase A validation on 3-object first | De-risk expensive 5-object run |

**Competitive inhibition mechanism:**

The key architectural addition. When multiple diagnostic DBs want to fire simultaneously:
- **DB1 (exploration_stagnation)** fires independently — it's a meta-intervention that prompts broader exploration, not a diagnostic about a specific condition.
- **DB2 (order_hypothesis), DB3 (stochasticity_hypothesis), DB4 (rule_change_hypothesis)** compete: all three are scored by the trigger model, the top scorer fires, and the runner-up fires only if it scores within 1.0 points of the winner.

This prevents the Run 2 problem: in order-sensitive episodes, `rule_change_hypothesis` used to outscore `order_hypothesis` and fire as a false positive. With competitive inhibition, `order_hypothesis` wins the competition on order-sensitive episodes (it has stronger evidence), suppressing the false-positive `rule_change_hypothesis` fire.

**Results (5-object extended conditions):**

| Condition | base_llm | cg_only | full_cg_db |
|-----------|----------|---------|------------|
| order_sensitive | 100.0% | 100.0% | 100.0% |
| stochastic | 76.7% | 76.7% | 73.3% |
| hidden_moderator | 76.7% | 76.7% | 93.3% |
| **Extended overall** | **84.4%** | **84.4%** | **88.9%** |

**What improved:**
- Order-sensitive: full_cg_db 100% (was 86.7%) — competitive inhibition eliminated false-positive DB drag. All three agents now solve order-sensitive perfectly with 5 objects.
- Hidden moderator: full_cg_db 93.3% (was 83.3%). In this run, 100% of full_cg_db episodes triggered the rule switch (all 30 of 30), and `rule_change_hypothesis` achieved perfect precision: fired in all 28 correct post-switch episodes, fired in 0 of the 2 wrong ones. The pre/post split for Run 3:

| Agent | Episodes switched | Pre-switch accuracy | Post-switch accuracy | Overall |
|-------|-------------------|--------------------|--------------------|---------|
| base_llm | 26/30 (87%) | 4/4 (100%) | 19/26 (73.1%) | 76.7% |
| cg_only | 28/30 (93%) | 2/2 (100%) | 21/28 (75.0%) | 76.7% |
| full_cg_db | 30/30 (100%) | N/A | 28/30 (93.3%) | 93.3% |

  With 5 objects, agents explore more before answering, so the switch triggers much more often across all architectures. The playing field is leveled — every full_cg_db episode now faces the hard problem. Yet full_cg_db still achieves 93.3% post-switch, compared to base_llm's 73.1% and cg_only's 75.0%. This is the strongest evidence for coordinate system revision capability.

- **Ranking inversion:** full_cg_db (88.9%) > base_llm (84.4%) = cg_only (84.4%). For the first time, the full compositional architecture outperforms the bare LLM.

**DB fire rates (full_cg_db, 5-object):**

| Behavior | Order-sensitive | Stochastic | Hidden moderator |
|----------|----------------|------------|------------------|
| exploration_stagnation (DB1) | 43.3% | 40.0% | 26.7% |
| order_hypothesis (DB2) | **96.7%** | 96.7% | 96.7% |
| stochasticity_hypothesis (DB3) | 43.3% | **76.7%** | 70.0% |
| rule_change_hypothesis (DB4) | 53.3% | 86.7% | **93.3%** |

Note that DB2 fires at 96.7% across *all* conditions — but competitive inhibition prevents it from dominating on stochastic or hidden moderator episodes. The bold entries show the intended "correct" DB for each condition. DB4 fires at 93.3% on hidden moderator (intended) vs 53.3% on order-sensitive (suppressed by competition with DB2).

**What stayed the same:**
- Stochastic: ~73-77% across all agents, all runs. A genuine difficulty ceiling.

### 5d. Cost Progression

| Run | Episodes | Total Cost | Avg/Episode |
|-----|----------|------------|-------------|
| Run 1 (3-obj, 450 eps) | 450 | $76.35 | $0.17 |
| Run 2 (3-obj, 300 eps) | 300 | $90.13 | $0.30 |
| Run 3 (5-obj, 330 eps) | 330 | ~$300 | ~$0.91 |
| **Total** | **1,080** | **~$466** | **$0.43** |

Cost per episode scales roughly with object count: more objects → larger search space → more steps → more tokens. The 5-object run cost ~5x per episode compared to 3-object Run 1. With ~$34 remaining of the $500 budget, future work requires either budget expansion or more cost-efficient designs.

### 5e. The Progression in One Table (Extended Conditions Only)

| Metric | Run 1 | Run 2 | Run 3 |
|--------|-------|-------|-------|
| Objects | 3 | 3 | 5 |
| Step budget | 50 | 50 | 75 |
| CG guidelines | Prescriptive | Objective-oriented | Objective-oriented |
| DB triggers | Reactive, binary | + Proactive, graded | + Competitive inhibition, 7.0 threshold |
| Best agent (extended) | base_llm (87.8%) | base_llm (91.1%) | **full_cg_db (88.9%)** |
| Order-sensitive (full_cg_db) | 0% | 86.7% | 100% |
| Hidden moderator (full_cg_db) | 86.7% | 83.3% | 93.3% |
| Stochastic (full_cg_db) | 80.0% | 76.7% | 73.3% |
| Cost | $76.35 | $90.13 | ~$300 |

### 5f. What This Progression Demonstrates

Each run isolates a different source of failure:

**Run 1 → Run 2:** Was the failure in the environment or the architecture? Answer: both. The `_placement_order` bug made order-sensitive unfair to systematic agents. But even after the fix, prescriptive CG guidelines and reactive-only DB triggers limited performance. The fix: objective-oriented CG + proactive/graded DBs.

**Run 2 → Run 3:** Was the remaining gap in DB sensitivity or DB selectivity? Answer: selectivity. DBs were firing when they should (sensitivity was adequate), but also firing when they shouldn't (false positives on wrong conditions). The fix: competitive inhibition so only the most relevant DB fires.

**Across all three runs:** Stochastic remains hard at ~73-77% regardless of architecture. This is a genuine difficulty ceiling: with p=0.7, probabilistic feedback creates ambiguity that no current mechanism resolves. The agent cannot easily distinguish "I have the wrong objects" from "I have the right objects but the coin flip went against me."

The three-run progression also demonstrates something about research methodology: the architecture's advantage was *invisible* until the environment bug was fixed (Run 2) and false-positive interference was eliminated (Run 3). A single-run evaluation would have concluded that compositional architecture hurts performance — the opposite of the true finding.

---

## 6. Future Extensions

### 6a. Extensions That Probe Deeper Coordinate System Revision

**Multi-step causality.** Activation requires a sequence of 3+ actions, not just placement order. For example: place A, then remove A, then place B. Tests whether the agent can discover that causal chains extend beyond pairwise relationships. Would require a new DB that detects multi-step dependencies.

**Temporal windows.** Objects must be placed within N steps of each other. Placing A at step 1 and B at step 20 fails; placing A at step 1 and B at step 3 succeeds. Tests whether the agent can discover time-dependent causality — a dimension orthogonal to order and identity.

**Compositional rules.** Activation requires {A AND B} OR {C AND D} — a compound rule where the disjuncts are themselves conjunctions. Tests whether the agent can discover hierarchical rule structure. Maps to overhypothesis hierarchies: the rule *type* is itself structured.

**Few-shot incongruent.** Provide examples establishing one frame (conjunctive), then test with a different frame (order-sensitive). This is the strongest test of coordinate system revision — can the agent escape an *established* coordinate system? Current experiments start each episode fresh; this extension tests escape from prior frames.

**Cyclic rule changes.** Instead of one permanent switch (hidden moderator), the rule cycles between multiple states: conjunctive → disjunctive → order-sensitive → conjunctive. Tests whether the agent can discover *periodicity* in rule changes — a second-order property of rule instability.

### 6b. Extensions That Test Overhypothesis Transfer

**Cross-episode memory.** Allow agents to retain information from prior episodes. Test whether successful revision in episode N improves revision speed in episode N+1 — the hallmark of overhypothesis formation. This would require adding RECALL states to the CG that access episodic memory.

**Abstract pattern transfer.** Train on order-sensitive with objects A, B, C, then test on new objects D, E, F. If the agent retains the overhypothesis "order can matter," it should check for order effects earlier in the new episode. This directly tests Gopnik's finding that children transfer overhypotheses across task domains.

**Mixed conditions within episodes.** Instead of one condition per episode, conditions change mid-episode beyond the single switch in hidden moderator. For example: conjunctive for the first 20 steps, then stochastic for the next 30. Tests how rapidly the agent can detect and adapt to frame shifts.

### 6c. Extensions That Test Architecture Limits

**Scaling beyond 5 objects.** Run with 7, 10, or 15 objects. At what point does the combinatorial explosion overwhelm the architecture? With 15 objects there are 32,767 possible subsets and 15! orderings. Does CG+DB degrade gracefully (accuracy drops slowly) or catastrophically (sudden failure)?

**Model capability ablation.** Run with Haiku (weaker model) as the action model instead of Sonnet. Does CG+DB help *more* when the base model is weaker? The composition thesis predicts yes: architectural structure compensates for capability deficits.

**Entropy stratification.** Vary guideline density per CG state — dense in VERIFICATION (low entropy, precise instructions), sparse in INITIAL_EXPLORATION (high entropy, open-ended). Test whether matching entropy to task phase improves performance over uniform density.

**Repeatable DB firing.** Allow Dynamic Behaviors to fire multiple times per episode with a cooldown period. Test whether cumulative interventions help on long episodes (reinforcing correct hypotheses) or cause information overload (too many state transitions).

### 6d. The Stochastic Problem

Stochastic remains the unsolved condition — approximately 75% accuracy across all three runs and all three architectures. No architectural change has moved this number. Extensions specifically targeting this difficulty:

**Explicit counting tool.** Give the agent a "repeat N times" action that runs the same configuration N times and returns the activation frequency. Tests whether tool use enables statistical reasoning that conversation-based exploration cannot. The hypothesis: agents struggle with stochastic because they reason about individual observations rather than distributions.

**Bayesian update state.** Add a CG state that explicitly prompts the agent to maintain and update probability estimates after each observation. For example: "Current estimate: P(A is blicket) = 0.7, P(B is blicket) = 0.9." Tests whether structured probabilistic tracking improves accuracy.

**Higher step budget.** Increase from 75 to 150 steps. Tests whether the limitation is exploration time (not enough observations to distinguish p=0.7 from p=0 or p=1.0) or reasoning capacity (the agent has enough data but can't process it correctly). If doubling the budget doesn't help, the problem is reasoning, not data.

---

## 7. Key Takeaways

1. **The blicket paradigm creates a controlled test of coordinate system revision.** Each condition requires discovering that a new dimension is causally relevant — not just updating weights on known variables.

2. **Standard conditions confirm causal reasoning capability.** All agents hit 100% on conjunctive and disjunctive, establishing that the LLM is a capable reasoner. The question is whether it can *revise its reasoning framework*.

3. **Extended conditions each probe a specific revision type.** Order-sensitive: discovering sequence dependence. Stochastic: discovering probabilistic rules. Hidden moderator: discovering rule instability plus re-learning.

4. **Compositional architecture enables revision — but only with both sensitivity and selectivity.** The three-run progression shows that DB triggers must fire when they should (sensitivity, addressed in Run 2) AND not fire when they shouldn't (selectivity, addressed in Run 3).

5. **Competitive inhibition was the key mechanism.** It preserved DB benefits on the right conditions (hidden moderator: +16.6pp over base_llm) while eliminating DB costs on wrong conditions (order-sensitive: from 86.7% to 100%).

6. **Overhypotheses connect to the architecture.** The CG embodies the current overhypothesis — what variables matter, what states exist, what transitions are allowed. DBs revise that overhypothesis at runtime when evidence demands it. Neither component alone produces revision: CG without DB is static; DB without CG has nothing to modify.

7. **Stochastic reveals a fundamental limitation.** None of the current mechanisms enable statistical reasoning about probabilistic causality. This is not an architecture failure — it's a boundary condition that points toward future work (explicit counting tools, Bayesian tracking states).

---

## Source Files

| File | Content |
|------|---------|
| `environment/blicket_env.py` | Base environment: actions, causal rules, activation logic |
| `environment/extended_env.py` | Extended conditions: order, stochastic, hidden moderator |
| `environment/blicket_env_5obj.py` | 5-object variant |
| `environment/extended_env_5obj.py` | 5-object extended conditions |
| `environment/text_interface.py` | State rendering, action parsing, observation formatting |
| `experiments/config.py` | Experiment parameters, cost tracking |
| `experiments/episode_runner.py` | Core episode loop |
| `experiments/run_standard.py` | Standard condition runner (3-object) |
| `experiments/run_extended.py` | Extended condition runner (3-object) |
| `experiments/run_standard_5obj.py` | Standard condition runner (5-object) |
| `experiments/run_extended_5obj.py` | Extended condition runner (5-object) |
| `experiments/run_phaseA.py` | Phase A competitive inhibition validation |
| `report-run1_2026-02-13.md` | Run 1 results and analysis |
| `report-run2_2026-02-16.md` | Run 2 results and analysis |
| `report-run3_2026-02-17.md` | Run 3 Phase A results |
| `report-run3_2026-02-18.md` | Run 3 5-object results |
| `agents/agent_explanations.md` | Companion document: agent architecture details |
| `artifacts/01_concept.md` | Theoretical framing |
| `artifacts/08_proposal.md` | Formal proposal with isomorphism mapping |
