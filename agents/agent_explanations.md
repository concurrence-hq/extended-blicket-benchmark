# Pilot-Blicket Agent Architectures

This document explains the three agent architectures used in the pilot-blicket experiment, how they work internally, and how they relate to the production platform from which they are derived. It is written for someone unfamiliar with the codebase.

---

## 1. Introduction

The pilot-blicket experiment tests whether compositional agent architecture helps with **coordinate system revision** — the problem of detecting that your model of causality has become invalid and building a new one. This concept comes from Alison Gopnik's developmental psychology research: when children encounter evidence that contradicts their causal theory, they don't just patch the old theory — they construct a new framework with different variables and relationships.

We test three agent architectures, each adding a layer of compositional structure:

1. **base_llm** — A bare LLM with a task description. No structured guidance.
2. **cg_only** — The same LLM plus a Context Graph (CG): a fixed state machine that structures exploration.
3. **full_cg_db** — The CG agent plus Dynamic Behaviors (DBs): runtime monitors that expand the CG when evidence demands it.

The experiment runs these agents across five causal learning conditions — two standard (conjunctive, disjunctive) and three extended (order-sensitive, stochastic, hidden moderator). The extended conditions require the agent to revise its causal framework: the standard conjunctive/disjunctive vocabulary is insufficient.

**Source files:** `agents/base_llm_agent.py`, `agents/cg_agent.py`, `agents/full_agent.py`, `agents/context_graph.py`, `agents/dynamic_behaviors.py`

---

## 2. The Three Agents at a Glance

| Property | base_llm | cg_only | full_cg_db |
|----------|----------|---------|------------|
| Task description in system prompt | Yes | Yes | Yes |
| Context Graph (exploration structure) | No | Yes (4 states) | Yes (4 + up to 4 dynamic) |
| Dynamic Behaviors (runtime adaptation) | No | No | Yes (4 behaviors) |
| State transitions declared by agent | No | Yes | Yes |
| System notifications injected mid-episode | No | No | Yes |
| Trigger evaluation model | — | — | Haiku 4.5 |
| Action/reasoning model | Sonnet 4.5 | Sonnet 4.5 | Sonnet 4.5 |

The hierarchy is additive — each layer adds structure without removing capabilities:

```
BaseLLMAgent ──────► CGAgent ──────────► FullAgent
(task prompt)        (+ Context Graph)   (+ Dynamic Behaviors)
```

"Compositional" here means that each layer contributes a distinct function. The CG provides vocabulary for reasoning about exploration phases. The DBs provide a mechanism for expanding that vocabulary when evidence demands it. Together they implement coordinate system revision: the agent's problem space literally grows to accommodate new phenomena.

---

## 3. The Base LLM Agent

**File:** `agents/base_llm_agent.py` — class `BaseLLMAgent`

The base agent receives only a task description in its system prompt:

> You are interacting with a 'blicket detector' — a device that glows when certain objects (called 'blickets') are placed on it.
>
> There are 3 objects: A, B, and C. Some may be blickets, some may not.
>
> The detector follows a hidden rule:
> - CONJUNCTIVE: ALL blicket objects must be on the detector for it to activate.
> - DISJUNCTIVE: ANY single blicket object is enough to activate the detector.
>
> Your goal: Figure out which objects are blickets AND whether the rule is conjunctive or disjunctive.

The agent sees numbered available actions each turn and responds with `REASONING:` and `ACTION:` fields. When it can't determine an action from the response, a **4-strategy parse cascade** attempts recovery:

1. `ACTION: N` — numbered selection
2. `ACTION: keyword` — keyword matching (e.g., "place A")
3. Last line keyword scan
4. Full text keyword scan

If all four strategies fail, a retry prompt is injected. After 3 consecutive failures, the episode terminates.

This agent establishes the baseline: what can a capable LLM do with raw task understanding and no structured exploration guidance?

---

## 4. Context Graph — The Problem Space Map

**File:** `agents/context_graph.py`

### 4a. What a Context Graph Is

A Context Graph is a directed graph of **typed states**. Each state carries:

- **Name** — identifier (e.g., `INITIAL_EXPLORATION`)
- **State type** — one of `ACTION`, `DECISION`, or `REFLECTION`
- **Objective** — what the agent should accomplish in this state
- **Guidelines** — ordered instructions for how to proceed
- **Transitions** — edges to other states with human-readable conditions

The CG is **injected into the system prompt every turn**. The agent sees where it currently is, what it should do there, and where it can go next. To change states, the agent includes `TRANSITION: STATE_NAME` in its response. The transition is validated against the graph topology — only edges that exist in the graph are allowed.

### 4b. The Base CG (4 States)

The default CG for the blicket task has four states arranged in a linear-with-loops progression:

```
┌──────────────────────────────┐
│     INITIAL_EXPLORATION      │  [action]
│  "Understand how individual  │
│   objects affect detector"   │
└──────────────┬───────────────┘
               │  "After testing all 3 individual objects"
               ▼
┌──────────────────────────────┐       "If results are confusing,
│     COMBINATION_TESTING      │        go back to singles"
│  "Discover how object        │◄──────────────────────────┐
│   combinations and           │  [action]                  │
│   arrangements affect        │                            │
│   activation"                │                            │
└──────────────┬───────────────┘                            │
               │  "After testing key combinations"          │
               ▼                                            │
┌──────────────────────────────┐                            │
│    HYPOTHESIS_EVALUATION     │  [reflection]              │
│  "Evaluate your evidence     │────────────────────────────┘
│   to determine the           │  "If hypothesis is unclear,
│   causal rule"               │   test more combinations"
└──────────────┬───────────────┘
               │  "When you have a clear hypothesis"
               ▼
┌──────────────────────────────┐
│        VERIFICATION          │  [decision]
│  "Confirm your hypothesis    │──► back to HYPOTHESIS_EVALUATION
│   with targeted tests        │    "If verification fails"
│   before submitting"         │
└──────────────────────────────┘
```

Walking through each state:

**INITIAL_EXPLORATION** (action) — The agent tests individual objects one at a time. Guidelines direct it to observe responses, pay attention to changes, and notice patterns. The transition condition ("after testing all 3 individual objects") is evaluated by the LLM, not enforced programmatically.

**COMBINATION_TESTING** (action) — The agent tests object groupings. Guidelines explicitly mention varying arrangement and sequence, not just presence/absence. A back-edge to INITIAL_EXPLORATION handles confusion.

**HYPOTHESIS_EVALUATION** (reflection) — The agent steps back from acting and evaluates its evidence. Guidelines pose questions: what rule type fits? Which objects are blickets? Are there inconsistencies? A back-edge to COMBINATION_TESTING handles insufficient evidence.

**VERIFICATION** (decision) — The agent designs discriminating tests and makes a final judgment. If verification fails, the agent returns to HYPOTHESIS_EVALUATION.

The deliberate progression — **explore → combine → reflect → verify** — mirrors how a structured investigator would approach the problem. The back-edges allow iteration when evidence is ambiguous.

### 4c. What the Agent Sees Each Turn

The `render_for_prompt()` method generates text like this (example with the agent in COMBINATION_TESTING):

```
=== Context Graph ===
Current State: COMBINATION_TESTING [action]
Objective: Discover how object combinations and arrangements affect activation.
Guidelines:
  - Test different groupings of objects on the detector
  - Vary not just which objects are present, but how you arrange and sequence them
  - Compare results across different configurations
  - If a configuration that should work doesn't, consider what else might matter
Transitions:
  -> HYPOTHESIS_EVALUATION: After testing key combinations
  -> INITIAL_EXPLORATION: If results are confusing, go back to singles

Graph overview:
  [action] INITIAL_EXPLORATION -> COMBINATION_TESTING
  [action] COMBINATION_TESTING -> HYPOTHESIS_EVALUATION, INITIAL_EXPLORATION <-- YOU ARE HERE
  [reflection] HYPOTHESIS_EVALUATION -> VERIFICATION, COMBINATION_TESTING
  [decision] VERIFICATION -> HYPOTHESIS_EVALUATION
```

Three layers of information:
1. **Current state details** — objective, guidelines, available transitions
2. **Graph overview** — condensed topology with a `<-- YOU ARE HERE` marker
3. **Modification log** — last 3 runtime changes (only present for full_cg_db when DBs have fired)

### 4d. Why the CG Matters for Coordinate System Revision

The base CG assumes a **conjunctive/disjunctive world**. Its vocabulary — "individual objects," "combinations," "rule type" — is designed for conditions where object identity is all that matters. For standard conditions, 4 states are sufficient.

For extended conditions, the 4-state graph has **no vocabulary** for the relevant phenomena:
- **Order-sensitive**: No concept of "placement sequence" vs. "object identity"
- **Stochastic**: No concept of "probabilistic activation" vs. "deterministic rule"
- **Hidden moderator**: No concept of "rule change" vs. "stable rule"

The CG agent handles these through the flexible guideline "If observations are inconsistent, consider whether factors beyond object identity might matter" — but it has no structured state to move into. It must reason about these phenomena within existing states.

This is where Dynamic Behaviors come in: they expand the graph **when evidence demands it**.

---

## 5. Dynamic Behaviors — Adaptive Problem Space Expansion

**File:** `agents/dynamic_behaviors.py`

### 5a. The Core Concept

Dynamic Behaviors are **runtime monitors** that watch the agent's exploration. When a DB detects evidence of a phenomenon the CG can't handle, it **fires**: adding new states and transitions to the CG and injecting a system notification into the conversation.

Each DB fires **at most once per episode** (one-shot). This is a deliberate simplification for experimental control — production behaviors can fire repeatedly (see Section 7).

This is **coordinate system revision** made literal: the agent's problem space physically expands. Before a DB fires, the agent has 4 states and no vocabulary for the phenomenon. After, it has 5+ states with explicit objectives and guidelines for investigating the new dimension.

### 5b. The Trigger Pipeline (3 Stages)

Each DB goes through three evaluation stages:

**Stage 1: Pre-screen** (cheap, deterministic)

Quick checks to avoid unnecessary LLM calls:
- Has the minimum step count been reached? (`min_steps`)
- For stagnation behaviors: has the detector ever activated? (If yes, stagnation doesn't apply)
- For stagnation behaviors: have enough steps passed without activation? (`stagnation_threshold`)
- Has the behavior already fired? (One-shot enforcement)

**Stage 2: LLM evaluation** (Claude Haiku 4.5, scores 0-10)

A condensed context is built from:
- The last 5 conversation messages (truncated to 300 chars each)
- Current CG state name
- Exploration summary (step count, activation count, CG state)

This context is sent to Haiku with the behavior's evaluation prompt. Haiku returns a score from 0-10. If the score meets or exceeds `fire_threshold`, the behavior fires.

The evaluation prompt template:

```
You are evaluating whether a specific condition has been met in an
agent's exploration of a causal learning task.

Condition to evaluate:
{behavior-specific evaluation prompt}

Current CG state: {current_state}
Exploration summary: {summary}
Recent conversation:
{last 5 messages, 300 chars each}

Rate how strongly this condition is met on a scale of 0-10,
where 0 = definitely not met and 10 = clearly met.
Respond with just the number.
```

**Stage 3: Modification application**

When a behavior fires:
1. Each modification in the behavior's list is applied to the CG
2. A notification message is injected into the conversation as a `user` role message
3. The behavior name is added to the `_fired` set (preventing re-firing)

### 5c. The Four Behaviors

#### DB1: exploration_stagnation

**Purpose:** Fires when the agent has explored for several steps without ever activating the detector, suggesting that standard exploration strategies are failing.

**Trigger:**
- `stagnation_mode: True` — only eligible when activation history has no `True` values
- `min_steps: 6`
- `stagnation_threshold: 6`
- `fire_threshold: 6.0`

**Evaluation prompt:**
> The agent has been exploring the detector for several steps but has NEVER achieved activation. Are they stuck in an ineffective exploration pattern? Consider whether they need to think about alternative causal mechanisms beyond simple conjunctive/disjunctive rules.

**CG modification:** Adds `DIMENSION_DISCOVERY` state (reflection)
- Objective: "The standard conjunctive/disjunctive framework doesn't explain your observations. Consider alternative causal dimensions: order of placement, probabilistic activation, or changing rules."
- Guidelines:
  1. Review ALL observations for patterns beyond which objects are present
  2. Consider: does the ORDER of placement matter?
  3. Consider: is activation PROBABILISTIC (same setup, different results)?
  4. Consider: has the RULE CHANGED over time?
  5. Form a new hypothesis about the causal mechanism
- Connected from: `HYPOTHESIS_EVALUATION`, `COMBINATION_TESTING`
- Transitions to: `COMBINATION_TESTING`, `VERIFICATION`

**Notification:**
> [SYSTEM] Your observations don't fit standard causal rules. A new exploration state DIMENSION_DISCOVERY is now available. Consider transitioning there to explore alternative causal dimensions.

---

#### DB2: order_hypothesis

**Purpose:** Fires when evidence suggests that something beyond object identity matters for activation — particularly placement order or sequence.

**Trigger:**
- `min_steps: 8`
- `fire_threshold: 6.0`
- No stagnation requirement — can fire even if activations have occurred

**Evaluation prompt:**
> Has the agent observed inconsistent or surprising results that could be explained by the ORDER or SEQUENCE of object placement mattering? For example: the same set of objects producing different results, or configurations that 'should' work based on object identity alone failing to activate.

**CG modification:** Adds `ORDER_TESTING` state (action)
- Objective: "Test whether the order of object placement affects activation. Systematically vary placement order while keeping the same objects."
- Guidelines:
  1. Clear detector completely before each test
  2. Place the SAME objects in DIFFERENT orders
  3. Record: which order activated, which didn't
  4. Try at least 3 different orderings
- Connected from: `DIMENSION_DISCOVERY`, `COMBINATION_TESTING`
- Transitions to: `HYPOTHESIS_EVALUATION`, `VERIFICATION`

**Notification:**
> [SYSTEM] Evidence suggests placement order may matter. A new ORDER_TESTING state is available for systematic order experiments.

---

#### DB3: stochasticity_hypothesis

**Purpose:** Fires when the same configuration gives inconsistent results across repetitions.

**Trigger:**
- `min_steps: 8`
- `fire_threshold: 6.0`

**Evaluation prompt:**
> Has the agent tried the EXACT same configuration of objects (same objects on detector) MORE THAN ONCE and gotten DIFFERENT results? This would suggest probabilistic/stochastic activation rather than a deterministic rule.

**CG modification:** Adds `RELIABILITY_TESTING` state (action)
- Objective: "Test whether activation is probabilistic by repeating the same configuration multiple times."
- Guidelines:
  1. Pick one configuration that activated before
  2. Clear and re-place the same objects 5+ times
  3. Count activations vs non-activations
  4. Estimate the activation probability
- Connected from: `DIMENSION_DISCOVERY`, `COMBINATION_TESTING`
- Transitions to: `HYPOTHESIS_EVALUATION`

**Notification:**
> [SYSTEM] Inconsistent results detected for same configurations. A new RELIABILITY_TESTING state is available to measure activation probability.

---

#### DB4: rule_change_hypothesis

**Purpose:** Fires when a previously working rule suddenly stops working, suggesting the underlying rule has changed.

**Trigger:**
- `min_steps: 10`
- `fire_threshold: 6.0`

**Evaluation prompt:**
> Has the agent found a rule that WORKED CONSISTENTLY for several trials but then STOPPED working? The same objects that reliably activated the detector no longer do so, suggesting the underlying rule has changed.

**CG modification:** Adds `MODERATOR_DETECTION` state (reflection)
- Objective: "The causal rule appears to have changed. Investigate what triggers rule changes and identify the new rule."
- Guidelines:
  1. Count how many successful activations occurred before the change
  2. Re-test all objects individually under the new regime
  3. Re-test combinations to identify the new rule
  4. Consider: does the rule cycle or switch permanently?
- Connected from: `DIMENSION_DISCOVERY`, `HYPOTHESIS_EVALUATION`
- Transitions to: `COMBINATION_TESTING`, `VERIFICATION`

**Notification:**
> [SYSTEM] The causal rule appears to have changed. A new MODERATOR_DETECTION state is available to investigate rule shifts.

### 5d. The Expanded CG After All DBs Fire

When all four DBs fire, the CG grows from 4 to 8 states. The expanded topology:

```
                        ┌────────────────────────────┐
                        │    INITIAL_EXPLORATION      │ [action]
                        └─────────────┬──────────────┘
                                      │
                                      ▼
┌──────────────────┐    ┌────────────────────────────┐    ┌──────────────────┐
│ DIMENSION_       │◄───│    COMBINATION_TESTING      │───►│ ORDER_TESTING    │
│ DISCOVERY        │    │                             │    │ (DB2)            │
│ (DB1)            │    └─────────────┬──────────────┘    └───────┬──────────┘
│ [reflection]     │                  │                           │
└───────┬──────────┘                  ▼                           │
        │               ┌────────────────────────────┐            │
        │               │  HYPOTHESIS_EVALUATION      │◄───────────┘
        │               │  [reflection]               │
        │               └─────────────┬──────────────┘
        │                             │
        │               ┌─────────────▼──────────────┐
        │               │     VERIFICATION            │ [decision]
        │               └────────────────────────────┘
        │
┌───────▼──────────┐    ┌────────────────────────────┐
│ RELIABILITY_     │    │   MODERATOR_DETECTION       │
│ TESTING          │    │   (DB4)                     │
│ (DB3)            │    │   [reflection]              │
│ [action]         │    └────────────────────────────┘
└──────────────────┘
```

Connection details:
- **DIMENSION_DISCOVERY** ← connected from HYPOTHESIS_EVALUATION, COMBINATION_TESTING
- **ORDER_TESTING** ← connected from DIMENSION_DISCOVERY, COMBINATION_TESTING
- **RELIABILITY_TESTING** ← connected from DIMENSION_DISCOVERY, COMBINATION_TESTING
- **MODERATOR_DETECTION** ← connected from DIMENSION_DISCOVERY, HYPOTHESIS_EVALUATION

### 5e. How DBs Transform the Problem Space

The graph growth is a progression of expanding causal vocabulary:

| States | What the Agent Can Reason About |
|--------|---------------------------------|
| 4 (base CG) | Object identity, conjunctive/disjunctive rules |
| + DIMENSION_DISCOVERY | "Something beyond identity matters" — meta-awareness |
| + ORDER_TESTING | Placement sequence as a causal variable |
| + RELIABILITY_TESTING | Probabilistic vs. deterministic activation |
| + MODERATOR_DETECTION | Rule changes over time, hidden moderators |

Not all behaviors fire in every episode. DB1 (stagnation) fires only when the agent has never activated the detector. DB4 (rule change) fires only after 10+ steps with evidence of a working-then-broken rule. The typical pattern:

- **Standard conditions** (conjunctive/disjunctive): 0-1 DBs fire (usually none)
- **Order-sensitive**: DB1 + DB2 fire (stagnation → order testing)
- **Stochastic**: DB1 + DB3 fire (stagnation → reliability testing)
- **Hidden moderator**: DB4 fires (rule change detection), sometimes DB1 first

### 5f. Score Logging and Observability

Every trigger evaluation is logged in the episode trace. The `db_activation_log` in each trace records:
- Which step the behavior fired at
- The notification message that was injected
- The cumulative set of fired behaviors at that point

The `db_trigger_tokens` field tracks total tokens spent on trigger evaluations (Haiku calls), enabling cost analysis of the DB apparatus.

---

## 6. The Agent-Environment Loop

**File:** `experiments/episode_runner.py` — function `run_episode()`

One turn of the full_cg_db agent follows this sequence:

```
episode_runner                    FullAgent                       Environment
     │                               │                                │
     │                               │                                │
     │── get_action(env) ───────────►│                                │
     │                               │                                │
     │                               │── db_engine.evaluate_and_apply()
     │                               │   ├── pre_screen (each DB)     │
     │                               │   ├── LLM evaluate (Haiku)     │
     │                               │   └── apply modifications      │
     │                               │                                │
     │                               │── inject notifications ────────│
     │                               │   (into conversation history)  │
     │                               │                                │
     │                               │── build_system_prompt()        │
     │                               │   (task desc + CG context)     │
     │                               │                                │
     │                               │── call LLM (Sonnet) ──────────│
     │                               │── parse CG transition          │
     │                               │── parse action                 │
     │                               │                                │
     │◄── action ───────────────────│                                │
     │                               │                                │
     │── env.step(action) ───────────────────────────────────────────►│
     │◄── observation ───────────────────────────────────────────────│
     │                               │                                │
     │── observe_activation(active) ►│                                │
     │   (records bool in history)   │                                │
     │                               │                                │
     │── [if detector changed]       │                                │
     │   mark_unexpected_result() ──►│                                │
     │   (sets flag for next DB eval)│                                │
     │                               │                                │
     │── add_observation(text) ─────►│                                │
     │   (appends to conversation)   │                                │
     │                               │                                │
```

Key data flows:

**`had_unexpected_result`**: The episode runner sets this flag whenever `obs.detector_changed` is True. It flows into the DB pre-screen as a gate for `requires_hypothesis_failure` behaviors (not used in current Run 2 behaviors, but available). The flag is reset to `False` after each DB evaluation cycle.

**`activation_history`**: A list of booleans, one per step, recording whether the detector was active after each action. This feeds the stagnation pre-screen: if `any(activation_history)` is True, DB1 (exploration_stagnation) is ineligible.

**Episode termination**: The loop runs until `env.done` (step count reaches `max_steps` or agent selects `CHECK`). If the agent chose `CHECK`, the runner calls `agent.get_answer()` to extract the final rule type and blicket set, then evaluates correctness against the environment's ground truth.

---

## 7. Production Comparison — Pilot vs. Production Architecture

### 7a. Shared Properties

The pilot faithfully reproduces several core properties of the production architecture:

| Property | Production | Pilot | Match |
|----------|-----------|-------|-------|
| Typed states with objectives + guidelines | Yes (6 types) | Yes (3 types) | Partial |
| Directed transitions with conditions | Yes | Yes | Yes |
| CG injected into system prompt each turn | Yes | Yes | Yes |
| Runtime CG expansion by Dynamic Behaviors | Yes | Yes | Yes |
| DB trigger → CG modification pipeline | Yes | Yes | Yes |
| Modification logging | Yes | Yes | Yes |
| Multiple DB types with distinct triggers | Yes | Yes | Yes |

### 7b. Key Differences

**1. State types**

Production has 6 state types: `action`, `decision`, `reflection`, `recall`, `annotation`, `side-effect` (docs-reference.txt:1101-1125). The pilot uses 3: `action`, `decision`, `reflection`. Missing types:

- **Recall** — pulls memory into active context (e.g., retrieving patient history). The pilot has no memory system, so recall states are unnecessary.
- **Annotation** — tags context for downstream processing (e.g., marking compliance events). The pilot has no downstream consumers.
- **Side-effect** — triggers external systems without user-facing output (e.g., scheduling follow-ups, calling APIs). The pilot has no external integrations.

**2. Trigger mechanism**

Production uses **multi-dimensional embedding vectors** (docs-reference.txt:1714-1734) — 7+ dimensions including agent thinking patterns, user input, tool calls, side-effects, and external signals. These vectors combine into a unified pool where behaviors are matched against multiple dimensions simultaneously.

The pilot uses **single-dimension LLM-as-judge scoring** with explicit evaluation prompts. Claude Haiku reads a condensed context and outputs a 0-10 score. This is a deliberate simplification that preserves the core property (evidence → trigger → expansion) while being tractable for a controlled experiment.

**3. Arc contracts**

Production has formal **entry predicates and exit guarantees** with variance bounds and measurement-backed validation (docs-reference.txt:6283-6284). Orchestration only runs arcs whose entry predicates are satisfied, and exit guarantees remain within measured bounds.

The pilot uses **human-readable transition conditions** that the LLM self-evaluates (e.g., "After testing all 3 individual objects"). The pilot preserves the graph topology (states + transitions) but not the formal verification.

**4. Memory integration**

Production CGs interact with a **hierarchical memory system** — working memory, conversation history, and long-term memory orchestrated through the Memory-Knowledge-Reasoning (M-K-R) cycle (docs-reference.txt:1066-1082). Recall states directly tap into stored memory frames.

Pilot agents have only **conversation history** within a single episode. No persistent memory exists across episodes, and there is no recall mechanism for prior experience.

**5. One-shot vs. repeatable firing**

Production behaviors have a **stickiness factor that gradually decreases over time** (docs-reference.txt:1730-1732), allowing behaviors to persist across turns and re-fire with decaying recency weight.

Pilot behaviors fire **at most once per episode**. This simplification provides cleaner experimental signal: we can unambiguously attribute CG expansion to specific evidence points.

**6. Entropy stratification**

Production varies **context density** across states — high-density (low entropy, few degrees of freedom) for safety-critical interactions, low-density (high entropy, many degrees of freedom) for open-ended exploration (docs-reference.txt:922-932).

The pilot uses **uniform medium-density guidance** across all states. Every state has roughly the same number and specificity of guidelines.

**7. Side-effect framework**

Production DBs can trigger a rich set of side-effects (docs-reference.txt:1834-1983): tool integration, reflective processing with specialized domain LLMs, enterprise integration (webhooks, handoffs), and cross-graph navigation between distinct CGs.

Pilot DBs can only **add states and modify guidelines**. No tool exposure, no external calls, no cross-graph navigation.

### 7c. Why the Simplifications Are Justified

The pilot tests one specific claim: **whether CG+DB enables coordinate system revision better than raw LLM capability or fixed CG alone.**

- **Memory, side-effects, and recall states** are irrelevant to this claim — the blicket task has no memory across episodes, no external systems, and no annotation needs.
- **Simplified triggers** preserve the key property: evidence-driven problem space expansion. Whether the evidence signal comes from embedding vectors or LLM-as-judge scoring, the functional pipeline is the same: observe → evaluate → expand.
- **One-shot firing** provides cleaner experimental signal than repeatable firing with decay. We can unambiguously track which evidence caused which expansion.
- **Uniform entropy** eliminates a confound. If the pilot varied context density, we couldn't distinguish CG+DB effects from entropy effects.

The simplifications remove confounds, not relevant capabilities.

### 7d. What This Means for Production Claims

**Results that should transfer:**
- If CG+DB helps with coordinate revision in the pilot, it should transfer to production — the relevant mechanisms (typed states, runtime expansion, evidence-triggered modification) are all present in production.
- If fixed CG outperforms bare LLM on structured exploration, this directly validates the production CG architecture's value.

**Results that are pilot-specific:**
- Specific trigger thresholds (e.g., fire_threshold=6.0) are calibrated for the pilot's LLM-as-judge mechanism. Production's embedding-based triggers have different scaling.
- Timing of DB fires (step counts) is specific to the blicket task's action space. Production triggers operate on turn-level signals in multi-turn conversations.

---

## 8. Modifiable Features for Future Experiments

A catalog of parameters and design choices that could be varied in future runs.

### CG Modifications

- **Number of base states**: Currently 4. Could add or remove states, or change the initial topology (e.g., skip INITIAL_EXPLORATION, start at COMBINATION_TESTING).
- **State types**: Only 3 of 6 production types are used. Adding RECALL states (with episodic memory across episodes) would test memory-augmented exploration. Adding SIDE_EFFECT states (with tool use) could enable calculator or statistics tools.
- **Transition conditions**: Currently human-readable strings the LLM interprets. Could add deterministic exit conditions (e.g., "after 5 steps in this state, auto-transition") to test structured vs. flexible pacing.
- **Guidelines density**: Could vary guideline specificity per state to test entropy stratification effects — e.g., dense prescriptive guidelines in VERIFICATION, sparse open-ended guidelines in COMBINATION_TESTING.
- **Initial state**: Could start agents in different states to test whether exploration ordering matters.

### DB Modifications

- **Fire thresholds**: Currently 6.0 for all behaviors. Tunable per behavior — Run 2 analysis suggests higher thresholds (7.0+) could reduce false-positive fires on order-sensitive conditions.
- **One-shot vs. repeatable**: Could allow behaviors to re-fire (with cooldown) to test cumulative intervention effects.
- **Trigger model**: Currently Haiku 4.5. Could use Sonnet for higher-fidelity trigger evaluation (at higher cost) or a future smaller model for cost reduction.
- **Pre-screen conditions**: `min_steps` and `stagnation_threshold` are tunable. Could add new pre-screen dimensions (e.g., minimum activation count, minimum state transitions).
- **New behaviors**: Could add behaviors for other phenomena (e.g., spatial arrangement, multi-step causality, quantity thresholds).
- **Behavior removal**: Ablation studies removing individual DBs to isolate their contribution.
- **Competitive inhibition**: Currently all eligible DBs are evaluated independently. A future run could implement winner-take-most competition: diagnostic DBs (order, stochastic, rule_change) compete, and only the top scorer fires, reducing cross-talk between behaviors.

### Agent-Level Modifications

- **Base model**: Currently Sonnet 4.5. Could test with Haiku (cheaper, weaker) or Opus (stronger, more expensive) to test composition vs. capability tradeoffs — does compositional structure help more when the base model is weaker?
- **Conversation window**: Currently full history. Could test sliding-window or summarized history to simulate bounded memory.
- **System prompt structure**: Could vary how CG context is presented (e.g., showing only current state vs. full graph, or omitting the graph overview).
- **Cross-episode memory**: Currently agents reset completely between episodes. Could add episode-to-episode learning to test whether prior experience improves coordinate system revision.

### Environment Modifications

- **Object count**: The experiment has already scaled from 3 to 5 objects (standard vs. extended conditions). Could go to 7+ to test scaling limits.
- **Step budget**: Currently 50 (standard) or 75 (extended). Could vary to test time pressure effects.
- **New condition types**: Could add conditions beyond the current 5 (e.g., spatial arrangement, multi-step causality, cyclic rules).

---

## 9. Traces — What They Are and How We Analyze Them

### 9a. What a Trace Is

Every episode produces a JSON trace file named `{condition}_{agent_type}_ep{NNN}.json`. A trace is a complete record of one episode: every action, observation, CG state, DB activation, and the final result. Traces are the raw data from which all experiment results are derived.

### 9b. Trace Structure

Each trace file has four top-level sections:

```json
{
  "result": {
    "condition": "stochastic",
    "agent_type": "full_cg_db",
    "episode_number": 22,
    "success": true,
    "answer_submitted": true,
    "steps_taken": 28,
    "rule_type_correct": true,
    "objects_correct": true,
    "true_rule_type": "conjunctive",
    "true_objects": ["A", "B"],
    "guessed_rule_type": "conjunctive",
    "guessed_objects": ["A", "B"],
    "total_input_tokens": 128539,
    "total_output_tokens": 4333,
    "trigger_tokens": 13919,
    "parse_failures": 0,
    "wall_time_seconds": 142.06,
    "error": null
  },
  "steps": [
    {
      "step": 0,
      "action": "place_a",
      "action_valid": true,
      "objects_on_detector": ["A"],
      "detector_active": false,
      "detector_changed": false,
      "observation_text": "You placed object A on the detector...",
      "cg_state": "INITIAL_EXPLORATION",
      "db_notifications": []
    }
  ],
  "agent_trace": {
    "agent_type": "full_cg_db",
    "model": "claude-sonnet-4-5-20250929",
    "conversation": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ],
    "raw_responses": ["..."],
    "token_usage": {
      "input_tokens": 128539,
      "output_tokens": 4333,
      "total_tokens": 132872
    },
    "context_graph_final": {
      "states": {"INITIAL_EXPLORATION": {...}, ...},
      "current_state": "VERIFICATION",
      "modification_log": [...]
    },
    "transition_log": [
      {"step": 3, "from": "INITIAL_EXPLORATION",
       "to": "COMBINATION_TESTING", "success": true}
    ],
    "db_activation_log": [
      {
        "step": 6,
        "notification": "[SYSTEM] Your observations don't fit...",
        "fired_behaviors": ["exploration_stagnation"]
      }
    ],
    "db_fired_behaviors": ["exploration_stagnation", "order_hypothesis"],
    "db_trigger_tokens": 13919
  },
  "env_info": {
    "condition": "stochastic",
    "rule_type": "conjunctive",
    "required_objects": ["A", "B"],
    "max_steps": 50,
    "step_count": 28,
    "done": true,
    "objects_on_detector": ["A", "B", "C"],
    "detector_active": true,
    "placement_order": ["C", "A", "B"]
  }
}
```

### 9c. Key Analysis Dimensions

**1. Accuracy analysis** — `result.success`, `rule_type_correct`, `objects_correct`

Did the agent get the right answer? When wrong, was it the rule type (conjunctive vs. disjunctive) or the object set? Object errors can be further classified as over-inclusion (guessed extra objects) or under-inclusion (missed a blicket).

**2. Step efficiency** — `result.steps_taken`

How many steps did the agent need? Fewer steps generally means more efficient exploration, but some conditions genuinely require more steps (e.g., stochastic requires repetition to estimate probabilities).

**3. CG state progression** — `steps[].cg_state`

Which states did the agent visit and in what order? This reveals exploration strategy. A typical successful conjunctive episode might show: INITIAL_EXPLORATION (steps 0-5) → COMBINATION_TESTING (6-10) → HYPOTHESIS_EVALUATION (11) → VERIFICATION (12-14). An extended condition might show DIMENSION_DISCOVERY and ORDER_TESTING insertions.

**4. DB activation patterns** — `agent_trace.db_activation_log`

Which behaviors fired, at what step, and with what notification? This reveals what evidence the system detected and when. The `fired_behaviors` list in each entry shows the cumulative set of fired behaviors up to that point.

**5. Transition log** — `agent_trace.transition_log`

Every CG transition the agent attempted, whether it succeeded or failed. Failed transitions reveal when the agent tried to move to a state that wasn't connected — this can indicate confusion or desire for states that don't yet exist.

**6. Post-switch analysis** (hidden moderator)

For hidden_moderator episodes, comparing agent behavior before vs. after the rule switch reveals whether the agent detected the change. Key signals: did DB4 (rule_change_hypothesis) fire? Did the agent transition to MODERATOR_DETECTION? Did it re-test objects under the new regime?

**7. Failure mode classification**

When `success=false`, the trace reveals the specific failure:
- Wrong rule type (conjunctive vs. disjunctive confusion)
- Wrong objects — over-inclusion (guessed C as blicket when it isn't)
- Wrong objects — under-inclusion (missed A or B)
- Premature termination (answer_submitted=true but insufficient exploration)
- No answer submitted (ran out of steps without calling CHECK)

**8. Conversation analysis**

Full LLM reasoning is visible in `agent_trace.conversation`. This allows inspection of exactly how the agent interpreted CG guidelines, how it responded to DB notifications, and where reasoning went wrong in failure cases.

### 9d. How Traces Are Aggregated

- **Individual traces** → per-condition accuracy rates, step distributions, DB fire rates
- **Cross-condition** (same agent, different conditions) → reveals which conditions are hardest for each architecture
- **Cross-agent** (same condition, different agents) → reveals architectural effects on specific causal phenomena
- **Cross-run** (Run 1 → Run 2 → ...) → shows how design changes (trigger tuning, CG redesign) affect behavior

Aggregation produces `extended_results.json` with per-condition, per-agent summary statistics used for the experiment reports.

---

## 10. Key Takeaways

The three agents form a clean ablation: **raw LLM → +structure → +adaptation**.

- The **base LLM** establishes what a capable model can do with only task understanding. It performs well on standard conditions but has no vocabulary for extended phenomena.
- The **CG** provides a structured exploration framework — phases, objectives, transition conditions. It improves systematic exploration but cannot adapt when the 4-state vocabulary is insufficient.
- **Dynamic Behaviors** provide the mechanism for expanding that vocabulary when evidence demands it. When a DB fires, the agent's problem space literally grows to accommodate the new phenomenon.

Together, CG+DB implements a form of **coordinate system revision**: the agent doesn't just patch its existing hypothesis — it gains new states with new objectives and guidelines for investigating dimensions it couldn't previously reason about. This maps to the composition thesis: domain sufficiency through structured interaction, not through scaling raw model capability.
