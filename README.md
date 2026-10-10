# Extended Blicket Benchmark

Codebase for "Separable Pathways for Causal Reasoning: How Architectural Scaffolding Enables Hypothesis-Space Restructuring in LLM Agents" (*Transactions on Machine Learning Research*, 2026; [arXiv:2604.20039](https://arxiv.org/abs/2604.20039)).

## Overview

We extend the blicket detector paradigm from developmental science into a parameterized evaluation suite for testing hypothesis-space restructuring in LLM agents. A blicket detector is a device governed by a hidden causal rule: the agent places and removes objects, observes activation patterns, and must infer both which objects are causally active and what type of rule governs the detector.

Three agent architectures are compared in an additive hierarchy:

| Agent | Architecture | Key Property |
|-------|-------------|--------------|
| `base_llm` | Claude Sonnet + task prompt | No structural scaffolding |
| `cg_only` | + Context Graph (4-state typed state machine) | Structures exploration as phased problem-solving |
| `full_cg_db` | + Dynamic Behaviors (4 runtime monitors) | Detects regime changes and expands hypothesis space |

### Experimental Conditions

- **Conjunctive / Disjunctive** (standard): Ceiling baselines; all agents achieve 100% accuracy
- **Hidden moderator** (core): Rule switches from conjunctive to disjunctive after N activations, requiring hypothesis-space restructuring
- **Order-sensitive** (boundary): Placement sequence matters; probes combinatorial search limits
- **Stochastic** (boundary): Probabilistic activation (p=0.70); probes statistical inference limits

### Key Findings (1,085 trials across Runs 03-08, 10g)

1. **Separable pathways**: Context graphs drive reasoning quality (94% of RE accuracy gain); dynamic behaviors drive reasoning eligibility (exactly-N avoidance). Neither substitutes for the other.
2. **Reasoning-eligible (RE) accuracy**: base_llm 73.3%, cg_only 93.9% (+20.6pp, p=0.038), full_cg_db 95.3% (Run 08, hard post-switch rule)
3. **Exactly-N trap avoidance**: cg_only 28.0% vs full_cg_db 6.0% (OR=6.09, p=0.003)
4. **Boundary conditions**: No reliable advantage on stochastic or order-sensitive conditions, establishing specificity of the architectural benefit to hypothesis-space restructuring

## Repository Structure

```
extended-blicket-benchmark/
├── agents/                  # Agent implementations
│   ├── base_llm_agent.py    # Base LLM agent (Sonnet + task prompt only)
│   ├── cg_agent.py          # Context Graph agent (+ state machine)
│   ├── full_agent.py        # Full CG+DB agent (+ dynamic behaviors)
│   ├── context_graph.py     # Context graph data model and rendering
│   └── dynamic_behaviors*.py # Dynamic behavior specifications (DB1-DB4)
├── environment/             # Blicket detector environment
│   ├── blicket_env.py       # 3-object environment
│   ├── blicket_env_5obj.py  # 5-object environment (primary)
│   ├── extended_env*.py     # Extended conditions (hidden moderator, order, stochastic)
│   └── text_interface.py    # State rendering and action parsing
├── experiments/             # Experiment runner scripts
│   ├── episode_runner.py    # Core episode execution loop
│   ├── config.py            # Shared configuration
│   ├── run_*.py             # Per-run experiment scripts
│   └── validate_*.py        # Seed validation scripts
├── analysis/                # Analysis and visualization
│   ├── metrics.py           # Episode and aggregate metric computation
│   ├── compare_to_children.py # Comparison with developmental data
│   └── visualize.py         # Plotting utilities
├── scripts/                 # Standalone analysis scripts
│   ├── paper1_statistical_tests.py  # All inferential statistics (Fisher, Bayes factors)
│   └── model_stability_audit.py     # Model string and parameter verification
├── tests/                   # Test suite (305 tests)
├── data/                    # Trace data and result summaries (~200MB)
│   ├── traces_run*/         # Raw JSON episode traces
│   └── results_run*/        # Aggregated result summaries
└── runs/                    # Run documentation
    ├── running-log.csv      # Master log of all experimental runs
    └── report_run*.md       # Post-run analysis reports
```

## Setup

```bash
pip install -e ".[dev]"
```

Requires `ANTHROPIC_API_KEY` in environment or `.env` file. All agents use Anthropic's Claude Sonnet 4.5 for reasoning; the full_cg_db agent additionally uses Haiku 4.5 for dynamic behavior trigger evaluations.

## Running Experiments

Each run has a dedicated script in `experiments/`:

```bash
# Example: Run 08 (powered separable-pathways comparison)
python -m experiments.run_8_powered_followup

# Smoke test (single episode, verifies setup)
python -m experiments.smoke_test
```

## Analysis

All statistical results reported in the paper are computed by a single reproducible script:

```bash
python scripts/paper1_statistical_tests.py
```

This script includes automated regression checks: pre-computed reference values for Run 08 are verified against recomputed values at runtime, and the script terminates if any discrepancy exceeds 0.2%.

## Tests

```bash
pytest tests/ -v
```

305 tests covering environment mechanics, agent behavior, context graph transitions, dynamic behavior triggers, competitive inhibition, metric computation, and episode runner integration.

## Trace Format

Each episode produces a structured JSON trace recording:
- Every action taken and environment response
- Context graph state at each step
- Dynamic behavior trigger evaluations and firings
- The agent's final answer and correctness

See `data/traces_run*/` for examples. The trace schema supports analysis at the turn level (not just episode-outcome level), enabling the reasoning-eligibility decomposition and mechanistic attribution reported in the paper.

## Cost

The primary experimental series (Runs 03-08; 1,085 episodes) cost approximately $807 in API fees. The full programme spanning 23 runs and 2,541 episodes cost approximately $1,950.

## Citation

```bibtex
@article{alderete2026separable,
  title   = {Separable Pathways for Causal Reasoning: How Architectural Scaffolding Enables Hypothesis-Space Restructuring in {LLM} Agents},
  author  = {Alderete, John and Benthall, Sebastian and Xu, Connie and Xing, John},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  eprint  = {2604.20039},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/2604.20039}
}
```

## License

Released under the [Apache License 2.0](LICENSE).
