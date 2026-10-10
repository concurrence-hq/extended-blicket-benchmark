# Data Directory

This directory contains all trace data and result summaries from the Extended Blicket Benchmark experimental programme (23 runs, 2,541 episodes).

## Directory Layout

There are two types of directories:

- **`traces/`** and **`results/`** (unnumbered) — Run 01, the initial 3-object pilot. These predate the run-numbering convention. The 486 trace files break down as: 180 standard conditions, 270 extended conditions, 30 pilot validation, and 6 smoke tests.
- **`traces_runNN/`** and **`results_runNN/`** — All subsequent runs (Run 02 onward), each containing the raw traces and aggregated results for that run.

Every `traces_runNN/` directory has a corresponding `results_runNN/` directory.

## Trace Format

All JSON trace files across every run use the same schema:

```json
{
  "result": {
    "condition": "hidden_moderator",
    "agent_type": "full_cg_db",
    "episode_number": 0,
    "success": true,
    "answer_submitted": true,
    "steps_taken": 42,
    "rule_type_correct": true,
    "objects_correct": true,
    "true_rule_type": "conjunctive",
    "true_objects": ["A", "B"],
    "guessed_rule_type": "conjunctive",
    "guessed_objects": ["A", "B"],
    "total_input_tokens": 52341,
    "total_output_tokens": 8712,
    "trigger_tokens": 1204,
    "parse_failures": 0,
    "wall_time_seconds": 38.2
  },
  "steps": [
    {
      "step": 0,
      "action": "place_a",
      "action_valid": true,
      "objects_on_detector": ["A"],
      "detector_active": false,
      "detector_changed": false,
      "observation_text": "...",
      "cg_state": "INITIAL_EXPLORATION",
      "db_notifications": []
    }
  ],
  "agent_trace": {
    "agent_type": "full_cg_db",
    "model": "claude-sonnet-4-5-20250929",
    "transition_log": [],
    "db_activation_log": [],
    "db_fired_behaviors": []
  },
  "env_info": {
    "rule_type": "conjunctive",
    "required_objects": ["A", "B"]
  }
}
```

File naming convention: `{condition}_{agent_type}_ep{NNN}.json`

## Run Index

Runs reported in the paper (Table 4):

| Run | Directory | Episodes | Condition | Description |
|-----|-----------|----------|-----------|-------------|
| 03 | `traces_run3/` | 300 | All 4 (5-obj) | Initial 5-object pilot across all conditions |
| 04 | `traces_run4_pilot/` | 90 | Hidden moderator | Pre-registered replication (n=30/cell) |
| 05 | — | 75 | Hidden moderator | Switch-point calibration (sw=5,7,9) |
| 06 | `traces_run6/` | 300 | Hidden moderator | Exploration depth x architecture (sw=3,5) |
| 07 | `traces_run7/` | 120 | Hidden moderator | Conjunctive post-switch rules (CD, AC, CDE) |
| 08 | `traces_run8/` | 200 | Hidden moderator | Powered 3-way comparison (primary result) |
| 10g | `traces_run10g/` | 51 | Order-sensitive | Powered 2-agent comparison (100 steps) |

Full experimental series (development + boundary exploration):

| Run | Directory | Episodes | Notes |
|-----|-----------|----------|-------|
| 01 | `traces/` | 450 | 3-object pilot (pre-numbering); excluded from paper (ceiling effects) |
| 02 | `traces_run2/` | 300 | Bug fixes and clean 3-object retest; excluded (ceiling effects) |
| 03 | `traces_run3/` | 300 | 5-object scaling with competitive inhibition |
| 03-phaseA | `traces_run3_phaseA/` | 30 | Phase A subset of Run 03 |
| 03b | `traces_run3b/` | 90 | Hidden moderator variant of Run 03 |
| 04 | `traces_run4_pilot/` | 75 | Hidden moderator replication + pilot validation |
| 05 | — | 75 | Switch-point calibration (traces not retained) |
| 06 | `traces_run6/` | 300 | Exploration depth study |
| 07 | `traces_run7/` | 120 | Post-switch difficulty pilot |
| 08 | `traces_run8/` | 200 | Powered separable-pathways comparison |
| 09 | `traces_run9/` | 60 | Order-sensitive complexity pilot |
| 10 | `traces_run10/` | 60 | Partial feedback order-sensitive pilot |
| 10b | `traces_run10b/` | 60 | Gradual periodic feedback pilot |
| 10c | `traces_run10c/` | 60 | Revised ORDER_TESTING + randomized labels |
| 10d | `traces_run10d/` | 160 | Powered 3-object replication + 4-object pilot |
| 10d-diag | `traces_run10d_diagnostic/` | 30 | Diagnostic replication of Run 10c seeds |
| 10e | `traces_run10e/` | 60 | Stability calibration + 4-object signal check |
| 10f | `traces_run10f/` | 50 | 4-object step budget pilot |
| 10g | `traces_run10g/` | 51 | Powered replication of 10f (100-step finding) |
| 11a | `traces_run11a/` | 30 | ORDER_STOCHASTIC combined condition pilot |
| 11b | `traces_run11b/` | 45 | 3-object ORDER_STOCHASTIC with v11b DB |
| 11c | `traces_run11c/` | 45 | v2 CG (OBJECT_RELEVANCE_SCREENING) |
| 11d-pre | `traces_run11d_prepilot/` | 10 | Static 7-state CG smoke test |
| 11e | `traces_run11e/` | 45 | v4 CG (subset-testing ORS) + lean DB |

See `runs/running-log.csv` for detailed notes on each run including seeds, cell structure, costs, and key findings.

## Result Summaries

Each `results_runNN/` directory contains one or more JSON files with aggregated episode-level results. File naming varies by run vintage:
- Early runs (01-06): `{testset}_results.json` (e.g., `extended_results.json`, `standard_results.json`)
- Run 07-08: `cell_{letter}.json` per experimental cell + analysis files
- Run 09+: Descriptive names (e.g., `powered_replication_results.json`, `order_complexity_results.json`)

## Reproducibility

All statistical analyses reported in the paper are computed from the raw trace files in this directory via `scripts/paper1_statistical_tests.py`. That script includes automated regression checks against pre-computed reference values for Run 08.
