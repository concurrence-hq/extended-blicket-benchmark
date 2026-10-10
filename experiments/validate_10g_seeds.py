"""Seed validation for Run 10g.

Validates seeds 1421-1440 (4-object, 6 labels) for the powered replication.

Checks:
  Batch 1 (seeds 1421-1430): Spot-check determinism (already validated at 125 steps in 10f)
  Batch 2 (seeds 1431-1440): Full validation:
    1. Determinism — identical output on 3 consecutive calls
    2. Print all 10 rules
    3. Variety — ≥5 different correct-object sets across 10 seeds
    4. Alphabetical rate — ≤1 of 10
    5. Environment activation — correct order activates, ≥3 wrong don't

  Cross-batch table: Print all 30 rules (1411-1440) to check for systematic patterns
"""

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.blicket_env import CausalRule, CausalRuleType
from environment.blicket_env_6obj import Action6
from environment.extended_env import ExtendedCondition, OrderConfig
from environment.extended_env_6obj import ExtendedBlicketEnv6

ALL_LABELS = ["A", "B", "C", "D", "E", "F"]
N_CORRECT = 4
PLACE_6 = {label: Action6[f"PLACE_{label}"] for label in "ABCDEF"}


def generate_randomized_order_rule(
    seed: int,
    all_labels: list[str] | None = None,
    n_correct: int = 3,
) -> dict:
    if all_labels is None:
        all_labels = ["A", "B", "C", "D", "E"]
    rng = random.Random(seed)
    correct = rng.sample(all_labels, n_correct)
    rng.shuffle(correct)
    distractors = set(all_labels) - set(correct)
    return {
        "correct_objects": set(correct),
        "required_order": list(correct),
        "distractors": distractors,
    }


def is_alphabetical(order: list[str]) -> bool:
    return order == sorted(order)


def validate_batch1_spot_check() -> bool:
    """Spot-check seeds 1421-1430 determinism (already validated in 10f at 125 steps)."""
    print(f"\n{'='*60}")
    print(f"  Batch 1 spot-check: seeds 1421-1430 (validated in 10f)")
    print(f"{'='*60}")

    all_ok = True
    seed = 1421
    r1 = generate_randomized_order_rule(seed, ALL_LABELS, N_CORRECT)
    r2 = generate_randomized_order_rule(seed, ALL_LABELS, N_CORRECT)
    r3 = generate_randomized_order_rule(seed, ALL_LABELS, N_CORRECT)
    match = (r1["required_order"] == r2["required_order"] == r3["required_order"])
    print(f"  Seed 1421: order={r1['required_order']}, deterministic={'OK' if match else 'FAIL'}")
    if not match:
        all_ok = False

    # Print all 10 rules for reference
    print(f"\n  {'Seed':<6} {'Correct Objects':<20} {'Required Order':<20} {'Distractors':<15}")
    print(f"  {'-'*6} {'-'*20} {'-'*20} {'-'*15}")
    for seed in range(1421, 1431):
        r = generate_randomized_order_rule(seed, ALL_LABELS, N_CORRECT)
        print(
            f"  {seed:<6} {str(sorted(r['correct_objects'])):<20} "
            f"{str(r['required_order']):<20} {str(sorted(r['distractors'])):<15}"
        )

    if all_ok:
        print(f"\n  Batch 1 spot-check: PASSED")
    return all_ok


def validate_batch2_full() -> bool:
    """Full validation of seeds 1431-1440 (new seeds)."""
    seeds = range(1431, 1441)

    print(f"\n{'='*60}")
    print(f"  Batch 2 full validation: seeds 1431-1440 (new)")
    print(f"{'='*60}")

    rules = []
    all_ok = True

    # 1. Determinism check
    print(f"\n--- Check 1: Determinism ---")
    for seed in seeds:
        r1 = generate_randomized_order_rule(seed, ALL_LABELS, N_CORRECT)
        r2 = generate_randomized_order_rule(seed, ALL_LABELS, N_CORRECT)
        r3 = generate_randomized_order_rule(seed, ALL_LABELS, N_CORRECT)
        match = (r1["required_order"] == r2["required_order"] == r3["required_order"])
        if not match:
            all_ok = False
            print(f"  Seed {seed}: FAIL -- {r1['required_order']} vs {r2['required_order']} vs {r3['required_order']}")
        rules.append(r1)
    if all_ok:
        print(f"  All 10 seeds deterministic: OK")

    # 2. Print all rules
    print(f"\n--- Check 2: All Rules ---")
    print(f"  {'Seed':<6} {'Correct Objects':<20} {'Required Order':<20} {'Distractors':<15}")
    print(f"  {'-'*6} {'-'*20} {'-'*20} {'-'*15}")
    for i, seed in enumerate(seeds):
        r = rules[i]
        print(
            f"  {seed:<6} {str(sorted(r['correct_objects'])):<20} "
            f"{str(r['required_order']):<20} {str(sorted(r['distractors'])):<15}"
        )

    # 3. Variety check — ≥5 different sets across 10 seeds
    unique_sets = set(frozenset(r["correct_objects"]) for r in rules)
    variety_ok = len(unique_sets) >= 5
    print(f"\n--- Check 3: Variety ---")
    print(f"  Unique correct-object sets: {len(unique_sets)} of 15 possible (need ≥5): {'OK' if variety_ok else 'FAIL'}")
    if not variety_ok:
        all_ok = False

    # 4. Alphabetical rate — ≤1/10
    alpha_count = sum(1 for r in rules if is_alphabetical(r["required_order"]))
    alpha_ok = alpha_count <= 1
    print(f"\n--- Check 4: Alphabetical Rate ---")
    print(f"  Alphabetical orderings: {alpha_count}/10 (need ≤1): {'OK' if alpha_ok else 'FAIL'}")
    if alpha_count > 0:
        for i, seed in enumerate(seeds):
            if is_alphabetical(rules[i]["required_order"]):
                print(f"    Seed {seed}: {rules[i]['required_order']} (alphabetical)")
    if not alpha_ok:
        all_ok = False

    # 5. Environment activation
    print(f"\n--- Check 5: Environment Activation ---")
    env_ok = True
    for i, seed in enumerate(seeds):
        r = rules[i]
        req_objects = r["correct_objects"]
        req_order = r["required_order"]

        rule = CausalRule(
            rule_type=CausalRuleType.ORDER_SENSITIVE,
            required_objects=req_objects,
        )
        env = ExtendedBlicketEnv6(
            causal_rule=rule,
            condition=ExtendedCondition.ORDER_SENSITIVE,
            max_steps=100,
            order_config=OrderConfig(required_order=req_order),
            rng_seed=seed,
        )

        # Test correct order activates
        env.reset()
        for obj in req_order:
            env.step(PLACE_6[obj])
        correct_activates = env.state.detector_active

        # Test ≥3 incorrect orders do NOT activate
        all_perms = list(itertools.permutations(req_order))
        wrong_perms = [p for p in all_perms if list(p) != req_order]
        wrong_count = 0
        wrong_tested = 0
        for wrong_order in wrong_perms[:5]:
            env.reset()
            for obj in wrong_order:
                env.step(PLACE_6[obj])
            if not env.state.detector_active:
                wrong_count += 1
            wrong_tested += 1

        seed_ok = correct_activates and wrong_count >= 3
        if not seed_ok:
            env_ok = False
            print(f"  Seed {seed}: FAIL (correct_activates={correct_activates}, wrong_inactive={wrong_count}/{wrong_tested})")

    if env_ok:
        print(f"  All 10 seeds: correct order activates, ≥3 wrong orders do not: OK")

    overall = all_ok and env_ok
    print(f"\n  Batch 2 overall: {'ALL CHECKS PASSED' if overall else 'SOME CHECKS FAILED'}")
    return overall


def cross_batch_table() -> None:
    """Print all 30 rules (seeds 1411-1440) for pattern inspection."""
    print(f"\n{'='*60}")
    print(f"  Cross-batch table: all 30 rules (seeds 1411-1440)")
    print(f"{'='*60}")

    print(f"\n  {'Seed':<6} {'Batch':<10} {'Correct Objects':<20} {'Required Order':<20} {'Distractors':<15} {'Alpha?':<6}")
    print(f"  {'-'*6} {'-'*10} {'-'*20} {'-'*20} {'-'*15} {'-'*6}")

    for seed in range(1411, 1441):
        r = generate_randomized_order_rule(seed, ALL_LABELS, N_CORRECT)
        if seed < 1421:
            batch = "pilot"
        elif seed < 1431:
            batch = "batch1"
        else:
            batch = "batch2"
        alpha = "YES" if is_alphabetical(r["required_order"]) else ""
        print(
            f"  {seed:<6} {batch:<10} {str(sorted(r['correct_objects'])):<20} "
            f"{str(r['required_order']):<20} {str(sorted(r['distractors'])):<15} {alpha:<6}"
        )

    # Summary stats across all 30
    all_rules = [generate_randomized_order_rule(s, ALL_LABELS, N_CORRECT) for s in range(1411, 1441)]
    unique_sets = set(frozenset(r["correct_objects"]) for r in all_rules)
    alpha_count = sum(1 for r in all_rules if is_alphabetical(r["required_order"]))
    print(f"\n  Summary: {len(unique_sets)} unique object sets across 30 seeds")
    print(f"  Alphabetical orderings: {alpha_count}/30")

    # Object frequency
    from collections import Counter
    obj_freq = Counter()
    for r in all_rules:
        for obj in r["correct_objects"]:
            obj_freq[obj] += 1
    print(f"  Object frequencies: {dict(sorted(obj_freq.items()))}")


if __name__ == "__main__":
    ok1 = validate_batch1_spot_check()
    ok2 = validate_batch2_full()
    cross_batch_table()

    print(f"\n{'='*60}")
    if ok1 and ok2:
        print("  ALL SEED VALIDATIONS PASSED")
    else:
        print("  SEED VALIDATION FAILURES DETECTED")
        sys.exit(1)
    print(f"{'='*60}")
