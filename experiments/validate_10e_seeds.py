"""Seed validation for Run 10e.

Validates seeds 2001-2010 (calibration, 3-obj) and 1401-1410 (signal check, 4-obj)
per planning prompt Section 4.3.

Checks:
  1. Determinism — identical output on 3 consecutive calls
  2. Print all rules
  3. Variety — ≥5 different sets of correct objects
  4. Alphabetical rate — ≤2/10 (3-obj), ≤1/10 (4-obj)
  5. Environment activation — correct order activates, incorrect orders do not
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.blicket_env import CausalRule, CausalRuleType
from environment.blicket_env_5obj import Action5
from environment.blicket_env_6obj import Action6
from environment.extended_env import ExtendedCondition, OrderConfig
from environment.extended_env_5obj import ExtendedBlicketEnv5
from environment.extended_env_6obj import ExtendedBlicketEnv6

# Map label to place action
PLACE_5 = {label: Action5[f"PLACE_{label}"] for label in "ABCDE"}
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


def validate_seed_set(
    label: str,
    seeds: range,
    all_labels: list[str],
    n_correct: int,
    max_alpha: int,
):
    print(f"\n{'='*60}")
    print(f"  {label}: seeds {seeds.start}-{seeds.stop - 1} ({n_correct}-obj, {len(all_labels)} labels)")
    print(f"{'='*60}")

    rules = []
    all_ok = True

    # 1. Determinism check
    print(f"\n--- Determinism Check ---")
    for seed in seeds:
        r1 = generate_randomized_order_rule(seed, all_labels, n_correct)
        r2 = generate_randomized_order_rule(seed, all_labels, n_correct)
        r3 = generate_randomized_order_rule(seed, all_labels, n_correct)
        match = (r1["required_order"] == r2["required_order"] == r3["required_order"])
        status = "OK" if match else "FAIL"
        if not match:
            all_ok = False
            print(f"  Seed {seed}: {status} -- {r1['required_order']} vs {r2['required_order']} vs {r3['required_order']}")
        rules.append(r1)
    if all_ok:
        print(f"  All {len(seeds)} seeds deterministic: OK")

    # 2. Print all rules
    print(f"\n--- All Rules ---")
    print(f"  {'Seed':<6} {'Correct Objects':<20} {'Required Order':<20} {'Distractors':<15}")
    print(f"  {'-'*6} {'-'*20} {'-'*20} {'-'*15}")
    for i, seed in enumerate(seeds):
        r = rules[i]
        print(
            f"  {seed:<6} {str(sorted(r['correct_objects'])):<20} "
            f"{str(r['required_order']):<20} {str(sorted(r['distractors'])):<15}"
        )

    # 3. Variety check
    unique_sets = set(frozenset(r["correct_objects"]) for r in rules)
    variety_ok = len(unique_sets) >= 5
    print(f"\n--- Variety Check ---")
    print(f"  Unique correct-object sets: {len(unique_sets)} (need ≥5): {'OK' if variety_ok else 'FAIL'}")
    if not variety_ok:
        all_ok = False

    # 4. Alphabetical rate
    alpha_count = sum(1 for r in rules if is_alphabetical(r["required_order"]))
    alpha_ok = alpha_count <= max_alpha
    print(f"\n--- Alphabetical Rate ---")
    print(f"  Alphabetical orderings: {alpha_count}/{len(rules)} (need ≤{max_alpha}): {'OK' if alpha_ok else 'FAIL'}")
    if not alpha_ok:
        all_ok = False

    # 5. Environment activation test
    print(f"\n--- Environment Activation ---")
    env_ok = True
    for i, seed in enumerate(seeds):
        r = rules[i]
        req_objects = r["correct_objects"]
        req_order = r["required_order"]

        rule = CausalRule(
            rule_type=CausalRuleType.ORDER_SENSITIVE,
            required_objects=req_objects,
        )

        if n_correct == 3:
            env = ExtendedBlicketEnv5(
                causal_rule=rule,
                condition=ExtendedCondition.ORDER_SENSITIVE,
                max_steps=40,
                order_config=OrderConfig(required_order=req_order),
                rng_seed=seed,
            )
        else:
            env = ExtendedBlicketEnv6(
                causal_rule=rule,
                condition=ExtendedCondition.ORDER_SENSITIVE,
                max_steps=75,
                order_config=OrderConfig(required_order=req_order),
                rng_seed=seed,
            )

        place_map = PLACE_5 if n_correct == 3 else PLACE_6

        # Test correct order activates
        env.reset()
        for obj in req_order:
            env.step(place_map[obj])
        correct_activates = env.state.detector_active

        # Test ≥2 (3-obj) or ≥3 (4-obj) incorrect orders do NOT activate
        min_wrong = 2 if n_correct == 3 else 3
        wrong_count = 0
        wrong_tested = 0

        # Generate some wrong orderings
        import itertools
        all_perms = list(itertools.permutations(req_order))
        wrong_perms = [p for p in all_perms if list(p) != req_order]
        for wrong_order in wrong_perms[:min_wrong + 2]:
            env.reset()
            for obj in wrong_order:
                env.step(place_map[obj])
            if not env.state.detector_active:
                wrong_count += 1
            wrong_tested += 1

        seed_ok = correct_activates and wrong_count >= min_wrong
        if not seed_ok:
            env_ok = False
            print(f"  Seed {seed}: FAIL (correct_activates={correct_activates}, wrong_inactive={wrong_count}/{wrong_tested})")

    if env_ok:
        print(f"  All {len(seeds)} seeds: correct order activates, ≥{min_wrong} wrong orders do not: OK")

    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_ok and env_ok else 'SOME CHECKS FAILED'}")
    return all_ok and env_ok


if __name__ == "__main__":
    ok1 = validate_seed_set(
        label="Calibration (3-object)",
        seeds=range(2001, 2011),
        all_labels=["A", "B", "C", "D", "E"],
        n_correct=3,
        max_alpha=2,
    )

    ok2 = validate_seed_set(
        label="Signal Check (4-object)",
        seeds=range(1401, 1411),
        all_labels=["A", "B", "C", "D", "E", "F"],
        n_correct=4,
        max_alpha=1,
    )

    print(f"\n{'='*60}")
    if ok1 and ok2:
        print("  ALL SEED VALIDATIONS PASSED")
    else:
        print("  SEED VALIDATION FAILURES DETECTED")
        sys.exit(1)
    print(f"{'='*60}")
