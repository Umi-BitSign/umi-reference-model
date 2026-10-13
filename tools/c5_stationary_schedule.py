"""Plan balanced replay and audit the completed finite-coverage training mixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from c5_all_source_training import domain
from c5_coverage_schedule import group


def stationary_schedule(
    rows: list[dict], masses: dict[str, float], seed: int, maximum_exposures: int = 100_000
) -> list[dict]:
    """Cover every unique clip while replaying exhausted groups with unit loss weight."""
    if not rows or len({(r["source"], r["id"]) for r in rows}) != len(rows):
        raise ValueError("stationary input must contain unique training examples")
    if any(r.get("role") != "training" or r.get("training_allowed") is not True for r in rows):
        raise ValueError("stationary replay accepts training-only examples")
    buckets = defaultdict(list)
    for row in rows:
        buckets[group(row)].append(row)
    groups_per_domain = Counter(name for name, _ in buckets)
    fractions = {name: Fraction(str(value)) for name, value in masses.items()}
    if (
        set(groups_per_domain) != set(fractions)
        or any(value <= 0 for value in fractions.values())
        or sum(fractions.values()) != 1
    ):
        raise ValueError("stationary replay requires complete normalized domain masses")
    weights = {key: fractions[key[0]] / groups_per_domain[key[0]] for key in buckets}
    period = math.lcm(*(weight.denominator for weight in weights.values()))
    minimum = max(math.ceil(len(bucket) / weights[key]) for key, bucket in buckets.items())
    total = math.ceil(minimum / period) * period
    if total > maximum_exposures:
        raise ValueError("stationary full coverage exceeds the explicit exposure budget")
    used = Counter()
    active = {}
    result = []
    for exposure in range(total):
        key = min(buckets, key=lambda item: (used[item] / weights[item], item))
        cycle, offset = divmod(used[key], len(buckets[key]))
        if offset == 0:
            active[key] = sorted(
                buckets[key],
                key=lambda row: hashlib.sha256(
                    f"stationary-{seed}-{cycle}:{row['source']}:{row['id']}".encode()
                ).hexdigest(),
            )
        row = active[key][offset]
        result.append(
            {
                **row,
                "original_corpus_loss_weight": row.get("loss_weight", 1.0),
                "loss_weight": 1.0,
                "exposure": exposure + 1,
                "group_exposure": used[key] + 1,
                "repeat_cycle": cycle,
            }
        )
        used[key] += 1
    if any(used[key] != weights[key] * total for key in buckets):
        raise AssertionError("stationary replay lost its exact overall mixture")
    return result


def mixture(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("mixture audit needs a nonempty window")
    weighted = defaultdict(float)
    counts = Counter()
    how = Counter()
    for row in rows:
        weight = row["loss_weight"]
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("mixture audit requires finite positive objective weights")
        name = domain(row)
        weighted[name] += weight
        counts[name] += 1
        if name == "how2sign":
            how[int(row["signer"])] += 1
    total = sum(weighted.values())
    return {
        "exposures": len(rows),
        "unique_examples": len({(r["source"], r["id"]) for r in rows}),
        "domain_counts": dict(sorted(counts.items())),
        "how_signer_counts": dict(sorted(how.items())),
        "objective_domain_mass": {key: value / total for key, value in sorted(weighted.items())},
    }


def main() -> None:
    from benchmark_fleurs_dev import sha256
    from c5_experiment_sources import archive_sources
    from c5_face_learning_pilot import ROOT, read_rows
    from c5_full_coverage_mps import RECIPE, checked_plan

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed-plan", type=Path, required=True)
    parser.add_argument("--completed-decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    path = args.completed_plan.resolve()
    plan = checked_plan(path)
    decision_path = args.completed_decision.resolve()
    decision = json.loads(decision_path.read_text())
    if (
        decision.get("schema") != "umi-studio-c5-full-coverage-mps-decision/1"
        or decision.get("plan_sha256") != sha256(path)
        or [r["examples"] for r in decision["evaluations"]] != RECIPE["checkpoint_after_examples"]
        or decision.get("decision") != "reject_full_coverage_no_open_generalization_gain"
        or decision.get("promoted") is not False
    ):
        raise ValueError("mixture proposal requires the completed rejected parent experiment")
    rows = read_rows(path.with_name("schedule.jsonl"))
    replay = stationary_schedule(rows, RECIPE["domain_mass"], plan["seed"])
    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("mixture comparison requires a new private runs/ directory")
    output.mkdir(parents=True)
    schedule = output / "proposed-schedule.jsonl"
    with schedule.open("x") as stream:
        for row in replay:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    sources = {**plan["source_sha256"], "c5_stationary_schedule.py": sha256(Path(__file__))}
    report = {
        "schema": "umi-studio-c5-stationary-mixture-proposal/1",
        "parent_plan_sha256": sha256(path),
        "completed_decision_sha256": sha256(decision_path),
        "proposed_schedule_sha256": sha256(schedule),
        "source_sha256": sources,
        "seed": plan["seed"],
        "intended_domain_mass": RECIPE["domain_mass"],
        "finite_coverage": {
            "full": mixture(rows),
            "first512": mixture(rows[:512]),
            "tail_after4096": mixture(rows[4096:]),
        },
        "stationary_replay": {
            "full": mixture(replay),
            "first512": mixture(replay[:512]),
            "last512": mixture(replay[-512:]),
            "maximum_repeat_count": max(Counter((r["source"], r["id"]) for r in replay).values()),
            "optimizer_updates_at_accumulation4": math.ceil(len(replay) / 4),
        },
        "planning_only": True,
        "quality_verified": False,
        "training_launched": False,
        "qualification": (
            "Repeated training-only exposures and unit loss weights preserve the intended "
            "sampling objective throughout training. This extends the total exposure budget; "
            "it is not an equal-update matched quality experiment. Minority replay may overfit. "
            "No quality gain is claimed, no evaluation data is trained on, and a new complete "
            "scientific plan and compatible trainer must be frozen before launching. "
            "The approved GPU expiry, memory limits and independent confirmation "
            "requirement remain."
        ),
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    archive_sources(report_path, output / "source-snapshot")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
