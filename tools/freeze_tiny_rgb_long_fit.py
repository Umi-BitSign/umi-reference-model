"""Extend only the raw-RGB fit horizon, reusing exact already-verified pixel bytes."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

from benchmark_fleurs_dev import sha256
from c5_experiment_sources import archive_sources
from prepare_tiny_rgb_fit import preview
from tiny_rgb_fit import (
    RECIPE,
    ROOT,
    SIDES,
    checked_backend,
    checked_plan,
    load_pixels,
    sources,
    storage,
)


def validate_horizon_change(previous: dict, current: dict) -> None:
    changes = {k for k, v in previous.items() if current.get(k) != v}
    if (
        previous.get("epochs") != 32
        or current.get("epochs") != 256
        or changes != {"epochs", "checks_at_epochs"}
        or set(current).difference(previous) != {"checkpoints_at_epochs"}
        or not set(previous["checks_at_epochs"]).issubset(current["checks_at_epochs"])
        or current["checks_at_epochs"][-1] != current["epochs"]
    ):
        raise ValueError("long RGB fit must change only training horizon and checkpoint cadence")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--completed-fit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_plan, completed = args.source_plan.resolve(), args.completed_fit_dir.resolve()
    output = args.output_dir.resolve()
    if (
        not source_plan.is_relative_to(ROOT / "runs")
        or not completed.is_relative_to(ROOT / "checkpoints")
        or output.exists()
        or not output.is_relative_to(ROOT / "runs")
    ):
        raise ValueError("long RGB fitting needs explicit private plan/results and a new run")
    old = json.loads(source_plan.read_text())
    validate_horizon_change(old["recipe"], RECIPE)
    decision_path = completed / "decision.json"
    decision = json.loads(decision_path.read_text())
    if (
        not decision["complete"]
        or decision["plan_sha256"] != sha256(source_plan)
        or decision["optimizer_updates_per_arm"] != 768
        or decision["training_examples_per_arm"] != 24
        or decision["held_out_quality_verified"]
        or decision["selected_model_changed"]
    ):
        raise ValueError("long RGB fitting requires the completed matched numerical repair control")
    snapshot_dir = source_plan.with_name("source-snapshot")
    snapshot = json.loads((snapshot_dir / "manifest.json").read_text())
    if snapshot["plan_sha256"] != sha256(source_plan) or any(
        sha256(snapshot_dir / name) != digest for name, digest in old["source_sha256"].items()
    ):
        raise ValueError("prior RGB fit source archive changed")
    for name, digest in {**old["dependency_sha256"], **old["language_asset_sha256"]}.items():
        if sha256(ROOT / name) != digest:
            raise ValueError("prior RGB fit data/language dependency changed")
    old_ledger = source_plan.with_name("pixels.jsonl")
    if sha256(old_ledger) != old["pixel_ledger_sha256"]:
        raise ValueError("prior RGB fit pixel ledger changed")
    bindings = [json.loads(s) for s in old_ledger.read_text().splitlines()]
    lookup = {(r["source"], r["example_id"]): r for r in bindings}
    if len(lookup) != len(old["examples"]) or len(bindings) != len(lookup):
        raise ValueError("prior RGB fit pixel ledger coverage differs")
    for row in old["examples"]:
        for side in SIDES:
            load_pixels(row, side, lookup[(row["source"], row["example_id"])])
    storage(ROOT / old["data_dir"])
    backend = checked_backend(old["examples"], ROOT / old["backend"]["original_plan"])
    if backend != old["backend"]:
        raise ValueError("long RGB fit numerical backend differs from completed control")
    new = copy.deepcopy(old)
    new["recipe"] = RECIPE
    new["source_sha256"] = sources()
    for dependency in (source_plan, decision_path, old_ledger, snapshot_dir / "manifest.json"):
        new["dependency_sha256"][str(dependency.relative_to(ROOT))] = sha256(dependency)
    new["prior_completed_fit"] = {
        "plan": str(source_plan.relative_to(ROOT)),
        "plan_sha256": sha256(source_plan),
        "decision": str(decision_path.relative_to(ROOT)),
        "decision_sha256": sha256(decision_path),
        "weights_reused": False,
        "optimizer_state_reused": False,
    }
    new["qualification"] = (
        "Horizon-only extension of the completed corrected32-epoch fit. All cases, frame indices, "
        "pixels, model configuration, fresh visual/bridge initialization, pinned language weights, "
        "rates, dropout and fit gates are unchanged.256 epochs per arm, no warm start from "
        "diagnostic weights. The prior pixel files are reused byte-for-byte. Training-only "
        "learnability research; never held-out quality or promotion. Full-corpus training and "
        "protected source-held-out generalization remain required."
    )
    output.mkdir(parents=True)
    shutil.copyfile(old_ledger, output / "pixels.jsonl")
    plan_path = output / "plan.json"
    plan_path.write_text(json.dumps(new, indent=2, sort_keys=True) + "\n")
    archive_sources(plan_path, output / "source-snapshot")
    checked_plan(plan_path)
    preview(bindings, output / "resolution-preview.png")
    print(
        json.dumps(
            {
                "plan_sha256": sha256(plan_path),
                "cases": len(old["examples"]),
                "epochs_per_arm": RECIPE["epochs"],
                "new_pixel_cache_bytes": 0,
                "pixel_ledger_identical": sha256(old_ledger) == sha256(output / "pixels.jsonl"),
            }
        )
    )


if __name__ == "__main__":
    main()
