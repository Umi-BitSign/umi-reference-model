"""Fresh64px appearance training on the expanded, reserved-signer-safe short corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from benchmark_fleurs_dev import sha256
from launch_terra_visual_job import write
from prepare_roi_short_corpus import POLICY as DATA_POLICY
from prepare_roi_short_corpus import checked as checked_data_plan
from roi_appearance_protocol import RECIPE as PILOT_RECIPE
from roi_appearance_protocol import ROOT, checked_cache, model_specification
from terra_training_recovery import artifact, verify_snapshot
from terra_training_storage import storage

DATA_PLAN = ROOT / "runs/roi-short-corpus-data-v1/plan.json"
DATA = ROOT / "checkpoints/roi-short-corpus-data-v1/data.json"
RECIPE = {**PILOT_RECIPE, "epochs": 8, "evaluation_epochs": [0, 1, 4, 8]}


def checked_data():
    plan, data, arrays = checked_cache(
        DATA_PLAN,
        DATA,
        validate_plan=checked_data_plan,
        expected_clips=90,
        cache_limit=DATA_POLICY["maximum_cache_RAM_bytes"],
    )
    if (
        data.get("counts") != {"training": 88, "feature_diagnostic": 2}
        or sum(r["frames"] for r in data["records"] if r["roi_training_allowed"]) != 7865
        or {r["original_row"]["signer"] for r in data["records"] if not r["roi_training_allowed"]}
        != {"P0", "P19"}
    ):
        raise ValueError("expanded ROI training/diagnostic coverage differs")
    return plan, data, arrays


def proof_indices(records):
    train = [i for i, r in enumerate(records) if r["roi_training_allowed"]]
    if not train:
        raise ValueError("expanded ROI lacks training-only proof inputs")
    longest = max(train, key=lambda i: records[i]["frames"])
    supplemental = next(
        i for i in train if records[i]["original_row"]["source"] == "asl_stem_supplemental"
    )
    return sorted({train[0], longest, supplemental})


def specification():
    data_plan, data, arrays = checked_data()
    plan = model_specification(data_plan, data, arrays, DATA_PLAN, DATA, RECIPE)
    plan.update(
        {
            "schema": "umi-studio-roi-short-corpus-training/1",
            "proof_training_record_indices": proof_indices(data["records"]),
            "fixed_final_epoch_not_selected_by_feature_diagnostics": True,
            "initial_pilot_vision_weights_reused": False,
            "data_change": (
                "six training clips become all88 eligible short-STEM clips/25 signers; "
                "same64px appearance encoder and loss"
            ),
            "schedule_change": (
                "eight complete passes instead of64 on six clips; exposure/update counts "
                "derived from the complete expanded frame inventory"
            ),
            "observed_crop_preview_review": (
                "expanded eight-row native crop preview reviewed before freeze; "
                "all90 caches have exact native-reference parity"
            ),
        }
    )
    plan["source_sha256"]["tools/train_roi_short_corpus.py"] = sha256(Path(__file__))
    return plan


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("expanded ROI immutable data/model/recipe/runtime bindings changed")
    verify_snapshot(path, plan)
    if sha256(ROOT / plan["resource_policy"]) != plan["resource_policy_sha256"]:
        raise ValueError("expanded ROI resource policy changed")
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("expanded ROI training requires a new owned plan")
    torch.set_num_threads(2)
    plan = specification()
    storage(plan["visual_parameters"] * 12 + 1024**2)
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(
        json.dumps(
            {
                "plan": artifact(path),
                "updates": plan["planned_optimizer_updates"],
                "frame_exposures": plan["planned_frame_exposures"],
            }
        ),
        flush=True,
    )


def main():
    from train_roi_appearance import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qualification-only", action="store_true")
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    elif args.plan and args.output:
        run(
            args.plan,
            args.output,
            qualification_only=args.qualification_only,
            qualification=args.qualification,
            protocol=sys.modules[__name__],
        )
    else:
        parser.error("provide --freeze or --plan and --output")


if __name__ == "__main__":
    main()
