"""Unchanged paired Terra-medium semantic gate for completed fixed ROI translations."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from benchmark_fleurs_dev import sha256
from evaluate_roi_short_translation import CONTROL, ROOT, prediction, require_control, terminal
from evaluate_roi_short_translation import checked as checked_inference
from launch_terra_visual_job import identity, now, write
from terra_short_curriculum_learning import comparison_items, score_comparison, semantic_decision
from terra_short_resume import RecoveryPacer
from terra_training_recovery import artifact, verify_snapshot
from terra_training_storage import storage

INFERENCE_PLAN = ROOT / "runs/roi-short-translation-evaluation-v1/plan.json"
INFERENCE = ROOT / "checkpoints/roi-short-translation-evaluation-v1/receipt.json"
POLICY = {
    "judge_model": "gpt-5.6-terra",
    "reasoning_effort": "medium",
    "rubric": "unchanged owner0-10, nonsense0 and exact semantic meaning10",
    "passes": 2,
    "maximum_pairs_per_request": 16,
    "minimum_seconds_between_request_starts": 60,
    "own_requests_in_flight": 1,
    "automatic_retries": 0,
    "endpoint_or_model_fallback": False,
    "GPU_use": False,
    "optimizer_updates": 0,
    "WER_is_diagnostic_only": True,
    "promotion_authorized": False,
}


def verified_prediction(row, case, tokenizer):
    tokens = case.get("tokens")
    if (
        not isinstance(tokens, list)
        or not tokens
        or len(tokens) > 2048
        or any(type(v) is not int or not 0 <= v < 384 for v in tokens)
        or case.get("native_crop_parity") is not True
        or tokenizer.decode(tokens, skip_special_tokens=True) != case.get("hypothesis")
    ):
        raise ValueError("ROI score needs complete actual native-parity decoded token evidence")
    expected = prediction(row, case["hypothesis"])
    if any(case.get(k) != v for k, v in expected.items()):
        raise ValueError("ROI decoded reference/roles/diagnostic measurement differs")
    return expected


def specification():
    from transformers import ByT5Tokenizer

    plan = checked_inference(INFERENCE_PLAN)
    decoded = json.loads(INFERENCE.read_text())
    terminal(INFERENCE)
    if (
        decoded.get("complete") is not True
        or decoded.get("plan_sha256") != sha256(INFERENCE_PLAN)
        or decoded.get("clips") != 34
        or decoded.get("all34_frozen_C5_controls_reproduced") is not True
        or decoded.get("all_fixed_weights_unchanged") is not True
        or decoded.get("optimizer_updates") != 0
        or decoded.get("new_Terra_requests") != 0
        or decoded.get("promotion_authorized") is not False
    ):
        raise ValueError("ROI scoring cannot run before complete fixed-model/control inference")
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / plan["reference"]["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )
    baseline = json.loads(CONTROL.read_text())["predictions"]
    indexed = {(r["source"], r["example_id"]): r for r in baseline}
    assets, predictions, rows = {}, [], []
    for i, record in enumerate(plan["records"]):
        row = record["development"]
        control, candidate = (
            INFERENCE.with_name(f"{prefix}-{i:04d}.json") for prefix in ("control", "candidate")
        )
        observed_control = json.loads(control.read_text())
        if observed_control.get("matches_frozen_control") is not True:
            raise ValueError("ROI semantic comparison lost its reproduced C5 control")
        if any(
            observed_control.get(k) != v
            for k, v in prediction(row, observed_control["hypothesis"]).items()
        ):
            raise ValueError("ROI reproduced control lost its actual identity/role/reference")
        require_control(
            observed_control["hypothesis"],
            indexed[(row["source"], row["example_id"])]["hypothesis"],
        )
        predictions.append(verified_prediction(row, json.loads(candidate.read_text()), tokenizer))
        rows.append(row)
        for path in (control, candidate):
            assets[str(path.relative_to(ROOT))] = sha256(path)
    if predictions != decoded["predictions"]:
        raise ValueError("ROI inference receipt differs from actual candidate cases")
    comparison = comparison_items(rows, baseline, predictions)
    if comparison != decoded["comparison"]:
        raise ValueError("ROI paired comparison differs from inference identities")
    sources = dict(plan["source_sha256"])
    for name in (
        "score_roi_short_translation.py",
        "terra_quality_training.py",
        "probe_jeff_terra_quality.py",
        "semantic_quality_judge.py",
        "terra_semantic_selection.py",
        "terra_short_curriculum_learning.py",
        "terra_short_resume.py",
    ):
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-roi-short-translation-semantic-score/1",
        "policy": POLICY,
        "inference_plan": artifact(INFERENCE_PLAN),
        "inference": artifact(INFERENCE),
        "assets": assets,
        "source_sha256": sources,
        "comparison": comparison,
        "baseline": baseline,
        "predictions": predictions,
        "planned_Terra_requests": 2 * math.ceil(len(comparison["items"]) / 16),
        "new_training_updates": 0,
        "promotion_authorized": False,
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("ROI semantic evaluation input/policy/source changed")
    verify_snapshot(path, plan)
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("ROI semantic evaluation needs a fresh owned plan")
    storage()
    plan = specification()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(
        json.dumps({"plan": artifact(path), "planned_requests": plan["planned_Terra_requests"]}),
        flush=True,
    )


def run(path, output):
    import torch

    torch.set_num_threads(2)
    plan = checked(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("ROI semantic scores need a fresh owned output")
    storage()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        scores = score_comparison(plan["comparison"], output / "semantic", RecoveryPacer())
        result = semantic_decision(
            plan["comparison"], scores, plan["baseline"], plan["predictions"]
        )
        checked(path)
        write(
            output / "decision.json",
            {
                **result,
                "plan_sha256": sha256(path),
                "optimizer_updates": 0,
                "new_Terra_requests": plan["planned_Terra_requests"],
                "C5_retained_unless_fully_confirmed": True,
                "promotion_authorized": False,
            },
        )
        print(json.dumps({"stage": "ROI_Terra_semantic_decision", **result}), flush=True)
        code = 0
    except BaseException as error:
        write(
            output / "failure.json",
            {
                "error_type": type(error).__name__,
                "message": str(error),
                "automatic_retries": 0,
                "complete_semantic_decision": False,
            },
        )
        raise
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    elif args.plan and args.output:
        run(args.plan, args.output)
    else:
        parser.error("provide --freeze or --plan and --output")


if __name__ == "__main__":
    main()
