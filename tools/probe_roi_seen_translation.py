"""Training-only decoded-quality diagnosis for two fixed ROI checkpoints, never a win gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch
from benchmark_fleurs_dev import sha256
from evaluate_roi_short_translation import decode, predicted_streams, prediction, terminal
from launch_terra_visual_job import identity, now, write
from prepare_roi_appearance import native_helpers
from probe_tiny_rgb_capacity import state_digest
from roi_appearance_model import STREAMS, RoiAppearance, feature_loss
from roi_appearance_protocol import ROOT, checked_data
from semantic_quality_judge import calibration_controls
from terra_short_curriculum_learning import score_comparison
from terra_short_resume import RecoveryPacer
from terra_training_recovery import artifact, verify_snapshot
from terra_training_storage import storage
from tiny_rgb_full_run import runtime_binding
from train_roi_short_corpus import checked as checked_expanded_fit

OLD_PLAN = ROOT / "runs/roi-appearance-fit-v2/plan.json"
OLD_RESULT = ROOT / "checkpoints/roi-appearance-fit-v2/decision.json"
BROAD_PLAN = ROOT / "runs/roi-short-corpus-training-v1/plan.json"
BROAD_RESULT = ROOT / "checkpoints/roi-short-corpus-training-v1/decision.json"
LABELS = ("C5", "ROI_six_clip_fit", "ROI_expanded_fit")
POLICY = {
    "scope": "all six predeclared pilot training clips, also trained by the expanded fit",
    "true_training_roles_preserved": True,
    "heldout_inputs_used": False,
    "checkpoints": "both fixed final checkpoints, not selected using probe outputs",
    "published_C5_generation_and_CPU_budget": True,
    "judge_model": "gpt-5.6-terra",
    "reasoning_effort": "medium",
    "passes": 2,
    "maximum_pairs_per_request": 16,
    "minimum_request_start_gap_seconds": 60,
    "automatic_retries": 0,
    "optimizer_updates": 0,
    "training_scores_are_not_development_or_generalization": True,
    "model_win_claimed": False,
    "promotion_authorized": False,
}


def selected_training(data, expanded):
    rows = [r for r in data["records"] if r["roi_training_allowed"] is True]
    allowed = {
        (r["original_row"]["source"], r["original_row"]["example_id"])
        for r in expanded["records"]
        if r["roi_training_allowed"] is True
    }
    if len(rows) != 6 or len({r["original_row"]["signer"] for r in rows}) != 6:
        raise ValueError("seen probe cannot replace the six predeclared training identities")
    if any(
        r["roi_role"] != "training"
        or r["original_row"]["role"] != "training"
        or r["original_row"]["training_allowed"] is not True
        or (r["original_row"]["source"], r["original_row"]["example_id"]) not in allowed
        for r in rows
    ):
        raise ValueError("seen probe cannot relabel heldout or untrained clips as seen")
    return rows


def fixed_checkpoint(plan_path, result_path, expected, current):
    plan, result = json.loads(plan_path.read_text()), json.loads(result_path.read_text())
    verify_snapshot(plan_path, plan)
    terminal(result_path)
    if artifact(ROOT / plan["data"]["path"]) != plan["data"]:
        raise ValueError("seen probe historical fitted-data artifact changed")
    if (
        result.get("complete") is not True
        or result.get("plan_sha256") != sha256(plan_path)
        or (result.get("optimizer_updates"), result.get("frame_exposures")) != expected
    ):
        raise ValueError("seen probe requires complete fixed-final checkpoint counters")
    if plan["runtime_binding"] != runtime_binding() or any(
        plan[k] != current[k]
        for k in (
            "initialization_sha256",
            "visual_parameters",
            "parameter_tensors",
            "torchvision_version",
            "resnet_source_sha256",
            "C5_reference_sha256",
        )
    ):
        raise ValueError("old and expanded ROI implementations/runtime differ")
    # Old orchestration was refactored after its terminal run. Validate its archive,
    # and require byte-identical implementations of the actual visual computation.
    for name in ("roi_appearance_model.py", "tiny_rgb_model.py", "tiny_rgb_mps_safe.py"):
        if plan["source_sha256"][f"tools/{name}"] != sha256(ROOT / "tools" / name):
            raise ValueError("historical ROI visual computation changed")
    path = ROOT / result["visual"]["path"]
    commit = json.loads(path.with_suffix(".json").read_text())
    if (
        artifact(path) != result["visual"]
        or commit.get("complete") is not True
        or commit.get("visual") != result["visual"]
    ):
        raise ValueError("seen probe fixed checkpoint bytes/commit differ")
    return {
        "plan": artifact(plan_path),
        "result": artifact(result_path),
        "visual": artifact(path),
        "frame_exposures": expected[1],
        "optimizer_updates": expected[0],
    }


def specification():
    broad = checked_expanded_fit(BROAD_PLAN)
    _, data, _ = checked_data()
    expanded = json.loads((ROOT / broad["data"]["path"]).read_text())
    rows = selected_training(data, expanded)
    models = {
        LABELS[1]: fixed_checkpoint(OLD_PLAN, OLD_RESULT, (2624, 20672), broad),
        LABELS[2]: fixed_checkpoint(BROAD_PLAN, BROAD_RESULT, (7872, 62920), broad),
    }
    source = dict(broad["source_sha256"])
    for name in (
        "probe_roi_seen_translation.py",
        "evaluate_roi_short_translation.py",
        "c5_terra_tail.py",
        "tiny_rgb_winner_student.py",
        "semantic_quality_judge.py",
        "terra_short_curriculum_learning.py",
        "terra_short_resume.py",
        "terra_quality_training.py",
        "probe_jeff_terra_quality.py",
    ):
        source[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-roi-seen-translation-probe/1",
        "policy": POLICY,
        "models": models,
        "records": rows,
        "source_sha256": source,
        "reference": json.loads((ROOT / "C5_BASELINE_REFERENCE.json").read_text()),
        "resource_policy": broad["resource_policy"],
        "resource_policy_sha256": broad["resource_policy_sha256"],
        "initial_reference_hashes": json.loads(
            (ROOT / "runs/terra-c5-tail-training-v1/plan.json").read_text()
        )["initial_reference_hashes"],
        "new_Terra_requests": 0,
        "optimizer_updates": 0,
        "development_quality_gate_computed": False,
        "promotion_authorized": False,
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("seen probe fixed inputs/model/scope/implementation changed")
    verify_snapshot(path, plan)
    return plan


def freeze(path, plan):
    from c5_experiment_sources import archive_sources

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("seen probe requires a fresh owned plan")
    storage()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(json.dumps({"plan": artifact(path)}), flush=True)


def infer(path, output, plan):
    import numpy as np
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
    from c5_terra_tail import load_model, native_memory
    from safetensors import safe_open
    from safetensors.torch import load_file
    from tiny_rgb_winner_student import load_frozen_winner_encoder
    from torch.nn import functional as F
    from transformers import ByT5Tokenizer

    native_helpers()  # Preserve the byte-identical pinned reader import path.
    budget = CertificationComputeWindow(ROOT / plan["resource_policy"])
    if (
        budget.duty_fraction() != 0.5
        or budget.cpu_threads() != 2
        or not torch.backends.mps.is_available()
    ):
        raise ValueError("seen probe requires50% Metal/CPU2")
    duty = WinnerDutyCycle(torch, budget)
    model, reference = load_model()
    model.requires_grad_(False).eval()
    encoder = load_frozen_winner_encoder(reference)
    if reference != plan["reference"]:
        raise ValueError("seen probe C5 reference differs")
    before = {
        "language": state_digest(model.language.state_dict()),
        "encoder": state_digest(encoder.state_dict()),
    }
    if before != plan["initial_reference_hashes"]:
        raise ValueError("seen probe did not load original C5 values")
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / reference["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )
    predictions = {label: [] for label in LABELS}
    memories, arrays = [], []
    with torch.no_grad():
        for i, record in enumerate(plan["records"]):
            row = record["original_row"]
            with np.load(ROOT / record["cache"]["path"], allow_pickle=False) as archive:
                arrays.append({n: archive[n].copy() for n in ("pixels", *STREAMS, "pose_features")})
            memory = native_memory(
                encoder, row["original_pair"], reference["translator_checkpoint_sha256"]
            )
            memories.append(memory)
            text, tokens = decode(model.language, tokenizer, reference["generation"], memory)
            result = prediction(row, text)
            predictions[LABELS[0]].append(result)
            write(output / f"C5-{i:02d}.json", {**result, "tokens": tokens})
        for label in LABELS[1:]:
            descriptor = plan["models"][label]
            visual = ROOT / descriptor["visual"]["path"]
            with safe_open(visual, framework="pt") as file:
                meta = file.metadata()
            if (
                meta["plan_sha256"] != descriptor["plan"]["sha256"]
                or meta["update"] != str(descriptor["frame_exposures"])
                or meta["optimizer_updates"] != str(descriptor["optimizer_updates"])
                or meta["fixed_winner_checkpoint"] != reference["translator_checkpoint_sha256"]
            ):
                raise ValueError("seen probe loaded another checkpoint's metadata")
            appearance = RoiAppearance().eval()
            appearance.load_state_dict(load_file(visual), strict=True)
            appearance.requires_grad_(False)
            visual_hash = state_digest(appearance.state_dict())
            appearance = duty.run(lambda appearance=appearance: appearance.to("mps"))
            if state_digest(appearance.state_dict()) != visual_hash:
                raise ValueError("seen probe transfer changed visual values")
            for i, (record, values) in enumerate(zip(plan["records"], arrays, strict=True)):
                storage(output=output)
                row = record["original_row"]
                target = {n: values[n] for n in (*STREAMS, "pose_features")}
                features = predicted_streams(appearance, values["pixels"], target, duty)
                memory = encoder(
                    **{n: v[None] for n, v in features.items()}, return_dict=True
                ).last_hidden_state
                if memory.shape != memories[i].shape or not torch.isfinite(memory).all():
                    raise ValueError("seen probe candidate lost complete semantic frame coverage")
                text, tokens = decode(model.language, tokenizer, reference["generation"], memory)
                result = prediction(row, text)
                predictions[label].append(result)
                features_loss = feature_loss(
                    {n: features[n] for n in STREAMS},
                    {n: torch.from_numpy(target[n]) for n in STREAMS},
                )
                write(
                    output / f"{label}-{i:02d}.json",
                    {
                        **result,
                        "tokens": tokens,
                        "training_feature_loss": float(features_loss["loss"]),
                        "training_latent_cosine_loss": float(
                            (1 - F.cosine_similarity(memory, memories[i], dim=-1)).mean()
                        ),
                    },
                )
                print(
                    json.dumps({"stage": "seen_decode", "model": label, "clips": i + 1}), flush=True
                )
            if state_digest(appearance.state_dict()) != visual_hash:
                raise ValueError("seen probe changed fixed visual weights")
            duty.run(lambda appearance=appearance: appearance.to("cpu"))
            del appearance
    if before != {
        "language": state_digest(model.language.state_dict()),
        "encoder": state_digest(encoder.state_dict()),
    }:
        raise ValueError("seen probe changed C5 backbone values")
    checked(path)
    write(
        output / "receipt.json",
        {
            "complete": True,
            "plan": artifact(path),
            "predictions": predictions,
            "all_fixed_weights_unchanged": True,
            "training_only": True,
            "development_quality_gate_computed": False,
            "model_win_claimed": False,
            "promotion_authorized": False,
            "new_Terra_requests": 0,
            "optimizer_updates": 0,
            "resources": duty.receipt(),
        },
    )


def comparison(records, predictions):
    if set(predictions) != set(LABELS) or any(
        len(predictions[label]) != len(records) for label in LABELS
    ):
        raise ValueError("seen score cannot drop a model or training clip")
    items, indices, cases = [], {}, []
    for i, record in enumerate(records):
        row = record["original_row"]
        if row["role"] != "training" or row["training_allowed"] is not True:
            raise ValueError("seen scores preserve true training roles")
        mapped = {}
        for label in LABELS:
            result = predictions[label][i]
            expected = prediction(row, result["hypothesis"])
            if result != expected:
                raise ValueError("seen comparison identity/role/reference differs")
            key = row["reference"], result["hypothesis"]
            if key not in indices:
                indices[key] = len(items)
                items.append({"reference": key[0], "candidate": key[1]})
            mapped[label] = indices[key]
        cases.append({"source": row["source"], "example_id": row["example_id"], "indices": mapped})
    controls = []
    for control in calibration_controls():
        controls.append({"index": len(items), "range": control["range"]})
        items.append({"reference": control["reference"], "candidate": control["candidate"]})
    return {"items": items, "cases": cases, "controls": controls}


def score_specification(decoded):
    receipt = json.loads(decoded.read_text())
    terminal(decoded)
    plan_path = ROOT / receipt["plan"]["path"]
    if artifact(plan_path) != receipt["plan"]:
        raise ValueError("seen probe decoded-plan artifact changed")
    plan = checked(plan_path)
    if (
        receipt.get("complete") is not True
        or receipt.get("all_fixed_weights_unchanged") is not True
        or receipt.get("training_only") is not True
        or receipt.get("model_win_claimed") is not False
        or receipt.get("optimizer_updates") != 0
    ):
        raise ValueError("seen scoring requires complete training-only fixed inference")
    from transformers import ByT5Tokenizer

    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / plan["reference"]["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )
    assets = {}
    for label in LABELS:
        for i, result in enumerate(receipt["predictions"][label]):
            path = decoded.with_name(f"{label}-{i:02d}.json")
            case = json.loads(path.read_text())
            verify_case(result, case, tokenizer)
            assets[str(path.relative_to(ROOT))] = sha256(path)
    paired = comparison(plan["records"], receipt["predictions"])
    return {
        "schema": "umi-studio-roi-seen-translation-scores/1",
        "policy": POLICY,
        "decoded": artifact(decoded),
        "source_sha256": plan["source_sha256"],
        "assets": assets,
        "comparison": paired,
        "planned_requests": 2 * math.ceil(len(paired["items"]) / 16),
        "promotion_authorized": False,
    }


def verify_case(result, case, tokenizer):
    tokens = case.get("tokens")
    if (
        not isinstance(tokens, list)
        or not tokens
        or len(tokens) > 2048
        or any(type(v) is not int or not 0 <= v < 384 for v in tokens)
        or any(case.get(k) != v for k, v in result.items())
        or tokenizer.decode(tokens, skip_special_tokens=True) != result["hypothesis"]
    ):
        raise ValueError("seen score differs from actual decoded token evidence")


def summarize(paired, scores):
    if len(scores) != 2 or any(
        len(s) != len(paired["items"]) or any(type(v) is not int or not 0 <= v <= 10 for v in s)
        for s in scores
    ):
        raise ValueError("seen diagnostic requires two complete actual score passes")
    calibrated = all(
        c["range"][0] <= s[c["index"]] <= c["range"][1] for s in scores for c in paired["controls"]
    )
    return {
        "calibration_passed_both_passes": calibrated,
        "training_mean_quality": {
            label: statistics.mean(
                statistics.mean(s[c["indices"][label]] for s in scores) for c in paired["cases"]
            )
            for label in LABELS
        },
        "training_only": True,
        "development_quality_gate_computed": False,
        "model_win_claimed": False,
        "promotion_authorized": False,
    }


def run(path, output, *, scoring=False):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if scoring:
        plan = json.loads(path.read_text())
        if plan != score_specification(ROOT / plan["decoded"]["path"]):
            raise ValueError("seen score plan changed")
        verify_snapshot(path, plan)
    else:
        plan = checked(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("seen probe requires a fresh owned output")
    storage()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        if scoring:
            scores = score_comparison(plan["comparison"], output / "semantic", RecoveryPacer())
            result = summarize(plan["comparison"], scores)
            if plan != score_specification(ROOT / plan["decoded"]["path"]):
                raise ValueError("seen score input/source changed during requests")
            verify_snapshot(path, plan)
            write(
                output / "diagnosis.json",
                {
                    **result,
                    "plan_sha256": sha256(path),
                    "new_Terra_requests": plan["planned_requests"],
                    "optimizer_updates": 0,
                },
            )
            print(json.dumps(result), flush=True)
        else:
            from train_terra_c5_tail import gpu_lease

            with gpu_lease():
                infer(path, output, plan)
        code = 0
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--freeze-scores", type=Path)
    parser.add_argument("--decoded", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.freeze_scores and args.decoded is None:
        parser.error("--freeze-scores requires --decoded")
    if args.freeze or args.freeze_scores:
        torch.set_num_threads(2)
        freeze(
            args.freeze or args.freeze_scores,
            specification() if args.freeze else score_specification(args.decoded),
        )
    elif args.plan and args.output:
        run(args.plan, args.output, scoring=args.score)
    else:
        parser.error("provide a freeze target or --plan and --output")


if __name__ == "__main__":
    main()
