"""Freeze real training candidates and qualify Terra-only gradients through ROI vision."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from benchmark_fleurs_dev import sha256
from launch_terra_visual_job import identity, now, write
from probe_roi_seen_translation import BROAD_PLAN, BROAD_RESULT, fixed_checkpoint
from probe_tiny_rgb_capacity import compare, state_digest
from roi_appearance_model import RoiAppearance
from roi_appearance_protocol import ROOT
from roi_semantic_student import RoiSemanticStudent, clip_input, visual_gradients
from semantic_quality_judge import calibration_controls, parse_terra_scores
from terra_short_resume import RecoveryPacer
from terra_training_recovery import artifact, verify_snapshot
from terra_training_storage import storage
from tiny_rgb_capacity_protocol import LIMITS, verified_comparison
from tiny_rgb_full_run import runtime_binding
from train_roi_short_corpus import checked as checked_fit

POLICY = {
    "scope": "three complete training clips selected by source and duration before scoring",
    "objective": "Terra expected semantic error; no caption/feature/latent auxiliary loss",
    "risk_alpha": 0.005,
    "caption_weight": 0,
    "augmentation": "none; isolate direct semantic gradients before perturbations",
    "ROI_chunk_frames": 8,
    "ROI_activation_checkpointing": True,
    "trainable_parameters": 32826496,
    "trainable_tensors": 116,
    "candidate_generation": "human anchor, reference-blind greedy/sample and native C5 teacher",
    "candidate_horizon": 512,
    "teacher_scores": "actual Terra scores, including zero, never assumed correct",
    "judge": "gpt-5.6-terra",
    "reasoning_effort": "medium",
    "maximum_pairs_per_request": 16,
    "minimum_request_start_gap_seconds": 60,
    "automatic_retries": 0,
    "optimizer_updates": 0,
    "WER_is_diagnostic_only": True,
    "development_or_feature_diagnostic_inputs_used": False,
    "model_win_claimed": False,
    "promotion_authorized": False,
}


def training_records(data, indices, reserved):
    rows = [data["records"][i] for i in indices]
    if len(rows) != 3 or len(set(indices)) != 3:
        raise ValueError("ROI risk proof requires the three predeclared real training clips")
    for row in rows:
        source = row["original_row"]
        if (
            row["roi_role"] != "training"
            or row["roi_training_allowed"] is not True
            or source["role"] != "training"
            or source["training_allowed"] is not True
            or source["signer"] in reserved
        ):
            raise ValueError("ROI risk proof cannot use reserved or diagnostic identities")
    return rows


def specification():
    broad = checked_fit(BROAD_PLAN)
    data = json.loads((ROOT / broad["data"]["path"]).read_text())
    reserved = broad["reserved_STEM_participants"] + broad["ROI_diagnostic_participants"]
    rows = training_records(data, broad["proof_training_record_indices"], reserved)
    source = dict(broad["source_sha256"])
    for name in (
        "roi_semantic_student.py",
        "probe_roi_semantic_learning.py",
        "probe_roi_seen_translation.py",
        "evaluate_roi_short_translation.py",
        "c5_terra_tail.py",
        "c5_winner_language_decoder.py",
        "tiny_rgb_winner_student.py",
        "tiny_rgb_feature_winner_student.py",
        "terra_visual_learning.py",
        "terra_teacher_candidates.py",
        "terra_quality_training.py",
        "semantic_quality_judge.py",
        "probe_jeff_terra_quality.py",
        "terra_short_curriculum_learning.py",
        "terra_short_resume.py",
    ):
        source[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-roi-semantic-gradient-proof/1",
        "policy": POLICY,
        "limits": LIMITS,
        "records": rows,
        "record_indices": broad["proof_training_record_indices"],
        "ROI_initializer": fixed_checkpoint(BROAD_PLAN, BROAD_RESULT, (7872, 62920), broad),
        "data": broad["data"],
        "source_sha256": source,
        "runtime_binding": runtime_binding(),
        "resource_policy": broad["resource_policy"],
        "resource_policy_sha256": broad["resource_policy_sha256"],
        "reference": json.loads((ROOT / "C5_BASELINE_REFERENCE.json").read_text()),
        "initial_reference_hashes": json.loads(
            (ROOT / "runs/terra-c5-tail-training-v1/plan.json").read_text()
        )["initial_reference_hashes"],
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("ROI risk data/model/numerical policy or sources changed")
    verify_snapshot(path, plan)
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("ROI risk proof requires a fresh owned plan")
    torch.set_num_threads(2)
    plan = specification()
    storage()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(json.dumps({"plan": artifact(path), "cases": len(plan["records"])}), flush=True)


def fixed_hashes(model):
    return {
        "language": state_digest(model.language.state_dict()),
        "encoder": state_digest(model.semantic_encoder.state_dict()),
    }


def load_student(plan):
    from c5_terra_tail import load_model
    from prepare_roi_appearance import native_helpers
    from safetensors import safe_open
    from safetensors.torch import load_file
    from tiny_rgb_winner_student import load_frozen_winner_encoder
    from transformers import ByT5Tokenizer

    native_helpers()
    original, reference = load_model()
    if reference != plan["reference"]:
        raise ValueError("ROI risk loaded another C5 reference")
    encoder = load_frozen_winner_encoder(reference)
    descriptor = plan["ROI_initializer"]
    path = ROOT / descriptor["visual"]["path"]
    if artifact(path) != descriptor["visual"]:
        raise ValueError("ROI risk initializer bytes changed")
    with safe_open(path, framework="pt") as saved:
        meta = saved.metadata()
    if (
        meta["plan_sha256"] != descriptor["plan"]["sha256"]
        or meta["update"] != str(descriptor["frame_exposures"])
        or meta["optimizer_updates"] != str(descriptor["optimizer_updates"])
        or meta["fixed_winner_checkpoint"] != reference["translator_checkpoint_sha256"]
    ):
        raise ValueError("ROI risk initializer provenance changed")
    vision = RoiAppearance()
    vision.load_state_dict(load_file(path), strict=True)
    model = RoiSemanticStudent(vision, encoder, original.language).eval()
    if (
        fixed_hashes(model) != plan["initial_reference_hashes"]
        or len(model.trainable_parameters()) != POLICY["trainable_tensors"]
        or sum(p.numel() for p in model.trainable_parameters().values())
        != POLICY["trainable_parameters"]
    ):
        raise ValueError("ROI risk backbone values or optimizer scope differ")
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / reference["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )
    return model, tokenizer


def load_clip(record):
    path = ROOT / record["cache"]["path"]
    if artifact(path) != record["cache"]:
        raise ValueError("ROI risk real-frame cache changed")
    with np.load(path, allow_pickle=False) as saved:
        values = {name: saved[name].copy() for name in ("pixels", "pose_features")}
    clip = clip_input(values)
    if len(clip.pixels) != record["frames"]:
        raise ValueError("ROI risk cannot truncate a training clip")
    return clip, torch.ones(1, len(clip.pixels), dtype=torch.bool)


def comparison(records, pools):
    from c5_terra_tail import candidate_labels

    items, known, cases = [], {}, []
    if len(records) != len(pools):
        raise ValueError("ROI risk candidate coverage differs")
    for record, pool in zip(records, pools, strict=True):
        row = record["original_row"]
        if (
            record["roi_role"] != "training"
            or record["roi_training_allowed"] is not True
            or row["role"] != "training"
            or row["training_allowed"] is not True
        ):
            raise ValueError("ROI candidate rewards cannot use heldout or diagnostic clips")
        candidate_labels(pool)
        if pool[0]["kind"] != "human_anchor" or pool[0]["text"] != row["reference"]:
            raise ValueError("ROI risk must retain its exact human anchor")
        indices = []
        for candidate in pool:
            key = (row["reference"], candidate["text"])
            if key not in known:
                known[key] = len(items)
                items.append({"reference": key[0], "candidate": key[1]})
            indices.append(known[key])
        cases.append({"cache": record["cache"], "indices": indices})
    controls = []
    for control in calibration_controls():
        controls.append({"index": len(items), "range": control["range"]})
        items.append({"reference": control["reference"], "candidate": control["candidate"]})
    return {"items": items, "controls": controls, "cases": cases}


def qualities(paired, scores):
    if len(scores) != len(paired["items"]) or any(
        type(value) is not int or not 0 <= value <= 10 for value in scores
    ):
        raise ValueError("ROI risk requires complete actual0-10 Terra grades")
    if any(not c["range"][0] <= scores[c["index"]] <= c["range"][1] for c in paired["controls"]):
        raise ValueError("ROI training judge failed the original rubric controls")
    result = [[scores[i] for i in case["indices"]] for case in paired["cases"]]
    if any(q[0] != 10 or len(set(q)) < 2 for q in result):
        raise ValueError("ROI proof needs actual contrasting Terra scores and exact anchors")
    return result


def prepare(path, output, plan):
    from terra_teacher_candidates import TeacherTextCache, teacher_augmented_pool

    model, tokenizer = load_student(plan)
    initial = state_digest(model.vision.state_dict())
    cache = TeacherTextCache(path.parent / "teacher-text-cache")
    pools = []
    for index, record in enumerate(plan["records"]):
        clip, mask = load_clip(record)
        row = record["original_row"]
        pool = teacher_augmented_pool(
            model,
            tokenizer,
            clip,
            mask,
            row["reference"],
            pair=row["original_pair"],
            seed=20261016 + index,
            checkpoint_sha=plan["reference"]["translator_checkpoint_sha256"],
            cache=cache,
        )
        pools.append(pool)
        write(output / f"pool-{index:02d}.json", {"record": record, "candidates": pool})
        print(
            json.dumps({"stage": "real_training_pool", "case": index, "candidates": len(pool)}),
            flush=True,
        )
    paired = comparison(plan["records"], pools)
    # One calibrated pass supplies integer training rewards. Development selection
    # remains the separate, unchanged two-pass paired semantic gate.
    paired_output = output / "semantic"
    paired_output.mkdir()
    write(paired_output / "blinded-inputs.json", paired)
    scores = [None] * len(paired["items"])
    import random

    indices = list(range(len(scores)))
    random.Random(20261016).shuffle(indices)
    pacer = RecoveryPacer()
    for offset in range(0, len(indices), 16):
        chosen = indices[offset : offset + 16]
        response = pacer.score(
            [paired["items"][i] for i in chosen], paired_output / f"batch-{offset // 16:03d}"
        )
        for i, value in zip(chosen, response["scores"], strict=True):
            scores[i] = value
    q = qualities(paired, scores)
    if (
        state_digest(model.vision.state_dict()) != initial
        or fixed_hashes(model) != plan["initial_reference_hashes"]
    ):
        raise ValueError("ROI preparation changed fixed model weights")
    write(output / "scores.json", scores)
    checked(path)
    write(
        output / "candidates.json",
        {
            "complete": True,
            "plan_sha256": sha256(path),
            "pools": pools,
            "comparison": paired,
            "qualities": q,
            "scores": artifact(output / "scores.json"),
            "vision_initialization_sha256": initial,
            "optimizer_updates": 0,
            "new_Terra_requests": (len(scores) + 15) // 16,
            "calibration_passed": True,
            "training_only": True,
        },
    )
    print(
        json.dumps(
            {"stage": "actual_training_rewards_complete", "qualities": q, "optimizer_updates": 0}
        ),
        flush=True,
    )


def checked_candidates(path, plan_path, plan, tokenizer):
    from evaluate_roi_short_translation import terminal
    from probe_jeff_terra_quality import terra_payload

    terminal(path)
    result = json.loads(path.read_text())
    if (
        result.get("complete") is not True
        or result.get("plan_sha256") != sha256(plan_path)
        or artifact(ROOT / result["scores"]["path"]) != result["scores"]
    ):
        raise ValueError("ROI risk candidates are incomplete or changed")
    paired = comparison(plan["records"], result["pools"])
    if paired != result["comparison"]:
        raise ValueError("ROI risk changed candidate-to-identity mapping")
    for pool in result["pools"]:
        for candidate in pool:
            text = tokenizer.decode(
                candidate["tokens"], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if text != candidate["text"]:
                raise ValueError("ROI training text differs from its actual likelihood tokens")
    import random

    indices = list(range(len(paired["items"])))
    random.Random(20261016).shuffle(indices)
    actual = [None] * len(indices)
    for offset in range(0, len(indices), 16):
        chosen = indices[offset : offset + 16]
        directory = path.parent / "semantic" / f"batch-{offset // 16:03d}"
        request = json.loads((directory / "request.json").read_text())
        response = json.loads((directory / "response.json").read_text())
        scored = json.loads((directory / "scores.json").read_text())
        if (
            request != terra_payload([paired["items"][i] for i in chosen])
            or scored["response_sha256"] != sha256(directory / "response.json")
            or response.get("model") != POLICY["judge"]
            or response.get("reasoning", {}).get("effort") != "medium"
        ):
            raise ValueError("ROI risk rewards lack authentic model/prompt/response bindings")
        parsed = parse_terra_scores(response, [f"pair_{i:02d}" for i in range(len(chosen))])
        for position, i in enumerate(chosen):
            actual[i] = parsed[f"pair_{position:02d}"]
    if (
        actual != json.loads((path.parent / "scores.json").read_text())
        or qualities(paired, actual) != result["qualities"]
    ):
        raise ValueError("ROI risk grade mapping changed")
    return result


def qualify(path, output, plan, candidates):
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
    from c5_terra_tail import candidate_labels
    from terra_visual_learning import backward_candidate_risk

    budget = CertificationComputeWindow(ROOT / plan["resource_policy"])
    if (
        budget.duty_fraction() != 0.5
        or budget.cpu_threads() != 2
        or not torch.backends.mps.is_available()
    ):
        raise ValueError("ROI risk proof requires CPU2 and50% Metal duty")
    duty = WinnerDutyCycle(torch, budget)
    model, tokenizer = load_student(plan)
    initial = state_digest(model.vision.state_dict())
    judged = checked_candidates(candidates, path, plan, tokenizer)
    if initial != judged["vision_initialization_sha256"]:
        raise ValueError("ROI risk proof loaded another visual initialization")
    comparisons = []
    for index, (record, pool, quality) in enumerate(
        zip(plan["records"], judged["pools"], judged["qualities"], strict=True)
    ):
        clip, mask = load_clip(record)
        labels = candidate_labels(pool)
        model.zero_grad(set_to_none=True)
        cpu = backward_candidate_risk(
            model, clip, mask, labels, quality, alpha=POLICY["risk_alpha"], caption_weight=0
        )
        gradients = visual_gradients(model)
        model.zero_grad(set_to_none=True)
        duty.run(lambda: model.to("mps"))

        def metal(clip=clip, mask=mask, labels=labels, quality=quality):
            measured = backward_candidate_risk(
                model, clip, mask, labels, quality, alpha=POLICY["risk_alpha"], caption_weight=0
            )
            return measured, visual_gradients(model)

        measured, metal_gradients = duty.run(metal)
        result = compare(
            gradients,
            metal_gradients,
            cpu["loss"],
            measured["loss"],
            cpu["gold_logits"],
            measured["gold_logits"],
        )
        write(
            output / f"comparison-{index:02d}.json",
            {"cache": record["cache"], "qualities": quality, "comparison": result},
        )
        worst = verified_comparison(result, POLICY["trainable_tensors"])
        if (
            fixed_hashes(model) != plan["initial_reference_hashes"]
            or state_digest(model.vision.state_dict()) != initial
        ):
            raise ValueError("ROI risk zero-update proof changed model values")
        comparisons.append(
            {
                "record": record["cache"],
                "worst_relative_gradient_error": worst,
                "cpu_risk": cpu["loss"],
                "actual_Terra_qualities": quality,
            }
        )
        print(
            json.dumps({"stage": "actual_ROI_semantic_gradient", "case": index, **comparisons[-1]}),
            flush=True,
        )
        model.zero_grad(set_to_none=True)
        duty.run(lambda: model.cpu())
        del gradients, metal_gradients, cpu, measured
        gc.collect()
        torch.mps.empty_cache()
    checked(path)
    write(
        output / "qualification.json",
        {
            "complete": True,
            "numerics_passed": True,
            "plan_sha256": sha256(path),
            "candidates": artifact(candidates),
            "vision_initialization_sha256": initial,
            "trainable_tensors": POLICY["trainable_tensors"],
            "comparisons": comparisons,
            "optimizer_updates": 0,
            "new_Terra_requests": 0,
            "pure_Terra_objective": True,
            "all_fixed_weights_unchanged": True,
            "resources": duty.receipt(),
            "model_win_claimed": False,
        },
    )
    print(
        json.dumps(
            {
                "stage": "ROI_semantic_proof_complete",
                "cases": len(comparisons),
                "optimizer_updates": 0,
            }
        ),
        flush=True,
    )


def run(path, output, mode, candidates=None):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    plan = checked(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("ROI risk worker requires a fresh owned output")
    storage()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        if mode == "candidates":
            prepare(path, output, plan)
        else:
            from train_terra_c5_tail import gpu_lease

            if candidates is None:
                raise ValueError("ROI semantic proof requires completed actual scoring")
            with gpu_lease():
                qualify(path, output, plan, candidates.resolve())
        code = 0
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("candidates", "qualify"))
    parser.add_argument("--candidates", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    elif args.plan and args.output and args.mode:
        run(args.plan, args.output, args.mode, args.candidates)
    else:
        parser.error("choose --freeze, or --plan --output --mode")


if __name__ == "__main__":
    main()
