"""Decode fixed ROI vision through released C5 backbones on reserved development."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from benchmark_fleurs_dev import distance, normalize, sha256
from launch_terra_visual_job import identity, now, write
from prepare_roi_appearance import ROOT, SampledRgbReader, native_helpers, rgb_crops
from prepare_roi_short_corpus import LAYOUT, NativeSampledReader
from prepare_terra_short_curriculum import checked as checked_curriculum
from probe_tiny_rgb_capacity import state_digest
from roi_appearance_model import STREAMS, RoiAppearance, pixel_tensor
from terra_short_curriculum_learning import comparison_items
from terra_training_recovery import artifact, verify_snapshot
from terra_training_storage import storage
from tiny_rgb_visual_teacher_data import load_targets
from train_roi_short_corpus import checked as checked_fit

CURRICULUM = ROOT / "runs/terra-short-curriculum-v1/plan.json"
CONTROL = ROOT / "runs/terra-short-curriculum-C5-control-v1/receipt.json"
FIT = ROOT / "runs/roi-short-corpus-training-v1/plan.json"
FITTED = ROOT / "checkpoints/roi-short-corpus-training-v1/decision.json"
POLICY = {
    "all34_original_development_identities": True,
    "reproduce_all34_C5_controls_before_candidate_decode": True,
    "generation": "same published C5 settings and existing CpuBudget, no reduced horizon",
    "vision": "fixed final epoch8 ROI checkpoint, no earlier checkpoint selection",
    "ROI_patch_side": 64,
    "appearance_batch_frames": 8,
    "pose": "original video-derived14D native stream; no caption-conditioned inputs",
    "frame_coverage": "complete original stride2 timeline with independent native crop parity",
    "backbones": "released C5 semantic encoder and language, frozen on CPU",
    "CPU_threads": 2,
    "GPU_duty_fraction": 0.5,
    "optimizer_updates": 0,
    "new_Terra_requests": 0,
    "new_downloads": 0,
    "WER_is_diagnostic_only": True,
    "promotion_authorized": False,
}


def reserved_rows(curriculum):
    rows = [r for r in curriculum["records"] if r["role"] == "development"]
    if (
        len(rows) != 34
        or len({(r["source"], r["example_id"]) for r in rows}) != 34
        or any(r["training_allowed"] is not False or r["source"] not in LAYOUT for r in rows)
        or any(r["signer"] not in curriculum["reserved_STEM_participants"] for r in rows)
    ):
        raise ValueError("ROI translation cannot replace/drop/repurpose reserved development")
    return rows


def terminal(path):
    result = json.loads(path.with_name("exit.json").read_text())
    worker = json.loads(path.with_name("worker.json").read_text())
    if result.get("exit_code") != 0 or result.get("pid") != worker.get("pid"):
        raise ValueError("ROI translation input lacks a complete same-worker exit0")


def specification():
    fit, curriculum = checked_fit(FIT), checked_curriculum(CURRICULUM)
    result, control = json.loads(FITTED.read_text()), json.loads(CONTROL.read_text())
    terminal(FITTED)
    terminal(CONTROL)
    if (
        result.get("complete") is not True
        or result.get("plan_sha256") != sha256(FIT)
        or result.get("optimizer_updates") != fit["planned_optimizer_updates"]
        or result.get("frame_exposures") != fit["planned_frame_exposures"]
        or result.get("translation_quality_evaluated") is not False
        or result.get("promotion_authorized") is not False
        or control.get("complete") is not True
        or control.get("plan_sha256") != sha256(CURRICULUM)
        or control.get("all_weights_unchanged") is not True
        or control.get("optimizer_updates") != 0
        or control.get("original_C5_generation") != curriculum["reference"]["generation"]
    ):
        raise ValueError("ROI translation requires fixed final vision and unchanged C5 control")
    visual = ROOT / result["visual"]["path"]
    commit = json.loads(visual.with_suffix(".json").read_text())
    if (
        artifact(visual) != result["visual"]
        or commit.get("visual") != result["visual"]
        or commit.get("complete") is not True
    ):
        raise ValueError("ROI final visual checkpoint commit/hash differs")
    rows = reserved_rows(curriculum)
    comparison_items(rows, control["predictions"], control["predictions"])
    assets, indexed, records = {}, {}, []
    baseline_reader = ROOT / "artifacts/baseline-6fe8df59/vendor/demo/video_reader.py"
    winner_reader = (
        ROOT / curriculum["reference"]["bundle_root"] / "model/vendor/demo/video_reader.py"
    )
    if sha256(baseline_reader) != sha256(winner_reader):
        raise ValueError("ROI/C5 shared native video readers are not byte-identical")
    for reader_path in (baseline_reader, winner_reader):
        assets[str(reader_path.relative_to(ROOT))] = sha256(reader_path)
    for source, (directory, _) in LAYOUT.items():
        indexed[source] = []
        for name in ("landmark-receipts.jsonl", "stream-cache-stride2.jsonl"):
            path = ROOT / "runs" / directory / name
            assets[str(path.relative_to(ROOT))] = sha256(path)
            indexed[source].append(
                {r["example_id"]: r for r in map(json.loads, path.read_text().splitlines())}
            )
    for row in rows:
        source, ident, pair = row["source"], row["example_id"], row["original_pair"]
        lm, stream = (index[ident] for index in indexed[source])
        run_dir, landmark_dir = LAYOUT[source]
        path = ROOT / landmark_dir / f"{ident}.npz"
        if (
            lm["clip_sha256"] != pair["raw"]["video_sha256"]
            or lm["frames"] != pair["header"]["source_frames"]
            or stream["landmarks_sha256"] != lm["landmarks_sha256"]
            or stream["landmark_receipts_sha256"]
            != assets[f"runs/{run_dir}/landmark-receipts.jsonl"]
            or stream["encoder_cache_sha256"] != pair["teacher"]["cache_sha256"]
            or sha256(path) != lm["landmarks_sha256"]
        ):
            raise ValueError("ROI development landmarks/raw/native target receipts differ")
        records.append(
            {
                "development": row,
                "landmarks": {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": lm["landmarks_sha256"],
                },
            }
        )
    sources = dict(fit["source_sha256"])
    for name in (
        "evaluate_roi_short_translation.py",
        "c5_terra_tail.py",
        "c5_winner_language_decoder.py",
        "tiny_rgb_winner_student.py",
        "evaluate_tiny_rgb_feature_winner.py",
        "terra_short_curriculum_learning.py",
    ):
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-roi-short-translation-evaluation/1",
        "policy": POLICY,
        "fit_plan": artifact(FIT),
        "fit_decision": artifact(FITTED),
        "visual": artifact(visual),
        "visual_commit": artifact(visual.with_suffix(".json")),
        "curriculum": artifact(CURRICULUM),
        "control": artifact(CONTROL),
        "records": records,
        "reference": curriculum["reference"],
        "source_sha256": sources,
        "assets": assets,
        "resource_policy": fit["resource_policy"],
        "resource_policy_sha256": fit["resource_policy_sha256"],
        "visual_parameters": fit["visual_parameters"],
        "initial_reference_hashes": json.loads(
            (ROOT / "runs/terra-c5-tail-training-v1/plan.json").read_text()
        )["initial_reference_hashes"],
        "C5_upstream_training_inventory_known": False,
        "promotion_authorized": False,
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("ROI translation fixed model/data/control/implementation changed")
    verify_snapshot(path, plan)
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("ROI translation needs a fresh owned plan")
    torch.set_num_threads(2)
    storage()
    plan = specification()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(
        json.dumps({"plan": artifact(path), "development_clips": len(plan["records"])}), flush=True
    )


def crops(record):
    from extract_mediapipe_landmarks import unpack_landmarks

    pair = record["development"]["original_pair"]
    header, video = pair["header"], ROOT / pair["raw"]["video"]
    landmarks, metadata = unpack_landmarks(ROOT / record["landmarks"]["path"])
    if (
        metadata["source_sha256"] != pair["raw"]["video_sha256"]
        or metadata["frames"] != header["source_frames"]
        or metadata["fps"] != 25
    ):
        raise ValueError("ROI development landmark source timeline changed")
    sampled = {j: landmarks[i] for j, i in enumerate(range(0, header["source_frames"], 2))}
    if any(
        not np.isfinite(np.asarray(points)).all()
        for value in sampled.values()
        if value is not None
        for points in value.values()
        if points is not None
    ):
        raise ValueError("ROI development has nonfinite active coordinates")
    hands, face, native_reader = native_helpers()
    with SampledRgbReader(video, header) as reader:
        left, right = hands(output_size=(64, 64)).extract_hand_frames(reader, sampled)
    with SampledRgbReader(video, header) as reader:
        facial = face(output_size=(64, 64)).extract_face_frames(reader, sampled)
    count = header["sampled_frames"]
    pixels = np.stack(
        [rgb_crops(facial, count), rgb_crops(left, count), rgb_crops(right, count)], axis=1
    )
    del left, right, facial
    reader = native_reader(video)
    try:
        left, right = hands(output_size=(64, 64)).extract_hand_frames(
            NativeSampledReader(reader, header), sampled
        )
        facial = face(output_size=(64, 64)).extract_face_frames(
            NativeSampledReader(reader, header), sampled
        )
    finally:
        del reader
    reference = np.stack(
        [rgb_crops(facial, count), rgb_crops(left, count), rgb_crops(right, count)], axis=1
    )
    if not np.array_equal(pixels, reference):
        raise ValueError("ROI development stream/native crop parity failed")
    return pixels


def predicted_streams(appearance, pixels, original, duty):
    if set(original) != {*STREAMS, "pose_features"} or original["pose_features"].shape != (
        len(pixels),
        14,
    ):
        raise ValueError("ROI inference cannot invent/drop pose or time coverage")
    parts = {name: [] for name in STREAMS}
    for start in range(0, len(pixels), 8):
        batch = pixel_tensor(pixels[start : start + 8])

        def forward(batch=batch):
            return {n: v.detach().cpu() for n, v in appearance(batch.to("mps")).items()}

        for name, value in duty.run(forward).items():
            parts[name].append(value)
    features = {n: torch.cat(v) for n, v in parts.items()}
    if any(v.shape != (len(pixels), 384) or not torch.isfinite(v).all() for v in features.values()):
        raise ValueError("ROI prediction shape/finite coverage differs")
    features["pose_features"] = torch.from_numpy(original["pose_features"].copy())
    return features


def decode(language, tokenizer, generation, memory):
    from evaluate_tiny_rgb_feature_winner import CpuBudget
    from transformers import StoppingCriteriaList
    from transformers.modeling_outputs import BaseModelOutput

    tokens = language.generate(
        encoder_outputs=BaseModelOutput(last_hidden_state=memory),
        **generation,
        early_stopping=True,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([CpuBudget()]),
    )
    return tokenizer.decode(tokens[0], skip_special_tokens=True), tokens[0].tolist()


def require_control(actual, expected):
    if actual != expected:
        raise ValueError("native-feature route did not reproduce frozen C5 control text")


def prediction(row, hypothesis):
    result = {
        k: row[k]
        for k in (
            "source",
            "example_id",
            "source_group",
            "signer",
            "reference",
            "role",
            "training_allowed",
        )
    }
    words = len(normalize(row["reference"]))
    edits = distance(normalize(row["reference"]), normalize(hypothesis))
    return {
        **result,
        "hypothesis": hypothesis,
        "reference_words": words,
        "edits": edits,
        "WER_diagnostic": edits / words,
    }


def evaluate(path, output, plan):
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
    from c5_terra_tail import load_model, native_memory
    from safetensors import safe_open
    from safetensors.torch import load_file
    from tiny_rgb_winner_student import load_frozen_winner_encoder
    from transformers import ByT5Tokenizer

    # Preload the pinned reader before C5 imports its byte-identical video_reader.
    # Otherwise Python's global module cache could bind the cropper to another path.
    native_helpers()
    budget = CertificationComputeWindow(ROOT / plan["resource_policy"])
    if (
        budget.duty_fraction() != 0.5
        or budget.cpu_threads() != 2
        or not torch.backends.mps.is_available()
    ):
        raise ValueError("ROI evaluation requires50% Metal duty/CPU2")
    duty = WinnerDutyCycle(torch, budget)
    model, reference = load_model()
    model.requires_grad_(False).eval()
    encoder = load_frozen_winner_encoder(reference)
    if reference != plan["reference"]:
        raise ValueError("ROI released language reference differs")
    before = {
        "language": state_digest(model.language.state_dict()),
        "encoder": state_digest(encoder.state_dict()),
    }
    if before != plan["initial_reference_hashes"]:
        raise ValueError("ROI evaluation did not load original C5 backbones")
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / reference["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )
    appearance = RoiAppearance().eval()
    visual = ROOT / plan["visual"]["path"]
    with safe_open(visual, framework="pt") as file:
        metadata = file.metadata()
    if (
        metadata["plan_sha256"] != plan["fit_plan"]["sha256"]
        or metadata["update"] != "62920"
        or metadata["optimizer_updates"] != "7872"
        or metadata["fixed_winner_checkpoint"] != reference["translator_checkpoint_sha256"]
    ):
        raise ValueError("ROI inference weight metadata differs from fixed final")
    appearance.load_state_dict(load_file(visual), strict=True)
    appearance.requires_grad_(False)
    visual_state = state_digest(appearance.state_dict())
    appearance = duty.run(lambda: appearance.to("mps"))
    if state_digest(appearance.state_dict()) != visual_state:
        raise ValueError("ROI inference transfer changed fixed visual weights")
    baseline = json.loads(CONTROL.read_text())["predictions"]
    controls = {(r["source"], r["example_id"]): r for r in baseline}
    with torch.no_grad():
        for i, record in enumerate(plan["records"]):
            storage(output=output)
            row = record["development"]
            memory = native_memory(
                encoder, row["original_pair"], reference["translator_checkpoint_sha256"]
            )
            hypothesis, tokens = decode(model.language, tokenizer, reference["generation"], memory)
            expected = controls[(row["source"], row["example_id"])]["hypothesis"]
            write(
                output / f"control-{i:04d}.json",
                {
                    **prediction(row, hypothesis),
                    "tokens": tokens,
                    "matches_frozen_control": hypothesis == expected,
                },
            )
            require_control(hypothesis, expected)
            print(json.dumps({"stage": "C5_control_reproduced", "clips": i + 1}), flush=True)
        predictions = []
        for i, record in enumerate(plan["records"]):
            storage(output=output)
            row = record["development"]
            pixels = crops(record)
            features = predicted_streams(
                appearance, pixels, load_targets(row["original_pair"]), duty
            )
            memory = encoder(
                **{n: v[None] for n, v in features.items()}, return_dict=True
            ).last_hidden_state
            if memory.shape != (1, len(pixels), 1536) or not torch.isfinite(memory).all():
                raise ValueError("ROI candidate semantic memory lost complete frame coverage")
            hypothesis, tokens = decode(model.language, tokenizer, reference["generation"], memory)
            result = prediction(row, hypothesis)
            predictions.append(result)
            write(
                output / f"candidate-{i:04d}.json",
                {
                    **result,
                    "tokens": tokens,
                    "native_crop_parity": True,
                    "predicted_streams_sha256": state_digest(features),
                    "semantic_memory_sha256": state_digest({"memory": memory}),
                },
            )
            print(json.dumps({"stage": "ROI_translation_decoded", "clips": i + 1}), flush=True)
    if state_digest(appearance.state_dict()) != visual_state or before != {
        "language": state_digest(model.language.state_dict()),
        "encoder": state_digest(encoder.state_dict()),
    }:
        raise ValueError("ROI inference changed a fixed model value")
    comparison = comparison_items(
        [r["development"] for r in plan["records"]], baseline, predictions
    )
    checked(path)
    write(
        output / "receipt.json",
        {
            "complete": True,
            "plan_sha256": sha256(path),
            "clips": 34,
            "all34_frozen_C5_controls_reproduced": True,
            "all_fixed_weights_unchanged": True,
            "predictions": predictions,
            "comparison": comparison,
            "mean_WER_diagnostic": sum(r["WER_diagnostic"] for r in predictions) / len(predictions),
            "semantic_scores_pending": True,
            "optimizer_updates": 0,
            "new_Terra_requests": 0,
            "promotion_authorized": False,
            "resources": duty.receipt(),
        },
    )


def run(path, output):
    from train_terra_c5_tail import gpu_lease

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;2")
    if os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] != "threads;2":
        raise ValueError("ROI inference requires two-thread FFmpeg")
    import cv2

    cv2.setNumThreads(2)
    plan = checked(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("ROI translation evaluation requires a fresh owned output")
    storage()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        with gpu_lease():
            evaluate(path, output, plan)
        code = 0
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
