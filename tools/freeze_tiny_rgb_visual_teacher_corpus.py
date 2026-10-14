"""Audit every duration-vetted raw/teacher pair before broad representation learning."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "runs/tiny-rgb-visual-teacher-pilot-v1/plan.json"
PILOT_OUTPUT = ROOT / "checkpoints/tiny-rgb-visual-teacher-pilot-v1"
SCHEMA = "umi-studio-raw-rgb-feature-teacher-corpus/1"


def usable_feature_signal(targets):
    import numpy as np

    return any(
        np.any(np.square(targets[name]).sum(axis=-1) > 1e-12)
        for name in ("face_features", "left_hand_features", "right_hand_features")
    )


def audit(output):
    from audit_tiny_rgb_duration import checked_duration_audit
    from benchmark_fleurs_dev import sha256
    from c5_experiment_sources import archive_sources
    from c5_full_coverage_mps import load_training
    from freeze_visual_lora_plan import SOURCE_LAYOUT
    from tiny_rgb_visual_teacher_data import (
        CORPUS,
        DURATION_AUDIT,
        PARENT,
        bind_pair,
        canonical_input,
        load_targets,
        read_rows,
    )
    from train_tiny_rgb_capacity_cpu import storage
    from train_tiny_rgb_visual_teacher import checked_plan, sources

    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("full feature-teacher audit requires a new runs/ directory")
    storage()
    pilot = checked_plan(PILOT)
    decision_path = PILOT_OUTPUT / "decision.json"
    proof_path = PILOT_OUTPUT / "numerical-receipt.json"
    decision, proof = (json.loads(path.read_text()) for path in (decision_path, proof_path))
    if (
        decision["plan_sha256"] != sha256(PILOT)
        or decision["complete"] is not True
        or decision["passes_feature_learning_gate"] is not True
        or proof["numerics_passed"] is not True
        or proof["plan_sha256"] != sha256(PILOT)
        or proof["real_cases"] != 6
        or proof["parameter_tensors"] != 50
        or proof["caption_or_language_updates"] != 0
        or proof["optimizer_updates"] != 0
        or decision["held_out_quality_verified"] is not False
    ):
        raise ValueError("full feature-teacher audit requires the unchanged passing pilot")
    duration, rows = checked_duration_audit(DURATION_AUDIT)
    parent = json.loads(PARENT.read_text())
    headers = {
        (r["source"], r["example_id"]): r
        for r in read_rows(ROOT / parent["headers"])
        if r["role"] == "training"
    }
    approved = {
        (r["source"], r["id"]): r
        for r in load_training(CORPUS / "audit.json", CORPUS / "training-streams.jsonl")
    }
    indexes = {source: layout["index"] for source, layout in SOURCE_LAYOUT.items()}
    indexes.update(
        {
            "how2sign_precut_expansion": (
                "runs/how2sign-precut-expansion-v1/stream-cache-stride2-index.jsonl"
            ),
            "asl_stem_original": "runs/asl-stem-wiki-v1/stream-cache-stride2.jsonl",
            "asl_stem_supplemental": "runs/asl-stem-wiki-scale-v1/stream-cache-stride2.jsonl",
            "openasl_scale": "runs/openasl-scale-v1/stream-cache-stride2.jsonl",
        }
    )
    indexed = {
        source: {r.get("example_id", r.get("sentence_name")): r for r in read_rows(ROOT / name)}
        for source, name in indexes.items()
    }
    sidecars = {
        source: {r["sentence_name"]: r for r in read_rows(ROOT / layout["sidecar_index"])}
        for source, layout in SOURCE_LAYOUT.items()
        if "sidecar_index" in layout
    }
    dependencies = dict(pilot["data"]["dependency_sha256"])
    for path in (PILOT, decision_path, proof_path):
        dependencies[str(path.relative_to(ROOT))] = sha256(path)
    dependencies.update({name: sha256(ROOT / name) for name in indexes.values()})
    for layout in SOURCE_LAYOUT.values():
        if layout.get("sidecar_index"):
            dependencies[layout["sidecar_index"]] = sha256(ROOT / layout["sidecar_index"])
    output.mkdir(parents=True)
    pairs_path = output / "native-training-pairs.jsonl"
    no_signal = []
    counts, native_counts = Counter(), Counter()
    rebound = 0
    max_frames, maximum = 0, None
    duration_digest = sha256(DURATION_AUDIT)
    with pairs_path.open("x") as stream:
        for position, raw in enumerate(rows, 1):
            identity = raw["source"], raw["example_id"]
            teacher = approved[identity]
            receipt = indexed[identity[0]][identity[1]]
            canonical, header = canonical_input(raw, headers[identity], receipt)
            pair = bind_pair(canonical, header, teacher, receipt, duration_digest)
            pair["original_raw_binding"] = raw
            pair["native_receipt_index"] = indexes[identity[0]]
            for name, digest in (
                (raw["video"], raw["video_sha256"]),
                (canonical["video"], canonical["video_sha256"]),
                (teacher["cache"], teacher["cache_sha256"]),
                (teacher.get("sidecar"), teacher.get("sidecar_sha256")),
            ):
                if name:
                    path = (ROOT / name).resolve()
                    if not path.is_relative_to(ROOT / "data") or sha256(path) != digest:
                        raise ValueError(f"full teacher bytes changed: {identity}")
                    dependencies[name] = digest
            if teacher.get("sidecar"):
                record = sidecars[identity[0]][identity[1]]
                if (
                    record["resampled_clip_sha256"] != canonical["video_sha256"]
                    or record["encoder_cache_sha256"] != teacher["cache_sha256"]
                    or record["stream_sha256"] != teacher["sidecar_sha256"]
                    or record["original_index_sha256"] != dependencies[indexes[identity[0]]]
                ):
                    raise ValueError(f"full native sidecar provenance differs: {identity}")
            targets = load_targets(pair)
            signal = usable_feature_signal(targets)
            pair["feature_pretraining_allowed"] = signal
            if not signal:
                # This is target availability, never a model score/gradient exclusion.
                # Original caption-training eligibility remains unchanged.
                no_signal.append(
                    {
                        "source": identity[0],
                        "example_id": identity[1],
                        "reason": "all_hand_face_feature_targets_zero",
                        "caption_training_allowed": True,
                    }
                )
            counts[identity[0]] += 1
            native_counts[canonical["video_fps"]] += 1
            rebound += canonical != raw
            if header["sampled_frames"] > max_frames:
                max_frames, maximum = header["sampled_frames"], list(identity)
            stream.write(json.dumps(pair, sort_keys=True) + "\n")
            if position % 256 == 0:
                stream.flush()
                storage()
                print(
                    json.dumps(
                        {
                            "native_pairs_checked": position,
                            "rebound_inputs": rebound,
                            "no_feature_signal": len(no_signal),
                        }
                    ),
                    flush=True,
                )
        stream.flush()
        os.fsync(stream.fileno())
    if sum(counts.values()) != duration["valid_training_clips"] or max_frames > 512:
        raise ValueError(
            "full teacher audit count/frame bounds differ from complete eligible corpus"
        )
    dependencies[str(pairs_path.relative_to(ROOT))] = sha256(pairs_path)
    plan = {
        "schema": SCHEMA,
        "planning_only": True,
        "eligible_caption_training_clips": sum(counts.values()),
        "feature_pretraining_clips": sum(counts.values()) - len(no_signal),
        "by_source": dict(counts),
        "canonical_input_fps_counts": dict(native_counts),
        "canonical25fps_rebound_inputs": rebound,
        "no_feature_signal": no_signal,
        "maximum_sampled_frames": max_frames,
        "maximum_frame_identity": maximum,
        "native_pairs": str(pairs_path.relative_to(ROOT)),
        "native_pairs_sha256": sha256(pairs_path),
        "dependency_sha256": dependencies,
        "pilot_plan_sha256": sha256(PILOT),
        "pilot_decision_sha256": sha256(decision_path),
        "source_sha256": {
            **sources(),
            "tools/freeze_tiny_rgb_visual_teacher_corpus.py": sha256(Path(__file__)),
        },
        "initial_diagnostic_weights_reused": False,
        "held_out_quality_verified": False,
        "baseline_beaten": False,
        "miner_modified": False,
        "qualification": "Native bytes/frame/protected-role provenance and feature availability "
        "only. Caption eligibility unchanged. A separately frozen full trainer, updated real "
        "extrema numerical proof and fresh initialization remain required before updates.",
    }
    plan_path = output / "plan.json"
    storage(len(json.dumps(plan).encode()) + 1024**2)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    archive_sources(plan_path, output / "source-snapshot")
    for name, digest in dependencies.items():
        if sha256(ROOT / name) != digest:
            raise ValueError("full teacher dependency changed during audit")
    print(
        json.dumps(
            {
                "complete": True,
                "plan_sha256": sha256(plan_path),
                "caption_training_clips": plan["eligible_caption_training_clips"],
                "feature_pretraining_clips": plan["feature_pretraining_clips"],
                "rebound_inputs": rebound,
                "maximum_sampled_frames": max_frames,
            }
        ),
        flush=True,
    )


def main():
    for name, value in {
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items():
        os.environ[name] = value
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    audit(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
