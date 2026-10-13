"""Evaluate an audited 83-point ByT5 pilot on public FSboard validation clips."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from benchmark_fleurs_dev import distance, miner_revision, sha256
from evaluate_fsboard_byt5 import dominant_hand, validation_rows
from evaluate_fsboard_coreml import comparable_units
from fsboard_arrow_training_data import AuditedTrainingShard
from fsboard_byt5_compact_model import FSboardByT5Compact
from fsboard_paper_like_features import paper_like_features
from train_fsboard_byt5_smoke import (
    PRETRAINED_CONFIG_SHA256, PRETRAINED_WEIGHTS_SHA256, eligible_positions,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, action="append", required=True)
    parser.add_argument("--landmarks-dir", type=Path, action="append", required=True)
    parser.add_argument("--landmark-receipts", type=Path, action="append", required=True)
    parser.add_argument("--training-index", type=Path, required=True)
    parser.add_argument("--training-shard", type=Path, required=True)
    parser.add_argument("--training-audit", type=Path, required=True)
    parser.add_argument("--training-steps", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.results.resolve().is_relative_to(ROOT / "runs")
        or not args.summary.resolve().is_relative_to(ROOT / "runs")
        or args.results.exists()
        or args.summary.exists()
    ):
        parser.error("new result and summary paths must be under runs/")
    training = json.loads(args.training_result.read_text())
    if (
        training.get("schema") != "umi-studio-fsboard-byt5-compact-pilot-training/1"
        or training.get("validation_or_test_used") is not False
        or training.get("checkpoint_sha256") != sha256(args.checkpoint)
        or training.get("training_index_sha256") != sha256(args.training_index)
        or training.get("source_shard_sha256") != sha256(args.training_shard)
        or training.get("source_audit_sha256") != sha256(args.training_audit)
        or training.get("steps_sha256") != sha256(args.training_steps)
        or training.get("trainer_sha256") != sha256(ROOT / "tools/train_fsboard_byt5_compact_pilot.py")
        or training.get("model_code_sha256") != sha256(ROOT / "tools/fsboard_byt5_compact_model.py")
        or training.get("features_code_sha256") != sha256(ROOT / "tools/fsboard_paper_like_features.py")
        or sha256(args.pretrained / "pytorch_model.bin") != PRETRAINED_WEIGHTS_SHA256
        or sha256(args.pretrained / "config.json") != PRETRAINED_CONFIG_SHA256
    ):
        raise ValueError("candidate checkpoint or training ledger differs")
    loader = AuditedTrainingShard(
        args.training_shard, args.training_index, args.training_audit
    )
    positions = [
        position for position in eligible_positions(loader.rows, training["seed"])
        if loader.rows[position]["frames"] <= 512
    ]
    steps = [json.loads(line) for line in args.training_steps.read_text().splitlines()]
    if len(steps) != len(positions) or training.get("eligible_examples") != len(steps):
        raise ValueError("candidate training step count differs from audited schedule")
    for number, (step, position) in enumerate(zip(steps, positions, strict=True), start=1):
        row = loader.rows[position]
        if (
            step.get("step") != number
            or step.get("clip_filename") != row["clip_filename"]
            or step.get("signer_id") != row["signer_id"]
            or step.get("source_row") != row["shard_row"]
        ):
            raise ValueError("candidate steps differ from audited schedule")
    sources = validation_rows(
        args.selection, args.landmarks_dir, args.landmark_receipts, args.training_index
    )
    torch.set_num_threads(2)
    revision = miner_revision(args.miner_health_url, None)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, local_files_only=True)
    model = FSboardByT5Compact(T5ForConditionalGeneration.from_pretrained(
        args.pretrained, local_files_only=True, use_safetensors=False
    ))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    revision = miner_revision(args.miner_health_url, revision)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    partial = args.results.with_name(args.results.name + ".partial")
    if partial.exists():
        raise ValueError("unfinished evaluation results exist")
    errors = []
    exact = 0
    errors_by_hand: dict[str, list[float]] = {"left": [], "right": [], "tie": []}
    with partial.open("x", encoding="utf-8") as stream, torch.inference_mode():
        for number, (row, source, receipt, selection_hash) in enumerate(sources, start=1):
            revision = miner_revision(args.miner_health_url, revision)
            with np.load(source, allow_pickle=False) as payload:
                landmarks = payload["landmarks"]
            features = paper_like_features(
                landmarks, frame_stride=2, max_frames=256, normalization="uniform"
            )
            tokens = model.language_model.generate(
                inputs_embeds=model.projection(torch.from_numpy(features)[None]),
                attention_mask=torch.ones((1, len(features)), dtype=torch.long),
                max_new_tokens=64,
                num_beams=1,
            )
            hypothesis = tokenizer.decode(tokens[0], skip_special_tokens=True)
            reference_units = comparable_units(row["reference"])
            hypothesis_units = comparable_units(hypothesis)
            cer = distance(reference_units, hypothesis_units) / max(1, len(reference_units))
            hand = dominant_hand(receipt)
            revision = miner_revision(args.miner_health_url, revision)
            errors.append(cer)
            errors_by_hand[hand].append(cer)
            exact += reference_units == hypothesis_units
            stream.write(json.dumps({
                "schema": "umi-studio-fsboard-byt5-validation/1",
                "clip_filename": row["clip_filename"],
                "signer_id": row["signer_id"],
                "reference": row["reference"],
                "hypothesis": hypothesis,
                "input_frames": len(features),
                "generated_tokens": int(tokens.shape[1]),
                "single_reference_cer": cer,
                "hand_dominance": hand,
                "selection_sha256": selection_hash,
                "landmarks_sha256": receipt["sha256"],
                "checkpoint_sha256": training["checkpoint_sha256"],
                "miner_model_revision": revision,
            }, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            if number % 10 == 0:
                print(f"evaluated {number}/{len(sources)}", flush=True)
    os.replace(partial, args.results)
    summary = {
        "schema": "umi-studio-fsboard-byt5-validation-summary/1",
        "split": "FSboard daun_v3 validation, 60 public development clips",
        "test_used": False,
        "single_reference_mean_cer": sum(errors) / len(errors),
        "exact_transcriptions": exact,
        "clips": len(errors),
        "hand_dominance": {
            hand: {"clips": len(values), "mean_cer": sum(values) / len(values) if values else None}
            for hand, values in errors_by_hand.items()
        },
        "results_sha256": sha256(args.results),
        "checkpoint_sha256": training["checkpoint_sha256"],
        "training_result_sha256": sha256(args.training_result),
        "evaluation_script_sha256": sha256(Path(__file__)),
        "miner_model_revision": revision,
        "live_miner_changed": False,
    }
    with args.summary.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
