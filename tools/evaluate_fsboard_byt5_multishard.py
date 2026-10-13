"""Evaluate an audited multishard ByT5 run on 60 public FSboard validation clips."""

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
from fsboard_byt5_model import FSboardByT5, normalized_features
from train_fsboard_byt5_multishard import schedule
from train_fsboard_byt5_smoke import PRETRAINED_CONFIG_SHA256, PRETRAINED_WEIGHTS_SHA256

ROOT = Path(__file__).resolve().parents[1]


def verify_training(
    plan_path: Path, result_path: Path, steps_path: Path, checkpoint: Path,
    pretrained: Path, shards: list[Path], indexes: list[Path], audits: list[Path],
) -> tuple[list[AuditedTrainingShard], dict]:
    plan = json.loads(plan_path.read_text())
    result = json.loads(result_path.read_text())
    if (
        plan.get("schema") != "umi-studio-fsboard-byt5-multishard-plan/1"
        or plan.get("validation_or_test_used") is not False
        or plan.get("trainer_sha256") != sha256(ROOT / "tools/train_fsboard_byt5_multishard.py")
        or result.get("schema") != "umi-studio-fsboard-byt5-multishard-training/1"
        or result.get("validation_or_test_used") is not False
        or result.get("plan_sha256") != sha256(plan_path)
        or result.get("steps_sha256") != sha256(steps_path)
        or result.get("checkpoint_sha256") != sha256(checkpoint)
        or sha256(pretrained / "pytorch_model.bin") != PRETRAINED_WEIGHTS_SHA256
        or sha256(pretrained / "config.json") != PRETRAINED_CONFIG_SHA256
        or not len(shards) == len(indexes) == len(audits) == len(plan["shards"])
    ):
        raise ValueError("candidate training ledger, checkpoint or pretrained source differs")
    loaders = []
    for identity, shard, index, audit in zip(
        plan["shards"], shards, indexes, audits, strict=True
    ):
        if (
            identity.get("shard_name") != shard.name
            or identity.get("shard_sha256") != sha256(shard)
            or identity.get("index_sha256") != sha256(index)
            or identity.get("audit_sha256") != sha256(audit)
        ):
            raise ValueError("training shard differs from frozen multishard plan")
        loaders.append(AuditedTrainingShard(shard, index, audit))
    expected = schedule(loaders, plan["seed"])
    steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
    if len(expected) != len(steps) or len(steps) != result.get("eligible_examples"):
        raise ValueError("training step count differs from audited schedule")
    for number, ((shard_number, position), step) in enumerate(
        zip(expected, steps, strict=True), start=1
    ):
        row = loaders[shard_number].rows[position]
        if (
            step.get("step") != number
            or step.get("shard_number") != shard_number
            or step.get("clip_filename") != row["clip_filename"]
            or step.get("signer_id") != row["signer_id"]
            or step.get("source_row") != row["shard_row"]
        ):
            raise ValueError("training steps differ from audited schedule")
    return loaders, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, action="append", required=True)
    parser.add_argument("--landmarks-dir", type=Path, action="append", required=True)
    parser.add_argument("--landmark-receipts", type=Path, action="append", required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--training-steps", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
        parser.error("new evaluation outputs must be under runs/")
    loaders, training = verify_training(
        args.training_plan, args.training_result, args.training_steps, args.checkpoint,
        args.pretrained, args.shard, args.index, args.audit,
    )
    sources = validation_rows(
        args.selection, args.landmarks_dir, args.landmark_receipts, args.index[0]
    )
    all_rows = [row for loader in loaders for row in loader.rows]
    signers = {row["signer_id"] for row in all_rows}
    references = {row["reference"] for row in all_rows}
    if any(row["signer_id"] in signers or row["reference"] in references for row, *_ in sources):
        raise ValueError("FSboard validation signer or exact phrase overlaps training")
    torch.set_num_threads(2)
    revision = miner_revision(args.miner_health_url, None)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, local_files_only=True)
    language_model = T5ForConditionalGeneration.from_pretrained(
        args.pretrained, local_files_only=True, use_safetensors=False
    )
    model = FSboardByT5(language_model)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    revision = miner_revision(args.miner_health_url, revision)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    partial = args.results.with_name(args.results.name + ".partial")
    if partial.exists():
        raise ValueError("unfinished evaluation output exists")
    errors = []
    exact = 0
    errors_by_hand: dict[str, list[float]] = {"left": [], "right": [], "tie": []}
    with partial.open("x", encoding="utf-8") as stream, torch.inference_mode():
        for number, (row, source, receipt, selection_hash) in enumerate(sources, start=1):
            revision = miner_revision(args.miner_health_url, revision)
            with np.load(source, allow_pickle=False) as payload:
                landmarks = payload["landmarks"]
            features = normalized_features(landmarks, frame_stride=2, max_frames=256)
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
