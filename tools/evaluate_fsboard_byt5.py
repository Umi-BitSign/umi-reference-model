"""Score an offline ByT5 candidate on frozen public FSboard validation clips.

This is a single-reference development diagnostic, not an official UMI score.
The live miner is checked for liveness only and never serves the candidate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from benchmark_fleurs_dev import distance, miner_revision, sha256
from evaluate_fsboard_coreml import comparable_units
from fetch_fsboard_validation import validate_selection
from fsboard_arrow_training_data import AuditedTrainingShard
from fsboard_byt5_model import FSboardByT5, normalized_features
from train_fsboard_byt5_smoke import (
    PRETRAINED_CONFIG_SHA256,
    PRETRAINED_WEIGHTS_SHA256,
    eligible_positions,
)

ROOT = Path(__file__).resolve().parents[1]


def dominant_hand(receipt: dict) -> str:
    left = receipt["left_hand_finite_scalars"]
    right = receipt["right_hand_finite_scalars"]
    return "left" if left > right else "right" if right > left else "tie"


def validate_training_steps(steps: list[dict], indexed_rows: tuple[dict, ...], seed: int) -> None:
    positions = eligible_positions(indexed_rows, seed)
    if len(steps) != len(positions):
        raise ValueError("candidate training step count differs from audited schedule")
    for number, (step, position) in enumerate(zip(steps, positions, strict=True), start=1):
        row = indexed_rows[position]
        if (
            step.get("step") != number
            or step.get("clip_filename") != row["clip_filename"]
            or step.get("signer_id") != row["signer_id"]
            or step.get("source_row") != row["shard_row"]
        ):
            raise ValueError("candidate training steps differ from audited schedule")


def validation_rows(
    selections: list[Path], landmarks_dirs: list[Path], receipt_paths: list[Path],
    training_index: Path,
) -> list[tuple[dict, Path, dict, str]]:
    if not len(selections) == len(landmarks_dirs) == len(receipt_paths) == 2:
        raise ValueError("exactly two frozen FSboard validation selections are required")
    train_rows = [json.loads(line) for line in training_index.read_text().splitlines()]
    training_signers = {row["signer_id"] for row in train_rows}
    training_references = {row["reference"] for row in train_rows}
    result = []
    names = set()
    for selection_path, landmarks_dir, receipts_path in zip(
        selections, landmarks_dirs, receipt_paths, strict=True
    ):
        selection_hash = sha256(selection_path)
        rows = validate_selection(json.loads(selection_path.read_text()))
        receipts = {
            record["clip_filename"]: record
            for record in map(json.loads, receipts_path.read_text().splitlines())
        }
        if len(receipts) != len(rows):
            raise ValueError("FSboard validation landmark receipts are incomplete")
        for row in rows:
            name = row["clip_filename"]
            receipt = receipts.get(name)
            source = landmarks_dir / f"{Path(name).stem}.npz"
            if (
                name in names
                or row["signer_id"] in training_signers
                or row["reference"] in training_references
                or receipt is None
                or receipt.get("schema") != "umi-studio-fsboard-landmarks/1"
                or receipt.get("selection_sha256") != selection_hash
                or receipt.get("mirror_input", False)
                or not source.is_file()
                or receipt.get("sha256") != sha256(source)
            ):
                raise ValueError(f"FSboard validation source differs from frozen receipt: {name}")
            names.add(name)
            result.append((row, source, receipt, selection_hash))
    if len(result) != 60:
        raise ValueError("expected 60 frozen public validation clips")
    return result


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
        training.get("schema") != "umi-studio-fsboard-byt5-shard-training/1"
        or training.get("checkpoint_sha256") != sha256(args.checkpoint)
        or training.get("training_index_sha256") != sha256(args.training_index)
        or training.get("source_shard_sha256") != sha256(args.training_shard)
        or training.get("source_audit_sha256") != sha256(args.training_audit)
        or training.get("steps_sha256") != sha256(args.training_steps)
        or sha256(args.pretrained / "pytorch_model.bin") != PRETRAINED_WEIGHTS_SHA256
        or sha256(args.pretrained / "config.json") != PRETRAINED_CONFIG_SHA256
    ):
        raise ValueError("candidate checkpoint or pretrained model differs from training ledger")
    loader = AuditedTrainingShard(
        args.training_shard, args.training_index, args.training_audit
    )
    steps = [json.loads(line) for line in args.training_steps.read_text().splitlines()]
    validate_training_steps(steps, loader.rows, training["seed"])
    if training.get("eligible_examples") != len(steps):
        raise ValueError("training result differs from audited step count")
    sources = validation_rows(
        args.selection, args.landmarks_dir, args.landmark_receipts, args.training_index
    )
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
        raise ValueError("unfinished evaluation results exist; inspect them before retrying")
    errors = []
    exact = 0
    errors_by_hand: dict[str, list[float]] = {"left": [], "right": [], "tie": []}
    with partial.open("x", encoding="utf-8") as stream, torch.inference_mode():
        for number, (row, source, receipt, selection_hash) in enumerate(sources, start=1):
            revision = miner_revision(args.miner_health_url, revision)
            with np.load(source, allow_pickle=False) as payload:
                landmarks = payload["landmarks"]
            features = normalized_features(landmarks, frame_stride=2, max_frames=256)
            tensor = torch.from_numpy(features)[None]
            tokens = model.language_model.generate(
                inputs_embeds=model.projection(tensor),
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
