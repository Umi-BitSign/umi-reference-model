"""Check whether a failed FSboard ByT5 checkpoint decodes clips it trained on."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from benchmark_fleurs_dev import distance, miner_revision, sha256
from evaluate_fsboard_byt5_multishard import verify_training
from evaluate_fsboard_coreml import comparable_units
from fsboard_byt5_model import FSboardByT5, normalized_features

ROOT = Path(__file__).resolve().parents[1]


def sample_positions(loaders: list, per_shard: int = 4) -> list[tuple[int, int]]:
    selected = []
    for shard_number, loader in enumerate(loaders):
        positions = [
            position for position, row in enumerate(loader.rows)
            if max(row["left_hand_detection_fraction"], row["right_hand_detection_fraction"]) >= 0.5
            and row["frames"] <= 512
        ]
        positions.sort(key=lambda position: hashlib.sha256(
            loader.rows[position]["clip_filename"].encode()
        ).hexdigest())
        if len(positions) < per_shard:
            raise ValueError("training shard has too few probe clips")
        selected.extend((shard_number, position) for position in positions[:per_shard])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--training-steps", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs"):
        parser.error("new output must be under runs/")
    loaders, training = verify_training(
        args.training_plan, args.training_result, args.training_steps, args.checkpoint,
        args.pretrained, args.shard, args.index, args.audit,
    )
    selected = sample_positions(loaders)
    torch.set_num_threads(2)
    revision = miner_revision(args.miner_health_url, None)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, local_files_only=True)
    model = FSboardByT5(T5ForConditionalGeneration.from_pretrained(
        args.pretrained, local_files_only=True, use_safetensors=False
    ))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    revision = miner_revision(args.miner_health_url, revision)
    rows = []
    with torch.inference_mode():
        for shard_number, position in selected:
            revision = miner_revision(args.miner_health_url, revision)
            landmarks, reference, row = loaders[shard_number].example(position)
            features = normalized_features(landmarks, frame_stride=2, max_frames=256)
            tokens = model.language_model.generate(
                inputs_embeds=model.projection(torch.from_numpy(features)[None]),
                attention_mask=torch.ones((1, len(features)), dtype=torch.long),
                max_new_tokens=64,
                num_beams=1,
            )
            hypothesis = tokenizer.decode(tokens[0], skip_special_tokens=True)
            reference_units = comparable_units(reference)
            hypothesis_units = comparable_units(hypothesis)
            cer = distance(reference_units, hypothesis_units) / max(1, len(reference_units))
            revision = miner_revision(args.miner_health_url, revision)
            rows.append({
                "shard_number": shard_number,
                "clip_filename": row["clip_filename"],
                "signer_id": row["signer_id"],
                "reference": reference,
                "hypothesis": hypothesis,
                "single_reference_cer": cer,
            })
    result = {
        "schema": "umi-studio-fsboard-byt5-training-fit-probe/1",
        "split": "FSboard daun_v3 train only",
        "validation_or_test_used": False,
        "sampling": "four clips per shard, smallest SHA-256 clip names",
        "checkpoint_sha256": training["checkpoint_sha256"],
        "training_result_sha256": sha256(args.training_result),
        "probe_script_sha256": sha256(Path(__file__)),
        "clips": len(rows),
        "mean_cer": sum(row["single_reference_cer"] for row in rows) / len(rows),
        "unique_hypotheses": len({row["hypothesis"] for row in rows}),
        "rows": rows,
        "miner_model_revision": revision,
        "live_miner_changed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("clips", "mean_cer", "unique_hypotheses")}))


if __name__ == "__main__":
    main()
