"""Train a bounded 83-point ByT5 pilot on one audited FSboard train shard."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.optimization import Adafactor

from benchmark_fleurs_dev import miner_revision, sha256
from fsboard_arrow_training_data import AuditedTrainingShard
from fsboard_byt5_compact_model import FSboardByT5Compact
from fsboard_paper_like_features import paper_like_features
from train_fsboard_byt5_smoke import (
    PRETRAINED_CONFIG_SHA256, PRETRAINED_REVISION, PRETRAINED_WEIGHTS_SHA256,
    eligible_positions,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    if (
        not 0 < args.learning_rate <= 1e-3
        or not args.run_dir.resolve().is_relative_to(ROOT / "runs")
        or not args.checkpoint.resolve().is_relative_to(ROOT / "checkpoints")
        or args.run_dir.exists()
        or args.checkpoint.exists()
    ):
        parser.error("new run/checkpoint paths and bounded learning rate are required")
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise RuntimeError("less than 20 GiB free; refusing training")
    if (
        sha256(args.pretrained / "pytorch_model.bin") != PRETRAINED_WEIGHTS_SHA256
        or sha256(args.pretrained / "config.json") != PRETRAINED_CONFIG_SHA256
    ):
        raise ValueError("ByT5-Small source differs from pinned revision")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    loader = AuditedTrainingShard(args.shard, args.index, args.audit)
    positions = [
        position for position in eligible_positions(loader.rows, args.seed)
        if loader.rows[position]["frames"] <= 512
    ]
    if not positions:
        raise ValueError("no bounded training examples")
    revision = miner_revision(args.miner_health_url, None)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, local_files_only=True)
    model = FSboardByT5Compact(T5ForConditionalGeneration.from_pretrained(
        args.pretrained, local_files_only=True, use_safetensors=False
    )).train()
    optimizer = Adafactor(
        model.parameters(), lr=args.learning_rate, scale_parameter=False, relative_step=False
    )
    revision = miner_revision(args.miner_health_url, revision)
    args.run_dir.mkdir(parents=True)
    steps_path = args.run_dir / "steps.jsonl"
    losses = []
    started = time.monotonic()
    with steps_path.open("x", encoding="utf-8") as stream:
        for step, position in enumerate(positions, start=1):
            revision = miner_revision(args.miner_health_url, revision)
            landmarks, reference, row = loader.example(position)
            features = paper_like_features(
                landmarks, frame_stride=2, max_frames=256, normalization="uniform"
            )
            labels = tokenizer(reference, return_tensors="pt").input_ids
            if not 0 < labels.shape[1] <= 256:
                raise ValueError("FSboard target exceeds byte-token limit")
            optimizer.zero_grad(set_to_none=True)
            output = model(
                torch.from_numpy(features)[None],
                torch.ones((1, len(features)), dtype=torch.long),
                labels,
            )
            if not torch.isfinite(output.loss):
                raise ValueError("FSboard training loss is not finite")
            output.loss.backward()
            optimizer.step()
            revision = miner_revision(args.miner_health_url, revision)
            loss = float(output.loss.detach())
            losses.append(loss)
            stream.write(json.dumps({
                "step": step,
                "clip_filename": row["clip_filename"],
                "signer_id": row["signer_id"],
                "source_row": row["shard_row"],
                "input_frames": len(features),
                "target_tokens": int(labels.shape[1]),
                "loss": loss,
            }, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            if step % 64 == 0 or step == len(positions):
                print(f"step {step}/{len(positions)}: loss {loss:.4f}", flush=True)
    revision = miner_revision(args.miner_health_url, revision)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    partial = args.checkpoint.with_name(args.checkpoint.name + ".partial")
    if partial.exists():
        raise ValueError("unfinished candidate checkpoint exists")
    torch.save(model.state_dict(), partial)
    revision = miner_revision(args.miner_health_url, revision)
    os.replace(partial, args.checkpoint)
    result = {
        "schema": "umi-studio-fsboard-byt5-compact-pilot-training/1",
        "split": "FSboard daun_v3 train only",
        "validation_or_test_used": False,
        "source_shard_sha256": sha256(args.shard),
        "source_audit_sha256": sha256(args.audit),
        "training_index_sha256": sha256(args.index),
        "pretrained_repo": "google/byt5-small",
        "pretrained_revision": PRETRAINED_REVISION,
        "pretrained_weights_sha256": PRETRAINED_WEIGHTS_SHA256,
        "pretrained_config_sha256": PRETRAINED_CONFIG_SHA256,
        "trainer_sha256": sha256(Path(__file__)),
        "model_code_sha256": sha256(ROOT / "tools/fsboard_byt5_compact_model.py"),
        "features_code_sha256": sha256(ROOT / "tools/fsboard_paper_like_features.py"),
        "steps_sha256": sha256(steps_path),
        "checkpoint_sha256": sha256(args.checkpoint),
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "optimizer": "Adafactor without relative scaling",
        "eligible_examples": len(positions),
        "first_64_mean_loss": sum(losses[:64]) / 64,
        "last_64_mean_loss": sum(losses[-64:]) / 64,
        "elapsed_seconds": time.monotonic() - started,
        "device": "cpu",
        "miner_model_revision": revision,
        "live_miner_changed": False,
    }
    with (args.run_dir / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
