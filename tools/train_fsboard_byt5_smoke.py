"""Run bounded, training-only ByT5 gradient steps on audited FSboard landmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, __version__ as transformers_version
from transformers.optimization import Adafactor

from benchmark_fleurs_dev import miner_revision, sha256
from fsboard_arrow_training_data import AuditedTrainingShard
from fsboard_byt5_model import FSboardByT5, normalized_features

ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_REVISION = "68377bdc18a2ffec8a0533fef03b1c513a4dd49d"
PRETRAINED_WEIGHTS_SHA256 = "5c5aaf56299d6f2d4eaadad550a40765198828ead4d74f0a15f91cbe0961931a"
PRETRAINED_CONFIG_SHA256 = "7845fb21b320f3fa05392ce151143502cf08729c8c89732da3293615885d3e83"


def eligible_positions(rows: tuple[dict, ...], seed: int) -> list[int]:
    eligible = [
        position for position, row in enumerate(rows)
        if max(row["left_hand_detection_fraction"], row["right_hand_detection_fraction"]) >= 0.5
    ]
    return sorted(
        eligible,
        key=lambda position: (
            hashlib.sha256(
                f"{seed}:{rows[position]['clip_filename']}".encode()
            ).hexdigest(),
            rows[position]["clip_filename"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if (
        not 1 <= args.steps <= 16
        or not 0 < args.learning_rate <= 1e-3
        or args.result.exists()
        or not args.result.resolve().is_relative_to(ROOT / "runs")
    ):
        parser.error("smoke run needs 1-16 steps and a new result under runs/")
    if (
        sha256(args.pretrained / "pytorch_model.bin") != PRETRAINED_WEIGHTS_SHA256
        or sha256(args.pretrained / "config.json") != PRETRAINED_CONFIG_SHA256
    ):
        raise ValueError("ByT5-Small source checkpoint differs from pinned revision")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    loader = AuditedTrainingShard(args.shard, args.index, args.audit)
    positions = eligible_positions(loader.rows, args.seed)[: args.steps]
    if len(positions) != args.steps:
        raise ValueError("not enough audited FSboard training examples")
    revision = miner_revision(args.miner_health_url, None)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, local_files_only=True)
    language_model = T5ForConditionalGeneration.from_pretrained(
        args.pretrained, local_files_only=True, use_safetensors=False
    )
    model = FSboardByT5(language_model).train()
    optimizer = Adafactor(
        model.parameters(), lr=args.learning_rate, scale_parameter=False, relative_step=False
    )
    load_seconds = time.monotonic() - started
    revision = miner_revision(args.miner_health_url, revision)
    records = []
    for position in positions:
        revision = miner_revision(args.miner_health_url, revision)
        landmarks, reference, row = loader.example(position)
        features = normalized_features(landmarks, frame_stride=2, max_frames=256)
        labels = tokenizer(reference, return_tensors="pt").input_ids
        if labels.shape[1] > 256:
            raise ValueError("FSboard target exceeds the byte-token limit")
        optimizer.zero_grad(set_to_none=True)
        before_projection = model.projection.weight.detach().clone()
        before_shared = model.language_model.shared.weight.detach().clone()
        step_started = time.monotonic()
        outputs = model(
            torch.from_numpy(features)[None],
            torch.ones((1, len(features)), dtype=torch.long),
            labels,
        )
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise ValueError("FSboard training loss is not finite")
        loss.backward()
        optimizer.step()
        step_seconds = time.monotonic() - step_started
        projection_change = float(
            torch.linalg.vector_norm(model.projection.weight.detach() - before_projection)
        )
        shared_change = float(
            torch.linalg.vector_norm(model.language_model.shared.weight.detach() - before_shared)
        )
        if projection_change <= 0 or shared_change <= 0:
            raise ValueError("ByT5 smoke step did not update projection and language weights")
        revision = miner_revision(args.miner_health_url, revision)
        records.append({
            "clip_filename": row["clip_filename"],
            "signer_id": row["signer_id"],
            "source_row": row["shard_row"],
            "input_frames": len(features),
            "target_tokens": int(labels.shape[1]),
            "loss": float(loss.detach()),
            "projection_weight_change_l2": projection_change,
            "shared_weight_change_l2": shared_change,
            "step_seconds": step_seconds,
        })
        print(f"step {len(records)}/{args.steps}: loss {records[-1]['loss']:.4f}", flush=True)
    result = {
        "schema": "umi-studio-fsboard-byt5-training-smoke/1",
        "split": "FSboard daun_v3 train only",
        "test_or_validation_used": False,
        "pretrained_repo": "google/byt5-small",
        "pretrained_revision": PRETRAINED_REVISION,
        "pretrained_weights_sha256": PRETRAINED_WEIGHTS_SHA256,
        "pretrained_config_sha256": PRETRAINED_CONFIG_SHA256,
        "source_audit_sha256": sha256(args.audit),
        "training_index_sha256": sha256(args.index),
        "source_shard_sha256": sha256(args.shard),
        "trainer_sha256": sha256(Path(__file__)),
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "optimizer": "Adafactor without relative scaling",
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "transformers_version": transformers_version,
        "load_seconds": load_seconds,
        "records": records,
        "miner_model_revision": revision,
        "candidate_checkpoint_written": False,
        "live_miner_changed": False,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    with args.result.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
