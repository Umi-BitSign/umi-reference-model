"""Train a separate ByT5 specialist on frozen, audited FSboard train shards.

This is a data-scale check of the first 543-point recipe. The active miner is
not used for training. Recoveries are retained if miner health aborts the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, __version__ as transformers_version
from transformers.optimization import Adafactor

from benchmark_fleurs_dev import miner_revision, sha256
from fsboard_arrow_training_data import AuditedTrainingShard
from fsboard_byt5_model import FSboardByT5, normalized_features
from train_fsboard_byt5_smoke import PRETRAINED_CONFIG_SHA256, PRETRAINED_REVISION, PRETRAINED_WEIGHTS_SHA256

ROOT = Path(__file__).resolve().parents[1]


def schedule(loaders: list[AuditedTrainingShard], seed: int) -> list[tuple[int, int]]:
    """Shuffle high-hand-coverage clips while respecting the 256-frame model cap."""
    if not loaders:
        raise ValueError("at least one audited FSboard shard is required")
    names: set[str] = set()
    choices = []
    for shard_number, loader in enumerate(loaders):
        for position, row in enumerate(loader.rows):
            name = row["clip_filename"]
            if name in names:
                raise ValueError("duplicate FSboard training clip across shards")
            names.add(name)
            if (
                max(row["left_hand_detection_fraction"], row["right_hand_detection_fraction"]) >= 0.5
                and row["frames"] <= 512
            ):
                choices.append((shard_number, position))
    return sorted(choices, key=lambda item: (
        hashlib.sha256(f"{seed}:{loaders[item[0]].rows[item[1]]['clip_filename']}".encode()).hexdigest(),
        loaders[item[0]].rows[item[1]]["clip_filename"],
    ))


def save_recovery(path: Path, model: FSboardByT5, optimizer: Adafactor, step: int) -> None:
    partial = path.with_name(path.name + ".partial")
    if partial.exists() or path.exists():
        raise ValueError("recovery destination already exists")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "torch_rng": torch.random.get_rng_state(),
    }, partial)
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--recovery-interval", type=int, default=1024)
    args = parser.parse_args()
    if (
        not len(args.shard) == len(args.index) == len(args.audit)
        or not 2 <= len(args.shard) <= 100
        or not 0 < args.learning_rate <= 1e-3
        or not 128 <= args.recovery_interval <= 4096
        or not args.run_dir.resolve().is_relative_to(ROOT / "runs")
        or not args.checkpoint.resolve().is_relative_to(ROOT / "checkpoints")
        or args.run_dir.exists()
        or args.checkpoint.exists()
    ):
        parser.error("expected new run/checkpoint paths, matching audited shards, and bounded settings")
    if shutil.disk_usage(ROOT).free < 30 * 1024**3:
        raise RuntimeError("less than 30 GiB free; refusing multishard training")
    if (
        sha256(args.pretrained / "pytorch_model.bin") != PRETRAINED_WEIGHTS_SHA256
        or sha256(args.pretrained / "config.json") != PRETRAINED_CONFIG_SHA256
    ):
        raise ValueError("ByT5-Small pretrained checkpoint differs from pinned revision")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    revision = miner_revision(args.miner_health_url, None)
    loaders = [
        AuditedTrainingShard(shard, index, audit)
        for shard, index, audit in zip(args.shard, args.index, args.audit, strict=True)
    ]
    identities = []
    for shard, index, audit in zip(args.shard, args.index, args.audit, strict=True):
        document = json.loads(audit.read_text())
        identities.append({
            "shard_name": shard.name,
            "shard_sha256": sha256(shard),
            "index_sha256": sha256(index),
            "audit_sha256": sha256(audit),
            "training_metadata_sha256": document["training_metadata_sha256"],
            "validation_metadata_sha256": document["validation_metadata_sha256"],
        })
    if (
        len({item["shard_name"] for item in identities}) != len(identities)
        or len({item["training_metadata_sha256"] for item in identities}) != 1
        or len({item["validation_metadata_sha256"] for item in identities}) != 1
    ):
        raise ValueError("training shards are duplicated or have different metadata sources")
    positions = schedule(loaders, args.seed)
    if not positions:
        raise ValueError("no eligible audited FSboard examples")
    revision = miner_revision(args.miner_health_url, revision)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, local_files_only=True)
    language_model = T5ForConditionalGeneration.from_pretrained(
        args.pretrained, local_files_only=True, use_safetensors=False
    )
    model = FSboardByT5(language_model).train()
    optimizer = Adafactor(
        model.parameters(), lr=args.learning_rate, scale_parameter=False, relative_step=False
    )
    revision = miner_revision(args.miner_health_url, revision)
    args.run_dir.mkdir(parents=True)
    plan = {
        "schema": "umi-studio-fsboard-byt5-multishard-plan/1",
        "split": "FSboard daun_v3 train only",
        "validation_or_test_used": False,
        "shards": identities,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "recovery_interval": args.recovery_interval,
        "eligible_examples": len(positions),
        "pretrained_repo": "google/byt5-small",
        "pretrained_revision": PRETRAINED_REVISION,
        "pretrained_weights_sha256": PRETRAINED_WEIGHTS_SHA256,
        "pretrained_config_sha256": PRETRAINED_CONFIG_SHA256,
        "trainer_sha256": sha256(Path(__file__)),
        "miner_model_revision": revision,
    }
    plan_path = args.run_dir / "plan.json"
    with plan_path.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    started = time.monotonic()
    losses = []
    steps_path = args.run_dir / "steps.jsonl"
    with steps_path.open("x", encoding="utf-8") as stream:
        for step, (shard_number, position) in enumerate(positions, start=1):
            revision = miner_revision(args.miner_health_url, revision)
            landmarks, reference, row = loaders[shard_number].example(position)
            features = normalized_features(landmarks, frame_stride=2, max_frames=256)
            labels = tokenizer(reference, return_tensors="pt").input_ids
            if not 0 < labels.shape[1] <= 256:
                raise ValueError("FSboard target exceeds byte-token limit")
            optimizer.zero_grad(set_to_none=True)
            step_started = time.monotonic()
            output = model(
                torch.from_numpy(features)[None],
                torch.ones((1, len(features)), dtype=torch.long),
                labels,
            )
            if not torch.isfinite(output.loss):
                raise ValueError("FSboard training loss is not finite")
            output.loss.backward()
            optimizer.step()
            elapsed = time.monotonic() - step_started
            revision = miner_revision(args.miner_health_url, revision)
            loss = float(output.loss.detach())
            losses.append(loss)
            stream.write(json.dumps({
                "step": step,
                "shard_number": shard_number,
                "clip_filename": row["clip_filename"],
                "signer_id": row["signer_id"],
                "source_row": row["shard_row"],
                "input_frames": len(features),
                "target_tokens": int(labels.shape[1]),
                "loss": loss,
                "step_seconds": elapsed,
            }, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            if step % args.recovery_interval == 0:
                save_recovery(args.run_dir / f"recovery-{step:05d}.pt", model, optimizer, step)
                revision = miner_revision(args.miner_health_url, revision)
            if step % 128 == 0 or step == len(positions):
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
        "schema": "umi-studio-fsboard-byt5-multishard-training/1",
        "plan_sha256": sha256(plan_path),
        "steps_sha256": sha256(steps_path),
        "checkpoint_sha256": sha256(args.checkpoint),
        "eligible_examples": len(positions),
        "first_64_mean_loss": sum(losses[:64]) / 64,
        "last_64_mean_loss": sum(losses[-64:]) / 64,
        "elapsed_seconds": time.monotonic() - started,
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "transformers_version": transformers_version,
        "miner_model_revision": revision,
        "live_miner_changed": False,
        "validation_or_test_used": False,
    }
    with (args.run_dir / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
