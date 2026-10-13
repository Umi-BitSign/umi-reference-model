"""Merge a trained SHuBERT LoRA adapter into a standalone checkpoint.

This writes only under this workspace's ignored checkpoints directory. The
frozen baseline and live miner bundle remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--minimum-free-gib", type=int, default=100)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    baseline = args.baseline.resolve()
    if baseline != workspace / "artifacts/baseline-6fe8df59":
        parser.error("merger requires the independent frozen baseline in this workspace")
    if not args.adapter.resolve().is_relative_to(workspace / "checkpoints"):
        parser.error("adapter must stay under this workspace's checkpoints directory")
    if not args.output_dir.resolve().is_relative_to(workspace / "checkpoints"):
        parser.error("output must stay under this workspace's checkpoints directory")
    if args.output_dir.exists() or args.minimum_free_gib < 1:
        parser.error("output must not exist and the free-space floor must be positive")
    if shutil.disk_usage(workspace).free < args.minimum_free_gib * 1024**3:
        raise RuntimeError("free disk space is below the checkpoint-merging floor")
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(baseline / "run/hf"),
        "TORCH_HOME": str(baseline / "run/torch"),
        "MPLCONFIGDIR": str(baseline / "run/matplotlib"),
        "SHUBERT_DINOV2_SOURCE": str(baseline / "vendor/dinov2-source"),
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }.items():
        os.environ[key] = value
    sys.path.insert(0, str(baseline / "vendor"))
    sys.path.insert(0, str(baseline / "vendor/demo"))
    from inference import SignLanguageByT5ForConditionalGeneration
    from safetensors import safe_open
    from safetensors.torch import load_file
    from shubert_lora import install_lora, load_adapter, merge_lora

    checkpoint = baseline / "models/checkpoint-11625"
    checkpoint_hash = sha256(checkpoint / "model.safetensors")
    adapter_hash = sha256(args.adapter)
    with safe_open(str(args.adapter), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    if (
        metadata.get("schema") != "umi-studio-shubert-lora/1"
        or metadata.get("baseline_checkpoint_sha256") != checkpoint_hash
    ):
        raise ValueError("adapter is not bound to the frozen baseline checkpoint")
    revision = miner_revision(args.miner_health_url, None)
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    original_keys = set(model.state_dict())
    install_lora(
        model,
        int(metadata["rank"]),
        float(metadata["alpha"]),
        metadata.get("targets", "cross_qv"),
    )
    load_adapter(model, load_file(str(args.adapter), device="cpu"))
    merge_lora(model)
    if set(model.state_dict()) != original_keys:
        raise RuntimeError("merged model does not have the original checkpoint topology")
    revision = miner_revision(args.miner_health_url, revision)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir), safe_serialization=True)
    revision = miner_revision(args.miner_health_url, revision)
    model_path = args.output_dir / "model.safetensors"
    if not model_path.is_file():
        raise RuntimeError("merged checkpoint is missing model.safetensors")
    receipt = {
        "schema": "umi-studio-shubert-lora-merge/1",
        "baseline_checkpoint_sha256": checkpoint_hash,
        "adapter_sha256": adapter_hash,
        "adapter_metadata": metadata,
        "merged_checkpoint_sha256": sha256(model_path),
        "merged_config_sha256": sha256(args.output_dir / "config.json"),
        "miner_model_revision_during_merge": revision,
    }
    with (args.output_dir / "MERGE_RECEIPT.json").open("x", encoding="utf-8") as output:
        json.dump(receipt, output, sort_keys=True, indent=2)
        output.write("\n")
    print(f"merged checkpoint sha256={receipt['merged_checkpoint_sha256']}")


if __name__ == "__main__":
    main()
