"""Merge the confirmed visual LoRA candidate into a conventional C5 checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from benchmark_fleurs_dev import sha256

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--confirmation-decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    checkpoint = baseline / "models/checkpoint-11625"
    if (
        baseline != ROOT / "artifacts/baseline-6fe8df59"
        or not args.adapter.is_file()
        or not args.confirmation_decision.is_file()
        or args.output_dir.exists()
        or not args.output_dir.resolve().is_relative_to((ROOT / "checkpoints").resolve())
    ):
        parser.error("confirmed export inputs or output are invalid")
    decision = json.loads(args.confirmation_decision.read_text())
    adapter_hash = sha256(args.adapter)
    baseline_hash = sha256(checkpoint / "model.safetensors")
    if (
        decision.get("schema")
        != "umi-studio-temporal-stride-confirmation-decision/1"
        or decision.get("decision") != "package_candidate"
        or decision.get("candidate_temporal_stride") != 2
        or decision.get("candidate_adapter_sha256") != adapter_hash
        or decision["how2sign"]["checkpoint_sha256"] != baseline_hash
        or decision["fleurs"]["checkpoint_sha256"] != baseline_hash
    ):
        raise ValueError("confirmation decision does not authorize this export")
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
    sys.path.insert(0, str(ROOT / "tools"))
    import torch
    from inference import SignLanguageByT5ForConditionalGeneration
    from safetensors import safe_open
    from safetensors.torch import load_file
    from shubert_lora import LoRALinear, merge_lora
    from shubert_visual_lora import install_visual_lora, load_visual_adapter

    torch.set_num_threads(2)
    with safe_open(str(args.adapter), framework="pt", device="cpu") as source:
        metadata = source.metadata()
    if (
        metadata.get("schema") != "umi-studio-shubert-visual-lora/1"
        or metadata.get("baseline_checkpoint_sha256") != baseline_hash
    ):
        raise ValueError("adapter metadata differs from the confirmed baseline")
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    baseline_keys = set(model.state_dict())
    install_visual_lora(
        model,
        int(metadata["rank"]),
        float(metadata["alpha"]),
        int(metadata["visual_last_layers"]),
    )
    load_visual_adapter(model, load_file(str(args.adapter), device="cpu"))
    merge_lora(model)
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise RuntimeError("merged checkpoint retains a LoRA module")
    if set(model.state_dict()) != baseline_keys:
        raise RuntimeError("merged checkpoint topology differs from the baseline")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    temporary = args.output_dir.with_name(f".{args.output_dir.name}.partial")
    if temporary.exists():
        raise ValueError("unfinished merged export must be inspected")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(
            temporary,
            safe_serialization=True,
            max_shard_size="10GB",
        )
        model_path = temporary / "model.safetensors"
        if not model_path.is_file():
            raise RuntimeError("merged export did not produce one safetensors file")
        manifest = {
            "schema": "umi-studio-c5-stride2-merged-checkpoint/1",
            "baseline_checkpoint_sha256": baseline_hash,
            "adapter_sha256": adapter_hash,
            "merged_checkpoint_sha256": sha256(model_path),
            "confirmation_decision_sha256": sha256(args.confirmation_decision),
            "temporal_stride": 2,
            "rank": int(metadata["rank"]),
            "alpha": float(metadata["alpha"]),
            "visual_last_layers": int(metadata["visual_last_layers"]),
            "exporter_sha256": sha256(Path(__file__)),
            "miner_changed": False,
        }
        with (temporary / "MERGED_CHECKPOINT.json").open("x", encoding="utf-8") as output:
            json.dump(manifest, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
