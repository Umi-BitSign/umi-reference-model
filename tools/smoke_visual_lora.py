"""Run one offline visual-adapter gradient step on public training features."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from evaluate_fleurs_cache import cached_visual_streams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    baseline = args.baseline.resolve()
    if baseline != workspace / "artifacts/baseline-6fe8df59":
        parser.error("smoke test requires the independent frozen baseline")
    if args.output.exists() or not args.output.resolve().is_relative_to(workspace / "runs"):
        parser.error("output must be new and remain under the workspace run ledger")
    with args.index.open(encoding="utf-8") as source:
        first = source.readline()
    if not first:
        raise ValueError("visual training index is empty")
    row = json.loads(first)
    if (
        row.get("schema") != "umi-studio-how2sign-encoder-cache-example/1"
        or row.get("split") != "training"
        or row.get("baseline_identity_sha256") != sha256(baseline / "MODEL_IDENTITY.json")
        or row.get("baseline_runtime_sha256") != sha256(baseline / "runtime.py")
        or row.get("clip_fps") != 25
    ):
        raise ValueError("visual smoke input is not verified public training data")
    name = row["sentence_name"]
    cache = args.cache_dir / f"{name}.npz"
    if sha256(cache) != row["encoder_cache_sha256"]:
        raise ValueError("visual smoke cache differs from the training index")
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
    import numpy as np
    import torch
    from inference import SignLanguageByT5ForConditionalGeneration
    from shubert_visual_lora import install_visual_lora
    from transformers import ByT5Tokenizer

    torch.set_num_threads(2)
    torch.manual_seed(17)
    revision = miner_revision(args.miner_health_url, None)
    checkpoint = baseline / "models/checkpoint-11625"
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    selected = install_visual_lora(model, rank=8, alpha=16, visual_last_layers=4)
    tokenizer = ByT5Tokenizer.from_pretrained(
        str(baseline / "models/byt5_base"), local_files_only=True
    )
    with np.load(cache, allow_pickle=False) as payload:
        state = payload["encoder_hidden_state"]
        streams = cached_visual_streams(payload, row)
    features = {
        name: torch.from_numpy(value).unsqueeze(0) for name, value in streams.items()
    }
    model.eval()
    with torch.no_grad():
        untouched = model.encoder(**features, return_dict=True).last_hidden_state
    if not np.array_equal(untouched[0].numpy(), state):
        raise ValueError("visual smoke inputs differ from the frozen baseline encoder")
    revision = miner_revision(args.miner_health_url, revision)
    labels = tokenizer(row["reference"], return_tensors="pt").input_ids
    if labels.shape[1] > 512:
        raise ValueError("visual smoke reference exceeds 512 ByT5 tokens")
    optimizer = torch.optim.AdamW(selected.values(), lr=0.0001, weight_decay=0.1)
    before = {name: parameter.detach().clone() for name, parameter in selected.items()}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(**features, labels=labels, use_cache=False, return_dict=True)
    loss = torch.nn.functional.cross_entropy(
        output.logits.reshape(-1, output.logits.shape[-1]),
        labels.reshape(-1),
        label_smoothing=0.1,
    )
    if not torch.isfinite(loss):
        raise RuntimeError("visual smoke loss is non-finite")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(selected.values(), max_norm=1.0)
    if not torch.isfinite(gradient_norm):
        raise RuntimeError("visual smoke gradient is non-finite")
    revision = miner_revision(args.miner_health_url, revision)
    optimizer.step()
    revision = miner_revision(args.miner_health_url, revision)
    changed = {
        "visual": any(
            not torch.equal(before[name], parameter.detach())
            for name, parameter in selected.items()
            if name.startswith("encoder.")
        ),
        "decoder_cross_attention": any(
            not torch.equal(before[name], parameter.detach())
            for name, parameter in selected.items()
            if name.startswith("decoder.")
        ),
    }
    if not all(changed.values()):
        raise RuntimeError("visual smoke step did not update both planned model branches")
    result = {
        "schema": "umi-studio-visual-lora-gradient-smoke/1",
        "example_id": name,
        "training_index_sha256": sha256(args.index),
        "training_cache_sha256": row["encoder_cache_sha256"],
        "baseline_checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
        "adapter_source_sha256": sha256(Path(__file__).with_name("shubert_visual_lora.py")),
        "smoke_source_sha256": sha256(Path(__file__)),
        "trainable_parameter_count": sum(parameter.numel() for parameter in selected.values()),
        "loss": float(loss.detach().cpu()),
        "gradient_norm_before_clipping": float(gradient_norm.detach().cpu()),
        "changed": changed,
        "miner_model_revision": revision,
        "checkpoint_saved": False,
        "live_miner_changed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as ledger:
        json.dump(result, ledger, sort_keys=True, indent=2)
        ledger.write("\n")
    print(f"visual gradient smoke passed for {name}; no checkpoint saved")


if __name__ == "__main__":
    main()
