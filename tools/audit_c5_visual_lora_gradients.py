"""Reproduce the fused-weight bypass and verify the corrected real C5 gradient path."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from benchmark_fleurs_dev import sha256
from interpolate_visual_adapter import DIRECT_NAMES

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/baseline-6fe8df59"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    args = parser.parse_args()
    if not args.miner_maintenance_mode or args.output.exists():
        parser.error("gradient audit needs maintenance mode and new output")
    for key, value in {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(BASELINE / "run/hf"), "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
    }.items():
        os.environ[key] = value
    sys.path.insert(0, str(BASELINE / "vendor"))
    sys.path.insert(0, str(BASELINE / "vendor/demo"))
    import numpy as np
    import torch
    from inference import SignLanguageByT5ForConditionalGeneration
    from safetensors.torch import load_file
    from shubert_lora import LoRALinear
    from shubert_visual_lora import install_visual_lora, load_visual_adapter
    from train_c5_stem_projection_frozen import freeze_direct_parameters
    from transformers import ByT5Tokenizer

    torch.set_num_threads(4)
    torch.manual_seed(307)
    row = json.loads(args.index.read_text().splitlines()[0])
    if row["split"] != "training":
        raise ValueError("audit input is not a training clip")
    cache = args.cache_dir / f"{row['clip_id']}.npz"
    if sha256(cache) != row["encoder_cache_sha256"]:
        raise ValueError("audit cache differs from its index")
    with np.load(cache, allow_pickle=False) as payload:
        features = {
            name: torch.from_numpy(payload[name]).unsqueeze(0)
            for name in ("face_features", "left_hand_features", "right_hand_features", "pose_features")
        }
    checkpoint = BASELINE / "models/checkpoint-11625"
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        str(checkpoint), local_files_only=True
    ).eval()
    selected = install_visual_lora(model, 16, 32.0, 4)
    weights = load_file(str(args.adapter), device="cpu")
    load_visual_adapter(model, weights)
    trainable = freeze_direct_parameters(selected)
    visual = {name: parameter for name, parameter in trainable.items() if "signhubert" in name}
    if len(visual) != 16 or any(
        torch.count_nonzero(parameter).item()
        for name, parameter in visual.items() if ".lora_b." in name
    ):
        raise ValueError("audit requires the confirmed adapter's zero visual LoRA deltas")
    tokenizer = ByT5Tokenizer.from_pretrained(str(BASELINE / "models/byt5_base"), local_files_only=True)
    labels = tokenizer(row["reference"], return_tensors="pt").input_ids
    corrected_weight = LoRALinear.weight
    try:
        LoRALinear.weight = property(lambda layer: layer.base.weight)
        legacy = model(**features, labels=labels, use_cache=False, return_dict=True)
        legacy_logits = legacy.logits.detach().clone()
        legacy.loss.backward()
        legacy_missing = [name for name, parameter in visual.items() if parameter.grad is None]
        model.zero_grad(set_to_none=True)
    finally:
        LoRALinear.weight = corrected_weight
    fixed = model(**features, labels=labels, use_cache=False, return_dict=True)
    zero_delta_logits_equal = torch.equal(fixed.logits.detach(), legacy_logits)
    fixed.loss.backward()
    fixed_missing = [name for name, parameter in visual.items() if parameter.grad is None]
    fixed_norms = {
        name: float(parameter.grad.norm())
        for name, parameter in visual.items() if parameter.grad is not None
    }
    if len(legacy_missing) != 16 or fixed_missing or not zero_delta_logits_equal:
        raise RuntimeError("real-model gradient audit failed")
    nonzero_b = sum(norm > 0 for name, norm in fixed_norms.items() if ".lora_b." in name)
    if nonzero_b != 8:
        raise RuntimeError("not every adapted visual projection receives a nonzero B gradient")
    direct = {name: selected[name].detach().clone() for name in DIRECT_NAMES}
    with torch.inference_mode():
        before = model.encoder(**features, return_dict=True).last_hidden_state.clone()
    optimizer = torch.optim.AdamW(trainable.values(), lr=4e-6, weight_decay=0.01)
    torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
    optimizer.step()
    with torch.inference_mode():
        after = model.encoder(**features, return_dict=True).last_hidden_state
    delta = float((after - before).abs().max())
    if delta <= 0 or any(not torch.equal(selected[name], value) for name, value in direct.items()):
        raise RuntimeError("visual update did not affect the encoder or changed frozen projection")
    result = {
        "schema": "umi-studio-c5-visual-lora-gradient-audit/1",
        "example_id": row["example_id"], "index_sha256": sha256(args.index),
        "cache_sha256": sha256(cache), "starting_adapter_sha256": sha256(args.adapter),
        "legacy_visual_parameters_without_gradient": legacy_missing,
        "fixed_visual_parameters_without_gradient": fixed_missing,
        "fixed_visual_gradient_norms": fixed_norms,
        "nonzero_visual_b_gradients": nonzero_b,
        "zero_delta_logits_bit_identical": zero_delta_logits_equal,
        "encoder_max_absolute_change_after_one_step": delta,
        "direct_projection_preserved": True,
        "adapter_implementation_sha256": sha256(ROOT / "tools/shubert_lora.py"),
        "checkpoint_saved": False, "confirmation_used": False,
        "miner_health_policy": "user_authorized_maintenance", "miner_changed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "nonzero_visual_b_gradients", "zero_delta_logits_bit_identical",
        "encoder_max_absolute_change_after_one_step", "direct_projection_preserved"
    )}))


if __name__ == "__main__":
    main()
