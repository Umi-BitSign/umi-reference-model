"""Compare Metal activation precision and isolate decoder error on a failed training case."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path


def main():
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    }.items():
        os.environ[key] = value
    sys.dont_write_bytecode = True
    import tiny_rgb_model
    import torch
    from benchmark_fleurs_dev import sha256
    from tiny_rgb_capacity import build_capacity_translator
    from tiny_rgb_capacity_activations import NativeForwardExactGelu
    from tiny_rgb_capacity_protocol import ROOT, checked_plan
    from tiny_rgb_data import pixel_tensor
    from tiny_rgb_full_training_tools import decode_rgb_video_fast, masked_length_buckets
    from torch import nn
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--failed-comparisons", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = checked_plan(args.plan)
    failed = json.loads(args.failed_comparisons.read_text())[-1]
    row = next(
        case["training"]
        for case in plan["cases"]
        if (case["training"]["source"], case["training"]["example_id"])
        == (failed["source"], failed["example_id"])
    )
    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("Metal isolation requires a new runs/ directory")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    language = T5ForConditionalGeneration.from_pretrained(
        ROOT / plan["language_path"], local_files_only=True, use_safetensors=True
    )
    torch.manual_seed(plan["seed"])
    cpu = build_capacity_translator(language, failed["capacity"]).eval()
    pixels, _, _ = decode_rgb_video_fast(row["video"], failed["short_side"])
    video = pixel_tensor(pixels)[None]
    tokenizer = AutoTokenizer.from_pretrained(ROOT / plan["language_path"], local_files_only=True)
    label = tokenizer(row["reference"], return_tensors="pt", truncation=False).input_ids
    padded, mask, target = masked_length_buckets(video, label, torch)
    with torch.no_grad():
        reference = cpu(padded, mask, target)
        reference_memory, reference_mask = cpu.encode(padded, mask)
    model = copy.deepcopy(cpu).to("mps")
    slots = [
        (module, name)
        for module in model.vision.modules()
        for name, child in module.named_children()
        if isinstance(child, NativeForwardExactGelu)
    ]

    class Formula(nn.Module):
        def __init__(self, mode):
            super().__init__()
            self.mode = mode

        def forward(self, value):
            value = value.contiguous()
            if self.mode == "native":
                return torch.nn.functional.gelu(value, approximate="none")
            if self.mode == "erf_multiply":
                return value * 0.5 * (1 + torch.erf(value * math.sqrt(0.5)))
            return value * 0.5 * (1 + torch.erf(value / math.sqrt(2)))

    reports = []
    for mode in ("native", "erf_multiply", "erf_divide"):
        for parent, name in slots:
            setattr(parent, name, Formula(mode))
        with torch.no_grad():
            actual = model(padded.to("mps"), mask.to("mps"), target.to("mps"))
            memory, _ = model.encode(padded.to("mps"), mask.to("mps"))
        reports.append(
            {
                "activation": mode,
                "loss_difference": float((actual.loss.cpu() - reference.loss).abs()),
                "logit_maximum_difference": float(
                    (
                        actual.logits[:, : label.shape[1]].cpu()
                        - reference.logits[:, : label.shape[1]]
                    )
                    .abs()
                    .max()
                ),
                "encoded_memory_maximum_difference": float(
                    (memory.cpu() - reference_memory).abs().max()
                ),
            }
        )
    from transformers.modeling_outputs import BaseModelOutput

    with torch.no_grad():
        decoder = model.language_model(
            encoder_outputs=BaseModelOutput(last_hidden_state=reference_memory.to("mps")),
            attention_mask=reference_mask.long().to("mps"),
            labels=target.to("mps"),
            use_cache=False,
            return_dict=True,
        )
    original_positions = tiny_rgb_model.positions
    length = padded.shape[1] * model.vision.config.spatial_queries
    width = model.vision.config.width
    position_difference = float(
        (original_positions(length, width, "cpu") - original_positions(length, width, "mps").cpu())
        .abs()
        .max()
    )
    for parent, name in slots:
        setattr(parent, name, NativeForwardExactGelu())
    # Process-local diagnostic override only; no model source or other process changes.
    tiny_rgb_model.positions = lambda length, width, device: original_positions(
        length, width, "cpu"
    ).to(device)
    with torch.no_grad():
        stable = model(padded.to("mps"), mask.to("mps"), target.to("mps"))
        stable_memory, _ = model.encode(padded.to("mps"), mask.to("mps"))
    tiny_rgb_model.positions = original_positions
    result = {
        "plan_sha256": sha256(args.plan),
        "failed_comparisons_sha256": sha256(args.failed_comparisons),
        "source_sha256": sha256(Path(__file__)),
        "training_only": True,
        "optimizer_updates": 0,
        "activation_reports": reports,
        "decoder_with_exact_cpu_memory_logit_difference": float(
            (decoder.logits[:, : label.shape[1]].cpu() - reference.logits[:, : label.shape[1]])
            .abs()
            .max()
        ),
        "cpu_metal_positional_encoding_maximum_difference": position_difference,
        "cpu_generated_positions_logit_difference": float(
            (stable.logits[:, : label.shape[1]].cpu() - reference.logits[:, : label.shape[1]])
            .abs()
            .max()
        ),
        "cpu_generated_positions_memory_difference": float(
            (stable_memory.cpu() - reference_memory).abs().max()
        ),
    }
    output.mkdir(parents=True)
    output.joinpath("source.py").write_text(Path(__file__).read_text())
    output.joinpath("summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
