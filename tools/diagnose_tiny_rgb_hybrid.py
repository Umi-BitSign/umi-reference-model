"""Check GPU vision with CPU pretrained decoding on a preserved failed training case."""

from __future__ import annotations

import argparse
import copy
import json
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
    import torch
    from benchmark_fleurs_dev import sha256
    from probe_tiny_rgb_capacity import compare, gradients
    from tiny_rgb_capacity import build_capacity_translator
    from tiny_rgb_capacity_protocol import ROOT, checked_plan
    from tiny_rgb_data import pixel_tensor
    from tiny_rgb_full_training_tools import decode_rgb_video_fast, masked_length_buckets
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutput

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
        raise ValueError("hybrid diagnosis needs fresh runs/ output")
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
    reference = cpu(padded, mask, target)
    reference.loss.backward()
    reference_gradients = gradients(cpu)
    hybrid = copy.deepcopy(cpu).eval()
    hybrid.zero_grad(set_to_none=True)
    hybrid.vision.to("mps")
    memory, encoded_mask = hybrid.vision(padded.to("mps"), mask.to("mps"))
    memory = hybrid.memory_norm(hybrid.visual_projection(memory.cpu()))
    actual = hybrid.language_model(
        encoder_outputs=BaseModelOutput(last_hidden_state=memory),
        attention_mask=encoded_mask.cpu().long(),
        labels=target,
        use_cache=False,
        return_dict=True,
    )
    actual.loss.backward()
    comparison = compare(
        reference_gradients,
        gradients(hybrid),
        float(reference.loss),
        float(actual.loss),
        reference.logits[:, : label.shape[1]].detach(),
        actual.logits[:, : label.shape[1]].detach(),
    )
    report = {
        "plan_sha256": sha256(args.plan),
        "failed_comparisons_sha256": sha256(args.failed_comparisons),
        "source_sha256": sha256(Path(__file__)),
        "capacity": failed["capacity"],
        "short_side": failed["short_side"],
        "training_only": True,
        "optimizer_updates": 0,
        "execution_mode": "vision_mps_language_and_bridge_cpu",
        "comparison": comparison,
    }
    output.mkdir(parents=True)
    output.joinpath("source.py").write_text(Path(__file__).read_text())
    output.joinpath("summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "metrics": {k: v for k, v in comparison.items() if k != "gradient_differences"},
                "worst_gradients": sorted(
                    comparison["gradient_differences"],
                    key=lambda g: g["relative_l2_error"],
                    reverse=True,
                )[:5],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
