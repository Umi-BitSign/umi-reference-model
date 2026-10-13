"""Check scratch-model learnability on synthetic colors; this is not an ASL quality test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tiny_rgb_model import PAD, TinyRgbTranslator, decode_text, encode_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(root / "runs"):
        raise ValueError("CPU smoke receipt requires a new private runs/ output")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(317)
    model = TinyRgbTranslator().cpu()
    video = torch.full((2, 2, 3, 32, 56), -1.0)
    video[0, :, 0] = 1
    video[1, :, 2] = 1
    mask = torch.ones(2, 2, dtype=torch.bool)
    references = ["red", "blue"]
    encoded = [encode_text(text) for text in references]
    targets = torch.zeros(2, max(map(len, encoded)), dtype=torch.long)
    for index, tokens in enumerate(encoded):
        targets[index, : len(tokens)] = torch.tensor(tokens)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    losses = []
    for _ in range(200):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(video, mask, targets)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten(), ignore_index=PAD
        )
        if not torch.isfinite(loss):
            raise RuntimeError("scratch CPU smoke loss is nonfinite")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("scratch CPU smoke gradient coverage differs")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss))
    model.eval()
    predictions = [
        decode_text(row.tolist()) for row in model.generate(video, mask, maximum_tokens=8)
    ]
    swapped = [
        decode_text(row.tolist()) for row in model.generate(video.flip(0), mask, maximum_tokens=8)
    ]
    passed = (
        predictions == references
        and swapped == list(reversed(references))
        and losses[-1] < losses[0] * 0.1
    )
    receipt = {
        "schema": "umi-studio-tiny-rgb-cpu-smoke/1",
        "seed": 317,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "optimizer_updates": len(losses),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "references": references,
        "predictions": predictions,
        "swapped_predictions": swapped,
        "passes_synthetic_learnability_check": passed,
        "pretrained_weights_loaded": False,
        "gpu_used": False,
        "weights_saved": False,
        "asl_examples_used": 0,
        "quality_gain_verified": False,
        "miner_changed": False,
        "model_source_sha256": hashlib.sha256(
            (root / "tools/tiny_rgb_model.py").read_bytes()
        ).hexdigest(),
        "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "qualification": (
            "Synthetic color-to-word conditioning only. This proves neither ASL fitting nor "
            "generalization, runtime speed, independent confirmation or miner performance. "
            "Real training-only ASL fitting and held-out evaluation remain required."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt), flush=True)
    if not passed:
        raise RuntimeError(
            "scratch CPU smoke failed its fixed synthetic fit and visual-conditioning check"
        )


if __name__ == "__main__":
    main()
