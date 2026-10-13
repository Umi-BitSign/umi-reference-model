"""Compare frozen zero-update raw-RGB loss and gradients on CPU and native Metal."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path


def main() -> None:
    for key, value in {
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items():
        os.environ[key] = value
    import torch
    from benchmark_fleurs_dev import sha256
    from c5_experiment_sources import archive_sources
    from tiny_rgb_data import pixel_tensor
    from tiny_rgb_fit import (
        ROOT,
        SEED,
        SIDES,
        RgbDutyCycle,
        checked_plan,
        load_pixels,
        metal_working_set_bytes,
    )
    from tiny_rgb_language_model import TinyRgbLanguageTranslator
    from tiny_rgb_model import RmsNorm, TinyRgbConfig
    from tiny_rgb_mps_safe import replace_native_gelu
    from train_tiny_rgb_fit import initialization_digest
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    from transformers.models.t5.modeling_t5 import T5LayerNorm

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--portable-gelu", action="store_true")
    args = parser.parse_args()
    plan = checked_plan(args.plan)
    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("RGB numerical probe needs a new runs/ directory")
    output.mkdir(parents=True)
    archive_sources(args.plan, output / "source-snapshot")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available():
        raise RuntimeError("RGB numerical probe requires native Metal")
    torch.mps.set_per_process_memory_fraction(6 * 1024**3 / metal_working_set_bytes())
    duty = RgbDutyCycle(torch, ROOT / plan["gpu_budget_policy"])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / plan["language_path"], local_files_only=True)
    ledger = [json.loads(s) for s in args.plan.with_name("pixels.jsonl").read_text().splitlines()]
    bindings = {(r["source"], r["example_id"]): r for r in ledger}
    examples = plan["examples"]
    sizes = [bindings[(r["source"], r["example_id"])]["views"]["128"]["shape"][0] for r in examples]
    selected = sorted(
        {
            min(range(len(examples)), key=lambda i: sizes[i]),
            max(range(len(examples)), key=lambda i: sizes[i]),
        }
    )
    reports = []
    for side in SIDES:
        for index in selected:
            row = examples[index]
            text_model = T5ForConditionalGeneration.from_pretrained(
                ROOT / plan["language_path"], local_files_only=True
            )
            torch.manual_seed(SEED)
            model = TinyRgbLanguageTranslator(text_model, TinyRgbConfig(**plan["visual_config"]))
            replacements = replace_native_gelu(model) if args.portable_gelu else 0
            digest = initialization_digest(model)
            model.eval()
            pixels, _ = load_pixels(row, side, bindings[(row["source"], row["example_id"])])
            video = pixel_tensor(pixels)[None]
            mask = torch.ones(1, len(pixels), dtype=torch.bool)
            target = tokenizer(row["reference"], return_tensors="pt", truncation=False).input_ids
            activations = {"cpu": {}, "mps": {}}

            def activation(name, inputs, result):
                value = inputs[0].detach().float()
                return {
                    "input_rms": float(value.square().mean().sqrt()),
                    "input_min_abs": float(value.abs().min()),
                    "output_max_abs": float(result.detach().float().abs().max()),
                    "module": name,
                }

            handles = []
            for name, module in model.named_modules():
                if isinstance(module, RmsNorm | T5LayerNorm):

                    def hook(module, inputs, result, name=name, activations=activations):
                        device = inputs[0].device.type
                        activations[device][name] = activation(name, inputs, result)

                    handles.append(module.register_forward_hook(hook))
            result = model(video, mask, target)
            cpu_loss = float(result.loss)
            cpu_logits = result.logits.detach().clone()
            result.loss.backward()
            cpu_gradients = {
                n: p.grad.detach().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
            cpu_norm = math.sqrt(
                sum(float(g.double().square().sum()) for g in cpu_gradients.values())
            )
            del result
            model.zero_grad(set_to_none=True)

            def metal_probe(
                model=model,
                video=video,
                mask=mask,
                target=target,
                cpu_gradients=cpu_gradients,
                cpu_logits=cpu_logits,
            ):
                model.to("mps")
                metal_result = model(video.to("mps"), mask.to("mps"), target.to("mps"))
                metal_result.loss.backward()
                differences = []
                for name, parameter in model.named_parameters():
                    if parameter.grad is None or name not in cpu_gradients:
                        raise RuntimeError(f"RGB numerical probe lacks paired gradient: {name}")
                    reference = cpu_gradients[name].double()
                    actual = parameter.grad.detach().cpu().double()
                    norm = float(reference.norm())
                    actual_norm = float(actual.norm())
                    relative_error = float((actual - reference).norm()) / max(norm, 1e-12)
                    differences.append(
                        {
                            "parameter": name,
                            "cpu_norm": norm,
                            "mps_norm": actual_norm,
                            "relative_l2_error": relative_error,
                            "finite": bool(torch.isfinite(actual).all()),
                        }
                    )
                return {
                    "mps_loss": float(metal_result.loss),
                    "logit_max_abs_difference": float(
                        (metal_result.logits.detach().cpu() - cpu_logits).abs().max()
                    ),
                    "mps_total_gradient_norm": math.sqrt(
                        sum(r["mps_norm"] ** 2 for r in differences)
                    ),
                    "gradient_differences": differences,
                }

            comparison = duty.run(metal_probe)
            report = {
                "short_side": side,
                "example_id": row["example_id"],
                "frames": len(pixels),
                "cpu_loss": cpu_loss,
                "cpu_total_gradient_norm": cpu_norm,
                "visual_initialization_sha256": digest,
                "optimizer_updates": 0,
                "dropout_enabled": False,
                "portable_gelu_replacements": replacements,
                "normalization_activations": activations,
                **comparison,
                **duty.receipt(),
            }
            reports.append(report)
            output.joinpath("comparisons.json").write_text(
                json.dumps(reports, indent=2, sort_keys=True) + "\n"
            )
            worst = sorted(
                comparison["gradient_differences"],
                key=lambda r: r["relative_l2_error"],
                reverse=True,
            )[:5]
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in report.items()
                        if k not in ("gradient_differences", "normalization_activations")
                    }
                    | {"worst_gradient_differences": worst}
                ),
                flush=True,
            )
            for handle in handles:
                handle.remove()
            del metal_probe
            del model, text_model, cpu_gradients, cpu_logits, video, mask, target
            gc.collect()
            torch.mps.empty_cache()
    checked_plan(args.plan)
    receipt = {
        "schema": "umi-studio-tiny-rgb-numerics/1",
        "plan_sha256": sha256(args.plan),
        "probe_source_sha256": sha256(Path(__file__)),
        "comparisons_sha256": sha256(output / "comparisons.json"),
        "optimizer_updates": 0,
        "weights_saved": False,
        "portable_gelu_enabled": args.portable_gelu,
        "portable_gelu_source_sha256": sha256(ROOT / "tools/tiny_rgb_mps_safe.py"),
        "training_only": True,
        "selected_model_changed": False,
        "miner_modified": False,
        "qualification": (
            "Backend and initialization diagnosis on protected-eligible fit cases only; "
            "no quality or generalization evidence."
        ),
        **duty.receipt(),
    }
    output.joinpath("receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
