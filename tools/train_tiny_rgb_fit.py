"""Fit two newly initialized RGB visual encoders with the same pretrained language decoder."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from benchmark_fleurs_dev import distance, normalize, sha256
from tiny_rgb_fit import (
    RECIPE,
    ROOT,
    SEED,
    SIDES,
    RgbDutyCycle,
    checked_plan,
    gradient_receipt,
    load_pixels,
    metal_working_set_bytes,
    rank,
    storage,
)


def append(path: Path, record: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def fit_summary(correct: list[dict], swapped: list[dict]) -> dict:
    if not correct or len(correct) != len(swapped):
        raise ValueError("RGB fit diagnostic requires paired correct and swapped inputs")
    if [r["example_id"] for r in correct] != [r["example_id"] for r in swapped]:
        raise ValueError("RGB swapped diagnostic identity order differs")
    mean = sum(r["wer"] for r in correct) / len(correct)
    swap_mean = sum(r["wer"] for r in swapped) / len(swapped)
    exact = sum(r["edits"] == 0 for r in correct)
    return {
        "training_mean_wer": mean,
        "training_exact_translations": exact,
        "wrong_video_mean_wer": swap_mean,
        "wrong_video_wer_increase": swap_mean - mean,
        "passes_training_fit_gate": (
            mean <= RECIPE["fit_gate_mean_wer"]
            and exact >= RECIPE["fit_gate_minimum_exact"]
            and swap_mean - mean >= RECIPE["fit_gate_minimum_swapped_wer_increase"]
        ),
        "held_out_quality_verified": False,
        "promotion_authorized": False,
    }


def initialization_digest(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if not name.startswith("language_model."):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
        raise ValueError("use the early RGB launcher before importing PyTorch")
    plan = checked_plan(args.plan)
    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("RGB fitting requires a new checkpoints/ directory")
    import torch
    from safetensors.torch import save_model
    from tiny_rgb_data import pixel_tensor
    from tiny_rgb_language_model import TinyRgbLanguageTranslator
    from tiny_rgb_model import TinyRgbConfig
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available():
        raise RuntimeError("the RGB fit protocol requires native Metal")
    working_set = metal_working_set_bytes()
    memory_fraction = min(0.5, RECIPE["maximum_process_mps_bytes"] / working_set)
    torch.mps.set_per_process_memory_fraction(memory_fraction)
    data_dir = ROOT / plan["data_dir"]
    storage(data_dir)
    output.mkdir(parents=True)
    duty = RgbDutyCycle(torch, ROOT / plan["gpu_budget_policy"])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / plan["language_path"], local_files_only=True)
    examples = plan["examples"]
    ledger = [json.loads(s) for s in args.plan.with_name("pixels.jsonl").read_text().splitlines()]
    bindings = {(r["source"], r["example_id"]): r for r in ledger}
    if len(ledger) != len(examples) or len(bindings) != len(ledger):
        raise ValueError("RGB pixel ledger must bind every fit identity exactly once")
    labels = {}
    for row in examples:
        tokens = tokenizer(row["reference"], return_tensors="pt", truncation=False).input_ids
        if tokens.shape[1] > RECIPE["max_text_tokens"]:
            raise ValueError("RGB fit reference exceeds bound; do not truncate its target")
        labels[(row["source"], row["example_id"])] = tokens
        views = bindings[(row["source"], row["example_id"])]
        first_indices = None
        for side in SIDES:
            pixels, indices = load_pixels(row, side, views)
            if first_indices is not None and not np.array_equal(first_indices, indices):
                raise ValueError("RGB fit arms differ in physical frame indices")
            first_indices = indices
            del pixels
    (output / "run.json").write_text(
        json.dumps(
            {
                "schema": "umi-studio-tiny-rgb-fit-run/1",
                "plan_sha256": sha256(args.plan),
                "training_cases": len(examples),
                "recipe": RECIPE,
                "gpu_budget": duty.budget.record,
                "miner_mode": plan["miner_mode"],
                "miner_health_claim": False,
                "pretrained_visual_weights": False,
                "learned_feature_inputs": False,
                "c5_teacher_targets": False,
                "language": plan["language"],
                "quality_verified": False,
                "metal_recommended_working_set_bytes": working_set,
                "allocator_memory_fraction": memory_fraction,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    summaries = []
    initial_digest = None
    for side in SIDES:
        text_model = T5ForConditionalGeneration.from_pretrained(
            ROOT / plan["language_path"], local_files_only=True
        )
        # Reset after loading language weights to isolate the visual initialization seed.
        torch.manual_seed(SEED)
        model = TinyRgbLanguageTranslator(text_model, TinyRgbConfig(**plan["visual_config"]))
        digest = initialization_digest(model)
        if initial_digest is not None and digest != initial_digest:
            raise RuntimeError("RGB resolution arms differ in initial visual/bridge weights")
        initial_digest = digest
        counts = model.parameter_counts()
        duty.run(lambda model=model: model.to("mps"))
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if not n.startswith("language_model.")
                    ],
                    "lr": RECIPE["visual_learning_rate"],
                },
                {
                    "params": list(model.language_model.parameters()),
                    "lr": RECIPE["language_learning_rate"],
                },
            ],
            weight_decay=RECIPE["weight_decay"],
        )

        def sample(row, side=side):
            binding = bindings[(row["source"], row["example_id"])]
            pixels, _ = load_pixels(row, side, binding)
            video = pixel_tensor(pixels)[None].to("mps")
            mask = torch.ones(1, len(pixels), dtype=torch.bool, device="mps")
            target = labels[(row["source"], row["example_id"])].to("mps")
            return video, mask, target

        def train_one(row, update: bool, model=model, optimizer=optimizer, sample=sample):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            video, mask, target = sample(row)
            result = model(video, mask, target)
            if not torch.isfinite(result.loss):
                raise RuntimeError("RGB fit loss is nonfinite")
            result.loss.backward()
            gradients = gradient_receipt(model, torch)
            norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE["gradient_clip"])
            )
            if not math.isfinite(norm):
                raise RuntimeError("RGB fit total gradient norm is nonfinite")
            loss = float(result.loss)
            if update:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            return {
                "loss": loss,
                "gradient_norm": norm,
                "gradient_coverage": gradients,
                "frames": video.shape[1],
                "pixel_shape": list(video.shape[-2:]),
            }

        def shape(row, side=side):
            return bindings[(row["source"], row["example_id"])]["views"][str(side)]["shape"]

        probes = {
            min(range(len(examples)), key=lambda i, shape=shape: shape(examples[i])[0]),
            max(range(len(examples)), key=lambda i, shape=shape: shape(examples[i])[0]),
            max(range(len(examples)), key=lambda i, shape=shape: math.prod(shape(examples[i])[:3])),
        }
        for index in sorted(probes):
            receipt = duty.run(
                lambda index=index, train_one=train_one: train_one(examples[index], False)
            )
            append(
                output / "preflight.jsonl",
                {
                    "short_side": side,
                    "example_id": examples[index]["example_id"],
                    "optimizer_updates": 0,
                    "parameter_counts": counts,
                    "visual_initialization_sha256": digest,
                    **receipt,
                    **duty.receipt(),
                },
            )

        def conditioning_probe(model=model, sample=sample):
            model.eval()
            video, mask, target = sample(examples[0])
            with torch.no_grad():
                normal = model(video, mask, target).logits
                perturbed = model(-video, mask, target).logits
                difference = float((normal - perturbed).abs().max())
            if not math.isfinite(difference) or difference <= 1e-6:
                raise RuntimeError("RGB pixels do not influence the pretrained decoder")
            return difference

        difference = duty.run(conditioning_probe)
        append(
            output / "preflight.jsonl",
            {
                "short_side": side,
                "optimizer_updates": 0,
                "maximum_logit_change_from_negated_pixels": difference,
                **duty.receipt(),
            },
        )

        for epoch in range(RECIPE["epochs"] + 1):
            if epoch:
                storage(data_dir)
                order = sorted(examples, key=lambda row, epoch=epoch: rank(row, f"epoch-{epoch}"))
                losses = []
                for index, row in enumerate(order):
                    receipt = duty.run(lambda row=row, train_one=train_one: train_one(row, True))
                    losses.append(receipt["loss"])
                    append(
                        output / "steps.jsonl",
                        {
                            "short_side": side,
                            "epoch": epoch,
                            "update": (epoch - 1) * len(examples) + index + 1,
                            "example_id": row["example_id"],
                            **receipt,
                            **duty.receipt(),
                        },
                    )
                print(
                    json.dumps(
                        {
                            "short_side": side,
                            "completed_epoch": epoch,
                            "mean_training_loss": sum(losses) / len(losses),
                            **duty.receipt(),
                        }
                    ),
                    flush=True,
                )
            if epoch not in RECIPE["checks_at_epochs"]:
                continue
            model.eval()
            correct, swapped = [], []
            for swap, results in ((False, correct), (True, swapped)):
                for index, row in enumerate(examples):
                    input_row = examples[(index + 1) % len(examples)] if swap else row

                    def predict(input_row=input_row, model=model, sample=sample):
                        video, mask, _ = sample(input_row)
                        with torch.no_grad():
                            tokens = model.generate(video, mask, **RECIPE["generation"])
                        return tokenizer.decode(tokens[0].cpu(), skip_special_tokens=True)

                    hypothesis = duty.run(predict)
                    reference = normalize(row["reference"])
                    edits = distance(reference, normalize(hypothesis))
                    result = {
                        "short_side": side,
                        "epoch": epoch,
                        "example_id": row["example_id"],
                        "input_example_id": input_row["example_id"],
                        "source": row["source"],
                        "training_only": True,
                        "wrong_video": swap,
                        "hypothesis": hypothesis,
                        "edits": edits,
                        "reference_words": len(reference),
                        "wer": edits / len(reference),
                    }
                    results.append(result)
                    append(output / "fit-predictions.jsonl", result)
            summary = {
                "short_side": side,
                "epoch": epoch,
                **fit_summary(correct, swapped),
                **duty.receipt(),
            }
            summaries.append(summary)
            (output / "fit-decisions.json").write_text(
                json.dumps(summaries, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps(summary), flush=True)
            if epoch in (16, 32):
                storage(data_dir, counts["total"] * 4 + 1024**2)
                checkpoint = output / f"rgb-cap{side}-epoch{epoch:02d}.safetensors"
                duty.run(
                    lambda checkpoint=checkpoint, model=model: save_model(model, str(checkpoint))
                )
                append(
                    output / "checkpoints.jsonl",
                    {
                        "short_side": side,
                        "epoch": epoch,
                        "path": str(checkpoint.relative_to(ROOT)),
                        "sha256": sha256(checkpoint),
                        "training_fit_summary": summary,
                    },
                )
        del sample, train_one, conditioning_probe, predict, shape
        del model, text_model, optimizer
        gc.collect()
        torch.mps.empty_cache()
    checked_plan(args.plan)
    decision = {
        "schema": "umi-studio-tiny-rgb-fit-decision/1",
        "plan_sha256": sha256(args.plan),
        "complete": True,
        "training_examples_per_arm": len(examples),
        "optimizer_updates_per_arm": RECIPE["epochs"] * len(examples),
        "checks": summaries,
        "parameter_counts": counts,
        "identical_visual_initialization_sha256": initial_digest,
        "held_out_quality_verified": False,
        "selected_model_changed": False,
        "miner_modified": False,
        "confirmation_decoded": False,
        **duty.receipt(),
        "qualification": (
            "Training-set learnability and pixel-conditioning diagnostics only. Full-corpus "
            "training, protected source-held-out generalization and independent confirmation "
            "remain required."
        ),
    }
    (output / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "complete": True,
                "decision_sha256": sha256(output / "decision.json"),
                **duty.receipt(),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
