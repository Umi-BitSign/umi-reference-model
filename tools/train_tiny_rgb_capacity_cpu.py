"""Matched CPU learnability controls; Metal failures never authorize GPU training."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMERICAL_PLAN = ROOT / "runs/tiny-rgb-capacity-numerics-v6-duration-integrity/plan.json"
SCHEMA = "umi-studio-tiny-rgb-capacity-cpu-fit/1"
SEED = 317
RECIPE = {
    "capacities": ["small", "medium", "large"],
    "short_side": 192,
    "epochs": 256,
    "checks_at_epochs": [0, 1, 4, 8, 16, 32, 64, 128, 256],
    "checkpoints_at_epochs": [32, 128, 256],
    "visual_learning_rate": 3e-4,
    "language_learning_rate": 5e-5,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "gradient_accumulation": 1,
    "target_fps": 12.5,
    "max_frames": 512,
    "max_text_tokens": 128,
    "fit_gate_mean_wer": 0.25,
    "fit_gate_minimum_exact": 12,
    "fit_gate_minimum_swapped_wer_increase": 0.10,
    "generation": {"num_beams": 1, "max_new_tokens": 128, "do_sample": False},
    "augmentation": "none",
    "execution_device": "cpu",
    "initial_weights_reused": False,
}


def sources():
    from benchmark_fleurs_dev import sha256
    from tiny_rgb_capacity_protocol import sources as numerical_sources

    return {**numerical_sources(), "tools/train_tiny_rgb_capacity_cpu.py": sha256(Path(__file__))}


def storage(prospective=0):
    if shutil.disk_usage(ROOT).free < 80 * 1024**3 + 128 * 1024**2 + prospective:
        raise RuntimeError("CPU capacity experiment reached physical disk floor")


def inputs():
    from audit_tiny_rgb_duration import checked_duration_audit
    from tiny_rgb_capacity_protocol import DURATION_AUDIT, checked_plan
    from tiny_rgb_fit import select_fit_cases

    numerical = checked_plan(NUMERICAL_PLAN)
    audit, rows = checked_duration_audit(DURATION_AUDIT)
    return numerical, audit, select_fit_cases(rows)


def checked_plan(path):
    from benchmark_fleurs_dev import sha256
    from tiny_rgb_capacity_protocol import BUDGET, DURATION_AUDIT
    from tiny_rgb_full_run import runtime_binding

    plan = json.loads(path.read_text())
    if (
        plan["schema"] != SCHEMA
        or plan["seed"] != SEED
        or plan["recipe"] != RECIPE
        or plan["source_sha256"] != sources()
        or plan["runtime_binding"] != runtime_binding()
        or plan["numerical_plan_sha256"] != sha256(NUMERICAL_PLAN)
        or plan["metal_training_qualified"] is not False
        or plan["gpu_used"] is not False
        or plan["miner_modified"] is not False
        or plan["resource_policy"] != str(BUDGET.relative_to(ROOT))
    ):
        raise ValueError("CPU capacity immutable recipe/source/runtime changed")
    for name, digest in plan["dependency_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("CPU capacity clip/language dependency changed")
    numerical, audit, examples = inputs()
    expected_dependencies = dict(numerical["dependency_sha256"])
    for source in (NUMERICAL_PLAN, DURATION_AUDIT, BUDGET):
        expected_dependencies[str(source.relative_to(ROOT))] = sha256(source)
    expected_dependencies.update({row["video"]: row["video_sha256"] for row in examples})
    if (
        examples != plan["examples"]
        or audit["valid_training_clips"] != plan["eligible_training_clips"]
        or numerical["cases"] != plan["preflight_cases"]
        or numerical["language_path"] != plan["language_path"]
        or expected_dependencies != plan["dependency_sha256"]
    ):
        raise ValueError("CPU capacity source-protected selection changed")
    snapshot = json.loads(path.with_name("source-snapshot").joinpath("manifest.json").read_text())
    if snapshot["plan_sha256"] != sha256(path):
        raise ValueError("CPU capacity plan differs from archived source")
    return plan


def freeze(path):
    from benchmark_fleurs_dev import sha256
    from c5_experiment_sources import archive_sources
    from tiny_rgb_capacity_protocol import BUDGET, DURATION_AUDIT
    from tiny_rgb_full_run import runtime_binding

    numerical, audit, examples = inputs()
    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("CPU capacity fit needs a new runs/ directory")
    storage()
    dependencies = dict(numerical["dependency_sha256"])
    for source in (NUMERICAL_PLAN, DURATION_AUDIT, BUDGET):
        dependencies[str(source.relative_to(ROOT))] = sha256(source)
    for row in examples:
        dependencies[row["video"]] = row["video_sha256"]
    plan = {
        "schema": SCHEMA,
        "recipe": dict(RECIPE),
        "examples": examples,
        "preflight_cases": numerical["cases"],
        "eligible_training_clips": audit["valid_training_clips"],
        "seed": SEED,
        "numerical_plan_sha256": sha256(NUMERICAL_PLAN),
        "source_sha256": sources(),
        "runtime_binding": runtime_binding(),
        "dependency_sha256": dependencies,
        "language_path": numerical["language_path"],
        "resource_policy": str(BUDGET.relative_to(ROOT)),
        "metal_training_qualified": False,
        "gpu_used": False,
        "miner_modified": False,
        "qualification": "CPU-native arithmetic and training-only learnability. Failed Metal "
        "qualification is preserved and does not authorize Metal updates. Fixed192 pixels, "
        "same human data/language/rates/steps across capacities; no held-out quality or promotion.",
    }
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    archive_sources(path, path.with_name("source-snapshot"))
    checked_plan(path)
    print(
        json.dumps(
            {
                "plan_sha256": sha256(path),
                "training_cases": len(examples),
                "optimizer_updates_per_capacity": 256 * len(examples),
            }
        ),
        flush=True,
    )


def run(path, output):
    import torch
    from benchmark_fleurs_dev import distance, normalize, sha256
    from c5_certification_budget import CertificationComputeWindow
    from probe_tiny_rgb_capacity import compare, gradients, native_gelus, state_digest
    from safetensors.torch import save_model
    from tiny_rgb_capacity import build_capacity_translator
    from tiny_rgb_capacity_protocol import verified_comparison
    from tiny_rgb_data import pixel_tensor
    from tiny_rgb_fit import gradient_receipt, rank
    from tiny_rgb_full_training_tools import decode_rgb_video_fast, masked_length_buckets
    from train_tiny_rgb_fit import fit_summary
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    plan = checked_plan(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("CPU capacity fit requires a fresh checkpoints/ output")
    storage()
    output.mkdir(parents=True)
    budget = CertificationComputeWindow(ROOT / plan["resource_policy"])
    torch.set_num_threads(budget.cpu_threads())
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / plan["language_path"], local_files_only=True)
    examples = plan["examples"]
    cache = {}
    all_rows = {(r["source"], r["example_id"]): r for r in examples}
    all_rows.update(
        {
            (c["training"]["source"], c["training"]["example_id"]): c["training"]
            for c in plan["preflight_cases"]
        }
    )
    for key, row in all_rows.items():
        pixels, _, _ = decode_rgb_video_fast(
            row["video"], RECIPE["short_side"], RECIPE["target_fps"], RECIPE["max_frames"]
        )
        video = pixel_tensor(pixels)[None]
        label = tokenizer(row["reference"], return_tensors="pt", truncation=False).input_ids
        if label.shape[1] > 256 or (row in examples and label.shape[1] > RECIPE["max_text_tokens"]):
            raise ValueError("CPU capacity never truncates targets")
        cache[key] = (video, label, *masked_length_buckets(video, label, torch))
    output.joinpath("run.json").write_text(
        json.dumps(
            {
                "plan_sha256": sha256(path),
                "device": "cpu",
                "training_cases": len(examples),
                "recipe": RECIPE,
                "gpu_used": False,
                "miner_modified": False,
                "metal_training_qualified": False,
            },
            indent=2,
        )
        + "\n"
    )

    def append(name, row):
        with output.joinpath(name).open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    summaries = []
    for capacity in RECIPE["capacities"]:
        language = T5ForConditionalGeneration.from_pretrained(
            ROOT / plan["language_path"], local_files_only=True, use_safetensors=True
        )
        language_digest = state_digest(
            {n: v for n, v in language.state_dict().items() if not n.startswith("encoder.")}
        )
        torch.manual_seed(plan["seed"])
        model = build_capacity_translator(language, capacity).eval()
        if state_digest(model.language_model.state_dict()) != language_digest:
            raise RuntimeError("CPU capacity changed supplied language weights")
        counts = model.parameter_counts()
        append(
            "initializations.jsonl",
            {
                "capacity": capacity,
                "seed": plan["seed"],
                "language_weights_sha256": language_digest,
                "visual_weights_sha256": state_digest(model.vision.state_dict()),
                "parameter_counts": counts,
                "gpu_used": False,
            },
        )
        # Every capacity must pass the complete real training-extrema CPU proof before learning.
        for case in plan["preflight_cases"]:
            torch.set_num_threads(budget.cpu_threads())
            row = case["training"]
            video, label, padded, mask, target = cache[(row["source"], row["example_id"])]
            native = copy.deepcopy(model)
            native_gelus(native.vision)
            plain_mask = torch.ones(1, video.shape[1], dtype=torch.bool)
            original = native(video, plain_mask, label)
            original.loss.backward()
            reference = gradients(native)
            actual = model(video, plain_mask, label)
            actual.loss.backward()
            first = gradients(model)
            equivalence = compare(
                reference,
                first,
                float(original.loss),
                float(actual.loss),
                original.logits.detach(),
                actual.logits.detach(),
            )
            append(
                "preflight.jsonl",
                {
                    "capacity": capacity,
                    "example_id": row["example_id"],
                    "stage": "native_cpu_equivalence",
                    "comparison": equivalence,
                    "optimizer_updates": 0,
                    "parameter_counts": counts,
                    "gpu_used": False,
                },
            )
            verified_comparison(equivalence, len(list(model.parameters())), cpu_equivalence=True)
            model.zero_grad(set_to_none=True)
            bucketed = model(padded, mask, target)
            bucketed.loss.backward()
            padding = compare(
                first,
                gradients(model),
                float(actual.loss),
                float(bucketed.loss),
                actual.logits.detach(),
                bucketed.logits[:, : label.shape[1]].detach(),
            )
            append(
                "preflight.jsonl",
                {
                    "capacity": capacity,
                    "example_id": row["example_id"],
                    "stage": "cpu_padding",
                    "comparison": padding,
                    "optimizer_updates": 0,
                    "parameter_counts": counts,
                    "gpu_used": False,
                },
            )
            verified_comparison(padding, len(list(model.parameters())))
            model.zero_grad(set_to_none=True)
            del native, original, actual, bucketed, reference, first
            gc.collect()
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
            foreach=False,
        )
        for epoch in range(RECIPE["epochs"] + 1):
            if epoch:
                model.train()
                losses = []
                for index, row in enumerate(
                    sorted(examples, key=lambda r, epoch=epoch: rank(r, f"epoch-{epoch}"))
                ):
                    torch.set_num_threads(budget.cpu_threads())
                    optimizer.zero_grad(set_to_none=True)
                    _, _, video, mask, label = cache[(row["source"], row["example_id"])]
                    started = time.perf_counter()
                    result = model(video, mask, label)
                    if not torch.isfinite(result.loss):
                        raise RuntimeError("CPU capacity loss is nonfinite")
                    result.loss.backward()
                    coverage = gradient_receipt(model, torch)
                    norm = float(
                        torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE["gradient_clip"])
                    )
                    if not math.isfinite(norm):
                        raise RuntimeError("CPU capacity gradient norm is nonfinite")
                    optimizer.step()
                    losses.append(float(result.loss))
                    append(
                        "steps.jsonl",
                        {
                            "capacity": capacity,
                            "epoch": epoch,
                            "update": (epoch - 1) * len(examples) + index + 1,
                            "loss": losses[-1],
                            "gradient_norm": norm,
                            "gradient_coverage": coverage,
                            "step_seconds": time.perf_counter() - started,
                            "cpu_threads": budget.cpu_threads(),
                        },
                    )
                    del result
                if epoch % 8 == 0:
                    storage()
                    print(
                        json.dumps(
                            {
                                "capacity": capacity,
                                "completed_epoch": epoch,
                                "mean_loss": sum(losses) / len(losses),
                            }
                        ),
                        flush=True,
                    )
            if epoch not in RECIPE["checks_at_epochs"]:
                continue
            model.eval()
            correct, swapped = [], []
            for swap, records in ((False, correct), (True, swapped)):
                for index, row in enumerate(examples):
                    torch.set_num_threads(budget.cpu_threads())
                    input_row = examples[(index + 1) % len(examples)] if swap else row
                    _, _, video, mask, _ = cache[(input_row["source"], input_row["example_id"])]
                    with torch.no_grad():
                        tokens = model.generate(video, mask, **RECIPE["generation"])
                    hypothesis = tokenizer.decode(tokens[0], skip_special_tokens=True)
                    words = len(normalize(row["reference"]))
                    edits = distance(normalize(row["reference"]), normalize(hypothesis))
                    record = {
                        "capacity": capacity,
                        "epoch": epoch,
                        "example_id": row["example_id"],
                        "input_example_id": input_row["example_id"],
                        "wrong_video": swap,
                        "training_only": True,
                        "hypothesis": hypothesis,
                        "edits": edits,
                        "reference_words": words,
                        "wer": edits / words,
                    }
                    records.append(record)
                    append("fit-predictions.jsonl", record)
            summary = {"capacity": capacity, "epoch": epoch, **fit_summary(correct, swapped)}
            expected_fit = (
                summary["training_mean_wer"] <= RECIPE["fit_gate_mean_wer"]
                and summary["training_exact_translations"] >= RECIPE["fit_gate_minimum_exact"]
                and summary["wrong_video_wer_increase"]
                >= RECIPE["fit_gate_minimum_swapped_wer_increase"]
            )
            if summary["passes_training_fit_gate"] != expected_fit:
                raise RuntimeError("CPU capacity fit gate differs from pinned recipe")
            summaries.append(summary)
            output.joinpath("fit-decisions.json").write_text(json.dumps(summaries, indent=2) + "\n")
            print(json.dumps(summary), flush=True)
            if epoch in RECIPE["checkpoints_at_epochs"]:
                storage(counts["total"] * 4 + 1024**2)
                checkpoint = output / f"{capacity}-epoch{epoch:03d}.safetensors"
                save_model(model, str(checkpoint))
                append(
                    "checkpoints.jsonl",
                    {
                        "capacity": capacity,
                        "epoch": epoch,
                        "checkpoint": str(checkpoint.relative_to(ROOT)),
                        "sha256": sha256(checkpoint),
                    },
                )
        del model, language, optimizer
        gc.collect()
    checked_plan(path)
    decision = {
        "complete": True,
        "plan_sha256": sha256(path),
        "checks": summaries,
        "training_cases": len(examples),
        "optimizer_updates_per_capacity": 256 * len(examples),
        "held_out_quality_verified": False,
        "selected_model_changed": False,
        "miner_modified": False,
        "gpu_used": False,
        "metal_training_qualified": False,
    }
    output.joinpath("decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(
        json.dumps({"complete": True, "decision_sha256": sha256(output / "decision.json")}),
        flush=True,
    )


def main():
    for key, value in {
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items():
        os.environ[key] = value
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    elif args.plan and args.output_dir:
        run(args.plan, args.output_dir)
    else:
        parser.error("use --freeze or --plan with --output-dir")


if __name__ == "__main__":
    main()
