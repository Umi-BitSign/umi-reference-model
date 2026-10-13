"""Train C5 visual LoRA with a low-rate, anchored direct projection."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_c5_anchored_projection_plan import (
    BASELINE,
    RECIPE,
    ROOT,
    SEED,
    frozen_rows,
    load_examples,
    load_expansion_examples,
    load_openasl_examples,
    select_schedule,
)
from interpolate_visual_adapter import DIRECT_NAMES
from train_c5_stem_continuation import stride2_streams


def relative_anchor_penalty(selected, anchors, variance_floor: float):
    import torch

    terms = []
    for name in sorted(DIRECT_NAMES):
        parameter = selected[name]
        anchor = anchors[name].to(parameter.device)
        scale = anchor.square().mean().clamp_min(variance_floor)
        terms.append(torch.nn.functional.mse_loss(parameter, anchor) / scale)
    return torch.stack(terms).mean()


def direct_drift(selected, anchors) -> dict[str, float]:
    import torch

    return {
        name: float(
            (selected[name].detach().cpu() - anchors[name]).norm()
            / anchors[name].norm().clamp_min(torch.finfo(anchors[name].dtype).eps)
        )
        for name in sorted(DIRECT_NAMES)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    args = parser.parse_args()
    if (
        not args.miner_maintenance_mode
        or not args.plan.is_file()
        or args.output_dir.exists()
        or not args.output_dir.resolve().is_relative_to((ROOT / "checkpoints").resolve())
    ):
        parser.error("anchored training requires maintenance mode and new output")
    plan = json.loads(args.plan.read_text())
    if (
        plan.get("schema") != "umi-studio-c5-anchored-projection-plan/1"
        or plan.get("recipe") != RECIPE
        or plan.get("seed") != SEED
        or plan.get("trainer_source_sha256") != sha256(Path(__file__))
    ):
        raise ValueError("anchored projection plan differs from the trainer")
    replay, stem, source_hashes = load_examples()
    expansion, expansion_hashes = load_expansion_examples()
    openasl, openasl_hashes = load_openasl_examples()
    if {**source_hashes, **expansion_hashes, **openasl_hashes} != plan["source_hashes"]:
        raise ValueError("anchored projection source indexes changed")
    schedule = select_schedule(replay, stem, expansion, openasl)
    schedule_path = args.plan.with_name("schedule.jsonl")
    if (
        sha256(schedule_path) != plan["schedule_sha256"]
        or [json.loads(line) for line in schedule_path.read_text().splitlines()]
        != frozen_rows(schedule)
    ):
        raise ValueError("anchored projection schedule changed")
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(BASELINE / "run/hf"),
        "TORCH_HOME": str(BASELINE / "run/torch"),
        "MPLCONFIGDIR": str(BASELINE / "run/matplotlib"),
        "SHUBERT_DINOV2_SOURCE": str(BASELINE / "vendor/dinov2-source"),
        "OMP_NUM_THREADS": str(RECIPE["cpu_threads"]),
        "OPENBLAS_NUM_THREADS": str(RECIPE["cpu_threads"]),
    }.items():
        os.environ[key] = value
    sys.path.insert(0, str(BASELINE / "vendor"))
    sys.path.insert(0, str(BASELINE / "vendor/demo"))
    import torch
    from inference import SignLanguageByT5ForConditionalGeneration
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from shubert_visual_lora import install_visual_lora, load_visual_adapter
    from transformers import ByT5Tokenizer

    torch.set_num_threads(RECIPE["cpu_threads"])
    torch.manual_seed(SEED)
    checkpoint = BASELINE / "models/checkpoint-11625"
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    selected = install_visual_lora(
        model, RECIPE["rank"], RECIPE["alpha"], RECIPE["visual_last_layers"]
    )
    starting_adapter = ROOT / plan["starting_adapter"]
    with safe_open(str(starting_adapter), framework="pt", device="cpu") as source:
        metadata = source.metadata()
    if (
        sha256(starting_adapter) != plan["starting_adapter_sha256"]
        or metadata.get("schema") != "umi-studio-shubert-visual-lora/1"
        or metadata.get("baseline_checkpoint_sha256") != plan["baseline_checkpoint_sha256"]
    ):
        raise ValueError("anchored projection starting adapter differs")
    load_visual_adapter(model, load_file(str(starting_adapter), device="cpu"))
    anchors = {name: selected[name].detach().cpu().clone() for name in DIRECT_NAMES}
    lora = {name: value for name, value in selected.items() if name not in DIRECT_NAMES}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(lora.values()),
                "lr": RECIPE["lora_learning_rate"],
                "weight_decay": RECIPE["weight_decay"],
            },
            {
                "params": [selected[name] for name in sorted(DIRECT_NAMES)],
                "lr": RECIPE["direct_learning_rate"],
                "weight_decay": 0.0,
            },
        ]
    )
    tokenizer = ByT5Tokenizer.from_pretrained(
        str(BASELINE / "models/byt5_base"), local_files_only=True
    )
    model.to("cpu")
    model.eval()
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan_hash = sha256(args.plan)
    run = {
        "schema": "umi-studio-c5-anchored-projection-training/1",
        "plan_sha256": plan_hash,
        "recipe_id": RECIPE["id"],
        "starting_adapter_sha256": sha256(starting_adapter),
        "baseline_checkpoint_sha256": plan["baseline_checkpoint_sha256"],
        "scheduled_examples": len(schedule),
        "trainable_parameter_count": sum(value.numel() for value in selected.values()),
        "anchored_direct_parameter_names": sorted(DIRECT_NAMES),
        "device": "cpu",
        "cpu_threads": RECIPE["cpu_threads"],
        "miner_health_policy": "user_authorized_maintenance",
        "miner_changed": False,
        "trainer_source_sha256": sha256(Path(__file__)),
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n"
    )
    ledger = args.output_dir / "steps.jsonl"
    checkpoint_index = args.output_dir / "checkpoint-index.jsonl"
    accumulation = RECIPE["gradient_accumulation"]
    checkpoint_steps = set(RECIPE["checkpoint_after_examples"])
    for offset in range(0, len(schedule), accumulation):
        batch = schedule[offset : offset + accumulation]
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        penalties = []
        for row in batch:
            streams = stride2_streams(row)
            features = {
                name: torch.from_numpy(value).unsqueeze(0) for name, value in streams.items()
            }
            labels = tokenizer(row["reference"], return_tensors="pt").input_ids
            output = model(**features, labels=labels, use_cache=False, return_dict=True)
            supervised = torch.nn.functional.cross_entropy(
                output.logits.reshape(-1, output.logits.shape[-1]),
                labels.reshape(-1),
                label_smoothing=RECIPE["label_smoothing"],
            )
            anchor = relative_anchor_penalty(
                selected, anchors, RECIPE["direct_anchor_variance_floor"]
            )
            loss = supervised + RECIPE["direct_anchor_strength"] * anchor
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite anchored loss: {row['id']}")
            (loss / len(batch)).backward()
            losses.append(float(supervised.detach()))
            penalties.append(float(anchor.detach()))
        gradient_norm = torch.nn.utils.clip_grad_norm_(selected.values(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("non-finite anchored projection gradient")
        optimizer.step()
        example_step = offset + len(batch)
        record = {
            "example_step": example_step,
            "optimizer_step": example_step // accumulation,
            "sources": [row["source"] for row in batch],
            "example_ids": [row["id"] for row in batch],
            "supervised_losses": losses,
            "anchor_penalties": penalties,
            "gradient_norm_before_clipping": float(gradient_norm),
            "direct_relative_drift": direct_drift(selected, anchors),
            "seconds": time.perf_counter() - started,
        }
        with ledger.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        if example_step in checkpoint_steps:
            target = args.output_dir / f"visual-lora-after-{example_step:06d}.safetensors"
            temporary = target.with_suffix(".partial.safetensors")
            save_file(
                {
                    name: value.detach().cpu().contiguous().clone()
                    for name, value in selected.items()
                },
                str(temporary),
                metadata={
                    "schema": "umi-studio-shubert-visual-lora/1",
                    "plan_sha256": plan_hash,
                    "recipe_id": RECIPE["id"],
                    "rank": str(RECIPE["rank"]),
                    "alpha": str(RECIPE["alpha"]),
                    "visual_last_layers": str(RECIPE["visual_last_layers"]),
                    "baseline_checkpoint_sha256": plan["baseline_checkpoint_sha256"],
                    "starting_adapter_sha256": sha256(starting_adapter),
                    "trainer_source_sha256": sha256(Path(__file__)),
                    "example_step": str(example_step),
                    "direct_parameters_frozen": "false",
                    "direct_anchor_strength": str(RECIPE["direct_anchor_strength"]),
                },
            )
            temporary.rename(target)
            checkpoint_row = {
                "schema": "umi-studio-c5-anchored-projection-checkpoint/1",
                "plan_sha256": plan_hash,
                "example_step": example_step,
                "checkpoint_name": target.name,
                "checkpoint_sha256": sha256(target),
                "direct_relative_drift": record["direct_relative_drift"],
                "tuning_checkpoint": True,
            }
            with checkpoint_index.open("a", encoding="utf-8") as output:
                output.write(json.dumps(checkpoint_row, sort_keys=True) + "\n")
                output.flush()
                os.fsync(output.fileno())
            print(
                f"saved {target.name} mean_loss={sum(losses)/len(losses):.4f} "
                f"max_direct_drift={max(record['direct_relative_drift'].values()):.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
