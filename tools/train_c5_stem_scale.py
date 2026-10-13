"""Train the full two-tranche STEM replay with direct projection frozen."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_c5_stem_scale_plan import (
    BASELINE,
    RECIPE,
    ROOT,
    SEED,
    STARTING_ADAPTER,
    load_examples,
    select_schedule,
)
from interpolate_visual_adapter import DIRECT_NAMES
from train_c5_stem_continuation import stride2_streams
from train_c5_stem_projection_frozen import (
    complete_adapter_state,
    freeze_direct_parameters,
)


def frozen_schedule_rows(schedule: list[dict]) -> list[dict]:
    return [
        {
            "position": position,
            "source": row["source"],
            "example_id": row["id"],
            "signer": row["signer"],
            "stored_temporal_stride": row.get("stored_temporal_stride", 1),
        }
        for position, row in enumerate(schedule)
    ]


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
        parser.error("training requires maintenance mode, a frozen plan and new output")
    plan = json.loads(args.plan.read_text())
    if (
        plan.get("schema") != "umi-studio-c5-stem-scale-plan/1"
        or plan.get("recipe") != RECIPE
        or plan.get("seed") != SEED
        or plan.get("starting_adapter_sha256") != sha256(STARTING_ADAPTER)
        or plan.get("trainer_source_sha256") != sha256(Path(__file__))
    ):
        raise ValueError("scaled STEM plan differs from the trainer")
    replay, stem, source_hashes = load_examples()
    if source_hashes != plan.get("source_hashes"):
        raise ValueError("scaled STEM source indexes changed")
    schedule = select_schedule(replay, stem)
    schedule_path = args.plan.with_name("schedule.jsonl")
    schedule_rows = [json.loads(line) for line in schedule_path.read_text().splitlines()]
    if (
        sha256(schedule_path) != plan["schedule_sha256"]
        or schedule_rows != frozen_schedule_rows(schedule)
    ):
        raise ValueError("scaled STEM schedule differs from the frozen plan")
    cpu_threads = RECIPE["cpu_threads"]
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(BASELINE / "run/hf"),
        "TORCH_HOME": str(BASELINE / "run/torch"),
        "MPLCONFIGDIR": str(BASELINE / "run/matplotlib"),
        "SHUBERT_DINOV2_SOURCE": str(BASELINE / "vendor/dinov2-source"),
        "OMP_NUM_THREADS": str(cpu_threads),
        "OPENBLAS_NUM_THREADS": str(cpu_threads),
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

    torch.set_num_threads(cpu_threads)
    torch.manual_seed(SEED)
    checkpoint = BASELINE / "models/checkpoint-11625"
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    selected = install_visual_lora(
        model,
        RECIPE["rank"],
        RECIPE["alpha"],
        RECIPE["visual_last_layers"],
    )
    with safe_open(str(STARTING_ADAPTER), framework="pt", device="cpu") as source:
        metadata = source.metadata()
    if (
        metadata.get("schema") != "umi-studio-shubert-visual-lora/1"
        or metadata.get("baseline_checkpoint_sha256")
        != plan["baseline_checkpoint_sha256"]
        or metadata.get("rank") != str(RECIPE["rank"])
        or metadata.get("alpha") != str(RECIPE["alpha"])
        or metadata.get("visual_last_layers") != str(RECIPE["visual_last_layers"])
    ):
        raise ValueError("starting adapter metadata differs")
    load_visual_adapter(model, load_file(str(STARTING_ADAPTER), device="cpu"))
    with safe_open(
        str(checkpoint / "model.safetensors"), framework="pt", device="cpu"
    ) as baseline:
        for name in DIRECT_NAMES:
            if not torch.equal(selected[name].detach().cpu(), baseline.get_tensor(name)):
                raise ValueError(f"confirmed direct parameter differs: {name}")
    frozen_direct = {
        name: selected[name].detach().cpu().clone() for name in DIRECT_NAMES
    }
    trainable = freeze_direct_parameters(selected)
    tokenizer = ByT5Tokenizer.from_pretrained(
        str(BASELINE / "models/byt5_base"), local_files_only=True
    )
    model.to("cpu")
    model.eval()
    optimizer = torch.optim.AdamW(
        trainable.values(),
        lr=RECIPE["learning_rate"],
        weight_decay=RECIPE["weight_decay"],
    )
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan_hash = sha256(args.plan)
    run = {
        "schema": "umi-studio-c5-stem-scale-training/1",
        "plan_sha256": plan_hash,
        "recipe_id": RECIPE["id"],
        "starting_adapter_sha256": sha256(STARTING_ADAPTER),
        "baseline_checkpoint_sha256": plan["baseline_checkpoint_sha256"],
        "scheduled_examples": len(schedule),
        "trainable_parameter_count": sum(value.numel() for value in trainable.values()),
        "frozen_direct_parameter_names": sorted(DIRECT_NAMES),
        "device": "cpu",
        "cpu_threads": cpu_threads,
        "miner_health_policy": "user_authorized_maintenance",
        "miner_changed": False,
        "trainer_source_sha256": sha256(Path(__file__)),
    }
    with (args.output_dir / "run.json").open("x", encoding="utf-8") as output:
        json.dump(run, output, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    ledger = args.output_dir / "steps.jsonl"
    checkpoint_index = args.output_dir / "checkpoint-index.jsonl"
    accumulation = RECIPE["gradient_accumulation"]
    checkpoint_steps = set(RECIPE["checkpoint_after_examples"])
    for offset in range(0, len(schedule), accumulation):
        batch = schedule[offset : offset + accumulation]
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for row in batch:
            streams = stride2_streams(row)
            features = {
                name: torch.from_numpy(value).unsqueeze(0)
                for name, value in streams.items()
            }
            labels = tokenizer(row["reference"], return_tensors="pt").input_ids
            output = model(**features, labels=labels, use_cache=False, return_dict=True)
            loss = torch.nn.functional.cross_entropy(
                output.logits.reshape(-1, output.logits.shape[-1]),
                labels.reshape(-1),
                label_smoothing=RECIPE["label_smoothing"],
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite scaled STEM loss: {row['id']}")
            (loss / len(batch)).backward()
            losses.append(float(loss.detach()))
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("non-finite scaled STEM gradient")
        if any(selected[name].grad is not None for name in DIRECT_NAMES):
            raise RuntimeError("frozen direct parameter received a gradient")
        optimizer.step()
        if any(
            not torch.equal(selected[name].detach().cpu(), frozen_direct[name])
            for name in DIRECT_NAMES
        ):
            raise RuntimeError("frozen direct parameter changed")
        example_step = offset + len(batch)
        record = {
            "example_step": example_step,
            "optimizer_step": example_step // accumulation,
            "sources": [row["source"] for row in batch],
            "example_ids": [row["id"] for row in batch],
            "losses": losses,
            "gradient_norm_before_clipping": float(gradient_norm),
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
                complete_adapter_state(selected),
                str(temporary),
                metadata={
                    "schema": "umi-studio-shubert-visual-lora/1",
                    "plan_sha256": plan_hash,
                    "recipe_id": RECIPE["id"],
                    "rank": str(RECIPE["rank"]),
                    "alpha": str(RECIPE["alpha"]),
                    "visual_last_layers": str(RECIPE["visual_last_layers"]),
                    "baseline_checkpoint_sha256": plan["baseline_checkpoint_sha256"],
                    "starting_adapter_sha256": sha256(STARTING_ADAPTER),
                    "trainer_source_sha256": sha256(Path(__file__)),
                    "example_step": str(example_step),
                    "direct_parameters_frozen": "true",
                },
            )
            temporary.rename(target)
            checkpoint_row = {
                "schema": "umi-studio-c5-stem-scale-checkpoint/1",
                "plan_sha256": plan_hash,
                "example_step": example_step,
                "checkpoint_name": target.name,
                "checkpoint_sha256": sha256(target),
                "tuning_checkpoint": True,
            }
            with checkpoint_index.open("a", encoding="utf-8") as output:
                output.write(json.dumps(checkpoint_row, sort_keys=True) + "\n")
                output.flush()
                os.fsync(output.fileno())
            print(
                f"saved {target.name} mean_loss={sum(losses) / len(losses):.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
