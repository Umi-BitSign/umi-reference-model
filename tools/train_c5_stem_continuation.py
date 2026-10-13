"""Train the frozen stride-2 ASL STEM continuation on CPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from benchmark_fleurs_dev import sha256
from cache_fleurs_encoder import checked_stream_features
from freeze_c5_stem_continuation_plan import (
    BASELINE,
    RECIPE,
    ROOT,
    SEED,
    SOURCE_LAYOUT,
    STARTING_ADAPTER,
    load_stem,
    select_schedule,
)
from train_visual_lora import training_streams
from visual_training_data import load_visual_examples

STREAM_NAMES = (
    "face_features",
    "left_hand_features",
    "right_hand_features",
    "pose_features",
)


def stride2_streams(row: dict) -> dict:
    import numpy as np

    if row.get("stored_temporal_stride") == 2:
        cache = Path(row["cache"])
        if sha256(cache) != row["cache_sha256"]:
            raise ValueError(f"stored stride-2 cache changed: {row['id']}")
        with np.load(cache, allow_pickle=False) as payload:
            if set(payload.files) != {"encoder_hidden_state", *STREAM_NAMES}:
                raise ValueError(f"stored stride-2 cache fields differ: {row['id']}")
            streams = {name: payload[name] for name in STREAM_NAMES}
        return checked_stream_features(streams, row["frames"])
    full_rate = training_streams(row)
    streams = {
        name: np.ascontiguousarray(value[::2]) for name, value in full_rate.items()
    }
    return checked_stream_features(streams, (row["frames"] + 1) // 2)


def load_examples() -> list[dict]:
    examples = []
    for source, layout in SOURCE_LAYOUT.items():
        examples.extend(
            load_visual_examples(
                ROOT / layout["index"],
                ROOT / layout["cache_dir"],
                BASELINE,
                source,
                sidecar_index=(ROOT / layout["sidecar_index"])
                if "sidecar_index" in layout
                else None,
                sidecar_dir=(ROOT / layout["sidecar_dir"])
                if "sidecar_dir" in layout
                else None,
            )
        )
    examples.extend(load_stem())
    return examples


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
        plan.get("schema") != "umi-studio-c5-stem-continuation-plan/1"
        or plan.get("recipe") != RECIPE
        or plan.get("seed") != SEED
        or plan.get("starting_adapter_sha256") != sha256(STARTING_ADAPTER)
        or plan.get("trainer_source_sha256") != sha256(Path(__file__))
    ):
        raise ValueError("continuation plan differs from the trainer")
    examples = load_examples()
    schedule = select_schedule(
        [row for row in examples if row["source"] != "asl_stem"],
        [row for row in examples if row["source"] == "asl_stem"],
    )
    schedule_rows = [
        json.loads(line)
        for line in args.plan.with_name("schedule.jsonl").read_text().splitlines()
    ]
    if (
        sha256(args.plan.with_name("schedule.jsonl")) != plan["schedule_sha256"]
        or len(schedule_rows) != len(schedule)
        or any(
            frozen
            != {
                "position": position,
                "source": row["source"],
                "example_id": row["id"],
                "signer": row["signer"],
                "stored_temporal_stride": row.get("stored_temporal_stride", 1),
            }
            for position, (frozen, row) in enumerate(zip(schedule_rows, schedule, strict=True))
        )
    ):
        raise ValueError("continuation schedule differs from the frozen plan")
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(BASELINE / "run/hf"),
        "TORCH_HOME": str(BASELINE / "run/torch"),
        "MPLCONFIGDIR": str(BASELINE / "run/matplotlib"),
        "SHUBERT_DINOV2_SOURCE": str(BASELINE / "vendor/dinov2-source"),
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }.items():
        os.environ[key] = value
    sys.path.insert(0, str(BASELINE / "vendor"))
    sys.path.insert(0, str(BASELINE / "vendor/demo"))
    import torch
    from inference import SignLanguageByT5ForConditionalGeneration
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from shubert_visual_lora import (
        install_visual_lora,
        load_visual_adapter,
        visual_adapter_state,
    )
    from transformers import ByT5Tokenizer

    torch.set_num_threads(2)
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
    tokenizer = ByT5Tokenizer.from_pretrained(
        str(BASELINE / "models/byt5_base"), local_files_only=True
    )
    model.to("cpu")
    model.eval()
    optimizer = torch.optim.AdamW(
        selected.values(),
        lr=RECIPE["learning_rate"],
        weight_decay=RECIPE["weight_decay"],
    )
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan_hash = sha256(args.plan)
    run = {
        "schema": "umi-studio-c5-stem-continuation-training/1",
        "plan_sha256": plan_hash,
        "recipe_id": RECIPE["id"],
        "starting_adapter_sha256": sha256(STARTING_ADAPTER),
        "baseline_checkpoint_sha256": plan["baseline_checkpoint_sha256"],
        "scheduled_examples": len(schedule),
        "trainable_parameter_count": sum(value.numel() for value in selected.values()),
        "device": "cpu",
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
                raise RuntimeError(f"non-finite continuation loss: {row['id']}")
            (loss / len(batch)).backward()
            losses.append(float(loss.detach()))
        gradient_norm = torch.nn.utils.clip_grad_norm_(selected.values(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("non-finite continuation gradient")
        optimizer.step()
        example_step = offset + len(batch)
        step = {
            "example_step": example_step,
            "optimizer_step": example_step // accumulation,
            "sources": [row["source"] for row in batch],
            "example_ids": [row["id"] for row in batch],
            "losses": losses,
            "gradient_norm_before_clipping": float(gradient_norm),
            "seconds": time.perf_counter() - started,
        }
        with ledger.open("a", encoding="utf-8") as output:
            output.write(json.dumps(step, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        if example_step % RECIPE["recovery_after_examples"] == 0:
            target = args.output_dir / f"visual-lora-after-{example_step:06d}.safetensors"
            temporary = target.with_suffix(".partial.safetensors")
            save_file(
                visual_adapter_state(model),
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
                },
            )
            temporary.rename(target)
            checkpoint_row = {
                "schema": "umi-studio-c5-stem-continuation-checkpoint/1",
                "plan_sha256": plan_hash,
                "example_step": example_step,
                "checkpoint_name": target.name,
                "checkpoint_sha256": sha256(target),
                "tuning_checkpoint": example_step in RECIPE["checkpoint_after_examples"],
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
