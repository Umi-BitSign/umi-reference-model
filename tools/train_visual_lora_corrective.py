"""Train the frozen short signer-corrective visual LoRA schedule."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_visual_lora_corrective_plan import (
    BASELINE,
    RECIPE,
    ROOT,
    build_plan,
    corrective_schedule,
    digest,
)
from freeze_visual_lora_plan import SOURCE_LAYOUT
from interpolate_visual_adapter import DIRECT_NAMES
from train_visual_lora import training_streams
from visual_training_data import load_visual_examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    args = parser.parse_args()
    if not args.miner_maintenance_mode:
        parser.error("corrective training requires the authorized maintenance mode")
    if (
        args.output_dir.exists()
        or not args.output_dir.resolve().is_relative_to(ROOT / "checkpoints")
        or not args.plan.resolve().is_relative_to(ROOT / "runs")
    ):
        parser.error("corrective trainer needs a new checkpoint directory")
    plan = json.loads(args.plan.read_text())
    reproduced = build_plan(
        ROOT / plan["parent_plan"], ROOT / plan["starting_adapter"]
    )
    if plan != json.loads(json.dumps(reproduced, sort_keys=True)):
        raise ValueError("corrective training inputs differ from the frozen plan")
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
    schedule = corrective_schedule(examples, plan["seed"])
    if plan["schedule_sha256"] != digest(
        [(row["source"], row["id"]) for row in schedule]
    ):
        raise ValueError("corrective training schedule differs")
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
    from shubert_visual_lora import install_visual_lora, load_visual_adapter
    from transformers import ByT5Tokenizer

    torch.set_num_threads(2)
    torch.manual_seed(plan["seed"])
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        str(BASELINE / "models/checkpoint-11625"), local_files_only=True
    )
    selected = install_visual_lora(
        model, plan["rank"], plan["alpha"], plan["visual_last_layers"]
    )
    load_visual_adapter(model, load_file(str(ROOT / plan["starting_adapter"]), device="cpu"))
    with safe_open(
        str(BASELINE / "models/checkpoint-11625/model.safetensors"),
        framework="pt",
        device="cpu",
    ) as baseline:
        for name in DIRECT_NAMES:
            if not torch.equal(selected[name].detach().cpu(), baseline.get_tensor(name)):
                raise ValueError(f"corrective projection differs from baseline: {name}")
            selected[name].requires_grad_(False)
    trainable = {name: value for name, value in selected.items() if name not in DIRECT_NAMES}
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
        "schema": "umi-studio-visual-lora-corrective-training/1",
        "plan_sha256": plan_hash,
        "starting_adapter_sha256": plan["starting_adapter_sha256"],
        "trainable_parameter_count": sum(value.numel() for value in trainable.values()),
        "scheduled_examples": len(schedule),
        "miner_health_policy": "user_authorized_maintenance",
        "trainer_source_sha256": sha256(Path(__file__)),
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    ledger = args.output_dir / "steps.jsonl"
    accumulation = RECIPE["gradient_accumulation"]
    for offset in range(0, len(schedule), accumulation):
        batch = schedule[offset : offset + accumulation]
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for row in batch:
            streams = training_streams(row)
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
                raise RuntimeError(f"non-finite corrective loss: {row['id']}")
            (loss / len(batch)).backward()
            losses.append(float(loss.detach()))
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("non-finite corrective gradient")
        optimizer.step()
        example_step = offset + len(batch)
        record = {
            "example_step": example_step,
            "optimizer_step": offset // accumulation + 1,
            "sources": [row["source"] for row in batch],
            "signers": [row["signer"] for row in batch],
            "example_ids": [row["id"] for row in batch],
            "losses": losses,
            "gradient_norm_before_clipping": float(gradient_norm),
            "seconds": time.perf_counter() - started,
        }
        with ledger.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, sort_keys=True) + "\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        if example_step % RECIPE["recovery_after_examples"] == 0:
            target = args.output_dir / f"visual-lora-after-{example_step:06d}.safetensors"
            save_file(
                {
                    name: value.detach().cpu().contiguous().clone()
                    for name, value in selected.items()
                },
                str(target),
                metadata={
                    "schema": "umi-studio-shubert-visual-lora/1",
                    "plan_sha256": plan_hash,
                    "recipe_id": RECIPE["id"],
                    "rank": str(plan["rank"]),
                    "alpha": str(plan["alpha"]),
                    "visual_last_layers": str(plan["visual_last_layers"]),
                    "baseline_checkpoint_sha256": plan["baseline_checkpoint_sha256"],
                    "starting_adapter_sha256": plan["starting_adapter_sha256"],
                    "example_step": str(example_step),
                },
            )
            print(
                f"step={example_step} mean_loss={sum(losses) / len(losses):.4f} "
                f"checkpoint_sha256={sha256(target)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
