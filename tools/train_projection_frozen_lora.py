"""Continue visual LoRA training while holding the baseline projection fixed."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from freeze_projection_frozen_lora_plan import BASELINE, RECIPE, ROOT, build_plan
from freeze_visual_lora_plan import SEED, SOURCE_LAYOUT, _digest
from interpolate_visual_adapter import DIRECT_NAMES
from train_visual_lora import training_streams
from visual_training_data import balanced_schedule, load_visual_examples


def complete_adapter_state(selected: dict) -> dict:
    return {
        name: parameter.detach().cpu().contiguous().clone()
        for name, parameter in selected.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    args = parser.parse_args()
    if (
        args.output_dir.exists()
        or not args.output_dir.resolve().is_relative_to(ROOT / "checkpoints")
        or not args.plan.resolve().is_relative_to(ROOT / "runs")
    ):
        parser.error("projection-frozen trainer needs a new checkpoint directory")
    plan = json.loads(args.plan.read_text())
    reproduced = build_plan(
        ROOT / plan["parent_plan"],
        ROOT / plan["ablation_plan"],
        ROOT / plan["ablation_decision"],
        ROOT / plan["starting_adapter"],
        args.miner_health_url,
    )
    if plan != json.loads(json.dumps(reproduced, sort_keys=True)) or plan["recipe"] != RECIPE:
        raise ValueError("projection-frozen training inputs differ from the frozen plan")
    examples = []
    for source, layout in SOURCE_LAYOUT.items():
        rows = load_visual_examples(
            ROOT / layout["index"],
            ROOT / layout["cache_dir"],
            BASELINE,
            source,
            sidecar_index=(ROOT / layout["sidecar_index"])
            if "sidecar_index" in layout else None,
            sidecar_dir=(ROOT / layout["sidecar_dir"])
            if "sidecar_dir" in layout else None,
            miner_health_url=args.miner_health_url,
        )
        if len(rows) != layout["expected"]:
            raise ValueError(f"projection-frozen {source} training coverage changed")
        examples.extend(rows)
    schedule = balanced_schedule(examples, SEED)[: RECIPE["train_examples"]]
    if _digest([(row["source"], row["id"]) for row in schedule]) != plan[
        "stage_one_schedule_sha256"
    ]:
        raise ValueError("projection-frozen training schedule differs from the frozen plan")
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
    torch.manual_seed(SEED)
    revision = miner_revision(args.miner_health_url, plan["miner_model_revision"])
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        str(BASELINE / "models/checkpoint-11625"), local_files_only=True
    )
    selected_all = install_visual_lora(
        model, RECIPE["rank"], RECIPE["alpha"], RECIPE["visual_last_layers"]
    )
    load_visual_adapter(
        model, load_file(str(ROOT / plan["starting_adapter"]), device="cpu")
    )
    if not DIRECT_NAMES.issubset(selected_all):
        raise ValueError("projection-frozen trainer lacks the three direct visual parameters")
    with safe_open(
        str(BASELINE / "models/checkpoint-11625/model.safetensors"),
        framework="pt", device="cpu",
    ) as baseline:
        for name in DIRECT_NAMES:
            if not torch.equal(selected_all[name].detach().cpu(), baseline.get_tensor(name)):
                raise ValueError(f"projection-frozen trainer changed baseline parameter: {name}")
    for name in DIRECT_NAMES:
        selected_all[name].requires_grad_(False)
    selected_trainable = {
        name: parameter for name, parameter in selected_all.items()
        if name not in DIRECT_NAMES
    }
    if not selected_trainable or any(
        name in DIRECT_NAMES for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ):
        raise ValueError("projection-frozen trainer has the wrong trainable parameters")
    tokenizer = ByT5Tokenizer.from_pretrained(
        str(BASELINE / "models/byt5_base"), local_files_only=True
    )
    model.to("cpu")
    model.eval()
    optimizer = torch.optim.AdamW(
        selected_trainable.values(),
        lr=RECIPE["learning_rate"],
        weight_decay=RECIPE["weight_decay"],
    )
    plan_hash = sha256(args.plan)
    revision = miner_revision(args.miner_health_url, revision)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run = {
        "schema": "umi-studio-visual-lora-projection-frozen-training/1",
        "plan_sha256": plan_hash,
        "recipe_id": RECIPE["id"],
        "starting_adapter_sha256": plan["starting_adapter_sha256"],
        "baseline_checkpoint_sha256": plan["baseline_checkpoint_sha256"],
        "trainer_source_sha256": sha256(Path(__file__)),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in selected_trainable.values()
        ),
        "frozen_direct_parameter_names": sorted(DIRECT_NAMES),
        "scheduled_examples": len(schedule),
        "miner_model_revision": revision,
    }
    with (args.output_dir / "run.json").open("x", encoding="utf-8") as output:
        json.dump(run, output, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    ledger = args.output_dir / "steps.jsonl"
    checkpoint_index = args.output_dir / "checkpoint-index.jsonl"
    accumulation = RECIPE["gradient_accumulation"]
    recovery_interval = RECIPE["recovery_after_examples"]
    milestones = set(RECIPE["checkpoint_after_examples"])
    for offset in range(0, len(schedule), accumulation):
        batch = schedule[offset : offset + accumulation]
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for row in batch:
            revision = miner_revision(args.miner_health_url, revision)
            streams = training_streams(row)
            features = {
                name: torch.from_numpy(value).unsqueeze(0)
                for name, value in streams.items()
            }
            labels = tokenizer(row["reference"], return_tensors="pt").input_ids
            if labels.shape[1] > 512:
                raise ValueError(f"projection-frozen reference exceeds 512 tokens: {row['id']}")
            output = model(**features, labels=labels, use_cache=False, return_dict=True)
            loss = torch.nn.functional.cross_entropy(
                output.logits.reshape(-1, output.logits.shape[-1]),
                labels.reshape(-1),
                label_smoothing=RECIPE["label_smoothing"],
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite projection-frozen loss: {row['id']}")
            (loss / len(batch)).backward()
            losses.append(float(loss.detach().cpu()))
            revision = miner_revision(args.miner_health_url, revision)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            selected_trainable.values(), max_norm=1.0
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("non-finite projection-frozen gradient")
        if any(selected_all[name].grad is not None for name in DIRECT_NAMES):
            raise RuntimeError("frozen visual projection received a gradient")
        revision = miner_revision(args.miner_health_url, revision)
        optimizer.step()
        revision = miner_revision(args.miner_health_url, revision)
        example_step = offset + len(batch)
        record = {
            "example_step": example_step,
            "optimizer_step": (offset // accumulation) + 1,
            "sources": [row["source"] for row in batch],
            "example_ids": [row["id"] for row in batch],
            "losses": losses,
            "gradient_norm_before_clipping": float(gradient_norm.detach().cpu()),
            "seconds": time.perf_counter() - started,
            "miner_model_revision": revision,
        }
        with ledger.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        if example_step % recovery_interval == 0:
            target = args.output_dir / f"visual-lora-after-{example_step:06d}.safetensors"
            temporary = target.with_suffix(".partial.safetensors")
            optimizer_target = target.with_suffix(".optimizer.pt")
            optimizer_temporary = target.with_suffix(".optimizer.partial.pt")
            if any(
                path.exists()
                for path in (target, temporary, optimizer_target, optimizer_temporary)
            ):
                raise RuntimeError("projection-frozen checkpoint path already exists")
            metadata = {
                "schema": "umi-studio-shubert-visual-lora/1",
                "plan_sha256": plan_hash,
                "recipe_id": RECIPE["id"],
                "rank": str(RECIPE["rank"]),
                "alpha": str(RECIPE["alpha"]),
                "visual_last_layers": str(RECIPE["visual_last_layers"]),
                "baseline_checkpoint_sha256": plan["baseline_checkpoint_sha256"],
                "trainer_source_sha256": run["trainer_source_sha256"],
                "adapter_source_sha256": plan["adapter_source_sha256"],
                "starting_adapter_sha256": plan["starting_adapter_sha256"],
                "example_step": str(example_step),
            }
            save_file(complete_adapter_state(selected_all), str(temporary), metadata=metadata)
            temporary.rename(target)
            optimizer_payload = {
                "schema": "umi-studio-visual-lora-projection-frozen-optimizer/1",
                "plan_sha256": plan_hash,
                "example_step": example_step,
                "selected_names": list(selected_trainable),
                "adapter_sha256": sha256(target),
                "optimizer": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
            }
            with optimizer_temporary.open("xb") as output:
                torch.save(optimizer_payload, output)
                output.flush()
                os.fsync(output.fileno())
            optimizer_temporary.rename(optimizer_target)
            checkpoint_record = {
                "schema": "umi-studio-visual-lora-projection-frozen-recovery/1",
                "plan_sha256": plan_hash,
                "recipe_id": RECIPE["id"],
                "example_step": example_step,
                "checkpoint_name": target.name,
                "checkpoint_sha256": sha256(target),
                "optimizer_name": optimizer_target.name,
                "optimizer_sha256": sha256(optimizer_target),
                "selected_names": list(selected_trainable),
                "tuning_checkpoint": example_step in milestones,
            }
            with checkpoint_index.open("a", encoding="utf-8") as output:
                output.write(json.dumps(checkpoint_record, sort_keys=True) + "\n")
                output.flush()
                os.fsync(output.fileno())
            revision = miner_revision(args.miner_health_url, revision)
            print(f"saved {target.name} sha256={checkpoint_record['checkpoint_sha256']}", flush=True)
        elif record["optimizer_step"] % 25 == 0:
            print(
                f"examples={example_step} updates={record['optimizer_step']} "
                f"mean_loss={sum(losses) / len(losses):.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
