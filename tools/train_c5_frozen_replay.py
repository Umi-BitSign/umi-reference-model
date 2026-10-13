"""Train a frozen, projection-preserving C5 replay plan on CPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from benchmark_fleurs_dev import sha256
from c5_retention import ALL_SOURCE_IDS, RETENTION_RECIPES, teacher_kl
from c5_scale_learning_rates import SPLIT_RATES, optimizer_groups
from freeze_c5_stem_scale_plan import BASELINE, ROOT, load_examples
from interpolate_visual_adapter import DIRECT_NAMES
from train_c5_stem_continuation import stride2_streams
from train_c5_stem_projection_frozen import (
    complete_adapter_state,
    freeze_direct_parameters,
)


def materialize_schedule(
    replay: list[dict], stem: list[dict], frozen_rows: list[dict]
) -> list[dict]:
    by_key = {(row["source"], row["id"]): row for row in (*replay, *stem)}
    if len(by_key) != len(replay) + len(stem):
        raise ValueError("frozen replay source keys are not unique")
    schedule = []
    for position, frozen in enumerate(frozen_rows):
        try:
            row = by_key[(frozen["source"], frozen["example_id"])]
        except KeyError as error:
            raise ValueError("frozen replay example is unavailable") from error
        expected = {
            "position": position,
            "source": row["source"],
            "example_id": row["id"],
            "signer": row["signer"],
            "stored_temporal_stride": row.get("stored_temporal_stride", 1),
        }
        if frozen != expected:
            raise ValueError("frozen replay schedule row differs")
        schedule.append(row)
    if len({(row["source"], row["id"]) for row in schedule}) != len(schedule):
        raise ValueError("frozen replay schedule repeats an example")
    return schedule


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
    recipe = plan.get("recipe", {})
    starting_adapter = ROOT / plan.get("starting_adapter", "")
    if (
        plan.get("schema") != "umi-studio-c5-frozen-replay-plan/1"
        or recipe.get("id")
        not in {
            "c5-signer-recovery-projection-frozen-v1",
            "c5-signer-recovery-projection-frozen-v2",
            "c5-signer-recovery-projection-frozen-v3",
            "c5-signer3-recovery-projection-frozen-v1",
            "c5-expanded-visual-lora-v1",
            "c5-openasl-diversity-projection-frozen-v1",
            "c5-openasl-diversity-projection-frozen-v2",
            "c5-openasl-bbox-projection-frozen-v1",
            "c5-how2sign-precut-expansion-projection-frozen-v1",
            "c5-how2sign-precut-expansion-projection-frozen-v2",
            "c5-how2sign-precut-expansion-projection-frozen-v3",
            "c5-how2sign-precut-expansion-scale-v1",
            "c5-how2sign-precut-expansion-scale-v2",
            "c5-openasl-scale-projection-frozen-v1",
            "c5-openasl-scale-fused-lora-fixed-v1",
            "c5-openasl-scale-fused-lora-high-lr-v1",
            "c5-openasl-scale-fused-lora-split-visual-fast-v1",
            "c5-openasl-scale-fused-lora-split-decoder-fast-v1",
            "c5-all-source-fused-lora-v1",
            "c5-all-source-fused-lora-retention-v1",
            "c5-all-source-fused-lora-retention-coverage-v1",
        }
        or recipe.get("frozen_direct_parameter_names") != sorted(DIRECT_NAMES)
        or recipe.get("train_examples") not in {512, 768, 3_112, 7_732}
        or recipe.get("gradient_accumulation") != 4
        or not starting_adapter.is_file()
        or not starting_adapter.resolve().is_relative_to((ROOT / "checkpoints").resolve())
        or plan.get("starting_adapter_sha256") != sha256(starting_adapter)
        or plan.get("trainer_source_sha256") != sha256(Path(__file__))
    ):
        raise ValueError("frozen replay plan differs from the trainer")
    implementation_hash = sha256(ROOT / "tools/shubert_lora.py")
    if recipe["id"].startswith("c5-openasl-scale-fused-lora-") and (
        plan.get("adapter_implementation_sha256") != implementation_hash
    ):
        raise ValueError("frozen effective-weight LoRA implementation changed")
    is_full_corpus = recipe["id"] in ALL_SOURCE_IDS
    use_retention = recipe["id"] in RETENTION_RECIPES
    if (recipe["id"] in SPLIT_RATES or is_full_corpus) and plan.get("optimizer_groups_source_sha256") != sha256(
        ROOT / "tools/c5_scale_learning_rates.py"
    ):
        raise ValueError("frozen optimizer group implementation changed")
    if is_full_corpus:
        from c5_all_source_training import RECIPE as ALL_RECIPE, load_corpus
        expected_recipe = RETENTION_RECIPES[recipe["id"]] if use_retention else ALL_RECIPE
        if recipe != expected_recipe or plan["all_source_implementation_sha256"] != sha256(ROOT / "tools/c5_all_source_training.py"):
            raise ValueError("full-corpus recipe or implementation changed")
        if plan["adapter_implementation_sha256"] != implementation_hash:
            raise ValueError("full-corpus visual adapter implementation changed")
        replay, stem, source_hashes = load_corpus(plan)
    else:
        replay, stem, source_hashes = load_examples()
    if recipe["id"] == "c5-openasl-bbox-projection-frozen-v1":
        from freeze_c5_openasl_bbox_plan import load_openasl_bbox_examples

        openasl, openasl_hashes = load_openasl_bbox_examples()
        stem.extend(openasl)
        source_hashes.update(openasl_hashes)
    elif recipe["id"].startswith("c5-openasl-diversity-projection-frozen-v"):
        from freeze_c5_openasl_pilot_plan import load_openasl_examples

        openasl, openasl_hashes = load_openasl_examples()
        stem.extend(openasl)
        source_hashes.update(openasl_hashes)
    elif recipe["id"].startswith("c5-how2sign-precut-expansion-"):
        from freeze_c5_how2sign_expansion_plan import load_expansion_examples

        expansion, expansion_hashes = load_expansion_examples()
        stem.extend(expansion)
        source_hashes.update(expansion_hashes)
    elif recipe["id"] in {
        "c5-openasl-scale-projection-frozen-v1",
        "c5-openasl-scale-fused-lora-fixed-v1",
        "c5-openasl-scale-fused-lora-high-lr-v1",
        "c5-openasl-scale-fused-lora-split-visual-fast-v1",
        "c5-openasl-scale-fused-lora-split-decoder-fast-v1",
    }:
        from bind_c5_openasl_scale_plan import load_scale_examples

        partition_path = ROOT / plan["scale_partition"]
        scale_index = ROOT / plan["scale_index"]
        scale_cache_dir = ROOT / plan["scale_cache_dir"]
        partition = json.loads(partition_path.read_text())
        if sha256(partition_path) != plan["scale_partition_sha256"]:
            raise ValueError("OpenASL scale training partition changed")
        scale, scale_hashes = load_scale_examples(
            scale_index, scale_cache_dir, partition
        )
        stem.extend(scale)
        source_hashes.update(scale_hashes)
    if source_hashes != plan.get("source_hashes"):
        raise ValueError("frozen replay source indexes changed")
    if use_retention and (
        plan.get("retention_implementation_sha256") != sha256(ROOT / "tools/c5_retention.py")
        or plan.get("teacher_adapter_sha256") != plan["starting_adapter_sha256"]
        or plan.get("teacher_evaluation_data_used") is not False
    ):
        raise ValueError("frozen retention implementation or teacher differs")
    if recipe.get("schedule_policy") and (
        plan.get("coverage_implementation_sha256") != sha256(ROOT / "tools/c5_coverage_schedule.py")
    ):
        raise ValueError("frozen coverage scheduling implementation differs")
    schedule_path = args.plan.with_name("schedule.jsonl")
    frozen_rows = [json.loads(line) for line in schedule_path.read_text().splitlines()]
    if sha256(schedule_path) != plan.get("schedule_sha256"):
        raise ValueError("frozen replay schedule hash differs")
    schedule = materialize_schedule(replay, stem, frozen_rows)
    if len(schedule) != recipe["train_examples"]:
        raise ValueError("frozen replay schedule count differs")
    cpu_threads = recipe["cpu_threads"]
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
    from shubert_visual_lora import (
        install_visual_lora,
        load_visual_adapter,
        load_visual_adapter_with_new_layers,
    )
    from transformers import ByT5Tokenizer

    torch.set_num_threads(cpu_threads)
    torch.manual_seed(plan["seed"])
    checkpoint = BASELINE / "models/checkpoint-11625"
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    selected = install_visual_lora(
        model,
        recipe["rank"],
        recipe["alpha"],
        recipe["visual_last_layers"],
    )
    with safe_open(str(starting_adapter), framework="pt", device="cpu") as source:
        metadata = source.metadata()
    if (
        metadata.get("schema") != "umi-studio-shubert-visual-lora/1"
        or metadata.get("baseline_checkpoint_sha256")
        != plan["baseline_checkpoint_sha256"]
        or metadata.get("rank") != str(recipe["rank"])
        or metadata.get("alpha") != str(recipe["alpha"])
        or metadata.get("visual_last_layers")
        != str(plan.get("starting_visual_last_layers", recipe["visual_last_layers"]))
        or (
            metadata.get("direct_parameters_frozen") != "true"
            and not recipe["id"].startswith("c5-openasl-")
            and not recipe["id"].startswith("c5-how2sign-precut-expansion-")
            and not is_full_corpus
        )
    ):
        raise ValueError("frozen replay starting adapter metadata differs")
    starting_weights = load_file(str(starting_adapter), device="cpu")
    if plan.get("starting_visual_last_layers", recipe["visual_last_layers"]) == recipe[
        "visual_last_layers"
    ]:
        load_visual_adapter(model, starting_weights)
    else:
        missing = load_visual_adapter_with_new_layers(model, starting_weights)
        expected_new_layers = recipe["visual_last_layers"] - plan[
            "starting_visual_last_layers"
        ]
        if len(missing) != expected_new_layers * 4:
            raise ValueError("expanded visual adapter added an unexpected topology")
    with safe_open(
        str(checkpoint / "model.safetensors"), framework="pt", device="cpu"
    ) as baseline:
        for name in DIRECT_NAMES:
            if not torch.equal(selected[name].detach().cpu(), baseline.get_tensor(name)):
                raise ValueError(f"frozen replay direct parameter differs: {name}")
    frozen_direct = {
        name: selected[name].detach().cpu().clone() for name in DIRECT_NAMES
    }
    trainable = freeze_direct_parameters(selected)
    tokenizer = ByT5Tokenizer.from_pretrained(
        str(BASELINE / "models/byt5_base"), local_files_only=True
    )
    model.to("cpu")
    model.eval()
    teacher = None
    if use_retention:
        teacher = SignLanguageByT5ForConditionalGeneration.from_pretrained(
            str(checkpoint), local_files_only=True
        )
        install_visual_lora(teacher, recipe["rank"], recipe["alpha"], recipe["visual_last_layers"])
        load_visual_adapter(teacher, starting_weights)
        teacher.requires_grad_(False)
        teacher.to("cpu")
        teacher.eval()
        if any(parameter.requires_grad for parameter in teacher.parameters()):
            raise RuntimeError("retention teacher is not frozen")
    groups = optimizer_groups(trainable, recipe)
    optimizer = torch.optim.AdamW(groups, weight_decay=recipe["weight_decay"])
    scheduler = None
    if is_full_corpus:
        from c5_all_source_training import learning_rate_factor
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: learning_rate_factor(step, len(schedule) // recipe["gradient_accumulation"])
        )
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan_hash = sha256(args.plan)
    run = {
        "schema": "umi-studio-c5-frozen-replay-training/1",
        "plan_sha256": plan_hash,
        "recipe_id": recipe["id"],
        "starting_adapter_sha256": sha256(starting_adapter),
        "baseline_checkpoint_sha256": plan["baseline_checkpoint_sha256"],
        "scheduled_examples": len(schedule),
        "trainable_parameter_count": sum(value.numel() for value in trainable.values()),
        "frozen_direct_parameter_names": sorted(DIRECT_NAMES),
        "device": "cpu",
        "cpu_threads": cpu_threads,
        "miner_health_policy": "user_authorized_maintenance",
        "miner_changed": False,
        "trainer_source_sha256": sha256(Path(__file__)),
        "adapter_implementation_sha256": implementation_hash,
        "retention": recipe.get("retention"),
        "teacher_adapter_sha256": plan.get("teacher_adapter_sha256"),
        "retention_implementation_sha256": plan.get("retention_implementation_sha256"),
        "coverage_implementation_sha256": plan.get("coverage_implementation_sha256"),
        "optimizer_groups": [
            {"name": group["name"], "learning_rate": group.get("initial_lr", group["lr"]),
             "initial_active_learning_rate": group["lr"],
             "parameters": sum(parameter.numel() for parameter in group["params"])}
            for group in groups
        ],
    }
    with (args.output_dir / "run.json").open("x", encoding="utf-8") as output:
        json.dump(run, output, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    ledger = args.output_dir / "steps.jsonl"
    checkpoint_index = args.output_dir / "checkpoint-index.jsonl"
    accumulation = recipe["gradient_accumulation"]
    checkpoint_steps = set(recipe["checkpoint_after_examples"])
    for offset in range(0, len(schedule), accumulation):
        batch = schedule[offset : offset + accumulation]
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        retention_losses = []
        objective_losses = []
        for row in batch:
            streams = stride2_streams(row)
            features = {
                name: torch.from_numpy(value).unsqueeze(0)
                for name, value in streams.items()
            }
            labels = tokenizer(row["reference"], return_tensors="pt").input_ids
            teacher_logits = None
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(**features, labels=labels, use_cache=False, return_dict=True).logits
            output = model(**features, labels=labels, use_cache=False, return_dict=True)
            loss = torch.nn.functional.cross_entropy(
                output.logits.reshape(-1, output.logits.shape[-1]),
                labels.reshape(-1),
                label_smoothing=recipe["label_smoothing"],
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite frozen replay loss: {row['id']}")
            objective = loss
            if teacher_logits is not None:
                retention = teacher_kl(output.logits, teacher_logits, labels, recipe["retention"]["temperature"])
                if not torch.isfinite(retention) or retention < -1e-5:
                    raise RuntimeError("invalid frozen-teacher retention loss")
                if offset == 0 and abs(float(retention.detach())) > 1e-6:
                    raise RuntimeError("retention teacher and starting student differ")
                objective = objective + recipe["retention"]["coefficient"] * retention
                retention_losses.append(float(retention.detach()))
            (objective * row.get("loss_weight", 1.0) / len(batch)).backward()
            losses.append(float(loss.detach()))
            objective_losses.append(float(objective.detach()))
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("non-finite frozen replay gradient")
        if any(selected[name].grad is not None for name in DIRECT_NAMES):
            raise RuntimeError("frozen direct parameter received a gradient")
        visual_gradients = {
            name: float(parameter.grad.norm()) if parameter.grad is not None else None
            for name, parameter in trainable.items() if "signhubert" in name
        }
        if (recipe["id"].startswith("c5-openasl-scale-fused-lora-") or is_full_corpus) and (
            len(visual_gradients) != 16 or any(value is None for value in visual_gradients.values())
        ):
            raise RuntimeError("an adapted visual projection was bypassed during training")
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
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
            "retention_losses": retention_losses,
            "objective_losses": objective_losses,
            "gradient_norm_before_clipping": float(gradient_norm),
            "seconds": time.perf_counter() - started,
            "visual_gradient_norms": visual_gradients,
            "loss_weights": [row.get("loss_weight", 1.0) for row in batch],
            "learning_rates_after_update": [group["lr"] for group in optimizer.param_groups],
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
                    "recipe_id": recipe["id"],
                    "rank": str(recipe["rank"]),
                    "alpha": str(recipe["alpha"]),
                    "visual_last_layers": str(recipe["visual_last_layers"]),
                    "baseline_checkpoint_sha256": plan["baseline_checkpoint_sha256"],
                    "starting_adapter_sha256": sha256(starting_adapter),
                    "trainer_source_sha256": sha256(Path(__file__)),
                    "adapter_implementation_sha256": implementation_hash,
                    **({"retention_implementation_sha256": plan["retention_implementation_sha256"],
                        "teacher_adapter_sha256": plan["teacher_adapter_sha256"]} if use_retention else {}),
                    "example_step": str(example_step),
                    "direct_parameters_frozen": "true",
                },
            )
            temporary.rename(target)
            checkpoint_row = {
                "schema": "umi-studio-c5-frozen-replay-checkpoint/1",
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
