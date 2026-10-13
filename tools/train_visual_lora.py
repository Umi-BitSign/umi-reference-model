"""Train the first SHuBERT visual adapter from a frozen public-data plan."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from cache_fleurs_encoder import checked_stream_features
from freeze_visual_lora_plan import RECIPE, SEED, SOURCE_LAYOUT, _digest, freeze_plan
from visual_training_data import balanced_schedule, load_visual_examples


def checked_plan(plan: dict, reproduced: dict) -> None:
    normalized = json.loads(json.dumps(reproduced, sort_keys=True))
    if plan != normalized or plan.get("recipe") != RECIPE or plan.get("seed") != SEED:
        raise ValueError("visual training inputs differ from the frozen plan")


def checked_resume_record(
    checkpoint: Path,
    plan: dict,
    plan_hash: str,
    selected_names: list[str],
    checkpoints_root: Path,
) -> dict:
    from safetensors import safe_open

    checkpoint = checkpoint.resolve()
    if not checkpoint.is_relative_to(checkpoints_root.resolve()):
        raise ValueError("visual recovery checkpoint is outside the checkpoint ledger")
    index = checkpoint.parent / "checkpoint-index.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    matching = [row for row in rows if row.get("checkpoint_name") == checkpoint.name]
    if len(matching) != 1:
        raise ValueError("visual recovery checkpoint has no unique committed record")
    record = matching[0]
    optimizer_path = checkpoint.parent / record["optimizer_name"]
    step = record.get("example_step")
    if (
        record.get("schema") != "umi-studio-visual-lora-recovery/1"
        or record.get("plan_sha256") != plan_hash
        or record.get("recipe_id") != RECIPE["id"]
        or record.get("selected_names") != selected_names
        or record.get("checkpoint_sha256") != sha256(checkpoint)
        or optimizer_path.resolve() != checkpoint.with_suffix(".optimizer.pt")
        or record.get("optimizer_sha256") != sha256(optimizer_path)
        or not isinstance(step, int)
        or step < RECIPE["recovery_after_examples"]
        or step >= RECIPE["train_examples"]
        or step % RECIPE["recovery_after_examples"] != 0
        or step % RECIPE["gradient_accumulation"] != 0
    ):
        raise ValueError("visual recovery record differs from the frozen plan")
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        if set(handle.keys()) != set(selected_names):
            raise ValueError("visual recovery adapter topology differs")
    if (
        metadata.get("schema") != "umi-studio-shubert-visual-lora/1"
        or metadata.get("plan_sha256") != plan_hash
        or metadata.get("recipe_id") != RECIPE["id"]
        or metadata.get("rank") != str(RECIPE["rank"])
        or metadata.get("alpha") != str(RECIPE["alpha"])
        or metadata.get("visual_last_layers") != str(RECIPE["visual_last_layers"])
        or metadata.get("baseline_checkpoint_sha256")
        != plan["baseline_checkpoint_sha256"]
        or metadata.get("trainer_source_sha256") != sha256(Path(__file__))
        or metadata.get("adapter_source_sha256") != plan["visual_adapter_source_sha256"]
        or metadata.get("example_step") != str(step)
    ):
        raise ValueError("visual recovery adapter metadata differs")
    return {**record, "checkpoint": checkpoint, "optimizer": optimizer_path}


def training_streams(row: dict) -> dict:
    import numpy as np

    cache = Path(row["cache"])
    if sha256(cache) != row["cache_sha256"]:
        raise ValueError(f"visual training cache changed after plan verification: {cache}")
    if row["sidecar"] is None:
        with np.load(cache, allow_pickle=False) as payload:
            names = set(payload.files)
            if names != {
                "encoder_hidden_state",
                "face_features",
                "left_hand_features",
                "right_hand_features",
                "pose_features",
            }:
                raise ValueError(f"visual training cache lacks embedded streams: {cache}")
            streams = {
                name: payload[name]
                for name in ("face_features", "left_hand_features", "right_hand_features", "pose_features")
            }
    else:
        sidecar = Path(row["sidecar"])
        if sha256(sidecar) != row["sidecar_sha256"]:
            raise ValueError(f"visual training sidecar changed after plan verification: {sidecar}")
        with np.load(sidecar, allow_pickle=False) as payload:
            if set(payload.files) != {
                "face_features",
                "left_hand_features",
                "right_hand_features",
                "pose_features",
            }:
                raise ValueError(f"visual training sidecar fields differ: {sidecar}")
            streams = {
                name: payload[name]
                for name in ("face_features", "left_hand_features", "right_hand_features", "pose_features")
            }
    if any(array.dtype != np.float32 for array in streams.values()):
        raise ValueError(f"visual training stream dtype differs: {row['id']}")
    return checked_stream_features(streams, row["frames"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    baseline = args.baseline.resolve()
    if (
        baseline != root / "artifacts/baseline-6fe8df59"
        or not args.plan.resolve().is_relative_to(root / "runs")
        or args.output_dir.exists()
        or not args.output_dir.resolve().is_relative_to(root / "checkpoints")
    ):
        parser.error("training needs the independent baseline, frozen plan and new checkpoint directory")
    plan = json.loads(args.plan.read_text())
    checked_plan(plan, freeze_plan(root, baseline, args.miner_health_url))
    examples = []
    for source, layout in SOURCE_LAYOUT.items():
        examples.extend(
            load_visual_examples(
                root / layout["index"],
                root / layout["cache_dir"],
                baseline,
                source,
                sidecar_index=(root / layout["sidecar_index"])
                if "sidecar_index" in layout
                else None,
                sidecar_dir=(root / layout["sidecar_dir"])
                if "sidecar_dir" in layout
                else None,
                miner_health_url=args.miner_health_url,
            )
        )
    schedule = balanced_schedule(examples, SEED)[: RECIPE["train_examples"]]
    if _digest([(row["source"], row["id"]) for row in schedule]) != plan[
        "stage_one_schedule_sha256"
    ]:
        raise ValueError("visual training schedule differs from the frozen plan")
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(baseline / "run/hf"),
        "TORCH_HOME": str(baseline / "run/torch"),
        "MPLCONFIGDIR": str(baseline / "run/matplotlib"),
        "SHUBERT_DINOV2_SOURCE": str(baseline / "vendor/dinov2-source"),
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }.items():
        os.environ[key] = value
    sys.path.insert(0, str(baseline / "vendor"))
    sys.path.insert(0, str(baseline / "vendor/demo"))
    import numpy as np
    import torch
    from inference import SignLanguageByT5ForConditionalGeneration
    from safetensors.torch import load_file, save_file
    from shubert_visual_lora import (
        install_visual_lora,
        load_visual_adapter,
        visual_adapter_state,
    )
    from transformers import ByT5Tokenizer

    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    revision = miner_revision(args.miner_health_url, None)
    checkpoint = baseline / "models/checkpoint-11625"
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    selected = install_visual_lora(
        model, RECIPE["rank"], RECIPE["alpha"], RECIPE["visual_last_layers"]
    )
    tokenizer = ByT5Tokenizer.from_pretrained(
        str(baseline / "models/byt5_base"), local_files_only=True
    )
    model.to("cpu")
    model.eval()  # Frozen backbone dropout stays disabled; selected weights still receive gradients.
    optimizer = torch.optim.AdamW(
        selected.values(),
        lr=RECIPE["learning_rate"],
        weight_decay=RECIPE["weight_decay"],
    )
    plan_hash = sha256(args.plan)
    recovery = None
    start_step = 0
    if args.resume_checkpoint is not None:
        recovery = checked_resume_record(
            args.resume_checkpoint,
            plan,
            plan_hash,
            list(selected),
            root / "checkpoints",
        )
        load_visual_adapter(model, load_file(str(recovery["checkpoint"]), device="cpu"))
        saved = torch.load(recovery["optimizer"], map_location="cpu", weights_only=True)
        if (
            saved.get("schema") != "umi-studio-visual-lora-optimizer/1"
            or saved.get("plan_sha256") != plan_hash
            or saved.get("example_step") != recovery["example_step"]
            or saved.get("selected_names") != list(selected)
            or saved.get("adapter_sha256") != recovery["checkpoint_sha256"]
            or not isinstance(saved.get("torch_rng_state"), torch.Tensor)
        ):
            raise ValueError("visual recovery optimizer differs from the checkpoint")
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng_state"])
        start_step = recovery["example_step"]
    revision = miner_revision(args.miner_health_url, revision)
    if recovery is None:
        first = schedule[0]
        first_streams = training_streams(first)
        with torch.no_grad():
            untouched = model.encoder(
                **{name: torch.from_numpy(value).unsqueeze(0) for name, value in first_streams.items()},
                return_dict=True,
            ).last_hidden_state
        with np.load(first["cache"], allow_pickle=False) as payload:
            old_state = payload["encoder_hidden_state"]
        if not np.array_equal(untouched[0].cpu().numpy(), old_state):
            raise ValueError("initial visual adapter changed the frozen baseline encoder")
    revision = miner_revision(args.miner_health_url, revision)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run = {
        "schema": "umi-studio-visual-lora-training/1",
        "plan_sha256": plan_hash,
        "recipe_id": RECIPE["id"],
        "baseline_checkpoint_sha256": plan["baseline_checkpoint_sha256"],
        "trainer_source_sha256": sha256(Path(__file__)),
        "adapter_source_sha256": plan["visual_adapter_source_sha256"],
        "trainable_parameter_count": sum(parameter.numel() for parameter in selected.values()),
        "scheduled_examples": len(schedule),
        "start_example_step": start_step,
        "resume_checkpoint_sha256": recovery["checkpoint_sha256"] if recovery else None,
        "resume_optimizer_sha256": recovery["optimizer_sha256"] if recovery else None,
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
    milestones = set(RECIPE["checkpoint_after_examples"])
    recovery_interval = RECIPE["recovery_after_examples"]
    for offset in range(start_step, len(schedule), accumulation):
        batch = schedule[offset : offset + accumulation]
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for row in batch:
            revision = miner_revision(args.miner_health_url, revision)
            streams = training_streams(row)
            features = {
                name: torch.from_numpy(value).unsqueeze(0) for name, value in streams.items()
            }
            labels = tokenizer(row["reference"], return_tensors="pt").input_ids
            if labels.shape[1] > 512:
                raise ValueError(f"visual training reference exceeds 512 tokens: {row['id']}")
            output = model(**features, labels=labels, use_cache=False, return_dict=True)
            loss = torch.nn.functional.cross_entropy(
                output.logits.reshape(-1, output.logits.shape[-1]),
                labels.reshape(-1),
                label_smoothing=RECIPE["label_smoothing"],
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite visual training loss: {row['id']}")
            (loss / len(batch)).backward()
            losses.append(float(loss.detach().cpu()))
            revision = miner_revision(args.miner_health_url, revision)
        gradient_norm = torch.nn.utils.clip_grad_norm_(selected.values(), max_norm=1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("non-finite visual training gradient")
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
            if any(path.exists() for path in (target, temporary, optimizer_target, optimizer_temporary)):
                raise RuntimeError("visual checkpoint path already exists")
            metadata = {
                "schema": "umi-studio-shubert-visual-lora/1",
                "plan_sha256": plan_hash,
                "recipe_id": RECIPE["id"],
                "rank": str(RECIPE["rank"]),
                "alpha": str(RECIPE["alpha"]),
                "visual_last_layers": str(RECIPE["visual_last_layers"]),
                "baseline_checkpoint_sha256": plan["baseline_checkpoint_sha256"],
                "trainer_source_sha256": run["trainer_source_sha256"],
                "adapter_source_sha256": run["adapter_source_sha256"],
                "example_step": str(example_step),
            }
            save_file(visual_adapter_state(model), str(temporary), metadata=metadata)
            temporary.rename(target)
            optimizer_payload = {
                "schema": "umi-studio-visual-lora-optimizer/1",
                "plan_sha256": plan_hash,
                "example_step": example_step,
                "selected_names": list(selected),
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
                "schema": "umi-studio-visual-lora-recovery/1",
                "plan_sha256": plan_hash,
                "recipe_id": RECIPE["id"],
                "example_step": example_step,
                "checkpoint_name": target.name,
                "checkpoint_sha256": sha256(target),
                "optimizer_name": optimizer_target.name,
                "optimizer_sha256": sha256(optimizer_target),
                "selected_names": list(selected),
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
