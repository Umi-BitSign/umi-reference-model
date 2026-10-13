"""Train a fresh compact translator from generic DINO and MediaPipe streams."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

from benchmark_fleurs_dev import sha256
from train_fresh_lowres import language_artifact_receipt


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def selected_rows(rows: list[dict], selection: Path | None, inventory_hash: str) -> list[dict]:
    if selection is None:
        return rows
    payload = json.loads(selection.read_text())
    if (
        payload.get("schema") != "umi-studio-fresh-stream-overfit-selection/1"
        or payload.get("inventory_sha256") != inventory_hash
        or len(payload.get("examples", [])) != 16
    ):
        raise ValueError("fresh stream selection differs")
    by_identity = {(row["source"], row["example_id"]): row for row in rows}
    chosen = [by_identity[(item["source"], item["example_id"])] for item in payload["examples"]]
    if len({(row["source"], row["example_id"]) for row in chosen}) != 16:
        raise ValueError("fresh stream selection repeats an example")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--initial-visual-checkpoint", type=Path)
    parser.add_argument("--language-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--training-stage", choices=("alignment", "decoder"), default="alignment")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=256)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--mps-idle-fraction", type=float, default=0.5)
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if (
        args.output_dir.exists()
        or not args.output_dir.resolve().is_relative_to((root / "checkpoints").resolve())
        or args.epochs < 1
        or args.learning_rate <= 0
        or args.weight_decay < 0
        or args.gradient_accumulation < 1
        or args.checkpoint_every < 1
        or not 0 <= args.mps_idle_fraction < 1
        or not args.miner_maintenance_mode
    ):
        parser.error("invalid fresh stream training bounds or miner policy")
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(root / "artifacts/fresh-stream-v1/huggingface"),
        "TORCH_HOME": str(root / "artifacts/fresh-stream-v1/torch"),
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }.items():
        os.environ[key] = value
    import torch
    from fresh_stream_data import load_inventory, load_stream_tensor
    from fresh_stream_model import (
        FreshStreamTranslator,
        configure_alignment_training,
        configure_decoder_training,
        configure_visual_pretraining,
    )
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    receipt = language_artifact_receipt(root, args.language_model)
    if receipt["initialization"] != "google_t5_small":
        parser.error("fresh stream training requires the pinned generic T5-small artifact")
    inventory_hash = sha256(args.inventory)
    rows = selected_rows(load_inventory(args.inventory, root), args.selection, inventory_hash)
    if args.sources:
        requested_sources = set(args.sources)
        available_sources = {row["source"] for row in rows}
        if not requested_sources <= available_sources:
            parser.error("requested training source is absent from the inventory")
        rows = [row for row in rows if row["source"] in requested_sources]
    if args.max_examples is not None:
        if not 1 <= args.max_examples <= len(rows):
            parser.error("max examples exceeds the selected inventory")
        rows = rows[: args.max_examples]
    language_model = T5ForConditionalGeneration.from_pretrained(
        args.language_model, local_files_only=True
    )
    model = FreshStreamTranslator(language_model)
    initial_visual_checkpoint_hash = None
    if args.initial_visual_checkpoint is not None:
        from safetensors import safe_open
        from safetensors.torch import load_file

        visual_checkpoint = args.initial_visual_checkpoint.resolve()
        if not visual_checkpoint.is_file() or not visual_checkpoint.is_relative_to(
            (root / "checkpoints").resolve()
        ):
            parser.error("initial visual checkpoint must be a checkpoint-ledger artifact")
        with safe_open(str(visual_checkpoint), framework="pt", device="cpu") as handle:
            visual_metadata = handle.metadata()
        visual_state = load_file(str(visual_checkpoint), device="cpu")
        expected_visual = configure_visual_pretraining(model)
        if (
            visual_metadata.get("schema") != "umi-studio-fresh-stream-distillation/1"
            or set(visual_state) != set(expected_visual)
        ):
            raise ValueError("initial visual distillation checkpoint differs")
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in visual_state.items():
                if parameters[name].shape != value.shape:
                    raise ValueError("initial visual tensor shape differs")
                parameters[name].copy_(value)
        initial_visual_checkpoint_hash = sha256(visual_checkpoint)
    trainable = (
        configure_alignment_training(model)
        if args.training_stage == "alignment"
        else configure_decoder_training(model)
    )
    initial_checkpoint_hash = None
    if args.initial_checkpoint is not None:
        from safetensors import safe_open
        from safetensors.torch import load_file

        checkpoint = args.initial_checkpoint.resolve()
        if not checkpoint.is_file() or not checkpoint.is_relative_to((root / "checkpoints").resolve()):
            parser.error("initial checkpoint must be a checkpoint-ledger artifact")
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            checkpoint_metadata = handle.metadata()
        initial = load_file(str(checkpoint), device="cpu")
        if (
            checkpoint_metadata.get("schema") != "umi-studio-fresh-stream-checkpoint/1"
            or set(initial) != set(trainable)
        ):
            raise ValueError("initial fresh stream checkpoint topology differs")
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in initial.items():
                if parameters[name].shape != value.shape:
                    raise ValueError("initial fresh stream tensor shape differs")
                parameters[name].copy_(value)
        initial_checkpoint_hash = sha256(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model, local_files_only=True)
    model.to(args.device)
    optimizer = torch.optim.AdamW(
        trainable.values(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    schedule = []
    for epoch in range(1, args.epochs + 1):
        ordered = list(rows)
        random.Random(args.seed + epoch).shuffle(ordered)
        schedule.extend((epoch, row) for row in ordered)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run = {
        "schema": "umi-studio-fresh-stream-training/1",
        "architecture": "generic-dino-mediapipe-temporal-t5-small",
        "training_stage": args.training_stage,
        "sign_specific_weights": None,
        "stream_feature_width": 1166,
        "effective_fps": 12.5,
        "inventory_sha256": inventory_hash,
        "selection_sha256": sha256(args.selection) if args.selection else None,
        "selected_sources": sorted(set(row["source"] for row in rows)),
        "initial_checkpoint_sha256": initial_checkpoint_hash,
        "initial_visual_checkpoint_sha256": initial_visual_checkpoint_hash,
        "language_model": receipt,
        "epochs": args.epochs,
        "unique_examples": len(rows),
        "example_updates": len(schedule),
        "source_counts": dict(sorted(Counter(row["source"] for row in rows).items())),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_accumulation": args.gradient_accumulation,
        "checkpoint_every": args.checkpoint_every,
        "seed": args.seed,
        "device": args.device,
        "mps_idle_fraction": args.mps_idle_fraction,
        "miner_health_policy": "user_authorized_maintenance",
        "model_source_sha256": sha256(Path(__file__).with_name("fresh_stream_model.py")),
        "data_source_sha256": sha256(Path(__file__).with_name("fresh_stream_data.py")),
        "trainer_source_sha256": sha256(Path(__file__)),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable.values()),
        "trainable_names": sorted(trainable),
    }
    run_path = args.output_dir / "run.json"
    run_path.write_text(json.dumps(run, sort_keys=True, indent=2) + "\n")

    def save_checkpoint(path: Path, update: int) -> None:
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in trainable.items()},
            str(path),
            metadata={
                "schema": "umi-studio-fresh-stream-checkpoint/1",
                "run_sha256": sha256(run_path),
                "example_update": str(update),
            },
        )

    verified = set()
    ledger = args.output_dir / "steps.jsonl"
    model.train()
    model.language_model.encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    for update, (epoch, row) in enumerate(schedule, start=1):
        started = time.perf_counter()
        identity = (row["source"], row["example_id"])
        values = load_stream_tensor(row, verify_hash=identity not in verified)
        verified.add(identity)
        streams = torch.from_numpy(values).unsqueeze(0).to(args.device)
        mask = torch.ones((1, values.shape[0]), dtype=torch.bool, device=args.device)
        labels = tokenizer(row["reference"], return_tensors="pt").input_ids.to(args.device)
        output = model(streams, mask, labels)
        loss = output.loss
        if not torch.isfinite(loss):
            raise RuntimeError("fresh stream loss is non-finite")
        group_start = ((update - 1) // args.gradient_accumulation) * args.gradient_accumulation + 1
        group_end = min(group_start + args.gradient_accumulation - 1, len(schedule))
        group_size = group_end - group_start + 1
        (loss / group_size).backward()
        boundary = update == group_end
        gradient_norm = None
        if boundary:
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if args.device == "mps":
            torch.mps.synchronize()
        active = time.perf_counter() - started
        idle = 0.0
        if args.device == "mps" and args.mps_idle_fraction:
            idle = active * args.mps_idle_fraction / (1 - args.mps_idle_fraction)
            time.sleep(idle)
        record = {
            "update": update,
            "epoch": epoch,
            "source": row["source"],
            "example_id": row["example_id"],
            "frames": values.shape[0],
            "label_tokens": labels.shape[1],
            "loss": float(loss.detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()) if gradient_norm is not None else None,
            "active_seconds": active,
            "idle_seconds": idle,
        }
        append_jsonl(ledger, record)
        print(f"update={update} epoch={epoch} source={row['source']} frames={values.shape[0]} loss={record['loss']:.4f} seconds={active + idle:.2f}", flush=True)
        if update % args.checkpoint_every == 0 and update != len(schedule):
            save_checkpoint(args.output_dir / f"trainable-after-{update:06d}.safetensors", update)
    checkpoint = args.output_dir / "trainable.safetensors"
    save_checkpoint(checkpoint, len(schedule))
    print(json.dumps({"checkpoint": str(checkpoint), "sha256": sha256(checkpoint)}, sort_keys=True))


if __name__ == "__main__":
    main()

