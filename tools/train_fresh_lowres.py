"""Train a bounded fresh low-resolution raw-video candidate.

The clean-pretrained stage uses generic ImageNet or Kinetics weights and a
pinned T5-family model. It freezes those base weights and trains only new
visual-temporal layers plus decoder cross-attention. The strict scratch control
initializes the same retained topology randomly and trains it end to end.
Neither mode accepts C5 weights or encoder caches.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256

_LANGUAGE_ARTIFACTS = {
    "fsboard-byt5-small-v1": {
        "initialization": "google_byt5_small",
        "upstream_model": "google/byt5-small",
        "upstream_revision": None,
        "weights_file": "pytorch_model.bin",
    },
    "t5-small-df1b051c": {
        "initialization": "google_t5_small",
        "upstream_model": "google-t5/t5-small",
        "upstream_revision": "df1b051c49625cf57a3d0d8d3863ed4d13564fe4",
        "weights_file": "model.safetensors",
    },
}


def language_artifact_receipt(workspace: Path, path: Path) -> dict:
    """Validate a pinned local language artifact and return its provenance."""
    resolved = path.resolve()
    artifacts = (workspace / "artifacts").resolve()
    try:
        relative = resolved.relative_to(artifacts)
    except ValueError as error:
        raise ValueError("language model must be a pinned artifact") from error
    if len(relative.parts) != 1 or relative.name not in _LANGUAGE_ARTIFACTS:
        raise ValueError("language model must be a supported pinned artifact")
    receipt = dict(_LANGUAGE_ARTIFACTS[relative.name])
    weights_path = resolved / receipt["weights_file"]
    if not weights_path.is_file():
        raise ValueError("pinned language-model weights are missing")
    receipt.update(
        {
            "artifact": relative.name,
            "weights_sha256": sha256(weights_path),
        }
    )
    return receipt


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--language-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--miner-health-url")
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    parser.add_argument(
        "--initialization", choices=("clean_pretrained", "strict_scratch"), required=True
    )
    parser.add_argument(
        "--training-stage",
        choices=("alignment", "encoder_alignment", "visual_alignment", "decoder"),
        default="alignment",
    )
    parser.add_argument("--use-language-encoder", action="store_true")
    parser.add_argument(
        "--visual-backbone",
        choices=("mobilenet_v3_small", "s3d"),
        default="mobilenet_v3_small",
    )
    parser.add_argument("--source-short-side", type=int, choices=(160, 192, 256), default=192)
    parser.add_argument("--target-fps", type=float, default=12.5)
    parser.add_argument(
        "--augmentation-profile", choices=("none", "mild_v1"), default="none"
    )
    parser.add_argument("--max-frames", type=int, default=192)
    parser.add_argument("--max-examples", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=128)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--initial-trainable", type=Path)
    parser.add_argument("--frame-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--ctc-weight", type=float, default=0.0)
    parser.add_argument("--semantic-weight", type=float, default=0.0)
    parser.add_argument("--semantic-negatives", type=int, default=7)
    parser.add_argument("--fleurs-repeat", type=int, default=1)
    parser.add_argument("--asl-stem-wiki-repeat", type=int, default=1)
    parser.add_argument("--mps-idle-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to(
        workspace / "checkpoints"
    ):
        parser.error("new output directory must stay under workspace checkpoints")
    if (
        args.max_examples < 1
        or args.start_offset < 0
        or args.max_frames < 1
        or args.epochs < 1
        or args.frame_batch_size < 1
        or args.gradient_accumulation_steps < 1
        or args.checkpoint_every < 1
        or args.learning_rate <= 0
        or args.weight_decay < 0
        or args.ctc_weight < 0
        or args.semantic_weight < 0
        or args.semantic_negatives < 1
        or not 1 <= args.fleurs_repeat <= 8
        or not 1 <= args.asl_stem_wiki_repeat <= 8
        or not 0 <= args.mps_idle_fraction < 1
    ):
        parser.error("invalid training bounds")
    if args.start_offset > 0 and args.initial_trainable is None:
        parser.error("a nonzero start offset requires initial weights")
    if args.initial_trainable is not None and not args.initial_trainable.is_file():
        parser.error("initial trainable weights are missing")
    if args.initialization == "strict_scratch" and args.training_stage != "alignment":
        parser.error("strict scratch uses its end-to-end training scope")
    if args.initialization == "strict_scratch" and args.semantic_weight:
        parser.error("strict scratch cannot use a pretrained semantic teacher")
    if not args.miner_maintenance_mode and not args.miner_health_url:
        parser.error("miner health URL is required outside maintenance mode")
    if args.miner_maintenance_mode and args.miner_health_url:
        parser.error("maintenance mode must not claim a miner health endpoint")
    if args.training_stage == "encoder_alignment" and not args.use_language_encoder:
        parser.error("encoder alignment requires --use-language-encoder")
    if args.training_stage == "visual_alignment" and args.visual_backbone != "mobilenet_v3_small":
        parser.error("visual alignment currently requires MobileNetV3-Small")
    try:
        language_receipt = language_artifact_receipt(workspace, args.language_model)
    except ValueError as error:
        parser.error(str(error))
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(workspace / "artifacts/fresh-lowres-v1/huggingface"),
        "TORCH_HOME": str(workspace / "artifacts/fresh-lowres-v1/torch"),
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }.items():
        os.environ[key] = value
    import torch
    from fresh_lowres_data import decode_video, load_inventory
    from fresh_lowres_model import (
        FreshLowResolutionTranslator,
        discard_unused_text_encoder,
        initialization_manifest,
        mobilenet_v3_small_encoder,
        s3d_video_encoder,
    )
    from fresh_lowres_training import (
        accumulation_group,
        allowed_cumulative_state_keys,
        configure_alignment_stage,
        configure_decoder_stage,
        configure_encoder_alignment_stage,
        configure_scratch_stage,
        configure_visual_alignment_stage,
        ctc_alignment_is_feasible,
        ctc_alignment_loss,
        ordered_examples,
        repeat_source_examples,
        semantic_alignment_loss,
        training_schedule,
    )
    from safetensors.torch import load_file, save_file
    from torchvision.models import MobileNet_V3_Small_Weights
    from torchvision.models.video import S3D_Weights
    from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutput

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    revision = (
        None
        if args.miner_maintenance_mode
        else miner_revision(args.miner_health_url, None)
    )

    def check_miner(expected: str | None) -> str | None:
        if args.miner_maintenance_mode:
            return None
        return miner_revision(args.miner_health_url, expected)

    rows = ordered_examples(load_inventory(args.inventory, workspace), args.seed)
    if args.start_offset + args.max_examples > len(rows):
        parser.error("requested example range exceeds the frozen inventory")
    if args.initialization == "clean_pretrained":
        language_model = T5ForConditionalGeneration.from_pretrained(
            args.language_model, local_files_only=True
        )
        language_initialization = language_receipt["initialization"]
    else:
        config = T5Config.from_pretrained(args.language_model, local_files_only=True)
        language_model = T5ForConditionalGeneration(config)
        language_initialization = "random"
    text_teacher = (
        copy.deepcopy(language_model.encoder).eval()
        if args.semantic_weight
        else None
    )
    if text_teacher is not None:
        for parameter in text_teacher.parameters():
            parameter.requires_grad_(False)
    removed_text_encoder_parameters = (
        0 if args.use_language_encoder else discard_unused_text_encoder(language_model)
    )
    if args.visual_backbone == "mobilenet_v3_small":
        vision_weights = (
            MobileNet_V3_Small_Weights.IMAGENET1K_V1
            if args.initialization == "clean_pretrained"
            else None
        )
        frame_encoder = mobilenet_v3_small_encoder(vision_weights, args.frame_batch_size)
        visual_initialization = (
            "imagenet1k_v1" if args.initialization == "clean_pretrained" else "random"
        )
        normalization = "imagenet"
    else:
        vision_weights = (
            S3D_Weights.KINETICS400_V1 if args.initialization == "clean_pretrained" else None
        )
        frame_encoder = s3d_video_encoder(vision_weights)
        visual_initialization = (
            "kinetics400_s3d_v1" if args.initialization == "clean_pretrained" else "random"
        )
        normalization = "kinetics400"
    model = FreshLowResolutionTranslator(
        language_model,
        frame_encoder,
        frame_encoder.output_width,
        max_frames=args.max_frames,
        use_language_encoder=args.use_language_encoder,
    )
    alignment = configure_alignment_stage(model)
    stage_scopes = {"alignment": set(alignment)}
    if args.initialization == "clean_pretrained":
        stage_scopes["decoder"] = set(configure_decoder_stage(model))
        if args.use_language_encoder:
            stage_scopes["encoder_alignment"] = set(
                configure_encoder_alignment_stage(model)
            )
        stage_scopes["visual_alignment"] = set(configure_visual_alignment_stage(model))
        if args.training_stage == "alignment":
            trainable = configure_alignment_stage(model)
        elif args.training_stage == "decoder":
            trainable = configure_decoder_stage(model)
        elif args.training_stage == "visual_alignment":
            trainable = configure_visual_alignment_stage(model)
        else:
            trainable = configure_encoder_alignment_stage(model)
    else:
        trainable = configure_scratch_stage(model)
        stage_scopes = {"scratch": set(trainable)}
    initial_state_names: set[str] = set()
    if args.initial_trainable is not None:
        initial = load_file(str(args.initial_trainable), device="cpu")
        allowed_initial_keys = allowed_cumulative_state_keys(stage_scopes)
        if frozenset(initial) not in allowed_initial_keys:
            raise ValueError("initial weights differ from the trainable model topology")
        model_parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, weight in initial.items():
                parameter = model_parameters[name]
                if weight.shape != parameter.shape or weight.dtype != parameter.dtype:
                    raise ValueError(f"initial trainable tensor differs: {name}")
                parameter.copy_(weight)
        initial_state_names = set(initial)
    model_parameters = dict(model.named_parameters())
    checkpoint_parameters = {
        name: model_parameters[name]
        for name in sorted(set(trainable) | initial_state_names)
    }
    tokenizer = AutoTokenizer.from_pretrained(args.language_model, local_files_only=True)
    model.to(device)
    visual_device = device
    if args.visual_backbone == "s3d" and device == "mps":
        model.frame_encoder.to("cpu")
        visual_device = "cpu"
    optimizer = torch.optim.AdamW(
        trainable.values(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    revision = check_miner(revision)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = initialization_manifest(
        visual_initialization=visual_initialization,
        language_initialization=language_initialization,
        source_short_side=args.source_short_side,
        temporal_sample_fps=args.target_fps,
        use_language_encoder=args.use_language_encoder,
    )
    selected_rows = rows[args.start_offset : args.start_offset + args.max_examples]
    source_repeats = {
        "asl_stem_wiki": args.asl_stem_wiki_repeat,
        "fleurs": args.fleurs_repeat,
    }
    scheduled_rows = repeat_source_examples(selected_rows, source_repeats, args.seed + 1)
    scheduled_source_counts = {
        source: sum(row["source"] == source for row in scheduled_rows)
        for source in sorted({row["source"] for row in scheduled_rows})
    }
    run = {
        "schema": "umi-studio-fresh-lowres-training/1",
        "initialization": args.initialization,
        "training_stage": args.training_stage,
        "initialization_manifest": manifest,
        "visual_backbone": args.visual_backbone,
        "video_normalization": normalization,
        "inventory_sha256": sha256(args.inventory),
        "language_model": language_receipt,
        "language_model_sha256": language_receipt["weights_sha256"],
        "trainer_source_sha256": sha256(Path(__file__)),
        "model_source_sha256": sha256(Path(__file__).with_name("fresh_lowres_model.py")),
        "data_source_sha256": sha256(Path(__file__).with_name("fresh_lowres_data.py")),
        "max_examples": args.max_examples,
        "epochs": args.epochs,
        "total_example_updates": len(scheduled_rows) * args.epochs,
        "total_optimizer_steps": (
            len(scheduled_rows) * args.epochs + args.gradient_accumulation_steps - 1
        )
        // args.gradient_accumulation_steps,
        "checkpoint_every": args.checkpoint_every,
        "start_offset": args.start_offset,
        "initial_trainable_sha256": (
            sha256(args.initial_trainable) if args.initial_trainable is not None else None
        ),
        "initial_checkpoint_parameter_count": len(initial_state_names),
        "checkpoint_parameter_count": len(checkpoint_parameters),
        "max_frames": args.max_frames,
        "frame_batch_size": args.frame_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "ctc_weight": args.ctc_weight,
        "semantic_weight": args.semantic_weight,
        "semantic_negatives": args.semantic_negatives,
        "source_repeats": source_repeats,
        "scheduled_examples_per_epoch": len(scheduled_rows),
        "scheduled_source_counts": scheduled_source_counts,
        "mps_idle_fraction": args.mps_idle_fraction,
        "augmentation_profile": args.augmentation_profile,
        "semantic_teacher": (
            language_receipt if args.semantic_weight else None
        ),
        "seed": args.seed,
        "device": device,
        "visual_device": visual_device,
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable.values()),
        "removed_text_encoder_parameter_count": removed_text_encoder_parameters,
        "miner_model_revision": revision,
        "miner_health_policy": (
            "user_authorized_maintenance"
            if args.miner_maintenance_mode
            else "required_stable_revision"
        ),
        "examples": [
            {"source": row["source"], "example_id": row["example_id"]}
            for row in selected_rows
        ],
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    ledger = args.output_dir / "steps.jsonl"

    def save_trainable(path: Path, optimizer_step: int) -> None:
        tensors = {
            name: parameter.detach().cpu().contiguous().clone()
            for name, parameter in checkpoint_parameters.items()
        }
        save_file(
            tensors,
            str(path),
            metadata={
                "schema": "umi-studio-fresh-lowres-trainable/1",
                "initialization": args.initialization,
                "training_stage": args.training_stage,
                "inventory_sha256": run["inventory_sha256"],
                "run_sha256": sha256(args.output_dir / "run.json"),
                "optimizer_step": str(optimizer_step),
            },
        )

    verified = set()
    model.train()
    if args.initialization == "clean_pretrained" and args.training_stage != "visual_alignment":
        model.frame_encoder.eval()
    inventory_positions = {
        (row["source"], row["example_id"]): args.start_offset + position
        for position, row in enumerate(selected_rows, start=1)
    }
    schedule = training_schedule(scheduled_rows, args.epochs)
    optimizer.zero_grad(set_to_none=True)
    for example_update, (epoch, local_example, row) in enumerate(schedule, start=1):
        inventory_position = inventory_positions[(row["source"], row["example_id"])]
        revision = check_miner(revision)
        video_path = row["video_path"]
        if video_path not in verified:
            if sha256(video_path) != row["video_sha256"]:
                raise ValueError(f"training video checksum changed: {video_path}")
            verified.add(video_path)
        started = time.perf_counter()
        frames = decode_video(
            video_path,
            short_side=args.source_short_side,
            target_fps=args.target_fps,
            max_frames=args.max_frames,
            normalization=normalization,
            augmentation_profile=args.augmentation_profile,
            augmentation_seed=(
                int.from_bytes(
                    hashlib.sha256(
                        f"{args.seed}:{epoch}:{row['source']}:{row['example_id']}".encode()
                    ).digest()[:8],
                    "big",
                )
                if args.augmentation_profile != "none"
                else None
            ),
        ).unsqueeze(0)
        frame_mask = torch.ones(1, frames.shape[1], dtype=torch.bool)
        labels = tokenizer(row["reference"], return_tensors="pt").input_ids
        if labels.shape[1] > 512:
            raise ValueError("training reference exceeds 512 language-model tokens")
        frames = frames.to(device)
        frame_mask = frame_mask.to(device)
        labels = labels.to(device)
        memory, memory_mask = model._encode_with_mask(frames, frame_mask)
        output = model.language_model(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=memory_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        cross_entropy_loss = output.loss
        ctc_feasible = bool(
            args.ctc_weight
            and ctc_alignment_is_feasible(
                memory_mask,
                labels,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        )
        if ctc_feasible:
            auxiliary_ctc_loss = ctc_alignment_loss(
                memory,
                memory_mask,
                labels,
                model.language_model.shared.weight,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            loss = cross_entropy_loss + args.ctc_weight * auxiliary_ctc_loss.to(device)
        else:
            auxiliary_ctc_loss = None
            loss = cross_entropy_loss
        if text_teacher is not None:
            negative_rows = [
                rows[(inventory_position - 1 + offset) % len(rows)]
                for offset in range(1, args.semantic_negatives + 1)
            ]
            text_batch = tokenizer(
                [row["reference"], *(item["reference"] for item in negative_rows)],
                padding=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                teacher_hidden = text_teacher(
                    input_ids=text_batch.input_ids,
                    attention_mask=text_batch.attention_mask,
                    return_dict=True,
                ).last_hidden_state
                teacher_mask = text_batch.attention_mask.unsqueeze(-1).to(
                    teacher_hidden.dtype
                )
                text_targets = (teacher_hidden * teacher_mask).sum(dim=1) / teacher_mask.sum(
                    dim=1
                ).clamp_min(1)
            auxiliary_semantic_loss = semantic_alignment_loss(
                memory, memory_mask, text_targets
            )
            loss = loss + args.semantic_weight * auxiliary_semantic_loss
        else:
            auxiliary_semantic_loss = None
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite fresh-model loss")
        optimizer_step, accumulation_group_size, accumulation_boundary = accumulation_group(
            example_update, len(schedule), args.gradient_accumulation_steps
        )
        (loss / accumulation_group_size).backward()
        gradient_norm = None
        if accumulation_boundary:
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("non-finite fresh-model gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if device == "mps":
            torch.mps.synchronize()
        revision = check_miner(revision)
        active_seconds = time.perf_counter() - started
        idle_seconds = 0.0
        if device == "mps" and args.mps_idle_fraction:
            idle_seconds = active_seconds * args.mps_idle_fraction / (
                1 - args.mps_idle_fraction
            )
            time.sleep(idle_seconds)
        record = {
            "local_step": example_update,
            "optimizer_step": optimizer_step,
            "accumulation_group_size": accumulation_group_size,
            "accumulation_boundary": accumulation_boundary,
            "epoch": epoch,
            "local_example": local_example,
            "example_step": inventory_position,
            "inventory_position": inventory_position,
            "source": row["source"],
            "example_id": row["example_id"],
            "frames": frames.shape[1],
            "label_tokens": labels.shape[1],
            "loss": float(loss.detach().cpu()),
            "cross_entropy_loss": float(cross_entropy_loss.detach().cpu()),
            "ctc_loss": (
                float(auxiliary_ctc_loss.detach().cpu())
                if auxiliary_ctc_loss is not None
                else None
            ),
            "ctc_feasible": ctc_feasible if args.ctc_weight else None,
            "semantic_loss": (
                float(auxiliary_semantic_loss.detach().cpu())
                if auxiliary_semantic_loss is not None
                else None
            ),
            "gradient_norm_before_clipping": (
                float(gradient_norm.detach().cpu()) if gradient_norm is not None else None
            ),
            "active_seconds": active_seconds,
            "idle_seconds": idle_seconds,
            "seconds": active_seconds + idle_seconds,
            "miner_model_revision": revision,
        }
        _append_jsonl(ledger, record)
        print(
            f"step={example_update} optimizer={optimizer_step} epoch={epoch} "
            f"inventory={inventory_position} "
            f"source={row['source']} frames={record['frames']} "
            f"loss={record['loss']:.4f} seconds={record['seconds']:.2f}",
            flush=True,
        )
        total_example_updates = len(schedule)
        if (
            example_update % args.checkpoint_every == 0
            and example_update != total_example_updates
        ):
            if not accumulation_boundary:
                raise ValueError("checkpoint interval splits an accumulation group")
            recovery = args.output_dir / f"trainable-step-{example_update}.safetensors"
            save_trainable(recovery, optimizer_step)
            print(f"saved recovery {recovery} sha256={sha256(recovery)}", flush=True)
    checkpoint = args.output_dir / "trainable.safetensors"
    final_optimizer_step = accumulation_group(
        len(schedule), len(schedule), args.gradient_accumulation_steps
    )[0]
    save_trainable(checkpoint, final_optimizer_step)
    print(f"saved {checkpoint} sha256={sha256(checkpoint)}", flush=True)


if __name__ == "__main__":
    main()
