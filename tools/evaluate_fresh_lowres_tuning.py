"""Evaluate a fresh low-resolution checkpoint on the frozen open tuning clips."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from benchmark_fleurs_dev import distance, miner_revision, normalize, sha256
from freeze_fresh_lowres_tuning import SCHEMA, read_jsonl


def load_tuning(path: Path, workspace: Path) -> list[dict]:
    rows = read_jsonl(path)
    seen = set()
    groups = defaultdict(set)
    for row in rows:
        identity = (row.get("source"), row.get("example_id"))
        video = (workspace / row.get("video", "")).resolve()
        try:
            video.relative_to(workspace.resolve())
        except ValueError as error:
            raise ValueError("fresh tuning video is outside the workspace") from error
        if (
            row.get("schema") != SCHEMA
            or row.get("split") != "tuning"
            or row.get("source") not in {"fleurs", "how2sign"}
            or identity in seen
            or any("cache" in key or "baseline" in key for key in row)
            or not video.is_file()
            or not row.get("reference", "").strip()
            or len(row.get("video_sha256", "")) != 64
        ):
            raise ValueError(f"invalid fresh tuning row: {identity}")
        row["video_path"] = video
        seen.add(identity)
        groups[row["source"]].add(row["source_group"])
    if {source: len(items) for source, items in groups.items()} != {
        "fleurs": 3,
        "how2sign": 16,
    } or len(rows) != 23:
        raise ValueError("fresh tuning inventory scope differs")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-inventory", type=Path, required=True)
    parser.add_argument("--tuning-inventory", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-file", type=Path)
    parser.add_argument("--language-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--miner-health-url")
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint_file or args.checkpoint_dir / "trainable.safetensors"
    run_path = args.checkpoint_dir / "run.json"
    if (
        args.output.exists()
        or not args.output.resolve().is_relative_to(workspace / "runs")
        or not checkpoint.is_file()
        or not checkpoint.resolve().is_relative_to(args.checkpoint_dir.resolve())
        or not run_path.is_file()
        or args.max_new_tokens < 1
        or not 1 <= args.num_beams <= 8
        or args.length_penalty <= 0
        or args.repetition_penalty < 1
        or not 0 <= args.no_repeat_ngram_size <= 8
    ):
        parser.error("evaluation paths or bounds are invalid")
    if not args.miner_maintenance_mode and not args.miner_health_url:
        parser.error("miner health URL is required outside maintenance mode")
    if args.miner_maintenance_mode and args.miner_health_url:
        parser.error("maintenance mode must not claim a miner health endpoint")
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(workspace / "artifacts/fresh-lowres-v1/huggingface"),
        "TORCH_HOME": str(workspace / "artifacts/fresh-lowres-v1/torch"),
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }.items():
        os.environ[key] = value
    import torch
    from fresh_lowres_data import decode_video
    from fresh_lowres_model import (
        FreshLowResolutionTranslator,
        discard_unused_text_encoder,
        mobilenet_v3_small_encoder,
        s3d_video_encoder,
    )
    from safetensors import safe_open
    from safetensors.torch import load_file
    from torchvision.models import MobileNet_V3_Small_Weights
    from torchvision.models.video import S3D_Weights
    from train_fresh_lowres import language_artifact_receipt
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutput

    torch.set_num_threads(2)
    run = json.loads(run_path.read_text())
    receipt = language_artifact_receipt(workspace, args.language_model)
    if (
        run.get("schema") != "umi-studio-fresh-lowres-training/1"
        or run.get("initialization") != "clean_pretrained"
        or run.get("training_stage")
        not in {"alignment", "encoder_alignment", "visual_alignment", "decoder"}
        or (run.get("visual_backbone"), run.get("video_normalization"))
        not in {
            ("mobilenet_v3_small", "imagenet"),
            ("s3d", "kinetics400"),
        }
        or run.get("language_model") != receipt
        or run.get("inventory_sha256") != sha256(args.training_inventory)
        or run.get("model_source_sha256")
        != sha256(Path(__file__).with_name("fresh_lowres_model.py"))
        or run.get("data_source_sha256")
        != sha256(Path(__file__).with_name("fresh_lowres_data.py"))
    ):
        raise ValueError("fresh tuning run provenance differs")
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    if (
        metadata.get("schema") != "umi-studio-fresh-lowres-trainable/1"
        or metadata.get("run_sha256") != sha256(run_path)
    ):
        raise ValueError("fresh tuning checkpoint provenance differs")
    rows = load_tuning(args.tuning_inventory, workspace)
    use_language_encoder = (
        run["initialization_manifest"].get("language_encoder_policy") != "discarded"
    )
    language_model = T5ForConditionalGeneration.from_pretrained(
        args.language_model, local_files_only=True
    )
    if not use_language_encoder:
        discard_unused_text_encoder(language_model)
    if run["visual_backbone"] == "mobilenet_v3_small":
        frame_encoder = mobilenet_v3_small_encoder(
            MobileNet_V3_Small_Weights.IMAGENET1K_V1, run["frame_batch_size"]
        )
    else:
        frame_encoder = s3d_video_encoder(S3D_Weights.KINETICS400_V1)
    model = FreshLowResolutionTranslator(
        language_model,
        frame_encoder,
        frame_encoder.output_width,
        max_frames=run["max_frames"],
        use_language_encoder=use_language_encoder,
    )
    state = load_file(str(checkpoint), device="cpu")
    parameters = dict(model.named_parameters())
    if any(
        name not in parameters or parameters[name].shape != value.shape
        for name, value in state.items()
    ):
        raise ValueError("fresh tuning checkpoint topology differs")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value)
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    model.to(device).eval()
    if run["visual_backbone"] == "s3d" and device == "mps":
        model.frame_encoder.to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.language_model, local_files_only=True)
    revision = (
        None
        if args.miner_maintenance_mode
        else miner_revision(args.miner_health_url, run["miner_model_revision"])
    )
    records = []
    for row in rows:
        if not args.miner_maintenance_mode:
            revision = miner_revision(args.miner_health_url, revision)
        if sha256(row["video_path"]) != row["video_sha256"]:
            raise ValueError("fresh tuning video checksum changed")
        frames = decode_video(
            row["video_path"],
            short_side=run["initialization_manifest"]["source_short_side"],
            target_fps=run["initialization_manifest"]["temporal_sample_fps"],
            max_frames=run["max_frames"],
            normalization=run["video_normalization"],
        ).unsqueeze(0).to(device)
        mask = torch.ones(1, frames.shape[1], dtype=torch.bool, device=device)
        with torch.inference_mode():
            memory, memory_mask = model._encode_with_mask(frames, mask)
            tokens = model.language_model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=memory),
                attention_mask=memory_mask,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
        hypothesis = tokenizer.decode(tokens[0], skip_special_tokens=True)
        reference_words = normalize(row["reference"])
        hypothesis_words = normalize(hypothesis)
        records.append(
            {
                "source": row["source"],
                "source_group": row["source_group"],
                "signer": row["signer"],
                "example_id": row["example_id"],
                "reference": row["reference"],
                "hypothesis": hypothesis,
                "frames": frames.shape[1],
                "wer": distance(reference_words, hypothesis_words) / len(reference_words),
            }
        )
        print(f"decoded {row['source']} {row['example_id']}: {hypothesis!r}", flush=True)
    by_source = defaultdict(list)
    for row in records:
        by_source[row["source"]].append(row)
    summaries = {
        source: {
            "examples": len(items),
            "mean_single_reference_wer": sum(row["wer"] for row in items) / len(items),
            "exact_hypotheses": sum(row["wer"] == 0 for row in items),
            "distinct_hypotheses": len({row["hypothesis"] for row in items}),
        }
        for source, items in sorted(by_source.items())
    }
    result = {
        "schema": "umi-studio-fresh-lowres-tuning/1",
        "split": "tuning",
        "confirmation_evaluated": False,
        "run_sha256": sha256(run_path),
        "checkpoint_sha256": sha256(checkpoint),
        "training_inventory_sha256": sha256(args.training_inventory),
        "tuning_inventory_sha256": sha256(args.tuning_inventory),
        "evaluator_source_sha256": sha256(Path(__file__)),
        "miner_model_revision": revision,
        "miner_health_policy": (
            "user_authorized_maintenance"
            if args.miner_maintenance_mode
            else "required_stable_revision"
        ),
        "decoding": {
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "length_penalty": args.length_penalty,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
        },
        "summaries": summaries,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(result, output, sort_keys=True, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(summaries, sort_keys=True))


if __name__ == "__main__":
    main()
