"""Probe seen-example decoding and video conditioning for a fresh low-resolution run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmark_fleurs_dev import distance, miner_revision, normalize, sha256


def _select_seen(rows: list[dict], examples: list[dict], count: int) -> list[dict]:
    if count < 2 or len(examples) < count:
        raise ValueError("training-fit probe needs at least two seen examples")
    by_identity = {(row["source"], row["example_id"]): row for row in rows}
    positions = [round(index * (len(examples) - 1) / (count - 1)) for index in range(count)]
    selected = []
    for position in positions:
        identity = (examples[position]["source"], examples[position]["example_id"])
        if identity not in by_identity:
            raise ValueError("training run references an example outside its inventory")
        selected.append(by_identity[identity])
    if len({(row["source"], row["example_id"]) for row in selected}) != count:
        raise ValueError("training-fit selection repeated an example")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-file", type=Path)
    parser.add_argument("--language-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--miner-health-url")
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
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
        or args.examples < 2
        or args.max_new_tokens < 1
    ):
        parser.error("probe paths or bounds are invalid")
    if not args.miner_maintenance_mode and not args.miner_health_url:
        parser.error("miner health URL is required outside maintenance mode")
    if args.miner_maintenance_mode and args.miner_health_url:
        parser.error("maintenance mode must not claim a miner health endpoint")
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
        mobilenet_v3_small_encoder,
        s3d_video_encoder,
    )
    from fresh_lowres_training import ctc_alignment_loss, ctc_token_logits
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
        or run.get("language_model_sha256") != receipt["weights_sha256"]
        or run.get("inventory_sha256") != sha256(args.inventory)
        or run.get("model_source_sha256")
        != sha256(Path(__file__).with_name("fresh_lowres_model.py"))
        or run.get("data_source_sha256")
        != sha256(Path(__file__).with_name("fresh_lowres_data.py"))
    ):
        raise ValueError("training-fit run provenance differs")
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    if (
        metadata.get("schema") != "umi-studio-fresh-lowres-trainable/1"
        or metadata.get("run_sha256") != sha256(run_path)
    ):
        raise ValueError("training-fit checkpoint provenance differs")
    rows = load_inventory(args.inventory, workspace)
    selected = _select_seen(rows, run["examples"], args.examples)
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
        raise ValueError("training-fit checkpoint topology differs")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value)
    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
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

    def check_miner(expected: str | None) -> str | None:
        if args.miner_maintenance_mode:
            return None
        return miner_revision(args.miner_health_url, expected)

    encoded = []
    records = []
    for row in selected:
        revision = check_miner(revision)
        if sha256(row["video_path"]) != row["video_sha256"]:
            raise ValueError("training-fit video checksum changed")
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
                num_beams=1,
            )
        hypothesis = tokenizer.decode(tokens[0], skip_special_tokens=True)
        ctc_token_ids = []
        ctc_hypothesis = None
        if run.get("ctc_weight", 0):
            frame_tokens = ctc_token_logits(
                memory, model.language_model.shared.weight
            ).argmax(dim=-1)[0]
            previous = None
            for token_id in frame_tokens.detach().cpu().tolist():
                if token_id != previous and token_id != tokenizer.pad_token_id:
                    ctc_token_ids.append(token_id)
                previous = token_id
            ctc_hypothesis = tokenizer.decode(ctc_token_ids, skip_special_tokens=True)
        reference_words = normalize(row["reference"])
        hypothesis_words = normalize(hypothesis)
        records.append(
            {
                "source": row["source"],
                "example_id": row["example_id"],
                "reference": row["reference"],
                "hypothesis": hypothesis,
                "generated_token_ids": tokens[0].detach().cpu().tolist(),
                "ctc_hypothesis": ctc_hypothesis,
                "ctc_token_ids": ctc_token_ids,
                "frames": frames.shape[1],
                "wer": distance(reference_words, hypothesis_words) / len(reference_words),
            }
        )
        encoded.append((memory, memory_mask))
        revision = check_miner(revision)
        print(
            f"decoded {row['source']} {row['example_id']}: {hypothesis!r} "
            f"ctc={ctc_hypothesis!r}",
            flush=True,
        )
    conditioning = []
    for index, row in enumerate(selected):
        other_index = (index + 1) % len(selected)
        own_memory, own_mask = encoded[index]
        swapped_memory, swapped_mask = encoded[other_index]
        labels = tokenizer(row["reference"], return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            own_logits = model.language_model(
                encoder_outputs=BaseModelOutput(last_hidden_state=own_memory),
                attention_mask=own_mask,
                labels=labels,
                return_dict=True,
            ).logits
            swapped_logits = model.language_model(
                encoder_outputs=BaseModelOutput(last_hidden_state=swapped_memory),
                attention_mask=swapped_mask,
                labels=labels,
                return_dict=True,
            ).logits
            if run.get("ctc_weight", 0):
                try:
                    own_ctc_loss = ctc_alignment_loss(
                        own_memory,
                        own_mask,
                        labels,
                        model.language_model.shared.weight,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                except ValueError as error:
                    if "exceeds its temporal sequence" not in str(error):
                        raise
                    own_ctc_loss = None
                try:
                    swapped_ctc_loss = ctc_alignment_loss(
                        swapped_memory,
                        swapped_mask,
                        labels,
                        model.language_model.shared.weight,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                except ValueError as error:
                    if "exceeds its temporal sequence" not in str(error):
                        raise
                    swapped_ctc_loss = None
            else:
                own_ctc_loss = None
                swapped_ctc_loss = None
        conditioning.append(
            {
                "example_id": row["example_id"],
                "swapped_from_example_id": selected[other_index]["example_id"],
                "mean_absolute_logit_change": float(
                    (own_logits - swapped_logits).abs().mean().cpu()
                ),
                "own_ctc_loss": (
                    float(own_ctc_loss.cpu()) if own_ctc_loss is not None else None
                ),
                "swapped_ctc_loss": (
                    float(swapped_ctc_loss.cpu()) if swapped_ctc_loss is not None else None
                ),
            }
        )
    feasible_ctc_swaps = [
        row
        for row in conditioning
        if row["own_ctc_loss"] is not None and row["swapped_ctc_loss"] is not None
    ]
    result = {
        "schema": "umi-studio-fresh-lowres-training-fit/1",
        "diagnostic_only": True,
        "split": "training",
        "run_sha256": sha256(run_path),
        "checkpoint_sha256": sha256(checkpoint),
        "inventory_sha256": sha256(args.inventory),
        "probe_source_sha256": sha256(Path(__file__)),
        "miner_model_revision": revision,
        "miner_health_policy": (
            "user_authorized_maintenance"
            if args.miner_maintenance_mode
            else "required_stable_revision"
        ),
        "mean_seen_wer": sum(row["wer"] for row in records) / len(records),
        "distinct_hypotheses": len({row["hypothesis"] for row in records}),
        "distinct_ctc_hypotheses": len(
            {
                row["ctc_hypothesis"]
                for row in records
                if row["ctc_hypothesis"] is not None
            }
        ),
        "mean_absolute_swapped_logit_change": sum(
            row["mean_absolute_logit_change"] for row in conditioning
        )
        / len(conditioning),
        "mean_swapped_minus_own_ctc_loss": (
            sum(
                row["swapped_ctc_loss"] - row["own_ctc_loss"]
                for row in feasible_ctc_swaps
            )
            / len(feasible_ctc_swaps)
            if feasible_ctc_swaps
            else None
        ),
        "feasible_ctc_swap_pairs": len(feasible_ctc_swaps),
        "records": records,
        "conditioning": conditioning,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    if temporary.exists():
        parser.error("unfinished training-fit probe needs inspection")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(result, output, sort_keys=True, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "mean_seen_wer": result["mean_seen_wer"],
                "distinct_hypotheses": result["distinct_hypotheses"],
                "distinct_ctc_hypotheses": result["distinct_ctc_hypotheses"],
                "mean_absolute_swapped_logit_change": result[
                    "mean_absolute_swapped_logit_change"
                ],
                "mean_swapped_minus_own_ctc_loss": result[
                    "mean_swapped_minus_own_ctc_loss"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
