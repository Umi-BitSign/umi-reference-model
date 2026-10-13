"""Evaluate a SHuBERT/ByT5 checkpoint on frozen validation encoder states.

This is a single-reference development diagnostic, not the official UMI metric.
Run from the independent baseline environment and keep confirmation sealed until
a candidate and all tuning choices are frozen.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from benchmark_fleurs_dev import distance, miner_revision, normalize, sha256
from cache_fleurs_encoder import checked_stream_features


def evaluation_identifiers(source_rows: list[dict], split: str) -> list[str]:
    source_split = (
        "training"
        if split in {"openasl_training", "how2sign_expansion_holdout"}
        else split
    )
    if not source_rows or any(row["split"] != source_split for row in source_rows):
        raise ValueError("encoder index is empty or uses another split")
    expected_schema = {
        "validation": "umi-studio-how2sign-encoder-cache-example/1",
        "tuning": "umi-studio-fleurs-dev-encoder-cache-example/1",
        "confirmation": "umi-studio-fleurs-dev-encoder-cache-example/1",
        "openasl_training": "umi-studio-openasl-stream-cache/1",
        "how2sign_expansion_holdout": (
            "umi-studio-how2sign-precut-expansion-stream-cache/1"
        ),
    }[split]
    if any(row["schema"] != expected_schema for row in source_rows):
        raise ValueError("encoder index schema differs from its evaluation split")
    identifiers = [
        (
            row["sentence_name"]
            if split in {"validation", "how2sign_expansion_holdout"}
            else row["clip_id"]
            if split == "openasl_training"
            else row["example_id"]
        )
        for row in source_rows
    ]
    if len(set(identifiers)) != len(source_rows):
        raise ValueError("encoder index contains duplicate examples")
    return identifiers


def cached_visual_streams(payload, row: dict) -> dict:
    expected = {
        "encoder_hidden_state",
        "face_features",
        "left_hand_features",
        "right_hand_features",
        "pose_features",
    }
    if set(payload.files) != expected or row.get("stream_feature_dtype") != "float32":
        raise ValueError("evaluation cache lacks the four float32 visual streams")
    names = expected - {"encoder_hidden_state"}
    if any(payload[name].dtype.name != "float32" for name in names):
        raise ValueError("evaluation visual streams have another dtype")
    streams = checked_stream_features(
        {name: payload[name] for name in names}, row["encoder_shape"][0]
    )
    if row.get("stream_feature_shapes") != {
        name: list(value.shape) for name, value in streams.items()
    }:
        raise ValueError("evaluation visual stream metadata differs from payload")
    return streams


def temporal_subsample(streams: dict, stride: int) -> dict:
    if stride < 1:
        raise ValueError("temporal stride must be positive")
    result = {name: value[::stride] for name, value in streams.items()}
    lengths = {value.shape[0] for value in result.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
        raise ValueError("temporal subsampling returned invalid stream lengths")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=(
            "tuning",
            "confirmation",
            "validation",
            "openasl_training",
            "how2sign_expansion_holdout",
        ),
        required=True,
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--miner-health-url")
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--use-stream-features", action="store_true")
    parser.add_argument(
        "--temporal-stride",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help="evaluate every Nth cached visual-stream frame",
    )
    parser.add_argument("--limit", type=int, default=1, help="maximum new examples")
    args = parser.parse_args()
    if args.miner_maintenance_mode == bool(args.miner_health_url):
        parser.error(
            "choose exactly one of --miner-health-url or --miner-maintenance-mode"
        )
    if args.limit < 1:
        parser.error("limit must be positive")
    if args.temporal_stride != 1 and not args.use_stream_features:
        parser.error("temporal stride requires --use-stream-features")
    if not args.cache_dir.is_dir() or not args.checkpoint.is_dir():
        parser.error("cache and checkpoint directories must exist")
    baseline = args.baseline.resolve()
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
    from safetensors import safe_open
    from safetensors.torch import load_file
    from shubert_cross_attention import load_cross_attention
    from shubert_full_decoder import load_full_decoder
    from shubert_lora import install_lora, load_adapter
    from shubert_visual_lora import install_visual_lora, load_visual_adapter
    from transformers import ByT5Tokenizer
    from transformers.generation.utils import GenerationMixin
    from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

    torch.set_num_threads(2)
    checkpoint_hash = sha256(args.checkpoint / "model.safetensors")
    adapter_hash = sha256(args.adapter) if args.adapter is not None else None
    evaluation_input = (
        "visual_streams" if args.use_stream_features else "frozen_encoder_hidden_state"
    )
    index_hash = sha256(args.index)
    source_rows = [json.loads(line) for line in args.index.read_text().splitlines()]
    identifiers = evaluation_identifiers(source_rows, args.split)
    completed = {}
    if args.results.exists():
        for line in args.results.read_text().splitlines():
            record = json.loads(line)
            if (
                record["checkpoint_sha256"] != checkpoint_hash
                or record["index_sha256"] != index_hash
                or record.get("adapter_sha256") != adapter_hash
                or record.get("evaluation_input", "frozen_encoder_hidden_state")
                != evaluation_input
                or record.get("temporal_stride", 1) != args.temporal_stride
            ):
                raise ValueError("existing evaluation results use other inputs")
            if record["example_id"] in completed:
                raise ValueError("evaluation results contain duplicate examples")
            completed[record["example_id"]] = record
    def check_miner(current_revision):
        if args.miner_maintenance_mode:
            return None
        return miner_revision(args.miner_health_url, current_revision)

    revision = check_miner(None)
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        str(args.checkpoint), local_files_only=True
    )
    if args.adapter is not None:
        with safe_open(str(args.adapter), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        if metadata.get("baseline_checkpoint_sha256") != checkpoint_hash:
            raise ValueError("adapter metadata does not match the frozen checkpoint")
        weights = load_file(str(args.adapter), device="cpu")
        if metadata.get("schema") == "umi-studio-shubert-lora/1":
            install_lora(
                model,
                int(metadata["rank"]),
                float(metadata["alpha"]),
                metadata.get("targets", "cross_qv"),
            )
            load_adapter(model, weights)
        elif metadata.get("schema") == "umi-studio-shubert-cross-attention/1":
            load_cross_attention(model, weights)
        elif metadata.get("schema") == "umi-studio-shubert-full-decoder/1":
            load_full_decoder(model, weights)
        elif metadata.get("schema") == "umi-studio-shubert-visual-lora/1":
            if not args.use_stream_features:
                raise ValueError("visual adapter evaluation requires visual stream features")
            install_visual_lora(
                model,
                int(metadata["rank"]),
                float(metadata["alpha"]),
                int(metadata["visual_last_layers"]),
            )
            load_visual_adapter(model, weights)
        else:
            raise ValueError("unsupported adapter schema")
    tokenizer = ByT5Tokenizer.from_pretrained(
        str(baseline / "models/byt5_base"), local_files_only=True
    )
    model.to(args.device)
    model.eval()
    revision = check_miner(revision)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    new_count = 0
    for row, example_id in zip(source_rows, identifiers, strict=True):
        if example_id in completed:
            continue
        if new_count >= args.limit:
            break
        revision = check_miner(revision)
        cache = args.cache_dir / f"{example_id}.npz"
        if sha256(cache) != row["encoder_cache_sha256"]:
            raise ValueError(f"encoder cache differs from index: {cache}")
        with np.load(cache, allow_pickle=False) as payload:
            state = payload["encoder_hidden_state"]
            streams = cached_visual_streams(payload, row) if args.use_stream_features else None
        if list(state.shape) != row["encoder_shape"] or state.dtype != np.float32:
            raise ValueError(f"encoder cache shape or dtype differs from index: {cache}")
        if streams is None:
            hidden = torch.from_numpy(state).unsqueeze(0).to(args.device)
        else:
            streams = temporal_subsample(streams, args.temporal_stride)
            with torch.no_grad():
                hidden = model.encoder(
                    face_features=torch.from_numpy(streams["face_features"])
                    .unsqueeze(0)
                    .to(args.device),
                    left_hand_features=torch.from_numpy(streams["left_hand_features"])
                    .unsqueeze(0)
                    .to(args.device),
                    right_hand_features=torch.from_numpy(streams["right_hand_features"])
                    .unsqueeze(0)
                    .to(args.device),
                    pose_features=torch.from_numpy(streams["pose_features"])
                    .unsqueeze(0)
                    .to(args.device),
                    return_dict=True,
                ).last_hidden_state
            expected_frames = (state.shape[0] + args.temporal_stride - 1) // args.temporal_stride
            valid_shape = hidden.shape == (1, expected_frames, state.shape[1])
            if not valid_shape or not torch.isfinite(hidden).all():
                raise ValueError("visual-stream encoder returned an invalid state")
            if (
                args.adapter is None
                and args.temporal_stride == 1
                and not np.array_equal(hidden[0].cpu().numpy(), state)
            ):
                raise ValueError("baseline visual streams do not reproduce the frozen encoder")
            revision = check_miner(revision)
        outputs = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden)
        with torch.no_grad():
            ids = GenerationMixin.generate(
                model,
                encoder_outputs=outputs,
                max_length=2048,
                num_beams=5,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        hypothesis = tokenizer.decode(ids[0], skip_special_tokens=True)
        revision = check_miner(revision)
        reference_words = normalize(row["reference"])
        edits = distance(reference_words, normalize(hypothesis))
        result = {
            "schema": (
                "umi-studio-how2sign-cached-evaluation-example/1"
                if args.split in {"validation", "how2sign_expansion_holdout"}
                else "umi-studio-openasl-training-fit-example/1"
                if args.split == "openasl_training"
                else "umi-studio-fleurs-cached-evaluation-example/1"
            ),
            "example_id": example_id,
            "split": args.split,
            "selection_sha256": row["selection_sha256"],
            "signer": row.get("signer", row.get("source_group")),
            "index_sha256": index_hash,
            "encoder_cache_sha256": row["encoder_cache_sha256"],
            "checkpoint_sha256": checkpoint_hash,
            "adapter_sha256": adapter_hash,
            "evaluation_input": evaluation_input,
            "temporal_stride": args.temporal_stride,
            "source_frame_count": int(state.shape[0]),
            "evaluated_frame_count": int(hidden.shape[1]),
            "reference": row["reference"],
            "hypothesis": hypothesis,
            "single_reference_wer": edits / len(reference_words),
            "single_reference_word_edits": edits,
            "reference_word_count": len(reference_words),
            "miner_model_revision": revision,
            "miner_health_policy": (
                "user_authorized_maintenance"
                if args.miner_maintenance_mode
                else "required_stable_revision"
            ),
        }
        with args.results.open("a", encoding="utf-8") as output:
            output.write(json.dumps(result, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        new_count += 1
        print(
            f"{len(completed) + new_count}: {example_id} WER={result['single_reference_wer']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
