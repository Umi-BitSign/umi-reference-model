"""Compare container landmarks with an existing frozen C5 visual cache."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from benchmark_fleurs_dev import sha256
from extract_mediapipe_landmarks import unpack_landmarks


def array_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict:
    if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
        raise ValueError("candidate and reference arrays differ in topology")
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("candidate or reference array is non-finite")
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    candidate_flat = candidate.astype(np.float64).ravel()
    reference_flat = reference.astype(np.float64).ravel()
    denominator = np.linalg.norm(candidate_flat) * np.linalg.norm(reference_flat)
    return {
        "shape": list(candidate.shape),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "root_mean_square_error": float(np.sqrt(np.mean(difference**2))),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "cosine_similarity": float(
            np.dot(candidate_flat, reference_flat) / denominator if denominator else 1.0
        ),
        "exact_fraction": float(np.mean(candidate == reference)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    if (
        not args.clip.is_file()
        or not args.landmarks.is_file()
        or not args.reference_cache.is_file()
        or not args.baseline.is_dir()
        or args.output.exists()
        or not args.output.resolve().is_relative_to((workspace / "runs").resolve())
    ):
        parser.error("input or output paths are invalid")
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
    sys.path.insert(0, str(baseline))

    import torch
    from runtime import SHuBERTInferenceRuntime, _rgb_crop_frames
    from transformers.generation.utils import GenerationMixin
    from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

    torch.set_num_threads(2)
    model = SHuBERTInferenceRuntime(baseline, device="cpu")
    reader = model._video_reader(args.clip)
    frames = reader.get_batch(range(len(reader))).asnumpy()
    landmarks, landmark_metadata = unpack_landmarks(args.landmarks)
    if (
        len(landmarks) != len(reader)
        or landmark_metadata["frames"] != len(reader)
        or landmark_metadata["source_sha256"] != sha256(args.clip)
    ):
        raise ValueError("container landmarks differ from the validation clip")
    left_frames, right_frames = model._hand_extractor.extract_hand_frames(frames, landmarks)
    face_frames = model._face_extractor.extract_face_frames(frames, landmarks)
    pose = model._process_pose_landmarks(landmarks).astype(np.float32)
    left = model._hand_model.extract_embeddings_from_frames(_rgb_crop_frames(left_frames)).astype(
        np.float32
    )
    right = model._hand_model.extract_embeddings_from_frames(_rgb_crop_frames(right_frames)).astype(
        np.float32
    )
    face = model._face_model.extract_embeddings_from_frames(_rgb_crop_frames(face_frames)).astype(
        np.float32
    )
    with torch.inference_mode():
        hidden = (
            model._translator.encoder(
                face_features=torch.from_numpy(face).unsqueeze(0),
                left_hand_features=torch.from_numpy(left).unsqueeze(0),
                right_hand_features=torch.from_numpy(right).unsqueeze(0),
                pose_features=torch.from_numpy(pose).unsqueeze(0),
                return_dict=True,
            )
            .last_hidden_state[0]
            .cpu()
            .numpy()
        )
    candidate = {
        "encoder_hidden_state": hidden,
        "face_features": face,
        "left_hand_features": left,
        "right_hand_features": right,
        "pose_features": pose,
    }
    with np.load(args.reference_cache, allow_pickle=False) as archive:
        if set(archive.files) != set(candidate):
            raise ValueError("reference cache stream set differs")
        reference = {name: archive[name] for name in archive.files}
    metrics = {name: array_metrics(candidate[name], reference[name]) for name in sorted(candidate)}

    def decode(state: np.ndarray) -> str:
        outputs = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=torch.from_numpy(state).unsqueeze(0)
        )
        with torch.inference_mode():
            identifiers = GenerationMixin.generate(
                model._translator,
                encoder_outputs=outputs,
                max_length=model.generation_max_length,
                num_beams=model.generation_num_beams,
                early_stopping=True,
                pad_token_id=model._tokenizer.pad_token_id,
                eos_token_id=model._tokenizer.eos_token_id,
            )
        return model._tokenizer.decode(identifiers[0], skip_special_tokens=True)

    hypotheses = {
        "container_landmarks": decode(candidate["encoder_hidden_state"]),
        "frozen_reference": decode(reference["encoder_hidden_state"]),
    }
    result = {
        "schema": "umi-studio-mediapipe-container-fidelity/1",
        "clip_sha256": sha256(args.clip),
        "landmarks_sha256": sha256(args.landmarks),
        "reference_cache_sha256": sha256(args.reference_cache),
        "baseline_identity_sha256": sha256(baseline / "MODEL_IDENTITY.json"),
        "landmark_metadata": landmark_metadata,
        "metrics": metrics,
        "hypotheses": hypotheses,
        "hypotheses_equal": len(set(hypotheses.values())) == 1,
        "all_cosine_at_least_0_999": all(
            row["cosine_similarity"] >= 0.999 for row in metrics.values()
        ),
    }
    numeric_metrics = (
        value for row in metrics.values() for value in row.values() if isinstance(value, float)
    )
    if not all(math.isfinite(value) for value in numeric_metrics):
        raise RuntimeError("fidelity metrics are non-finite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
