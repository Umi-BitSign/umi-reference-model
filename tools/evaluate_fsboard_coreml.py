"""Offline right-hand FSboard diagnostic for the pinned Core ML CTC checkpoint.

The checkpoint's left-hand mirroring map is not yet independently implemented,
so left-hand-dominant clips are explicitly skipped rather than mis-scored.
This single-reference validation CER is not an official UMI evaluation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import regex
from benchmark_fleurs_dev import distance, miner_revision, normalize, sha256
from fetch_fsboard_validation import validate_selection


def model_input(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    if landmarks.ndim != 3 or landmarks.shape[1:] != (543, 3) or landmarks.dtype != np.float32:
        raise ValueError("unexpected MediaPipe Holistic landmark shape")
    frames = landmarks[~np.isnan(landmarks).all(axis=(1, 2))][:768]
    count = len(frames)
    if not count:
        raise ValueError("no usable landmark frames")
    features = np.zeros((1, 768, 1629), dtype=np.float32)
    mask = np.zeros((1, 768), dtype=np.float32)
    for axis in range(3):
        values = frames[:, :, axis]
        finite = np.isfinite(values)
        observed = values[finite]
        mean = float(observed.mean()) if observed.size else (0.0 if axis == 2 else 0.5)
        std = float(observed.std()) if observed.size else 0.0
        if std > 0 and np.isfinite(std):
            scaled = (values - mean) / std
            features[0, :count, axis::3] = np.where(np.isfinite(scaled), scaled, 0)
    mask[0, :count] = 1
    return features, mask, count


def ctc_decode(logits: np.ndarray, frame_count: int, vocab: list[str]) -> str:
    if logits.shape != (1, 384, 62) or len(vocab) != 60:
        raise ValueError("unexpected Core ML CTC output or vocabulary")
    labels = logits[0, : (frame_count + 1) // 2].argmax(axis=-1)
    return "".join(
        vocab[int(label)] for label, _ in itertools.groupby(labels) if 0 < label < len(vocab)
    )


def comparable_units(value: str) -> tuple[str, ...]:
    """Use UMI 0.1-style normalization and ignore whitespace for CER."""
    return tuple(regex.findall(r"\X", "".join(normalize(value))))


def main() -> None:
    import coremltools as ct

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--landmarks-dir", type=Path, required=True)
    parser.add_argument("--landmark-receipts", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    if args.limit < 1 or not args.model.is_dir():
        parser.error("limit must be positive and model must exist")
    selection = json.loads(args.selection.read_text())
    rows = validate_selection(selection)
    selection_hash = sha256(args.selection)
    vocab_hash = sha256(args.vocab)
    weights_hash = sha256(args.model / "Data/com.apple.CoreML/weights/weight.bin")
    selected_names = {row["clip_filename"] for row in rows}
    receipts = {
        row["clip_filename"]: row
        for row in map(json.loads, args.landmark_receipts.read_text().splitlines())
    }
    vocab = args.vocab.read_text().splitlines()
    completed = set()
    if args.results.exists():
        for line in args.results.read_text().splitlines():
            result = json.loads(line)
            name = result["clip_filename"]
            if (
                name in completed
                or name not in selected_names
                or result["schema"] != "umi-studio-fsboard-coreml-diagnostic/2"
                or result["selection_sha256"] != selection_hash
                or result["vocab_sha256"] != vocab_hash
                or result["model_weights_sha256"] != weights_hash
                or result["landmarks_sha256"] != receipts[name]["sha256"]
            ):
                raise ValueError("existing FSboard evaluation differs from inputs")
            completed.add(name)
    revision = miner_revision(args.miner_health_url, None)
    model = ct.models.MLModel(str(args.model), compute_units=ct.ComputeUnit.CPU_ONLY)
    revision = miner_revision(args.miner_health_url, revision)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    evaluated = 0
    for row in rows:
        name = row["clip_filename"]
        if name in completed or evaluated >= args.limit:
            continue
        receipt = receipts.get(name)
        source = args.landmarks_dir / f"{Path(name).stem}.npz"
        if (
            receipt is None
            or receipt["selection_sha256"] != selection_hash
            or not source.is_file()
            or receipt["sha256"] != sha256(source)
        ):
            continue
        revision = miner_revision(args.miner_health_url, revision)
        with np.load(source, allow_pickle=False) as payload:
            landmarks = payload["landmarks"]
        status = "right_hand_evaluated"
        reference = comparable_units(row["reference"])
        result = {
            "schema": "umi-studio-fsboard-coreml-diagnostic/2",
            "clip_filename": name,
            "selection_sha256": selection_hash,
            "landmarks_sha256": receipt["sha256"],
            "vocab_sha256": vocab_hash,
            "model_weights_sha256": weights_hash,
            "mirror_input": receipt.get("mirror_input", False),
            "reference": row["reference"],
            "signer_id": row["signer_id"],
        }
        if receipt["left_hand_finite_scalars"] > receipt["right_hand_finite_scalars"]:
            status = "skipped_left_hand_dominant"
        else:
            features, mask, count = model_input(landmarks)
            logits = model.predict({"features": features, "mask": mask})["Identity"]
            hypothesis = ctc_decode(logits, count, vocab)
            comparable_hypothesis = comparable_units(hypothesis)
            result.update(
                hypothesis=hypothesis,
                frame_count=count,
                single_reference_cer=distance(reference, comparable_hypothesis)
                / max(1, len(reference)),
            )
        revision = miner_revision(args.miner_health_url, revision)
        result.update(status=status, miner_model_revision=revision)
        with args.results.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        evaluated += 1
        print(f"{len(completed) + evaluated}: {status} {name}", flush=True)


if __name__ == "__main__":
    main()
