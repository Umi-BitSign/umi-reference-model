"""Offline FSboard validation diagnostic for the pinned second-place TFLite model.

The model embeds its own preprocessing and left-hand mirroring. The published
submission column declaration orders x/y/z separately within face, left hand,
right hand and pose. The training TFRecord reader instead reshapes global x/y/z
blocks with face, left hand, pose, right hand within each block. Local caches
store face, left hand, pose, right hand. Both layouts are diagnostic choices.
This diagnostic uses public validation clips only and cannot promote a model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from benchmark_fleurs_dev import distance, miner_revision, sha256
from evaluate_fsboard_coreml import comparable_units
from fetch_fsboard_validation import validate_selection

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "dd299ab7bbde6be5cc482fc991d7cbd9d7466395781a4e9b029f4e40bcdbd4b4"
VOCAB_SHA256 = "e9d4123f6ddec0e1cb0c2057922d4bff69d119c29a388cdcf11b98bc19143545"
SCHEMA = "umi-studio-fsboard-tflite-diagnostic/2"
LAYOUTS = ("submission_columns", "training_tensor")


def model_input(landmarks: np.ndarray, layout: str = "submission_columns") -> np.ndarray:
    """Map local (face, left, pose, right) points to one publisher layout."""
    if landmarks.ndim != 3 or landmarks.shape[1:] != (543, 3) or landmarks.dtype != np.float32:
        raise ValueError("unexpected MediaPipe Holistic landmark shape")
    if not len(landmarks) or np.isnan(landmarks).all():
        raise ValueError("no usable landmark frames")
    if layout not in LAYOUTS:
        raise ValueError("unknown TFLite input layout")
    if layout == "training_tensor":
        return np.ascontiguousarray(
            np.concatenate([landmarks[:, :, axis] for axis in range(3)], axis=1)
        )
    parts = (
        landmarks[:, :468, :],
        landmarks[:, 468:489, :],
        landmarks[:, 522:543, :],
        landmarks[:, 489:522, :],
    )
    return np.ascontiguousarray(
        np.concatenate([part[:, :, axis] for part in parts for axis in range(3)], axis=1)
    )


def decode_output(output: np.ndarray, vocab: list[str]) -> str:
    if (
        output.ndim != 2
        or output.shape[1] != 59
        or not len(output)
        or len(vocab) != 60
        or vocab[0] != "<blank>"
        or not np.isfinite(output).all()
        or not np.allclose(output.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("unexpected TFLite decoded output or vocabulary")
    return "".join(vocab[int(label) + 1] for label in output.argmax(axis=1))


def checked_existing(
    path: Path, selected: set[str], selection_hash: str, model_hash: str, vocab_hash: str,
    receipts: dict[str, dict], layout: str,
) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    for line in path.read_text().splitlines():
        record = json.loads(line)
        name = record["clip_filename"]
        if (
            name in completed
            or name not in selected
            or record.get("schema") != SCHEMA
            or record.get("selection_sha256") != selection_hash
            or record.get("model_sha256") != model_hash
            or record.get("vocab_sha256") != vocab_hash
            or record.get("landmarks_sha256") != receipts[name]["sha256"]
            or record.get("input_layout") != layout
        ):
            raise ValueError("existing TFLite evaluation differs from inputs")
        completed.add(name)
    return completed


def checked_landmark_columns(
    row: dict, receipt: dict, landmarks_dir: Path, selection_hash: str, layout: str
) -> np.ndarray:
    name = row["clip_filename"]
    source = landmarks_dir / f"{Path(name).stem}.npz"
    if (
        receipt.get("selection_sha256") != selection_hash
        or receipt.get("mirror_input", False)
        or not source.is_file()
        or receipt.get("sha256") != sha256(source)
    ):
        raise ValueError(f"FSboard landmark receipt mismatch: {name}")
    with np.load(source, allow_pickle=False) as payload:
        landmarks = payload["landmarks"]
    if receipt.get("shape") != list(landmarks.shape):
        raise ValueError(f"FSboard landmark shape differs from receipt: {name}")
    return model_input(landmarks, layout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--landmarks-dir", type=Path, required=True)
    parser.add_argument("--landmark-receipts", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--miner-health-url")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--input-layout", choices=LAYOUTS, default="submission_columns")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.limit < 1 or not args.results.resolve().is_relative_to(ROOT / "runs"):
        parser.error("limit must be positive and results must stay under runs/")
    if not args.preflight_only and not args.miner_health_url:
        parser.error("--miner-health-url is required for inference")
    model_hash = sha256(args.model)
    if model_hash != MODEL_SHA256:
        raise ValueError("TFLite model differs from the pinned source artifact")
    selection = json.loads(args.selection.read_text())
    rows = validate_selection(selection)
    selection_hash = sha256(args.selection)
    vocab_hash = sha256(args.vocab)
    if vocab_hash != VOCAB_SHA256:
        raise ValueError("TFLite vocabulary differs from the pinned comparison artifact")
    vocab = args.vocab.read_text().splitlines()
    receipt_rows = list(map(json.loads, args.landmark_receipts.read_text().splitlines()))
    receipts = {row["clip_filename"]: row for row in receipt_rows}
    selected_names = {row["clip_filename"] for row in rows}
    if (
        len(receipt_rows) != len(rows)
        or len(receipts) != len(rows)
        or set(receipts) != selected_names
    ):
        raise ValueError("missing or duplicated landmark receipts")
    completed = checked_existing(
        args.results, selected_names, selection_hash,
        model_hash, vocab_hash, receipts, args.input_layout,
    )
    if args.preflight_only:
        frames = 0
        for row in rows:
            frames += len(checked_landmark_columns(
                row, receipts[row["clip_filename"]], args.landmarks_dir,
                selection_hash, args.input_layout,
            ))
        print(f"Preflight passed: {len(rows)} clips, {frames} input frames; no inference run")
        return

    # The live miner has priority. Refuse to load a separate inference runtime
    # while its health endpoint is unavailable, slow or on another revision.
    revision = miner_revision(args.miner_health_url, None)
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(args.model), num_threads=2)
    runner = interpreter.get_signature_runner("serving_default")
    revision = miner_revision(args.miner_health_url, revision)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    evaluated = 0
    for row in rows:
        name = row["clip_filename"]
        if name in completed or evaluated >= args.limit:
            continue
        receipt = receipts[name]
        revision = miner_revision(args.miner_health_url, revision)
        columns = checked_landmark_columns(
            row, receipt, args.landmarks_dir, selection_hash, args.input_layout
        )
        output = runner(inputs=columns)["outputs"]
        hypothesis = decode_output(output, vocab)
        revision = miner_revision(args.miner_health_url, revision)
        reference = comparable_units(row["reference"])
        record = {
            "schema": SCHEMA,
            "clip_filename": name,
            "selection_sha256": selection_hash,
            "landmarks_sha256": receipt["sha256"],
            "model_sha256": model_hash,
            "vocab_sha256": vocab_hash,
            "input_layout": args.input_layout,
            "signer_id": row["signer_id"],
            "reference": row["reference"],
            "hypothesis": hypothesis,
            "input_frames": int(len(columns)),
            "single_reference_cer": distance(reference, comparable_units(hypothesis))
            / max(1, len(reference)),
            "miner_model_revision": revision,
        }
        with args.results.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        evaluated += 1
        print(f"{len(completed) + evaluated}: decoded {name}", flush=True)


if __name__ == "__main__":
    main()
