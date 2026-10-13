"""Verify TFLite/Core ML parity on two public FSboard validation samples."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from benchmark_fleurs_dev import distance, sha256
from evaluate_fsboard_coreml import comparable_units
from evaluate_fsboard_tflite import MODEL_SHA256, SCHEMA, VOCAB_SHA256
from fetch_fsboard_validation import validate_selection

ROOT = Path(__file__).resolve().parents[1]
COREML_SCHEMA = "umi-studio-fsboard-coreml-diagnostic/2"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def summarize_sample(
    selection: dict, selection_hash: str, tflite_rows: list[dict],
    coreml_right: list[dict], coreml_left: list[dict],
) -> dict:
    selected = validate_selection(selection)
    names = {row["clip_filename"] for row in selected}
    if len(names) != len(selected) or len(tflite_rows) != len(selected):
        raise ValueError("FSboard selection or TFLite results contain repeated clips")
    tflite = {row["clip_filename"]: row for row in tflite_rows}
    if set(tflite) != names:
        raise ValueError("TFLite results do not cover the selection exactly")
    coreml_rows = [
        (row, mirrored)
        for records, mirrored in ((coreml_right, False), (coreml_left, True))
        for row in records
        if row["status"] == "right_hand_evaluated"
    ]
    coreml = {row["clip_filename"]: row for row, _ in coreml_rows}
    if len(coreml) != len(coreml_rows) or set(coreml) != names:
        raise ValueError("Core ML results do not cover the selection exactly")
    for row, mirrored in coreml_rows:
        if row.get("mirror_input", False) != mirrored:
            raise ValueError("Core ML mirroring differs from its source ledger")

    scores = []
    exact = 0
    for selection_row in selected:
        name = selection_row["clip_filename"]
        lite = tflite[name]
        ml = coreml[name]
        if (
            lite.get("schema") != SCHEMA
            or lite.get("input_layout") != "training_tensor"
            or lite.get("selection_sha256") != selection_hash
            or lite.get("model_sha256") != MODEL_SHA256
            or lite.get("vocab_sha256") != VOCAB_SHA256
            or ml.get("schema") != COREML_SCHEMA
            or ml.get("selection_sha256") != selection_hash
            or ml.get("vocab_sha256") != VOCAB_SHA256
            or lite.get("reference") != selection_row["reference"]
            or ml.get("reference") != selection_row["reference"]
            or lite.get("signer_id") != selection_row["signer_id"]
            or ml.get("signer_id") != selection_row["signer_id"]
        ):
            raise ValueError(f"FSboard result provenance differs: {name}")
        reference = comparable_units(selection_row["reference"])
        lite_cer = distance(reference, comparable_units(lite["hypothesis"])) / max(1, len(reference))
        ml_cer = distance(reference, comparable_units(ml["hypothesis"])) / max(1, len(reference))
        if (
            abs(lite_cer - lite["single_reference_cer"]) > 1e-9
            or abs(ml_cer - ml["single_reference_cer"]) > 1e-9
        ):
            raise ValueError(f"FSboard CER differs from hypothesis: {name}")
        scores.append((lite_cer, ml_cer))
        exact += lite["hypothesis"] == ml["hypothesis"]
    return {
        "clips": len(scores),
        "signers": len({row["signer_id"] for row in selected}),
        "tflite_mean_cer": statistics.fmean(item[0] for item in scores),
        "coreml_mean_cer": statistics.fmean(item[1] for item in scores),
        "exact_transcription_matches": exact,
        "tflite_cer_wins": sum(left < right for left, right in scores),
        "coreml_cer_wins": sum(right < left for left, right in scores),
        "cer_ties": sum(left == right for left, right in scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-selection", type=Path, required=True)
    parser.add_argument("--pilot-tflite", type=Path, required=True)
    parser.add_argument("--pilot-coreml-right", type=Path, required=True)
    parser.add_argument("--pilot-coreml-left", type=Path, required=True)
    parser.add_argument("--followup-selection", type=Path, required=True)
    parser.add_argument("--followup-tflite", type=Path, required=True)
    parser.add_argument("--followup-coreml-right", type=Path, required=True)
    parser.add_argument("--followup-coreml-left", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs"):
        parser.error("summary needs a new output under runs/")
    samples = {}
    source_hashes = {}
    selections = {}
    for label in ("pilot", "followup"):
        selection_path = getattr(args, f"{label}_selection")
        lite_path = getattr(args, f"{label}_tflite")
        right_path = getattr(args, f"{label}_coreml_right")
        left_path = getattr(args, f"{label}_coreml_left")
        selection = json.loads(selection_path.read_text())
        selections[label] = selection
        samples[label] = summarize_sample(
            selection, sha256(selection_path), load_rows(lite_path),
            load_rows(right_path), load_rows(left_path),
        )
        source_hashes[label] = {
            "selection": sha256(selection_path),
            "tflite": sha256(lite_path),
            "coreml_right": sha256(right_path),
            "coreml_left": sha256(left_path),
        }
    pilot_rows = validate_selection(selections["pilot"])
    followup_rows = validate_selection(selections["followup"])
    for key in ("clip_filename", "reference", "source_recording"):
        if {row[key] for row in pilot_rows} & {row[key] for row in followup_rows}:
            raise ValueError(f"pilot and follow-up share {key}")
    total = sum(sample["clips"] for sample in samples.values())
    result = {
        "schema": "umi-studio-fsboard-tflite-parity/1",
        "source_split": "public FSboard validation",
        "test_used": False,
        "input_layout": "training_tensor",
        "model_sha256": MODEL_SHA256,
        "source_sha256": source_hashes,
        "samples": samples,
        "combined": {
            "clips": total,
            "tflite_mean_cer": sum(sample["clips"] * sample["tflite_mean_cer"] for sample in samples.values()) / total,
            "coreml_mean_cer": sum(sample["clips"] * sample["coreml_mean_cer"] for sample in samples.values()) / total,
            "exact_transcription_matches": sum(sample["exact_transcription_matches"] for sample in samples.values()),
        },
        "decision": "portable_inference_lead_not_promoted",
        "limits": [
            "The same signers appear in both public validation samples.",
            "This is single-reference CER, not the official UMI multi-reference evaluation.",
            "Rights for the TFLite file and public redistribution remain unresolved.",
            "The native worker does not yet route the fingerspelling stratum to a specialist.",
        ],
        "live_miner_changed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result["combined"], sort_keys=True))


if __name__ == "__main__":
    main()
