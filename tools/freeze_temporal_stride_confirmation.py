"""Freeze the sealed two-domain confirmation for the stride-2 C5 candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmark_fleurs_dev import sha256

ROOT = Path(__file__).resolve().parents[1]


def build_plan(
    *,
    decision: dict,
    how2sign_selection: dict,
    how2sign_partition: dict,
    how2sign_receipts: list[dict],
    fleurs_selection: dict,
    fleurs_partition: dict,
    fleurs_receipt: dict,
    hashes: dict[str, str],
) -> dict:
    if (
        decision.get("schema") != "umi-studio-visual-lora-corrective-decision/1"
        or decision.get("decision") != "proceed_to_confirmation"
        or decision.get("confirmation_evaluated") is not False
        or decision.get("candidate_temporal_stride") != 2
    ):
        raise ValueError("candidate decision is not the frozen stride-2 winner")
    if (
        how2sign_selection.get("schema")
        != "umi-studio-how2sign-validation-selection/1"
        or how2sign_partition.get("schema")
        != "umi-studio-how2sign-validation-partition/1"
        or how2sign_partition.get("selection_sha256")
        != hashes["how2sign_selection"]
        or how2sign_partition.get("confirmation_predictions_inspected") is not False
    ):
        raise ValueError("How2Sign confirmation partition is not sealed")
    h_names = how2sign_partition.get("confirmation_sentence_names")
    h_rows = {row["sentence_name"]: row for row in how2sign_selection["rows"]}
    h_receipts = {row["sentence_name"]: row for row in how2sign_receipts}
    if (
        len(h_names) != 42
        or len(set(h_names)) != 42
        or not set(h_names) <= h_rows.keys()
        or set(h_receipts) != set(h_rows)
        or any(
            receipt.get("schema") != "umi-studio-how2sign-validation-clip/1"
            or receipt.get("selection_sha256") != hashes["how2sign_selection"]
            or receipt.get("split") != "validation"
            for receipt in how2sign_receipts
        )
    ):
        raise ValueError("How2Sign confirmation sources differ from the frozen partition")
    if (
        fleurs_selection.get("schema") != "umi-studio-fleurs-dev-split/1"
        or fleurs_partition.get("schema")
        != "umi-studio-fleurs-dev-evaluation-partition/1"
        or fleurs_partition.get("selection_sha256") != hashes["fleurs_selection"]
        or fleurs_receipt.get("schema") != "umi-studio-fleurs-dev-source-files/1"
        or fleurs_receipt.get("selection_sha256") != hashes["fleurs_selection"]
        or fleurs_receipt.get("split") != "validation"
    ):
        raise ValueError("FLEURS confirmation sources differ from the frozen partition")
    tuning_documents = set(fleurs_partition["tuning_documents"])
    f_rows = [
        row for row in fleurs_selection["validation"] if row["document_id"] not in tuning_documents
    ]
    f_sources = {row["path"]: row for row in fleurs_receipt["files"]}
    if (
        len(f_rows) != 46
        or len({row["example_id"] for row in f_rows}) != 46
        or len({row["document_id"] for row in f_rows}) != 23
        or any(row["source_video"] not in f_sources for row in f_rows)
    ):
        raise ValueError("FLEURS confirmation scope differs from the frozen partition")
    records = [
        {
            "domain": "how2sign",
            "example_id": name,
            "signer": h_rows[name]["signer"],
            "source_path": f"data/how2sign/validation-clips/{name}.mp4",
            "source_sha256": h_receipts[name]["clip_sha256"],
            "start_seconds": 0.0,
            "end_seconds": None,
        }
        for name in h_names
    ]
    records.extend(
        {
            "domain": "fleurs",
            "example_id": row["example_id"],
            "signer": row["signer"],
            "document_id": row["document_id"],
            "source_path": f"data/fleurs-asl/source/{Path(row['source_video']).name}",
            "source_sha256": f_sources[row["source_video"]]["sha256"],
            "start_seconds": row["start_time_sec"],
            "end_seconds": row["end_time_sec"],
        }
        for row in f_rows
    )
    return {
        "schema": "umi-studio-temporal-stride-confirmation-plan/1",
        "candidate_decision_sha256": hashes["decision"],
        "candidate_adapter_sha256": decision["candidate_adapter_sha256"],
        "candidate_temporal_stride": 2,
        "checkpoint_sha256": decision["how2sign"]["checkpoint_sha256"],
        "decode": {"num_beams": 5, "max_length": 2048, "early_stopping": True},
        "source_hashes": {
            key: value for key, value in hashes.items() if key != "decision"
        },
        "confirmation_examples": {"how2sign": 42, "fleurs": 46},
        "confirmation_gate": {
            "how2sign_max_mean_wer_change": -0.025,
            "how2sign_min_improved_signers": 3,
            "fleurs_max_mean_wer_change": 0.0,
            "equal_domain_max_mean_wer_change": -0.025,
        },
        "selection_policy": (
            "Evaluate the fixed baseline and the single frozen stride-2 candidate once. "
            "Report the complete result whether or not the confirmation gate passes; do not "
            "select another checkpoint, stride or decoder setting from confirmation."
        ),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--how2sign-selection", type=Path, required=True)
    parser.add_argument("--how2sign-partition", type=Path, required=True)
    parser.add_argument("--how2sign-receipts", type=Path, required=True)
    parser.add_argument("--fleurs-selection", type=Path, required=True)
    parser.add_argument("--fleurs-partition", type=Path, required=True)
    parser.add_argument("--fleurs-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        "decision": args.decision,
        "how2sign_selection": args.how2sign_selection,
        "how2sign_partition": args.how2sign_partition,
        "how2sign_receipts": args.how2sign_receipts,
        "fleurs_selection": args.fleurs_selection,
        "fleurs_partition": args.fleurs_partition,
        "fleurs_receipt": args.fleurs_receipt,
    }
    if (
        args.output.exists()
        or not args.output.resolve().is_relative_to((ROOT / "runs").resolve())
        or not args.adapter.is_file()
        or not all(path.is_file() for path in inputs.values())
    ):
        parser.error("confirmation plan inputs or output are invalid")
    decision = json.loads(args.decision.read_text())
    if sha256(args.adapter) != decision.get("candidate_adapter_sha256"):
        raise ValueError("candidate adapter differs from the frozen decision")
    plan = build_plan(
        decision=decision,
        how2sign_selection=json.loads(args.how2sign_selection.read_text()),
        how2sign_partition=json.loads(args.how2sign_partition.read_text()),
        how2sign_receipts=[
            json.loads(line) for line in args.how2sign_receipts.read_text().splitlines()
        ],
        fleurs_selection=json.loads(args.fleurs_selection.read_text()),
        fleurs_partition=json.loads(args.fleurs_partition.read_text()),
        fleurs_receipt=json.loads(args.fleurs_receipt.read_text()),
        hashes={key: sha256(path) for key, path in inputs.items()},
    )
    for row in plan["records"]:
        source = ROOT / row["source_path"]
        if not source.is_file() or sha256(source) != row["source_sha256"]:
            raise ValueError(f"confirmation source is missing or altered: {source}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(plan, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(plan["confirmation_examples"], sort_keys=True))


if __name__ == "__main__":
    main()
