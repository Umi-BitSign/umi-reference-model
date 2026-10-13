"""Freeze a paired decision for a C5 source-resolution tuning candidate."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from benchmark_fleurs_dev import sha256
from evaluate_c5_resolution_tuning import SCHEMA, summarize


def load_results(path: Path) -> dict[tuple[str, str], dict]:
    records = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        identity = (record.get("source"), record.get("example_id"))
        if record.get("schema") != SCHEMA or identity in records:
            raise ValueError(f"invalid C5 resolution result: {identity}")
        records[identity] = record
    if len(records) != 23:
        raise ValueError("C5 resolution results do not cover 23 tuning examples")
    return records


def paired_summary(
    control: dict[tuple[str, str], dict], candidate: dict[tuple[str, str], dict]
) -> dict:
    if set(control) != set(candidate):
        raise ValueError("resolution result identities differ")
    immutable = (
        "tuning_inventory_sha256",
        "bundle_identity_sha256",
        "bundle_runtime_sha256",
        "stride_runtime_sha256",
        "model_sha256",
        "landmark_receipts_sha256",
        "landmarks_sha256",
        "temporal_stride",
    )
    for identity in control:
        if any(control[identity].get(key) != candidate[identity].get(key) for key in immutable):
            raise ValueError(f"paired resolution provenance differs: {identity}")
    if (
        {row.get("source_resize_policy") for row in control.values()} != {"native"}
        or {row.get("source_short_side") for row in control.values()} != {None}
        or {row.get("visual_crop_size") for row in control.values()} != {224}
        or len({row.get("source_short_side") for row in candidate.values()}) != 1
        or len({row.get("visual_crop_size") for row in candidate.values()}) != 1
        or (
            next(iter(candidate.values())).get("source_short_side") is None
            and next(iter(candidate.values())).get("visual_crop_size") == 224
        )
    ):
        raise ValueError("control or candidate resolution policy differs")

    by_source = {}
    for source in ("how2sign", "fleurs"):
        identities = [identity for identity in control if identity[0] == source]
        deltas = [
            candidate[identity]["single_reference_wer"]
            - control[identity]["single_reference_wer"]
            for identity in identities
        ]
        by_source[source] = {
            "examples": len(identities),
            "mean_wer_delta": sum(deltas) / len(deltas),
            "improved": sum(delta < 0 for delta in deltas),
            "tied": sum(delta == 0 for delta in deltas),
            "regressed": sum(delta > 0 for delta in deltas),
        }
    signer_deltas = defaultdict(list)
    for identity, control_row in control.items():
        if identity[0] == "how2sign":
            signer_deltas[str(control_row["signer"])].append(
                candidate[identity]["single_reference_wer"]
                - control_row["single_reference_wer"]
            )
    by_signer = {
        signer: {
            "examples": len(deltas),
            "mean_wer_delta": sum(deltas) / len(deltas),
        }
        for signer, deltas in sorted(signer_deltas.items(), key=lambda item: int(item[0]))
    }
    control_active = sum(row["active_seconds"] for row in control.values())
    candidate_active = sum(row["active_seconds"] for row in candidate.values())
    speed_ratio = candidate_active / control_active
    quality_pass = all(
        summary["mean_wer_delta"] <= 0.01
        and summary["regressed"] <= summary["improved"]
        for summary in by_source.values()
    )
    speed_pass = speed_ratio <= 0.95
    return {
        "control_summaries": summarize(list(control.values())),
        "candidate_summaries": summarize(list(candidate.values())),
        "paired_by_source": by_source,
        "paired_how2sign_by_signer": by_signer,
        "control_active_seconds": control_active,
        "candidate_active_seconds": candidate_active,
        "candidate_to_control_active_time_ratio": speed_ratio,
        "quality_gate_passed": quality_pass,
        "speed_gate_passed": speed_pass,
        "promotion_eligible": quality_pass and speed_pass,
        "decision": (
            "retain_resolution_candidate"
            if quality_pass and speed_pass
            else "reject_resolution_candidate"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("decision output already exists")
    control = load_results(args.control)
    candidate = load_results(args.candidate)
    comparison = paired_summary(control, candidate)
    result = {
        "schema": "umi-studio-c5-resolution-tuning-decision/1",
        "split": "tuning",
        "confirmation_evaluated": False,
        "control_results_sha256": sha256(args.control),
        "candidate_results_sha256": sha256(args.candidate),
        "candidate_source_short_side": next(iter(candidate.values()))[
            "source_short_side"
        ],
        "candidate_visual_crop_size": next(iter(candidate.values()))[
            "visual_crop_size"
        ],
        **comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
