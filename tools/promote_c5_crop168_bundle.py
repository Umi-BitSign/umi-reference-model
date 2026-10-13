"""Promote the calibrated C5 crop bundle after frozen OpenASL confirmation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from benchmark_fleurs_dev import sha256
from build_c5_stride2_bundle import ignored, inventory

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_BUNDLE = ROOT / "artifacts/c5-corrective-stride2-crop168-calibrated-v1"


def validate_confirmation(decision: dict) -> None:
    if (
        decision.get("schema")
        != "umi-studio-c5-crop-openasl-confirmation-decision/1"
        or decision.get("decision") != "promote_crop_candidate"
        or decision.get("quality_gate_passed") is not True
        or decision.get("fresh_external_confirmation") is not True
        or decision.get("source_videos") != 60
        or decision.get("miner_installed") is not False
        or decision.get("mean_wer_delta", 1) > -0.005
        or decision.get("word_weighted_wer_delta", 1) > 0
        or decision.get("paired", {}).get("improved", 0)
        < decision.get("paired", {}).get("regressed", 61)
    ):
        raise ValueError("crop candidate lacks passing fresh confirmation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.decision.is_file()
        or not RESEARCH_BUNDLE.is_dir()
        or args.output.exists()
        or not args.output.resolve().is_relative_to((ROOT / "artifacts").resolve())
    ):
        parser.error("promotion inputs or output are invalid")
    decision = json.loads(args.decision.read_text())
    validate_confirmation(decision)
    research_identity = json.loads(
        (RESEARCH_BUNDLE / "MODEL_IDENTITY.json").read_text()
    )
    research_evidence = json.loads(
        (RESEARCH_BUNDLE / "CROP168_CANDIDATE_EVIDENCE.json").read_text()
    )
    if (
        decision["calibrator_sha256"] != research_identity["calibrator_sha256"]
        or decision["calibrator_sha256"] != research_evidence["calibrator_sha256"]
        or decision["native_results_sha256"]
        != sha256(args.decision.with_name("native224.jsonl"))
        or decision["calibrated_results_sha256"]
        != sha256(args.decision.with_name("calibrated168.jsonl"))
    ):
        raise ValueError("confirmation is not bound to the research bundle or results")
    temporary = args.output.with_name(f".{args.output.name}.partial")
    if temporary.exists():
        raise ValueError("unfinished promoted bundle must be inspected")
    try:
        shutil.copytree(RESEARCH_BUNDLE, temporary, ignore=ignored)
        for stale in ("FILE_LIST.json", "SHA256SUMS"):
            (temporary / stale).unlink(missing_ok=True)
        identity = {
            **research_identity,
            "name": "UMI C5 Corrective Stride-2 Crop-168 Confirmed v1",
            "model_id": "umi-c5-corrective-stride2-crop168-confirmed-v1",
            "research_bundle_identity_sha256": sha256(
                RESEARCH_BUNDLE / "MODEL_IDENTITY.json"
            ),
            "qualification": (
                "Passed open tuning, a 134-clip internal holdout, and one-shot "
                "cross-domain confirmation on 60 distinct OpenASL source videos"
            ),
        }
        (temporary / "MODEL_IDENTITY.json").write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n"
        )
        research_evidence.update(
            {
                "fresh_external_confirmation_required": False,
                "fresh_external_confirmation_passed": True,
                "openasl_confirmation_decision_sha256": sha256(args.decision),
                "miner_installed": False,
            }
        )
        (temporary / "CROP168_CANDIDATE_EVIDENCE.json").write_text(
            json.dumps(research_evidence, indent=2, sort_keys=True) + "\n"
        )
        confirmation = {
            "schema": "umi-c5-crop168-openasl-confirmation-evidence/1",
            "decision_sha256": sha256(args.decision),
            "plan_sha256": decision["confirmation_plan_sha256"],
            "native_results_sha256": decision["native_results_sha256"],
            "calibrated_results_sha256": decision["calibrated_results_sha256"],
            "source_videos": decision["source_videos"],
            "baseline": decision["baseline"],
            "calibrated": decision["calibrated"],
            "mean_wer_delta": decision["mean_wer_delta"],
            "word_weighted_wer_delta": decision["word_weighted_wer_delta"],
            "active_time_ratio": decision["active_time_ratio"],
            "paired": decision["paired"],
            "quality_gate_passed": True,
            "miner_installed": False,
        }
        (temporary / "OPENASL_CONFIRMATION_EVIDENCE.json").write_text(
            json.dumps(confirmation, indent=2, sort_keys=True) + "\n"
        )
        (temporary / "README.md").write_text(
            "# UMI C5 Corrective Stride-2 Crop-168 Confirmed v1\n\n"
            "This standalone bundle combines the confirmed stride-2 C5 checkpoint "
            "with 168-pixel DINO hand and face crops and a training-only affine feature "
            "calibrator. It improved How2Sign and FLEURS open tuning, a 134-clip "
            "internal holdout, and a one-shot cross-domain confirmation containing 60 "
            "distinct OpenASL source videos. On that confirmation, mean WER improved "
            "from 0.6320 to 0.6145 and active time fell 6.7%. It has not been installed "
            "in the miner and is not an accessibility interpreter. No source videos "
            "or annotations are included.\n"
        )
        provenance = (temporary / "PROVENANCE.md").read_text().rstrip()
        (temporary / "PROVENANCE.md").write_text(
            provenance
            + "\n\nFresh confirmation used one frozen clip from each of 60 public OpenASL "
            "training source videos. The selection and quality gates were fixed before "
            "either native-224 or calibrated-168 inference was run. OpenASL data and "
            "transformed artifacts are not included in this bundle.\n"
        )
        derivation = {
            "schema": "umi-c5-crop168-confirmed-bundle-derivation/1",
            "research_bundle_identity_sha256": sha256(
                RESEARCH_BUNDLE / "MODEL_IDENTITY.json"
            ),
            "research_bundle_sums_sha256": sha256(RESEARCH_BUNDLE / "SHA256SUMS"),
            "confirmation_decision_sha256": sha256(args.decision),
            "promoter_sha256": sha256(Path(__file__)),
        }
        (temporary / "CROP168_CONFIRMATION_DERIVATION.json").write_text(
            json.dumps(derivation, indent=2, sort_keys=True) + "\n"
        )
        files = inventory(temporary)
        (temporary / "FILE_LIST.json").write_text(
            json.dumps(
                {
                    "schema": "umi-model-bundle-file-list/2",
                    "model_id": identity["model_id"],
                    "files": files,
                    "inventory_files": (
                        "FILE_LIST.json and SHA256SUMS are integrity metadata."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        sums = [
            f"{sha256(temporary / 'FILE_LIST.json')}  FILE_LIST.json",
            *(f"{row['sha256']}  {row['path']}" for row in files),
        ]
        (temporary / "SHA256SUMS").write_text("\n".join(sums) + "\n")
        temporary.rename(args.output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    subprocess.run(
        ["shasum", "-a", "256", "-c", "SHA256SUMS"],
        cwd=args.output,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "files": len(files),
                "bytes": sum(row["bytes"] for row in files),
                "identity_sha256": sha256(args.output / "MODEL_IDENTITY.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
