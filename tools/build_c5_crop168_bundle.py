"""Build a standalone research bundle for calibrated 168px C5 inference."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from benchmark_fleurs_dev import sha256
from build_c5_stride2_bundle import ignored, inventory

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "tools/c5_crop168_umi_inference.py"
CALIBRATOR_RUNTIME = ROOT / "tools/c5_crop_calibrator.py"
RESOLUTION_RUNTIME = ROOT / "tools/resolution_policy.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-bundle", type=Path, required=True)
    parser.add_argument("--calibrator", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--tuning-decision", type=Path, required=True)
    parser.add_argument("--holdout-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent = args.parent_bundle.resolve()
    if (
        parent != ROOT / "artifacts/c5-corrective-stride2-confirmed"
        or not all(
            path.is_file()
            for path in (
                args.calibrator,
                args.fit_report,
                args.tuning_decision,
                args.holdout_decision,
                ENTRYPOINT,
                CALIBRATOR_RUNTIME,
                RESOLUTION_RUNTIME,
            )
        )
        or args.output.exists()
        or not args.output.resolve().is_relative_to(ROOT / "artifacts")
    ):
        parser.error("calibrated bundle inputs or output are invalid")
    fit = json.loads(args.fit_report.read_text())
    tuning = json.loads(args.tuning_decision.read_text())
    holdout = json.loads(args.holdout_decision.read_text())
    calibrator_hash = sha256(args.calibrator)
    if (
        fit.get("calibrator_sha256") != calibrator_hash
        or fit.get("per_stream_frame_cap") != 32
        or tuning.get("quality_gate_passed") is not True
        or holdout.get("quality_gate_passed") is not True
        or holdout.get("promotion_eligible") is not False
    ):
        raise ValueError("calibrated candidate evidence differs")
    temporary = args.output.with_name(f".{args.output.name}.partial")
    if temporary.exists():
        raise ValueError("unfinished calibrated bundle must be inspected")
    try:
        shutil.copytree(parent, temporary, ignore=ignored)
        for stale in ("FILE_LIST.json", "SHA256SUMS"):
            (temporary / stale).unlink(missing_ok=True)
        shutil.copy2(args.calibrator, temporary / "crop_calibrator.safetensors")
        shutil.copy2(CALIBRATOR_RUNTIME, temporary / "c5_crop_calibrator.py")
        shutil.copy2(RESOLUTION_RUNTIME, temporary / "resolution_policy.py")
        shutil.copy2(ENTRYPOINT, temporary / "umi_inference.py")
        parent_identity_hash = sha256(parent / "MODEL_IDENTITY.json")
        parent_identity = json.loads((parent / "MODEL_IDENTITY.json").read_text())
        identity = {
            "name": "UMI C5 Corrective Stride-2 Crop-168 Calibrated v1",
            "model_id": "umi-c5-corrective-stride2-crop168-calibrated-v1",
            "format": "safetensors",
            "parent_model": parent_identity["name"],
            "parent_identity_sha256": parent_identity_hash,
            "checkpoint_sha256": parent_identity["checkpoint_sha256"],
            "calibrator_sha256": calibrator_hash,
            "inference_sha256": sha256(temporary / "umi_inference.py"),
            "generation": {"num_beams": 5, "max_length": 2048},
            "temporal_sampling": parent_identity["temporal_sampling"],
            "visual_crop": {
                "pixels": 168,
                "dino_patch_size": 14,
                "patch_tokens_per_stream_frame": 144,
                "calibration_target_pixels": 224,
            },
            "qualification": "Passed open tuning and internal training-split holdout; fresh external confirmation required",
        }
        (temporary / "MODEL_IDENTITY.json").write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n"
        )
        evidence = {
            "schema": "umi-c5-crop168-calibrated-evidence/1",
            "fit_report_sha256": sha256(args.fit_report),
            "tuning_decision_sha256": sha256(args.tuning_decision),
            "holdout_decision_sha256": sha256(args.holdout_decision),
            "calibrator_sha256": calibrator_hash,
            "open_tuning": {
                "how2sign_baseline_mean_wer": tuning["control_summaries"]["how2sign"]["mean_single_reference_wer"],
                "how2sign_candidate_mean_wer": tuning["candidate_summaries"]["how2sign"]["mean_single_reference_wer"],
                "fleurs_baseline_mean_wer": tuning["control_summaries"]["fleurs"]["mean_single_reference_wer"],
                "fleurs_candidate_mean_wer": tuning["candidate_summaries"]["fleurs"]["mean_single_reference_wer"],
            },
            "internal_holdout": {
                "examples": 134,
                "baseline_mean_wer": holdout["baseline_mean_wer"],
                "candidate_mean_wer": holdout["candidate_mean_wer"],
                "paired_improved": holdout["paired_examples_improved"],
                "paired_tied": holdout["paired_examples_tied"],
                "paired_regressed": holdout["paired_examples_regressed"],
            },
            "fresh_external_confirmation_required": True,
            "miner_installed": False,
        }
        (temporary / "CROP168_CANDIDATE_EVIDENCE.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        )
        (temporary / "README.md").write_text(
            "# UMI C5 Corrective Stride-2 Crop-168 Calibrated v1\n\n"
            "This research bundle reduces hand and face DINO crops from 224 to 168 "
            "pixels and applies a small affine feature calibrator before the confirmed "
            "stride-2 C5 model. It improved both open tuning domains and a 134-clip "
            "internal holdout. Fresh external confirmation is still required. It has "
            "not been installed in the miner and is not an accessibility interpreter.\n"
        )
        (temporary / "PROVENANCE.md").write_text(
            "# Provenance\n\n"
            f"Parent bundle identity SHA-256: `{parent_identity_hash}`.\n\n"
            f"Crop calibrator SHA-256: `{calibrator_hash}`.\n\n"
            "The affine calibrator was fit on paired 168/224 DINO features from 169 "
            "training-only How2Sign clips, capped per signer and per stream. No source "
            "videos or annotations are included.\n"
        )
        derivation = {
            "schema": "umi-c5-crop168-calibrated-bundle-derivation/1",
            "parent_identity_sha256": parent_identity_hash,
            "calibrator_sha256": calibrator_hash,
            "calibrator_runtime_sha256": sha256(CALIBRATOR_RUNTIME),
            "resolution_runtime_sha256": sha256(RESOLUTION_RUNTIME),
            "entrypoint_source_sha256": sha256(ENTRYPOINT),
            "builder_sha256": sha256(Path(__file__)),
        }
        (temporary / "CROP168_BUNDLE_DERIVATION.json").write_text(
            json.dumps(derivation, indent=2, sort_keys=True) + "\n"
        )
        files = inventory(temporary)
        (temporary / "FILE_LIST.json").write_text(
            json.dumps(
                {
                    "schema": "umi-model-bundle-file-list/2",
                    "model_id": identity["model_id"],
                    "files": files,
                    "inventory_files": "FILE_LIST.json and SHA256SUMS are integrity metadata.",
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
                "calibrator_sha256": calibrator_hash,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
