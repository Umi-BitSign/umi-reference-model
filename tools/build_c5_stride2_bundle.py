"""Build a standalone offline bundle for the confirmed merged C5 candidate."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from benchmark_fleurs_dev import sha256

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE = ROOT / "tools/c5_stride2_runtime.py"
ENTRYPOINT_SOURCE = ROOT / "tools/c5_stride2_umi_inference.py"


def ignored(_directory: str, names: list[str]) -> set[str]:
    excluded = {".runtime", ".cache", "run", "__pycache__"}
    return {name for name in names if name in excluded or name.endswith(".pyc")}


def role(path: str) -> str:
    if path.endswith((".safetensors", ".task")):
        return "weights"
    if path.startswith("licenses/") or path == "LICENSE":
        return "license"
    if path.endswith(".py") or path.endswith(".sh"):
        return "inference"
    if path.endswith((".json", ".md", ".txt")):
        return "provenance"
    return "dependency"


def inventory(bundle: Path) -> list[dict]:
    rows = []
    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(bundle).as_posix()
        if relative in {"FILE_LIST.json", "SHA256SUMS"}:
            continue
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "role": role(relative),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--merged-checkpoint", type=Path, required=True)
    parser.add_argument("--confirmation-decision", type=Path, required=True)
    parser.add_argument("--merged-equivalence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    merged = args.merged_checkpoint.resolve()
    if (
        baseline != ROOT / "artifacts/baseline-6fe8df59"
        or not (merged / "MERGED_CHECKPOINT.json").is_file()
        or not args.confirmation_decision.is_file()
        or not args.merged_equivalence.is_file()
        or args.output.exists()
        or not args.output.resolve().is_relative_to((ROOT / "artifacts").resolve())
    ):
        parser.error("bundle inputs or output are invalid")
    merge_manifest = json.loads((merged / "MERGED_CHECKPOINT.json").read_text())
    decision = json.loads(args.confirmation_decision.read_text())
    equivalence = json.loads(args.merged_equivalence.read_text())
    model_hash = sha256(merged / "model.safetensors")
    if (
        merge_manifest.get("merged_checkpoint_sha256") != model_hash
        or decision.get("decision") != "package_candidate"
        or equivalence.get("all_hypotheses_equal") is not True
        or equivalence.get("merged_checkpoint_sha256") != model_hash
        or equivalence.get("examples") != 88
    ):
        raise ValueError("merged checkpoint lacks confirmed equivalence")
    temporary = args.output.with_name(f".{args.output.name}.partial")
    if temporary.exists():
        raise ValueError("unfinished candidate bundle must be inspected")
    try:
        shutil.copytree(baseline, temporary, ignore=ignored)
        for stale in (
            "FILE_LIST.json",
            "SHA256SUMS",
            "LOCAL_RUNTIME_DERIVATION.json",
        ):
            (temporary / stale).unlink(missing_ok=True)
        checkpoint_dir = temporary / "models/checkpoint-11625"
        shutil.copy2(merged / "model.safetensors", checkpoint_dir / "model.safetensors")
        shutil.copy2(merged / "config.json", checkpoint_dir / "config.json")
        shutil.copy2(RUNTIME_SOURCE, temporary / "c5_stride2_runtime.py")
        shutil.copy2(ENTRYPOINT_SOURCE, temporary / "umi_inference.py")
        runtime_path = temporary / "runtime.py"
        runtime_text = runtime_path.read_text()
        old_hash = merge_manifest["baseline_checkpoint_sha256"]
        if runtime_text.count(old_hash) != 1:
            raise ValueError("baseline runtime model identity is unexpected")
        runtime_path.write_text(runtime_text.replace(old_hash, model_hash))
        identity = {
            "name": "UMI C5 Corrective Stride-2 v1",
            "model_id": "umi-c5-corrective-stride2-v1",
            "format": "safetensors",
            "parent_model": "UMI Community Baseline v0.2",
            "parent_checkpoint_sha256": old_hash,
            "checkpoint_sha256": model_hash,
            "inference_sha256": sha256(temporary / "umi_inference.py"),
            "generation": {"num_beams": 5, "max_length": 2048},
            "temporal_sampling": {
                "stride": 2,
                "phase": 0,
                "placement": "after_mediapipe_before_crops_dino_pose_and_c5",
            },
            "qualification": "Passed one-shot source/document-disjoint confirmation",
        }
        (temporary / "MODEL_IDENTITY.json").write_text(
            json.dumps(identity, sort_keys=True, indent=2) + "\n"
        )
        evidence = {
            "schema": "umi-c5-corrective-stride2-evidence/1",
            "confirmation_decision_sha256": sha256(args.confirmation_decision),
            "merged_equivalence_sha256": sha256(args.merged_equivalence),
            "checkpoint_sha256": model_hash,
            "temporal_stride": 2,
            "confirmation_examples": {"how2sign": 42, "fleurs": 46},
            "how2sign_mean_wer": {
                "baseline": decision["how2sign"]["baseline_mean_wer"],
                "candidate": decision["how2sign"]["candidate_mean_wer"],
            },
            "fleurs_mean_wer": {
                "baseline": decision["fleurs"]["baseline_mean_wer"],
                "candidate": decision["fleurs"]["candidate_mean_wer"],
            },
            "all_merged_hypotheses_equal": True,
            "miner_installed": False,
        }
        (temporary / "CANDIDATE_EVIDENCE.json").write_text(
            json.dumps(evidence, sort_keys=True, indent=2) + "\n"
        )
        (temporary / "README.md").write_text(
            "# UMI C5 Corrective Stride-2 v1\n\n"
            "This standalone offline bundle contains the confirmed merged C5 checkpoint. "
            "It runs MediaPipe on every decoded frame, then retains frames 0, 2, 4, ... "
            "for crops, DINO, pose features and translation. Decoding remains beam 5 with "
            "a 2,048-token ceiling.\n\n"
            "The model improved both frozen confirmation domains, but absolute WER remains "
            "high. It is a research/mining candidate, not an accessibility interpreter. "
            "No source videos or annotations are included.\n"
        )
        (temporary / "PROVENANCE.md").write_text(
            "# Provenance\n\n"
            f"Parent checkpoint SHA-256: `{old_hash}`.\n\n"
            f"Confirmed adapter SHA-256: `{decision['candidate_adapter_sha256']}`.\n\n"
            f"Merged checkpoint SHA-256: `{model_hash}`.\n\n"
            "Training used public-model-track How2Sign and FLEURS-ASL training material. "
            "The stride-2 configuration was selected on frozen open tuning data and then "
            "evaluated once on source-disjoint How2Sign and document-disjoint FLEURS-ASL "
            "confirmation. See `CANDIDATE_EVIDENCE.json` for aggregate results.\n"
        )
        derivation = {
            "schema": "umi-c5-corrective-stride2-bundle-derivation/1",
            "parent_identity_sha256": sha256(baseline / "MODEL_IDENTITY.json"),
            "merged_manifest_sha256": sha256(merged / "MERGED_CHECKPOINT.json"),
            "runtime_source_sha256": sha256(RUNTIME_SOURCE),
            "entrypoint_source_sha256": sha256(ENTRYPOINT_SOURCE),
            "builder_sha256": sha256(Path(__file__)),
        }
        (temporary / "BUNDLE_DERIVATION.json").write_text(
            json.dumps(derivation, sort_keys=True, indent=2) + "\n"
        )
        files = inventory(temporary)
        file_list = {
            "schema": "umi-model-bundle-file-list/2",
            "model_id": identity["model_id"],
            "files": files,
            "inventory_files": "FILE_LIST.json and SHA256SUMS are integrity metadata.",
        }
        (temporary / "FILE_LIST.json").write_text(
            json.dumps(file_list, sort_keys=True, indent=2) + "\n"
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
        [
            str(baseline / ".runtime/bin/python"),
            "-c",
            (
                "import sys; from pathlib import Path; "
                f"sys.path[:0]=[{str(args.output)!r}, {str(args.output / 'vendor')!r}, "
                f"{str(args.output / 'vendor/demo')!r}]; "
                "from runtime import SHuBERTInferenceRuntime; "
                f"m=SHuBERTInferenceRuntime(Path({str(args.output)!r}), device='cpu', "
                "verify_assets=True); print(m.component_devices)"
            ),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "checkpoint_sha256": model_hash,
                "files": len(files),
                "bytes": sum(row["bytes"] for row in files),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
