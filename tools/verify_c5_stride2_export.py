"""Verify that the merged checkpoint reproduces every confirmed candidate decode."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from benchmark_fleurs_dev import sha256

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/baseline-6fe8df59"
RUNTIME = BASELINE / ".runtime/bin/python"
CACHE = ROOT / "data/temporal-stride-confirmation/stream-cache-25fps"
RUN_DIR = ROOT / "runs/temporal-stride-v1"
DOMAINS = {
    "how2sign": {
        "index": RUN_DIR / "confirmation-how2sign-stream-index.jsonl",
        "split": "validation",
        "examples": 42,
    },
    "fleurs": {
        "index": RUN_DIR / "confirmation-fleurs-stream-index.jsonl",
        "split": "confirmation",
        "examples": 46,
    },
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def compare_results(expected: list[dict], actual: list[dict]) -> None:
    if len(expected) != len(actual):
        raise ValueError("merged export result count differs")
    for first, second in zip(expected, actual, strict=True):
        if (
            first["example_id"] != second["example_id"]
            or first["reference"] != second["reference"]
            or first["hypothesis"] != second["hypothesis"]
            or first["single_reference_wer"] != second["single_reference_wer"]
            or first["single_reference_word_edits"]
            != second["single_reference_word_edits"]
            or first["source_frame_count"] != second["source_frame_count"]
            or first["evaluated_frame_count"] != second["evaluated_frame_count"]
            or first.get("temporal_stride") != 2
            or second.get("temporal_stride") != 2
        ):
            raise ValueError(f"merged export decode differs: {first['example_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.checkpoint / "MERGED_CHECKPOINT.json"
    if (
        not manifest_path.is_file()
        or args.output.exists()
        or not args.output.resolve().is_relative_to((ROOT / "runs").resolve())
    ):
        parser.error("merged checkpoint or output is invalid")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "umi-studio-c5-stride2-merged-checkpoint/1"
        or manifest.get("temporal_stride") != 2
        or sha256(args.checkpoint / "model.safetensors")
        != manifest.get("merged_checkpoint_sha256")
    ):
        raise ValueError("merged checkpoint manifest differs")
    result_hashes = {}
    for domain, config in DOMAINS.items():
        actual_path = RUN_DIR / f"merged-equivalence-{domain}.jsonl"
        subprocess.run(
            [
                "nice",
                "-n",
                "20",
                str(RUNTIME),
                "tools/evaluate_fleurs_cache.py",
                "--index",
                str(config["index"]),
                "--cache-dir",
                str(CACHE),
                "--split",
                str(config["split"]),
                "--baseline",
                str(BASELINE),
                "--checkpoint",
                str(args.checkpoint),
                "--results",
                str(actual_path),
                "--miner-maintenance-mode",
                "--device",
                "cpu",
                "--use-stream-features",
                "--temporal-stride",
                "2",
                "--limit",
                str(config["examples"]),
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        expected_path = RUN_DIR / f"confirmation-{domain}-candidate-stride2.jsonl"
        compare_results(read_jsonl(expected_path), read_jsonl(actual_path))
        result_hashes[domain] = {
            "confirmed_candidate_results_sha256": sha256(expected_path),
            "merged_checkpoint_results_sha256": sha256(actual_path),
            "examples": config["examples"],
        }
    result = {
        "schema": "umi-studio-c5-stride2-merged-equivalence/1",
        "merged_checkpoint_sha256": manifest["merged_checkpoint_sha256"],
        "merged_manifest_sha256": sha256(manifest_path),
        "temporal_stride": 2,
        "examples": 88,
        "all_hypotheses_equal": True,
        "domains": result_hashes,
        "miner_changed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(result, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
