"""Remove only enumerated rejected experiment weights under standing owner permission."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from benchmark_fleurs_dev import sha256
from probe_jeff_terra_quality import write

ROOT = Path(__file__).resolve().parents[1]


def targets():
    result = []
    for name in (
        "fsboard-byt5-ten-shard-v1",
        "fsboard-byt5-shard00-v1",
        "fsboard-byt5-compact-shard00-v1",
    ):
        receipt = ROOT / "runs" / name / "result.json"
        row = json.loads(receipt.read_text())
        result.append(
            (
                f"checkpoints/{name}/model.pt",
                row["checkpoint_sha256"],
                str(receipt.relative_to(ROOT)),
            )
        )
    decision = ROOT / "checkpoints/c5-winner-full-v2-early/decision.json"
    rows = json.loads(decision.read_text())
    if rows["complete"] is not True or any(
        row["passes_open_development_gate"] or row["promotion_authorized"]
        for row in rows["evaluations"]
    ):
        raise ValueError("full-winner recipe has an unrejected or incomplete checkpoint")
    early = {row["exposures"]: row for row in rows["evaluations"]}
    for count in (2048, 8192, 16384, 32768):
        row = early[count]
        result.append(
            (row["checkpoint"], row["checkpoint_sha256"], str(decision.relative_to(ROOT)))
        )
    decoder = ROOT / "runs/full-decoder-v1/decision.json"
    rows = json.loads(decoder.read_text())
    if rows["decision"] != "retain_frozen_baseline" or any(
        row["decision"] != "retain_frozen_baseline" for row in rows["checkpoints"]
    ):
        raise ValueError("full-decoder recipe has a retained checkpoint")
    for row in rows["checkpoints"]:
        result.append(
            (
                f"checkpoints/full-decoder-v1/full-decoder-after-{row['example_step']:06d}.safetensors",
                row["adapter_sha256"],
                str(decoder.relative_to(ROOT)),
            )
        )
    # Preserve the failed final endpoint unless the earlier obsolete weights do
    # not restore sufficient real capacity for this run's checkpoint reserve.
    row = early[48800]
    result.append((row["checkpoint"], row["checkpoint_sha256"], str(decision.relative_to(ROOT))))
    return result


def run(receipt):
    if receipt.exists() or not receipt.resolve().is_relative_to(ROOT / "runs"):
        raise ValueError("cleanup needs a new private runs/ receipt")
    dependencies = set()
    for plan in (
        "runs/tiny-rgb-feature-winner-full-v1/plan.json",
        "runs/tiny-rgb-visual-teacher-corpus-v1/plan.json",
        "runs/terra-visual-training-v1/plan.json",
    ):
        row = json.loads((ROOT / plan).read_text())
        dependencies.update(row.get("dependency_sha256", {}))
    verified = []
    for name, digest, proof in targets():
        path = ROOT / name
        if (
            name in dependencies
            or path.is_symlink()
            or path.resolve() != path
            or not path.is_file()
            or sha256(path) != digest
        ):
            raise ValueError(f"obsolete weight scope or checksum differs: {name}")
        verified.append(
            {
                "path": name,
                "sha256": digest,
                "logical_bytes": path.stat().st_size,
                "rejection_or_provenance": proof,
            }
        )
    before = shutil.disk_usage(ROOT).free
    receipt.parent.mkdir(parents=True, exist_ok=True)
    write(
        receipt,
        {
            "standing_owner_permission": (
                "old model files created by this work may be deleted when needed"
            ),
            "free_bytes_before": before,
            "target_free_bytes": 82 * 1024**3,
            "verified_candidates": verified,
            "local_recovery_copy_promised": False,
            "baseline_selected_data_logs_and_results_preserved": True,
        },
    )
    removed = []
    for row in verified:
        if shutil.disk_usage(ROOT).free >= 82 * 1024**3:
            break
        (ROOT / row["path"]).unlink()
        removed.append(row)
        write(
            receipt.with_name(f"removed-{len(removed):02d}.json"),
            {**row, "free_bytes_after": shutil.disk_usage(ROOT).free},
        )
    after = shutil.disk_usage(ROOT).free
    result = {
        "removed": removed,
        "logical_bytes_removed": sum(row["logical_bytes"] for row in removed),
        "free_bytes_before": before,
        "free_bytes_after": after,
        "physical_floor_restored": after >= 80 * 1024**3 + 128 * 1024**2 + 215000000,
        "local_recovery_copy_promised": False,
    }
    write(receipt.with_name("completion.json"), result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    run(args.receipt.resolve())
