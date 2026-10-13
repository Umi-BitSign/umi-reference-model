"""Freeze a full-corpus C5 update without touching any evaluation outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmark_fleurs_dev import sha256
from c5_all_source_training import RECIPE, load_corpus
from freeze_c5_stem_continuation_plan import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to(ROOT / "runs"):
        parser.error("full-corpus plan needs a new runs/ output")
    control_path = ROOT / "runs/c5-openasl-scale-fixed-v1/plan.json"
    control = json.loads(control_path.read_text())
    audit_path = ROOT / "runs/c5-all-source-corpus-v1/audit.json"
    manifest = ROOT / "runs/c5-all-source-corpus-v1/training-streams.jsonl"
    plan = {
        "schema": "umi-studio-c5-frozen-replay-plan/1", "seed": 401, "recipe": RECIPE,
        "objective": "Test the complete eligible corpus with balanced dataset/signer losses and warmup/cosine decay.",
        "corpus_audit": str(audit_path.relative_to(ROOT)), "corpus_audit_sha256": sha256(audit_path),
        "training_manifest": str(manifest.relative_to(ROOT)), "training_manifest_sha256": sha256(manifest),
        "all_source_implementation_sha256": sha256(ROOT / "tools/c5_all_source_training.py"),
        "adapter_implementation_sha256": sha256(ROOT / "tools/shubert_lora.py"),
        "trainer_source_sha256": sha256(ROOT / "tools/train_c5_frozen_replay.py"),
        "optimizer_groups_source_sha256": sha256(ROOT / "tools/c5_scale_learning_rates.py"),
        "starting_adapter": control["starting_adapter"],
        "starting_adapter_sha256": control["starting_adapter_sha256"],
        "baseline_checkpoint_sha256": control["baseline_checkpoint_sha256"],
        "scale_partition": control["scale_partition"],
        "scale_partition_sha256": control["scale_partition_sha256"],
        "open_tuning": control["open_tuning"], "screen": control["screen"],
        "confirmation_policy": control["confirmation_policy"],
        "miner_health_policy": "user_authorized_maintenance",
        "diagnostic_checkpoint_used_as_initializer": False,
    }
    rows, _, hashes = load_corpus(plan)
    rows.sort(key=lambda row: hashlib.sha256(f"401\0{row['source']}\0{row['id']}".encode()).hexdigest())
    args.output_dir.mkdir(parents=True)
    schedule_path = args.output_dir / "schedule.jsonl"
    with schedule_path.open("x") as output:
        for position, row in enumerate(rows):
            output.write(json.dumps({
                "position": position, "source": row["source"], "example_id": row["id"],
                "signer": row["signer"], "stored_temporal_stride": row.get("stored_temporal_stride", 1),
            }, sort_keys=True) + "\n")
    plan["schedule_sha256"] = sha256(schedule_path)
    plan["source_hashes"] = hashes
    plan_path = args.output_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"plan_sha256": sha256(plan_path), "examples": len(rows), "recipe": RECIPE}))


if __name__ == "__main__":
    main()
