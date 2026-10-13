"""Archive local training tools, checking the implementations pinned by a plan."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_c5_stem_continuation_plan import ROOT

SOURCE_KEYS = {
    "trainer_source_sha256": "train_c5_frozen_replay.py",
    "adapter_implementation_sha256": "shubert_lora.py",
    "optimizer_groups_source_sha256": "c5_scale_learning_rates.py",
    "all_source_implementation_sha256": "c5_all_source_training.py",
    "retention_implementation_sha256": "c5_retention.py",
    "coverage_implementation_sha256": "c5_coverage_schedule.py",
}


def archive_sources(plan_path: Path, output_dir: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    if output_dir.exists() or not output_dir.resolve().is_relative_to(ROOT / "runs"):
        raise ValueError("source archive needs a new runs/ directory")
    for key, filename in SOURCE_KEYS.items():
        if key in plan and sha256(ROOT / "tools" / filename) != plan[key]:
            raise ValueError(f"cannot archive a changed pinned implementation: {filename}")
    shutil.copytree(ROOT / "tools", output_dir / "tools", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    manifest = {
        "schema": "umi-studio-c5-experiment-sources/1",
        "plan_sha256": sha256(plan_path),
        "source_sha256": {
            str(path.relative_to(output_dir)): sha256(path)
            for path in sorted((output_dir / "tools").rglob("*")) if path.is_file()
        },
        "reproduction": "Use these tools in an isolated copy of the original workspace; their ROOT paths expect tools/ under the workspace root. Frozen data, baseline runtime and checkpoint bytes remain separate plan dependencies.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = archive_sources(args.plan, args.output_dir)
    print(json.dumps({"plan_sha256": report["plan_sha256"], "archived_files": len(report["source_sha256"])}))


if __name__ == "__main__":
    main()
