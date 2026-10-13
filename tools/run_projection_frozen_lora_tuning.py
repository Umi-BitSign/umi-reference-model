"""Evaluate the projection-frozen LoRA checkpoints on the two tuning sets."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from freeze_projection_frozen_lora_plan import RECIPE, ROOT
from freeze_visual_lora_plan import TUNING_LAYOUT
from run_bounded_visual_extraction import rss_kib, stop_worker
from run_visual_lora_tuning import (
    BASELINE,
    DOMAINS,
    MAX_RSS_GIB,
    MIN_FREE_GIB,
    POLL_SECONDS,
    RUNTIME_PYTHON,
    checked_results,
    evaluation_command,
    result_path,
)


def checked_checkpoints(checkpoint_dir: Path, plan: dict, plan_hash: str) -> dict[int, Path]:
    from safetensors import safe_open

    run = json.loads((checkpoint_dir / "run.json").read_text())
    records = [
        json.loads(line)
        for line in (checkpoint_dir / "checkpoint-index.jsonl").read_text().splitlines()
    ]
    expected_steps = list(range(
        RECIPE["recovery_after_examples"],
        RECIPE["train_examples"] + 1,
        RECIPE["recovery_after_examples"],
    ))
    if (
        run.get("schema") != "umi-studio-visual-lora-projection-frozen-training/1"
        or run.get("plan_sha256") != plan_hash
        or run.get("recipe_id") != RECIPE["id"]
        or run.get("starting_adapter_sha256") != plan["starting_adapter_sha256"]
        or run.get("trainer_source_sha256") != plan["trainer_source_sha256"]
        or [row.get("example_step") for row in records] != expected_steps
    ):
        raise ValueError("projection-frozen training ledger is incomplete or differs")
    selected = {}
    for row in records:
        step = row["example_step"]
        path = checkpoint_dir / f"visual-lora-after-{step:06d}.safetensors"
        optimizer = path.with_suffix(".optimizer.pt")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        if (
            row.get("schema") != "umi-studio-visual-lora-projection-frozen-recovery/1"
            or row.get("plan_sha256") != plan_hash
            or row.get("recipe_id") != RECIPE["id"]
            or row.get("checkpoint_name") != path.name
            or row.get("checkpoint_sha256") != sha256(path)
            or row.get("optimizer_name") != optimizer.name
            or row.get("optimizer_sha256") != sha256(optimizer)
            or row.get("tuning_checkpoint") is not (step in RECIPE["checkpoint_after_examples"])
            or metadata.get("schema") != "umi-studio-shubert-visual-lora/1"
            or metadata.get("plan_sha256") != plan_hash
            or metadata.get("recipe_id") != RECIPE["id"]
            or metadata.get("example_step") != str(step)
            or metadata.get("rank") != str(RECIPE["rank"])
            or metadata.get("alpha") != str(RECIPE["alpha"])
            or metadata.get("visual_last_layers") != str(RECIPE["visual_last_layers"])
            or metadata.get("baseline_checkpoint_sha256") != plan["baseline_checkpoint_sha256"]
            or metadata.get("trainer_source_sha256") != plan["trainer_source_sha256"]
            or metadata.get("adapter_source_sha256") != plan["adapter_source_sha256"]
            or metadata.get("starting_adapter_sha256") != plan["starting_adapter_sha256"]
        ):
            raise ValueError(f"projection-frozen checkpoint differs at {step}")
        if step in RECIPE["checkpoint_after_examples"]:
            selected[step] = path
    return selected


def run_tuning(plan_path: Path, checkpoint_dir: Path, output_dir: Path, health_url: str) -> None:
    plan = json.loads(plan_path.read_text())
    if (
        plan.get("schema") != "umi-studio-visual-lora-projection-frozen-plan/1"
        or plan.get("recipe") != RECIPE
        or sha256(ROOT / plan["parent_plan"]) != plan["parent_plan_sha256"]
        or sha256(ROOT / plan["starting_adapter"]) != plan["starting_adapter_sha256"]
        or sha256(ROOT / "tools/train_projection_frozen_lora.py") != plan["trainer_source_sha256"]
        or sha256(BASELINE / "models/checkpoint-11625/model.safetensors")
        != plan["baseline_checkpoint_sha256"]
    ):
        raise ValueError("projection-frozen tuning inputs differ from the plan")
    revision = miner_revision(health_url, plan["miner_model_revision"])
    checkpoints = checked_checkpoints(checkpoint_dir, plan, sha256(plan_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    for step, adapter in checkpoints.items():
        adapter_hash = sha256(adapter)
        for domain, layout in DOMAINS.items():
            index = ROOT / TUNING_LAYOUT[domain]["index"]
            cache_dir = ROOT / layout["cache_dir"]
            if (
                not cache_dir.is_dir()
                or sha256(index) != plan["tuning"][domain]["index_sha256"]
                or sha256(ROOT / TUNING_LAYOUT[domain]["baseline"])
                != plan["tuning"][domain]["baseline_sha256"]
            ):
                raise ValueError(f"projection-frozen {domain} tuning input changed")
            results = result_path(output_dir, domain, step)
            complete = checked_results(
                results, index, layout["split"],
                plan["baseline_checkpoint_sha256"], adapter_hash,
            )
            target = TUNING_LAYOUT[domain]["examples"]
            if complete == target:
                print(f"{domain} {step}: already verified {target}/{target}", flush=True)
                continue
            if shutil.disk_usage(ROOT).free < MIN_FREE_GIB * 1024**3:
                raise RuntimeError("free disk is below the projection-frozen tuning guard")
            revision = miner_revision(health_url, revision)
            print(f"{domain} {step}: starting {complete}/{target}", flush=True)
            worker = subprocess.Popen(
                evaluation_command(
                    index, cache_dir, layout["split"], adapter, results,
                    health_url, target - complete,
                ),
                cwd=ROOT,
            )
            try:
                while True:
                    try:
                        exit_code = worker.wait(timeout=POLL_SECONDS)
                        break
                    except subprocess.TimeoutExpired:
                        revision = miner_revision(health_url, revision)
                        if rss_kib(worker.pid) > MAX_RSS_GIB * 1024**2:
                            raise RuntimeError("projection-frozen tuning worker exceeded its RSS guard")
            finally:
                stop_worker(worker)
            if exit_code != 0:
                raise RuntimeError(f"projection-frozen {domain} tuning failed at {step}: {exit_code}")
            if checked_results(
                results, index, layout["split"],
                plan["baseline_checkpoint_sha256"], adapter_hash,
            ) != target:
                raise RuntimeError(f"projection-frozen {domain} tuning stopped early at {step}")
            revision = miner_revision(health_url, revision)
            print(f"{domain} {step}: verified {target}/{target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    args = parser.parse_args()
    if (
        not RUNTIME_PYTHON.is_file()
        or not args.plan.resolve().is_relative_to(ROOT / "runs")
        or not args.checkpoint_dir.resolve().is_relative_to(ROOT / "checkpoints")
        or not args.output_dir.resolve().is_relative_to(ROOT / "runs")
    ):
        parser.error("projection-frozen tuning needs workspace ledgers and the separate runtime")
    run_tuning(args.plan, args.checkpoint_dir, args.output_dir, args.miner_health_url)


if __name__ == "__main__":
    main()
