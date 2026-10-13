"""Evaluate the four frozen visual-adapter checkpoints on both tuning domains."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

from benchmark_fleurs_dev import distance, miner_revision, normalize, sha256
from evaluate_fleurs_cache import evaluation_identifiers
from freeze_visual_lora_plan import RECIPE, TUNING_LAYOUT
from run_bounded_visual_extraction import rss_kib, stop_worker

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/baseline-6fe8df59"
RUNTIME_PYTHON = BASELINE / ".runtime/bin/python"
POLL_SECONDS = 10
MAX_RSS_GIB = 64
MIN_FREE_GIB = 80
DOMAINS = {
    "how2sign": {
        "cache_dir": "data/how2sign/validation-stream-encoder-cache-25fps",
        "split": "validation",
    },
    "fleurs": {
        "cache_dir": "data/fleurs-asl/encoder-cache-25fps/tuning-stream",
        "split": "tuning",
    },
}


def result_path(output_dir: Path, domain: str, step: int) -> Path:
    return output_dir / f"tuning-{domain}-{step:06d}.jsonl"


def checked_results(
    path: Path,
    index: Path,
    split: str,
    checkpoint_hash: str,
    adapter_hash: str,
) -> int:
    if not path.exists():
        return 0
    source = [json.loads(line) for line in index.read_text().splitlines()]
    identifiers = evaluation_identifiers(source, split)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) > len(source) or [row.get("example_id") for row in rows] != identifiers[: len(rows)]:
        raise ValueError(f"visual tuning results are not a source-index prefix: {path}")
    index_hash = sha256(index)
    schema = (
        "umi-studio-how2sign-cached-evaluation-example/1"
        if split == "validation"
        else "umi-studio-fleurs-cached-evaluation-example/1"
    )
    for result, original in zip(rows, source, strict=False):
        reference_words = normalize(original["reference"])
        edits = distance(reference_words, normalize(result["hypothesis"]))
        if (
            result.get("schema") != schema
            or result.get("split") != split
            or result.get("selection_sha256") != original["selection_sha256"]
            or result.get("signer") != original["signer"]
            or result.get("index_sha256") != index_hash
            or result.get("encoder_cache_sha256") != original["encoder_cache_sha256"]
            or result.get("checkpoint_sha256") != checkpoint_hash
            or result.get("adapter_sha256") != adapter_hash
            or result.get("evaluation_input") != "visual_streams"
            or result.get("reference") != original["reference"]
            or result.get("reference_word_count") != len(reference_words)
            or result.get("single_reference_word_edits") != edits
            or not math.isfinite(result.get("single_reference_wer", float("nan")))
            or not math.isclose(result["single_reference_wer"], edits / len(reference_words))
        ):
            raise ValueError(f"visual tuning result differs from source or score: {path}")
    return len(rows)


def evaluation_command(
    index: Path,
    cache_dir: Path,
    split: str,
    adapter: Path,
    results: Path,
    health_url: str,
    limit: int,
) -> list[str]:
    return [
        "nice", "-n", "20", str(RUNTIME_PYTHON),
        "tools/evaluate_fleurs_cache.py",
        "--index", str(index),
        "--cache-dir", str(cache_dir),
        "--split", split,
        "--baseline", str(BASELINE),
        "--checkpoint", str(BASELINE / "models/checkpoint-11625"),
        "--adapter", str(adapter),
        "--results", str(results),
        "--miner-health-url", health_url,
        "--device", "cpu",
        "--use-stream-features",
        "--limit", str(limit),
    ]


def checked_checkpoints(checkpoint_dir: Path, plan: dict, plan_hash: str) -> dict[int, Path]:
    from safetensors import safe_open

    ledger = checkpoint_dir / "checkpoint-index.jsonl"
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    by_step = {row["example_step"]: row for row in records}
    if len(by_step) != len(records) or len(records) != RECIPE["train_examples"] // RECIPE["recovery_after_examples"]:
        raise ValueError("visual recovery checkpoint ledger is incomplete or repeated")
    selected = {}
    for step in RECIPE["checkpoint_after_examples"]:
        row = by_step[step]
        path = checkpoint_dir / f"visual-lora-after-{step:06d}.safetensors"
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        if (
            row.get("schema") != "umi-studio-visual-lora-recovery/1"
            or row.get("plan_sha256") != plan_hash
            or row.get("recipe_id") != RECIPE["id"]
            or row.get("checkpoint_name") != path.name
            or row.get("checkpoint_sha256") != sha256(path)
            or row.get("tuning_checkpoint") is not True
            or metadata.get("schema") != "umi-studio-shubert-visual-lora/1"
            or metadata.get("plan_sha256") != plan_hash
            or metadata.get("recipe_id") != RECIPE["id"]
            or metadata.get("example_step") != str(step)
            or metadata.get("rank") != str(RECIPE["rank"])
            or metadata.get("alpha") != str(RECIPE["alpha"])
            or metadata.get("visual_last_layers") != str(RECIPE["visual_last_layers"])
            or metadata.get("baseline_checkpoint_sha256") != plan["baseline_checkpoint_sha256"]
            or metadata.get("trainer_source_sha256") != plan["trainer_source_sha256"]
            or metadata.get("adapter_source_sha256") != plan["visual_adapter_source_sha256"]
        ):
            raise ValueError(f"visual tuning checkpoint differs from the frozen plan: {step}")
        selected[step] = path
    return selected


def run_tuning(plan_path: Path, checkpoint_dir: Path, output_dir: Path, health_url: str) -> None:
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != "umi-studio-visual-lora-plan/1" or plan.get("recipe") != RECIPE:
        raise ValueError("visual tuning plan schema or recipe differs")
    revision = miner_revision(health_url, plan["miner_model_revision"])
    checkpoint_hash = sha256(BASELINE / "models/checkpoint-11625/model.safetensors")
    if checkpoint_hash != plan["baseline_checkpoint_sha256"]:
        raise ValueError("visual tuning baseline checkpoint changed")
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
                raise ValueError(f"visual {domain} tuning input differs from the plan")
            results = result_path(output_dir, domain, step)
            complete = checked_results(
                results, index, layout["split"], checkpoint_hash, adapter_hash
            )
            target = TUNING_LAYOUT[domain]["examples"]
            if complete == target:
                print(f"{domain} {step}: already verified {target}/{target}", flush=True)
                continue
            if shutil.disk_usage(ROOT).free < MIN_FREE_GIB * 1024**3:
                raise RuntimeError("free disk is below the tuning guard")
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
                            raise RuntimeError("visual tuning worker exceeded its RSS guard")
            finally:
                stop_worker(worker)
            if exit_code != 0:
                raise RuntimeError(f"visual {domain} tuning failed at {step}: exit={exit_code}")
            if checked_results(results, index, layout["split"], checkpoint_hash, adapter_hash) != target:
                raise RuntimeError(f"visual {domain} tuning stopped early at {step}")
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
        parser.error("visual tuning needs the independent runtime and workspace ledgers")
    run_tuning(args.plan, args.checkpoint_dir, args.output_dir, args.miner_health_url)


if __name__ == "__main__":
    main()
