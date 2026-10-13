"""Apply the frozen two-domain gate to one corrective visual LoRA checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmark_fleurs_dev import sha256
from decide_aligned_mixed import build_decision
from freeze_visual_lora_plan import TUNING_LAYOUT
from run_visual_lora_tuning import DOMAINS, checked_results
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def check_temporal_stride(rows: list[dict], expected_stride: int) -> None:
    if expected_stride < 1:
        raise ValueError("candidate temporal stride must be positive")
    for row in rows:
        source_frames = row.get("source_frame_count")
        evaluated_frames = row.get("evaluated_frame_count")
        if expected_stride == 1 and source_frames is None and evaluated_frames is None:
            if row.get("temporal_stride", 1) == 1:
                continue
        if (
            row.get("temporal_stride", 1) != expected_stride
            or source_frames is None
            or evaluated_frames != (source_frames + expected_stride - 1) // expected_stride
        ):
            raise ValueError("candidate results use another temporal stride")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--example-step", type=int, required=True)
    parser.add_argument("--fleurs-candidate", type=Path, required=True)
    parser.add_argument("--how2sign-candidate", type=Path, required=True)
    parser.add_argument("--candidate-temporal-stride", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs"):
        parser.error("corrective decision needs a new run-ledger output")
    plan = json.loads(args.plan.read_text())
    plan_hash = sha256(args.plan)
    adapter_hash = sha256(args.adapter)
    with safe_open(str(args.adapter), framework="pt", device="cpu") as source:
        metadata = source.metadata()
    if (
        plan.get("schema") != "umi-studio-visual-lora-corrective-plan/1"
        or metadata.get("schema") != "umi-studio-shubert-visual-lora/1"
        or metadata.get("plan_sha256") != plan_hash
        or metadata.get("recipe_id") != plan["recipe"]["id"]
        or metadata.get("example_step") != str(args.example_step)
        or metadata.get("baseline_checkpoint_sha256")
        != plan["baseline_checkpoint_sha256"]
    ):
        raise ValueError("corrective candidate differs from the frozen plan")
    candidates = {
        "fleurs": args.fleurs_candidate,
        "how2sign": args.how2sign_candidate,
    }
    baselines = {}
    candidate_rows = {}
    for domain, path in candidates.items():
        index = ROOT / TUNING_LAYOUT[domain]["index"]
        baseline = ROOT / TUNING_LAYOUT[domain]["baseline"]
        if (
            sha256(index) != plan["tuning"][domain]["index_sha256"]
            or sha256(baseline) != plan["tuning"][domain]["baseline_sha256"]
            or checked_results(
                path,
                index,
                DOMAINS[domain]["split"],
                plan["baseline_checkpoint_sha256"],
                adapter_hash,
            )
            != TUNING_LAYOUT[domain]["examples"]
        ):
            raise ValueError(f"corrective {domain} tuning results are incomplete")
        baselines[domain] = read_jsonl(baseline)
        candidate_rows[domain] = read_jsonl(path)
        check_temporal_stride(candidate_rows[domain], args.candidate_temporal_stride)
    compatible_plan = {
        "schema": "umi-studio-aligned-mixed-plan/1",
        "fleurs_tuning_baseline_sha256": plan["tuning"]["fleurs"][
            "baseline_sha256"
        ],
        "how2sign_tuning_baseline_sha256": plan["tuning"]["how2sign"][
            "baseline_sha256"
        ],
        "baseline_mean_single_reference_wer": {
            domain: plan["tuning"][domain]["baseline_mean_single_reference_wer"]
            for domain in candidates
        },
    }
    hashes = {
        "plan": plan_hash,
        "adapter": adapter_hash,
        "fleurs_baseline": sha256(ROOT / TUNING_LAYOUT["fleurs"]["baseline"]),
        "how2sign_baseline": sha256(
            ROOT / TUNING_LAYOUT["how2sign"]["baseline"]
        ),
        "fleurs_candidate": sha256(args.fleurs_candidate),
        "how2sign_candidate": sha256(args.how2sign_candidate),
    }
    decision = build_decision(
        compatible_plan,
        baselines["fleurs"],
        candidate_rows["fleurs"],
        baselines["how2sign"],
        candidate_rows["how2sign"],
        hashes,
    )
    decision.update(
        schema="umi-studio-visual-lora-corrective-decision/1",
        example_step=args.example_step,
        starting_adapter_sha256=plan["starting_adapter_sha256"],
        candidate_temporal_stride=args.candidate_temporal_stride,
        decision_source_sha256=sha256(Path(__file__)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(decision, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(decision["decision"])


if __name__ == "__main__":
    main()
