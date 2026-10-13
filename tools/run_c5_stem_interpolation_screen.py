"""Evaluate the frozen C5-to-STEM adapter interpolation grid on open tuning."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from benchmark_fleurs_dev import sha256
from run_c5_stem_tuning import CHECKPOINT, LAYOUT, RUNTIME, read_jsonl, summarize_candidate

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_WEIGHTS = (0.25, 0.5, 0.75)


def select_candidate(summaries: dict[str, dict], screen: dict) -> str | None:
    passing = []
    for label, row in summaries.items():
        how2sign = row["how2sign"]
        fleurs = row["fleurs"]
        h_delta = how2sign["candidate_minus_baseline_mean_wer"]
        f_delta = fleurs["candidate_minus_baseline_mean_wer"]
        improved_signers = sum(
            value["candidate_mean_wer"] < value["baseline_mean_wer"]
            for value in how2sign["signer_means"].values()
        )
        row["equal_domain_mean_wer_change"] = (h_delta + f_delta) / 2
        row["how2sign_signers_improved"] = improved_signers
        row["passes_screen"] = (
            h_delta <= screen["how2sign_max_mean_wer_change"]
            and f_delta <= screen["fleurs_max_mean_wer_change"]
            and row["equal_domain_mean_wer_change"]
            <= screen["equal_domain_max_mean_wer_change"]
            and improved_signers >= screen["minimum_improved_how2sign_signers"]
        )
        if row["passes_screen"]:
            passing.append(label)
    if not passing:
        return None
    return min(
        passing,
        key=lambda label: (
            summaries[label]["equal_domain_mean_wer_change"],
            summaries[label]["second_weight"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.plan.is_file()
        or args.output.exists()
        or not args.plan.resolve().is_relative_to((ROOT / "runs").resolve())
        or not args.output.resolve().is_relative_to((ROOT / "runs").resolve())
    ):
        parser.error("interpolation screen needs a frozen plan and new run output")
    plan = json.loads(args.plan.read_text())
    candidates = plan.get("candidates", [])
    if (
        plan.get("schema") != "umi-studio-c5-stem-interpolation-screen-plan/1"
        or plan.get("temporal_stride") != 2
        or plan.get("miner_health_policy") != "user_authorized_maintenance"
        or plan.get("miner_changed") is not False
        or [row.get("second_weight") for row in candidates] != list(EXPECTED_WEIGHTS)
        or "Do not reuse" not in plan.get("confirmation_policy", "")
    ):
        raise ValueError("interpolation screen plan differs")
    baseline_rows = {}
    for domain in LAYOUT:
        source = ROOT / plan["open_tuning"][domain]["results"]
        if sha256(source) != plan["open_tuning"][domain]["results_sha256"]:
            raise ValueError(f"open {domain} tuning result changed")
        baseline_rows[domain] = read_jsonl(source)
    summaries = {}
    result_hashes = {}
    for candidate in candidates:
        weight = candidate["second_weight"]
        label = f"weight-{round(weight * 100):03d}"
        adapter = ROOT / candidate["adapter"]
        blend_plan_path = ROOT / candidate["plan"]
        if (
            sha256(adapter) != candidate["adapter_sha256"]
            or sha256(blend_plan_path) != candidate["plan_sha256"]
        ):
            raise ValueError(f"interpolation candidate changed: {label}")
        blend_plan = json.loads(blend_plan_path.read_text())
        if (
            blend_plan.get("second_weight") != weight
            or blend_plan.get("first_adapter_sha256") != plan["first_adapter_sha256"]
            or blend_plan.get("second_adapter_sha256") != plan["second_adapter_sha256"]
            or blend_plan.get("direct_parameter_rule")
            != plan.get("direct_parameter_rule", "linear_interpolation")
            or blend_plan.get("confirmation_scored") is not False
        ):
            raise ValueError(f"interpolation derivation differs: {label}")
        summaries[label] = {"second_weight": weight}
        result_hashes[label] = {}
        for domain, layout in LAYOUT.items():
            results = args.output.parent / f"{label}-{domain}.jsonl"
            subprocess.run(
                [
                    "nice",
                    "-n",
                    "20",
                    str(RUNTIME),
                    "tools/evaluate_fleurs_cache.py",
                    "--index",
                    str(layout["index"]),
                    "--cache-dir",
                    str(layout["cache"]),
                    "--split",
                    str(layout["split"]),
                    "--baseline",
                    str(ROOT / "artifacts/baseline-6fe8df59"),
                    "--checkpoint",
                    str(CHECKPOINT),
                    "--adapter",
                    str(adapter),
                    "--results",
                    str(results),
                    "--miner-maintenance-mode",
                    "--device",
                    "cpu",
                    "--use-stream-features",
                    "--temporal-stride",
                    "2",
                    "--limit",
                    str(layout["examples"]),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            candidate_rows = read_jsonl(results)
            if {row.get("adapter_sha256") for row in candidate_rows} != {
                candidate["adapter_sha256"]
            }:
                raise ValueError(f"interpolation result adapter differs: {label}")
            summaries[label][domain] = summarize_candidate(
                baseline_rows[domain], candidate_rows
            )
            result_hashes[label][domain] = sha256(results)
    selected = select_candidate(summaries, plan["screen"])
    result = {
        "schema": "umi-studio-c5-stem-interpolation-screen-decision/1",
        "plan_sha256": sha256(args.plan),
        "candidates": summaries,
        "result_sha256": result_hashes,
        "selected_candidate": selected,
        "decision": "collect_fresh_heldout" if selected else "retain_confirmed_candidate",
        "prior_confirmation_reused": False,
        "confirmation_variants_evaluated": 0,
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
    print(result["decision"])


if __name__ == "__main__":
    main()
