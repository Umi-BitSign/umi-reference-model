"""Owner-authorized semantic-first research selection; WER is diagnostic only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

POLICY = {
    "authority": "Owner direction2026-10-16: LLM scores decisive; WER side measurement",
    "judge": "gpt-5.6-terra",
    "reasoning_effort": "medium",
    "primary_metric": "mean semantic quality0-10, higher is better",
    "minimum_relative_gain": 0.05,
    "minimum_absolute_gain": 0.1,
    "paired_source_group_bootstrap95_lower_bound_must_be_positive": True,
    "calibration_controls_required": True,
    "WER_is_diagnostic_only": True,
    "automatic_promotion_authorized": False,
}


def decision(audit, wer_diagnostics=None):
    complete = audit.get("complete") is True
    calibrated = audit.get("calibration_controls_passed_both_passes") is True
    a, b = audit["C5_mean_quality"], audit["native4096_mean_quality"]
    interval = audit["source_group_bootstrap95_interval"]
    if (
        len(interval) != 2
        or any(type(x) not in (float, int) or not math.isfinite(x) for x in (a, b, *interval))
        or not 0 <= a <= 10
        or not 0 <= b <= 10
        or not -10 <= interval[0] <= interval[1] <= 10
        or type(audit["clips"]) is not int
        or audit["clips"] < 1
    ):
        raise ValueError("semantic selection requires valid paired diagnostic scores")
    gain = b - a
    minimum = max(POLICY["minimum_absolute_gain"], a * POLICY["minimum_relative_gain"])
    passed = complete and calibrated and gain >= minimum and interval[0] > 0
    return {
        "policy": POLICY,
        "clips": audit["clips"],
        "C5_mean_quality": a,
        "candidate_mean_quality": b,
        "absolute_gain": gain,
        "relative_gain": gain / a if a else None,
        "required_absolute_gain": minimum,
        "paired_source_group_bootstrap95_interval": interval,
        "calibration_passed": calibrated,
        "complete_scoring": complete,
        "passes_semantic_research_gate": passed,
        "WER_is_diagnostic_only": True,
        "WER_veto_used": False,
        "WER_diagnostics": wer_diagnostics,
        "complete_benchmark_score": audit.get("complete_benchmark_score", False),
        "independent_human_validation": False,
        "scope": "matched_already_open_How2Sign_prefix_only",
        "promotion_authorized": False,
        "serving_runtime_and_independent_confirmation_still_required": True,
    }


def main():
    from audit_native_terra_semantics import ROOT, checked, summarize
    from launch_terra_visual_job import digest, write

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs"):
        parser.error("semantic decision requires a fresh owned runs/ file")
    plan = checked(args.plan)
    scores = [json.loads((args.results / f"pass-{i}-scores.json").read_text()) for i in (1, 2)]
    expected = {**summarize(plan, scores), "plan_sha256": digest(args.plan)}
    if json.loads((args.results / "decision.json").read_text()) != expected:
        raise ValueError("semantic decision differs from the complete frozen scoring evidence")
    rows = plan["cases"]
    wer = {
        "C5_mean_WER": sum(r["WER_C5_then_native4096"][0] for r in rows) / len(rows),
        "candidate_mean_WER": sum(r["WER_C5_then_native4096"][1] for r in rows) / len(rows),
    }
    result = {
        **decision(expected, wer),
        "audit_plan_sha256": digest(args.plan),
        "audit_decision_sha256": digest(args.results / "decision.json"),
        "selection_source_sha256": digest(Path(__file__)),
    }
    write(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
