"""Apply the frozen confirmation gate to the fixed stride-2 C5 candidate."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

from benchmark_fleurs_dev import sha256

ROOT = Path(__file__).resolve().parents[1]


def summarize_domain(
    baseline: list[dict], candidate: list[dict], *, expected_stride: int
) -> dict:
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("confirmation result counts differ")
    if [row["example_id"] for row in baseline] != [
        row["example_id"] for row in candidate
    ]:
        raise ValueError("confirmation result identities differ")
    by_signer = defaultdict(lambda: {"baseline": [], "candidate": []})
    for first, second in zip(baseline, candidate, strict=True):
        if (
            first["reference"] != second["reference"]
            or first["index_sha256"] != second["index_sha256"]
            or first["checkpoint_sha256"] != second["checkpoint_sha256"]
            or first.get("adapter_sha256") is not None
            or second.get("adapter_sha256") is None
            or first.get("temporal_stride", 1) != 1
            or second.get("temporal_stride") != expected_stride
            or first.get("source_frame_count") != second.get("source_frame_count")
            or first.get("evaluated_frame_count") != first.get("source_frame_count")
            or second.get("evaluated_frame_count")
            != (second["source_frame_count"] + expected_stride - 1) // expected_stride
            or not math.isfinite(first["single_reference_wer"])
            or not math.isfinite(second["single_reference_wer"])
        ):
            raise ValueError("confirmation results use different inputs")
        signer = str(first["signer"])
        by_signer[signer]["baseline"].append(first["single_reference_wer"])
        by_signer[signer]["candidate"].append(second["single_reference_wer"])
    baseline_mean = statistics.fmean(row["single_reference_wer"] for row in baseline)
    candidate_mean = statistics.fmean(row["single_reference_wer"] for row in candidate)
    signer_means = {
        signer: {
            "examples": len(values["baseline"]),
            "baseline_mean_wer": statistics.fmean(values["baseline"]),
            "candidate_mean_wer": statistics.fmean(values["candidate"]),
        }
        for signer, values in sorted(by_signer.items())
    }
    return {
        "examples": len(baseline),
        "baseline_mean_wer": baseline_mean,
        "candidate_mean_wer": candidate_mean,
        "candidate_minus_baseline_mean_wer": candidate_mean - baseline_mean,
        "baseline_word_weighted_wer": sum(
            row["single_reference_word_edits"] for row in baseline
        )
        / sum(row["reference_word_count"] for row in baseline),
        "candidate_word_weighted_wer": sum(
            row["single_reference_word_edits"] for row in candidate
        )
        / sum(row["reference_word_count"] for row in candidate),
        "signer_means": signer_means,
        "index_sha256": baseline[0]["index_sha256"],
        "checkpoint_sha256": baseline[0]["checkpoint_sha256"],
    }


def build_decision(
    plan: dict,
    how2sign_baseline: list[dict],
    how2sign_candidate: list[dict],
    fleurs_baseline: list[dict],
    fleurs_candidate: list[dict],
) -> dict:
    if plan.get("schema") != "umi-studio-temporal-stride-confirmation-plan/1":
        raise ValueError("unexpected confirmation plan schema")
    stride = plan["candidate_temporal_stride"]
    how2sign = summarize_domain(
        how2sign_baseline, how2sign_candidate, expected_stride=stride
    )
    fleurs = summarize_domain(fleurs_baseline, fleurs_candidate, expected_stride=stride)
    if (
        how2sign["examples"] != 42
        or fleurs["examples"] != 46
        or set(how2sign["signer_means"]) != {"1", "3", "5", "8"}
        or set(fleurs["signer_means"]) != {"4"}
        or how2sign["checkpoint_sha256"] != plan["checkpoint_sha256"]
        or fleurs["checkpoint_sha256"] != plan["checkpoint_sha256"]
        or {row["adapter_sha256"] for row in [*how2sign_candidate, *fleurs_candidate]}
        != {plan["candidate_adapter_sha256"]}
    ):
        raise ValueError("confirmation scope differs from the frozen plan")
    improved_signers = sum(
        row["candidate_mean_wer"] < row["baseline_mean_wer"]
        for row in how2sign["signer_means"].values()
    )
    equal_domain_change = (
        how2sign["candidate_minus_baseline_mean_wer"]
        + fleurs["candidate_minus_baseline_mean_wer"]
    ) / 2
    gate = plan["confirmation_gate"]
    passed = (
        how2sign["candidate_minus_baseline_mean_wer"]
        <= gate["how2sign_max_mean_wer_change"]
        and improved_signers >= gate["how2sign_min_improved_signers"]
        and fleurs["candidate_minus_baseline_mean_wer"]
        <= gate["fleurs_max_mean_wer_change"]
        and equal_domain_change <= gate["equal_domain_max_mean_wer_change"]
    )
    return {
        "schema": "umi-studio-temporal-stride-confirmation-decision/1",
        "confirmation_evaluated": True,
        "candidate_temporal_stride": stride,
        "candidate_adapter_sha256": plan["candidate_adapter_sha256"],
        "how2sign": how2sign,
        "fleurs": fleurs,
        "how2sign_signers_improved": improved_signers,
        "equal_domain_mean_wer_change": equal_domain_change,
        "confirmation_gate": gate,
        "decision": "package_candidate" if passed else "retain_frozen_baseline",
        "selection_policy_followed": True,
        "additional_confirmation_variants_evaluated": False,
        "live_miner_changed": False,
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--how2sign-baseline", type=Path, required=True)
    parser.add_argument("--how2sign-candidate", type=Path, required=True)
    parser.add_argument("--fleurs-baseline", type=Path, required=True)
    parser.add_argument("--fleurs-candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {
        "plan": args.plan,
        "how2sign_baseline": args.how2sign_baseline,
        "how2sign_candidate": args.how2sign_candidate,
        "fleurs_baseline": args.fleurs_baseline,
        "fleurs_candidate": args.fleurs_candidate,
    }
    if (
        args.output.exists()
        or not args.output.resolve().is_relative_to((ROOT / "runs").resolve())
        or not all(path.is_file() for path in sources.values())
    ):
        parser.error("confirmation decision inputs or output are invalid")
    plan = json.loads(args.plan.read_text())
    decision = build_decision(
        plan,
        read_jsonl(args.how2sign_baseline),
        read_jsonl(args.how2sign_candidate),
        read_jsonl(args.fleurs_baseline),
        read_jsonl(args.fleurs_candidate),
    )
    decision.update(
        confirmation_plan_sha256=sha256(args.plan),
        result_sha256={key: sha256(path) for key, path in sources.items() if key != "plan"},
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
