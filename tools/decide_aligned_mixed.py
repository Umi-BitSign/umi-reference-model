"""Apply the frozen two-domain tuning gate to one aligned LoRA checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from benchmark_fleurs_dev import sha256


def domain_summary(baseline: list[dict], candidate: list[dict], schema: str) -> dict:
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("baseline and candidate result counts differ")
    baseline_ids = [row["example_id"] for row in baseline]
    candidate_ids = [row["example_id"] for row in candidate]
    if baseline_ids != candidate_ids or len(set(baseline_ids)) != len(baseline_ids):
        raise ValueError("baseline and candidate example IDs differ or repeat")
    expected_split = (
        "tuning" if schema == "umi-studio-fleurs-cached-evaluation-example/1" else "validation"
    )
    if (
        len({row["index_sha256"] for row in [*baseline, *candidate]}) != 1
        or len({row["checkpoint_sha256"] for row in [*baseline, *candidate]}) != 1
        or any(row["split"] != expected_split for row in [*baseline, *candidate])
    ):
        raise ValueError("baseline and candidate result scopes differ")
    if any(
        first["schema"] != schema
        or second["schema"] != schema
        or first["reference"] != second["reference"]
        or first["split"] != second["split"]
        or first["index_sha256"] != second["index_sha256"]
        or first["checkpoint_sha256"] != second["checkpoint_sha256"]
        or first.get("adapter_sha256") is not None
        or second.get("adapter_sha256") is None
        or (first.get("signer") is not None and first["signer"] != second.get("signer"))
        or not math.isfinite(first["single_reference_wer"])
        or not math.isfinite(second["single_reference_wer"])
        for first, second in zip(baseline, candidate, strict=True)
    ):
        raise ValueError("baseline and candidate results use different inputs")
    baseline_mean = statistics.fmean(row["single_reference_wer"] for row in baseline)
    candidate_mean = statistics.fmean(row["single_reference_wer"] for row in candidate)
    by_signer = defaultdict(lambda: {"baseline": [], "candidate": []})
    for first, second in zip(baseline, candidate, strict=True):
        signer = first.get("signer", second.get("signer"))
        by_signer[signer]["baseline"].append(first["single_reference_wer"])
        by_signer[signer]["candidate"].append(second["single_reference_wer"])
    signer_means = {
        str(signer): {
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
        "signer_means": signer_means,
        "index_sha256": baseline[0]["index_sha256"],
        "checkpoint_sha256": baseline[0]["checkpoint_sha256"],
    }


def build_decision(
    plan: dict,
    fleurs_baseline: list[dict],
    fleurs_candidate: list[dict],
    how2sign_baseline: list[dict],
    how2sign_candidate: list[dict],
    hashes: dict[str, str],
) -> dict:
    if plan.get("schema") != "umi-studio-aligned-mixed-plan/1":
        raise ValueError("unexpected aligned-mixed plan schema")
    if (
        hashes["fleurs_baseline"] != plan["fleurs_tuning_baseline_sha256"]
        or hashes["how2sign_baseline"] != plan["how2sign_tuning_baseline_sha256"]
    ):
        raise ValueError("tuning baseline files differ from the frozen plan")
    fleurs = domain_summary(
        fleurs_baseline,
        fleurs_candidate,
        "umi-studio-fleurs-cached-evaluation-example/1",
    )
    how2sign = domain_summary(
        how2sign_baseline,
        how2sign_candidate,
        "umi-studio-how2sign-cached-evaluation-example/1",
    )
    if (
        fleurs["examples"] != 7
        or how2sign["examples"] != 16
        or set(how2sign["signer_means"]) != {"1", "2", "3", "5", "8"}
        or not math.isclose(
            fleurs["baseline_mean_wer"], plan["baseline_mean_single_reference_wer"]["fleurs"]
        )
        or not math.isclose(
            how2sign["baseline_mean_wer"],
            plan["baseline_mean_single_reference_wer"]["how2sign"],
        )
    ):
        raise ValueError("tuning result scope differs from the frozen plan")
    adapter_hash = hashes["adapter"]
    if any(
        row["adapter_sha256"] != adapter_hash for row in [*fleurs_candidate, *how2sign_candidate]
    ):
        raise ValueError("candidate results use another adapter")
    if fleurs["checkpoint_sha256"] != how2sign["checkpoint_sha256"]:
        raise ValueError("tuning domains use different frozen checkpoints")
    improved_signers = sum(
        values["candidate_mean_wer"] < values["baseline_mean_wer"]
        for values in how2sign["signer_means"].values()
    )
    joint_delta = (
        fleurs["candidate_minus_baseline_mean_wer"] + how2sign["candidate_minus_baseline_mean_wer"]
    ) / 2
    passed = (
        how2sign["candidate_minus_baseline_mean_wer"] <= -0.05
        and improved_signers >= 3
        and fleurs["candidate_minus_baseline_mean_wer"] <= 0.03
        and joint_delta <= -0.025
    )
    return {
        "schema": "umi-studio-aligned-mixed-tuning-decision/1",
        "plan_sha256": hashes["plan"],
        "candidate_adapter_sha256": adapter_hash,
        "fleurs_baseline_results_sha256": hashes["fleurs_baseline"],
        "fleurs_candidate_results_sha256": hashes["fleurs_candidate"],
        "how2sign_baseline_results_sha256": hashes["how2sign_baseline"],
        "how2sign_candidate_results_sha256": hashes["how2sign_candidate"],
        "fleurs": fleurs,
        "how2sign": how2sign,
        "how2sign_signers_improved": improved_signers,
        "equal_domain_mean_wer_change": joint_delta,
        "decision": "proceed_to_confirmation" if passed else "retain_frozen_baseline",
        "confirmation_evaluated": False,
        "live_miner_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fleurs-baseline", type=Path, required=True)
    parser.add_argument("--fleurs-candidate", type=Path, required=True)
    parser.add_argument("--how2sign-baseline", type=Path, required=True)
    parser.add_argument("--how2sign-candidate", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {
        "plan": args.plan,
        "fleurs_baseline": args.fleurs_baseline,
        "fleurs_candidate": args.fleurs_candidate,
        "how2sign_baseline": args.how2sign_baseline,
        "how2sign_candidate": args.how2sign_candidate,
        "adapter": args.adapter,
    }
    hashes = {name: sha256(path) for name, path in sources.items()}
    from safetensors import safe_open

    with safe_open(str(args.adapter), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    plan = json.loads(args.plan.read_text())
    if (
        metadata.get("schema") != "umi-studio-shubert-lora/1"
        or metadata.get("plan_sha256") != hashes["plan"]
        or metadata.get("recipe_id") != plan["recipes"][0]["id"]
    ):
        raise ValueError("candidate adapter metadata differs from frozen plan")

    def read_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines()]

    decision = build_decision(
        plan,
        read_jsonl(args.fleurs_baseline),
        read_jsonl(args.fleurs_candidate),
        read_jsonl(args.how2sign_baseline),
        read_jsonl(args.how2sign_candidate),
        hashes,
    )
    if metadata.get("baseline_checkpoint_sha256") != decision["fleurs"]["checkpoint_sha256"]:
        raise ValueError("candidate adapter was trained from another baseline checkpoint")
    if args.output.exists():
        if json.loads(args.output.read_text()) != decision:
            raise ValueError("frozen tuning decision differs from current results")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(decision, output, sort_keys=True, indent=2)
            output.write("\n")
    print(decision["decision"])


if __name__ == "__main__":
    main()
