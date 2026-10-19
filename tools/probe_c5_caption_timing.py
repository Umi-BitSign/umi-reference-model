"""Qualify caption-supervised timing gradients using authenticated existing grades.

Original public C5, all532 initial feature identities, four fixed training cases,
two existing complete grading passes and two declared timing states. No API,
rollout, optimizer or held-out input participates in this experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from c5_timing_feature_data import (
    ROOT,
    TimingFeatureDataset,
    artifact,
    bound,
    owned,
    read,
    require,
    sha,
    terminal,
)
from launch_terra_visual_job import identity, now, write

POLICY = {
    "case_indices": [0, 1, 214, 215],
    "timing_states": [[0.0, 0.0, 0.0, 0.0], [0.1, -0.1, 0.05, -0.05]],
    "reused_complete_grading_passes": 2,
    "comparisons": 16,
    "risk_alpha": 1.0,
    "caption_weight": 0.1,
    "sequence_score_normalization": "mean_valid_token_log_probability",
    "gradient_relative_tolerance": 0.005,
    "gradient_absolute_tolerance": 1e-7,
    "loss_absolute_tolerance": 1e-5,
    "probability_absolute_tolerance": 1e-6,
    "all532_bitwise_initial_feature_identities_required": True,
    "gradient_signal_required_every_case_state_pass": True,
    "initializer": "original public C5 plus zero timing; no trained checkpoint inheritance",
    "reference": "independent full autograd graph with one encoder call per candidate",
    "CPU_threads": 2,
    "GPU_use": False,
    "new_Terra_requests": 0,
    "optimizer_updates": 0,
    "training_only": True,
    "promotion_authorized": False,
}


def specification(parent_plan, parent_result, parent_poll):
    from train_c5_timing import qualification

    parent, result, authority = qualification(parent_plan, parent_result, parent_poll)
    require(
        [record["index"] for record in parent["records"]] == POLICY["case_indices"]
        and all(
            record["row"]["role"] == "training" and record["row"]["training_allowed"] is True
            for record in parent["records"]
        ),
        "caption timing qualification requires the four original training cases",
    )
    pools = read(bound(result["pools"]))
    require(
        len(pools) == 4 and len(result["grading"]) == 2,
        "caption timing omitted complete original pools or grading passes",
    )
    sources = dict(parent["source_sha256"])
    for name in ("train_c5_timing.py", "c5_caption_timing_risk.py", "probe_c5_caption_timing.py"):
        sources["tools/" + name] = sha(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-caption-timing-gradient-plan/1",
        "parent_plan": artifact(parent_plan),
        "parent_result": artifact(parent_result),
        "parent_terminal_poll": artifact(parent_poll),
        "parent_authority": authority,
        "data_audit": parent["data_audit"],
        "model_plan": parent["model_plan"],
        "records": parent["records"],
        "pools": result["pools"],
        "paired_inputs": result["comparison"],
        "grading": result["grading"],
        "raw_grading_evidence": result["raw_grading_evidence"],
        "grading_provenance": {
            "complete_authentic_source_passes": 2,
            "inherited_API_attempts_including_predecessor": result.get(
                "API_attempts_including_predecessor", result["new_Terra_requests"]
            ),
            "inherited_orchestrated_failed_request_replays": result.get(
                "orchestrated_failed_request_replays", 0
            ),
            "new_Terra_requests": 0,
        },
        "policy": POLICY,
        "source_sha256": sources,
        "optimizer_training_authorized": False,
        "quality_gain_established": False,
        "promotion_authorized": False,
    }


def checked(path):
    from terra_training_recovery import verify_snapshot

    plan = read(path)
    require(
        plan
        == specification(
            *(bound(plan[k]) for k in ("parent_plan", "parent_result", "parent_terminal_poll"))
        ),
        "caption timing changed its exact original grades, inputs, initializer or objective",
    )
    verify_snapshot(path, plan)
    return plan


def freeze(path, parent_plan, parent_result, parent_poll):
    from c5_experiment_sources import archive_sources

    owned(path, "runs")
    require(not path.parent.exists(), "caption timing proof needs a fresh plan")
    plan = specification(parent_plan, parent_result, parent_poll)
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))


def reference_backward(model, clip, mask, labels, quality):
    """Direct softmax objective, independent of the optimized risk implementation."""
    import torch
    from terra_quality_training import sequence_log_probability

    outputs = [model(clip, mask, value)["language"] for value in labels]
    logs = torch.stack(
        [
            sequence_log_probability(output.logits, value)
            for output, value in zip(outputs, labels, strict=True)
        ]
    )
    counts = logs.new_tensor([int((value != -100).sum()) for value in labels])
    probabilities = torch.softmax(POLICY["risk_alpha"] * logs / counts, dim=0)
    expected_error = (probabilities * (10 - logs.new_tensor(quality))).sum()
    loss = expected_error + POLICY["caption_weight"] * outputs[0].loss
    if loss.requires_grad:
        loss.backward()
    else:
        (model.timing.rate_weights.sum() * 0).backward()
    return {
        "loss": float(loss.detach()),
        "expected_terra_error": float(expected_error.detach()),
        "caption_loss": float(outputs[0].loss.detach()),
        "candidate_probabilities": probabilities.detach().tolist(),
        "candidate_token_counts": counts.long().tolist(),
        "differentiable_encoder_calls": len(labels),
    }


def compare_gradients(reference, optimized):
    a = np.asarray(reference["gradient_values"], dtype=np.float32)
    b = np.asarray(optimized["gradient_values"], dtype=np.float32)
    require(
        a.shape == b.shape == (4,) and np.isfinite(a).all() and np.isfinite(b).all(),
        "caption timing changed its finite four-gradient scope",
    )
    require(
        np.allclose(
            b,
            a,
            rtol=POLICY["gradient_relative_tolerance"],
            atol=POLICY["gradient_absolute_tolerance"],
        ),
        "caption timing optimized derivative differs from full autograd reference",
    )
    for value in (reference, optimized):
        require(
            value["parameters"] == 4
            and value["frozen_backbone_gradients_absent"] is True
            and math.isfinite(value["gradient_norm"])
            and value["gradient_norm"] > 0
            and math.isclose(
                value["gradient_norm"],
                float(np.linalg.norm(np.asarray(value["gradient_values"], dtype=np.float64))),
                rel_tol=1e-5,
                abs_tol=1e-8,
            ),
            "caption timing lacks a finite signal or changed frozen backbone gradients",
        )
    return float(np.max(np.abs(b - a)))


def compare_losses(reference, optimized):
    require(
        all(
            abs(reference[key] - optimized[key]) <= POLICY["loss_absolute_tolerance"]
            for key in ("loss", "expected_terra_error", "caption_loss")
        )
        and reference["candidate_token_counts"] == optimized["candidate_token_counts"]
        and len(reference["candidate_probabilities"]) == len(optimized["candidate_probabilities"])
        and np.allclose(
            reference["candidate_probabilities"],
            optimized["candidate_probabilities"],
            rtol=0,
            atol=POLICY["probability_absolute_tolerance"],
        )
        and optimized["differentiable_encoder_calls"] == 1
        and optimized["risk_alpha"] == POLICY["risk_alpha"]
        and optimized["caption_weight"] == POLICY["caption_weight"]
        and optimized["sequence_score_normalization"] == POLICY["sequence_score_normalization"],
        "caption timing objective, valid-token normalization or probabilities disagree",
    )


def qualify(path, output, plan):
    import probe_c5_timing_semantic as original
    import torch
    from c5_caption_timing_risk import backward_caption_timing_risk
    from c5_terra_tail import candidate_labels
    from c5_timing_semantic_model import clip_input, gradient_report, load_student, require_model

    dataset = TimingFeatureDataset(bound(plan["data_audit"]))
    model, tokenizer = load_student(plan["model_plan"])
    original.require_zero(model)
    write(output / "feature-identity.json", original.all_feature_identities(model, dataset))
    values = original.selected_values(dataset, plan)
    pools = read(bound(plan["pools"]))
    original.validate_pools(values, pools, tokenizer)
    require(
        original.paired_inputs(values, pools) == read(bound(plan["paired_inputs"])),
        "caption timing candidate tokens differ from the authentic graded pairs",
    )
    files = []
    for state, weights in enumerate(POLICY["timing_states"]):
        with torch.no_grad():
            model.timing.rate_weights.copy_(torch.tensor(weights, dtype=torch.float32))
        for grading in plan["grading"]:
            for case, (value, pool, quality) in enumerate(
                zip(values, pools, grading["qualities"], strict=True)
            ):
                original.guard(output)
                clip = clip_input(value)
                mask = torch.ones(1, len(clip.streams["pose_features"]), dtype=torch.bool)
                labels = candidate_labels(pool)
                model.zero_grad(set_to_none=True)
                reference = reference_backward(model, clip, mask, labels, quality)
                reference_gradient = gradient_report(model)
                model.zero_grad(set_to_none=True)
                optimized = backward_caption_timing_risk(
                    model,
                    clip,
                    mask,
                    labels,
                    quality,
                    alpha=POLICY["risk_alpha"],
                    caption_weight=POLICY["caption_weight"],
                )
                optimized_gradient = gradient_report(model)
                difference = compare_gradients(reference_gradient, optimized_gradient)
                compare_losses(reference, optimized)
                target = output / f"case-{state}-{grading['pass']}-{case}.json"
                write(
                    target,
                    {
                        "state": state,
                        "weights": model.timing.rate_weights.detach().tolist(),
                        "pass": grading["pass"],
                        "case": case,
                        "index": POLICY["case_indices"][case],
                        "row": value["row"],
                        "record": plan["records"][case],
                        "actual_grades": quality,
                        "reference": reference,
                        "optimized": optimized,
                        "reference_gradient": reference_gradient,
                        "optimized_gradient": optimized_gradient,
                        "maximum_absolute_gradient_difference": difference,
                    },
                )
                files.append(artifact(target))
                print(
                    json.dumps(
                        {"stage": "caption_timing_gradient", "comparisons": len(files), "of": 16}
                    ),
                    flush=True,
                )
    with torch.no_grad():
        model.timing.rate_weights.zero_()
    model.zero_grad(set_to_none=True)
    original.require_zero(model)
    require_model(model, plan["model_plan"])
    require(
        len(files) == 16 and plan == checked(path), "caption timing proof changed during execution"
    )
    write(
        output / "qualification.json",
        {
            "schema": "umi-studio-caption-timing-gradient-qualification/1",
            "complete": True,
            "plan": artifact(path),
            "worker": artifact(output / "worker.json"),
            "feature_identity": artifact(output / "feature-identity.json"),
            "comparisons": files,
            "grading_provenance": plan["grading_provenance"],
            "original_model_values_unchanged": True,
            "zero_timing_initializer_restored": True,
            "training_only": True,
            "new_Terra_requests": 0,
            "optimizer_updates": 0,
            "quality_gain_established": False,
            "promotion_authorized": False,
        },
    )


def run(path, output):
    import torch
    from prepare_c5_timing_features import require_environment
    from train_terra_c5_tail import gpu_lease

    require_environment()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    plan = checked(path)
    owned(output, "checkpoints")
    require(
        not output.exists() and not path.with_name("execution-claim.json").exists(),
        "caption timing proof needs one fresh execution",
    )
    from probe_c5_timing_semantic import guard

    guard(output)
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        write(
            path.with_name("execution-claim.json"),
            {
                "plan": artifact(path),
                "worker": artifact(output / "worker.json"),
                "claimed_at": now(),
            },
        )
        with gpu_lease():
            qualify(path, output, plan)
        code = 0
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def checked_qualification(plan_path, result_path, poll_path):
    for path, folder in ((plan_path, "runs"), (result_path, "checkpoints"), (poll_path, "runs")):
        owned(path, folder)
    plan = checked(plan_path)
    result, poll = read(result_path), read(poll_path)
    directory = result_path.parent
    authority = terminal(directory)
    worker = read(bound(authority["worker.json"]))
    argv = worker["argv"]
    flag = "--plan" if "--plan" in argv else "--freeze"
    require(
        poll["tool"] == "write_stdin"
        and type(poll["requested_session_id"]) is int
        and poll["actual_owned_pid"] == worker["pid"]
        and type(poll["tool_result"].get("exit_code")) is int
        and poll["tool_result"]["exit_code"] == 0
        and poll["tool_result"].get("session_id") is None
        and Path(argv[1]).absolute() == ROOT / "tools/probe_c5_caption_timing.py"
        and Path(argv[argv.index(flag) + 1]).absolute() == plan_path
        and Path(argv[argv.index("--output") + 1]).absolute() == directory,
        "caption timing needs actual terminal worker/program/plan evidence",
    )
    claim_path = plan_path.with_name("execution-claim.json")
    claim = read(claim_path)
    require(
        claim["plan"] == artifact(plan_path) and claim["worker"] == authority["worker.json"],
        "caption timing lost its single execution claim",
    )
    require(
        result_path.name == "qualification.json"
        and result["schema"] == "umi-studio-caption-timing-gradient-qualification/1"
        and result["complete"] is True
        and result["plan"] == artifact(plan_path)
        and result["worker"] == authority["worker.json"]
        and result["grading_provenance"] == plan["grading_provenance"]
        and result["original_model_values_unchanged"] is True
        and result["zero_timing_initializer_restored"] is True
        and result["training_only"] is True
        and result["new_Terra_requests"] == result["optimizer_updates"] == 0
        and result["quality_gain_established"] is result["promotion_authorized"] is False,
        "caption timing proof changed its complete original-model/grade/scope binding",
    )
    identity_path = bound(result["feature_identity"])
    require(
        identity_path == directory / "feature-identity.json"
        and read(identity_path) == {"clips": 532, "bitwise_identity_including_signed_zero": True}
        and len(result["comparisons"]) == 16,
        "caption timing proof omitted full initial identity or gradients",
    )
    expected = {result_path, directory / "worker.json", directory / "exit.json", identity_path}
    pools = read(bound(plan["pools"]))
    order = [
        (state, grading, case)
        for state in range(2)
        for grading in plan["grading"]
        for case in range(4)
    ]
    for descriptor, (state, grading, case) in zip(result["comparisons"], order, strict=True):
        path = bound(descriptor)
        value = read(path)
        require(
            path == directory / f"case-{state}-{grading['pass']}-{case}.json"
            and value["state"] == state
            and value["pass"] == grading["pass"]
            and value["case"] == case
            and value["index"] == POLICY["case_indices"][case]
            and value["weights"]
            == np.asarray(POLICY["timing_states"][state], dtype=np.float32).tolist()
            and value["row"] == plan["records"][case]["row"]
            and value["record"] == plan["records"][case]
            and value["actual_grades"] == grading["qualities"][case],
            "caption timing proof changed actual case/state/grade coverage",
        )
        require(
            compare_gradients(value["reference_gradient"], value["optimized_gradient"])
            == value["maximum_absolute_gradient_difference"],
            "caption timing gradient difference changed",
        )
        compare_losses(value["reference"], value["optimized"])
        optimized = value["optimized"]
        counts = [len(candidate["tokens"]) for candidate in pools[case]]
        logs = np.asarray(optimized["candidate_sequence_log_probabilities"], dtype=np.float32)
        require(
            optimized["candidate_token_counts"] == counts
            and logs.shape == (len(counts),)
            and np.isfinite(logs).all()
            and np.array_equal(
                np.asarray(optimized["candidate_normalized_log_probabilities"], dtype=np.float32),
                logs / np.asarray(counts, dtype=np.float32),
            )
            and abs(optimized["caption_loss"] + float(logs[0]) / counts[0]) <= 1e-5,
            "caption timing proof changed complete target-token normalization",
        )
        require(
            value["reference"]["differentiable_encoder_calls"] == len(grading["qualities"][case]),
            "caption timing reference omitted complete candidates",
        )
        expected.add(path)
    paths = list(directory.rglob("*"))
    require(
        all(not p.is_symlink() and (not p.is_file() or p.stat().st_nlink == 1) for p in paths)
        and {p for p in paths if p.is_file()} == expected,
        "caption timing contains partial, extra or aliased output evidence",
    )
    return (
        plan,
        result,
        [
            artifact(plan_path),
            artifact(poll_path),
            artifact(claim_path),
            *(artifact(p) for p in sorted(expected)),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "freeze",
        "plan",
        "output",
        "parent-plan",
        "parent-result",
        "parent-terminal-poll",
    ):
        parser.add_argument("--" + name, type=lambda p: Path(p).absolute())
    args = parser.parse_args()
    if (
        args.freeze
        and all((args.parent_plan, args.parent_result, args.parent_terminal_poll))
        and not args.plan
    ):
        freeze(args.freeze, args.parent_plan, args.parent_result, args.parent_terminal_poll)
        if args.output:
            run(args.freeze, args.output)
    elif args.plan and args.output and not args.freeze:
        run(args.plan, args.output)
    else:
        parser.error("freeze from complete existing grading qualification, or run one frozen proof")


if __name__ == "__main__":
    main()
