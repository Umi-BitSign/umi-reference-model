"""Qualify a local temporal adapter against full autograd on authentic training grades.

This proof starts from public C5, preserves every semantic-memory position and
uses both complete historical grading passes. No optimizer or new grader runs.
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

CONFIGURATION = {
    "schema": "umi-studio-local-temporal-memory-adapter/1",
    "width": 1536,
    "rank": 8,
    "context_frames": 3,
    "seed": 20261019,
    "maximum_relative_residual": 0.1,
    "scale": "per-frame original memory RMS with floor1e-6",
    "normalization_epsilon": 1e-5,
    "output_time_steps": "all original positions",
    "unknown_interval_policy": "no added neighbor mixing across any false interval",
    "initial_inference": "bitwise identity including signed zero",
    "trainable_parameters": 24600,
}
SHAPES = {
    "memory_adapter.input_projection.weight": (8, 1536),
    "memory_adapter.output_projection.weight": (1536, 8),
    "memory_adapter.temporal_weights": (8, 3),
}
POLICY = {
    "case_indices": [0, 1, 214, 215],
    "states": ["original_identity", "fixed_nonzero_adapter"],
    "nonzero_state": "output=0.03*sin(arange); temporal left=0.2 center=1 right=-0.1",
    "reused_complete_grading_passes": 2,
    "comparisons": 16,
    "risk_alpha": 1.0,
    "caption_weight": 0.1,
    "residual_weight": 0.01,
    "sequence_score_normalization": "mean_valid_token_log_probability",
    "gradient_relative_tolerance": 0.005,
    "gradient_absolute_tolerance": 1e-7,
    "loss_absolute_tolerance": 1e-5,
    "probability_absolute_tolerance": 1e-6,
    "all532_bitwise_initial_memory_identities_required": True,
    "gradient_signal_required_every_case_state_pass": True,
    "initializer": "original public C5 plus zero-output temporal adapter; no trained inheritance",
    "reference": "independent full autograd graph, one original encoder call per candidate",
    "CPU_threads": 2,
    "GPU_use": False,
    "new_Terra_requests": 0,
    "optimizer_updates": 0,
    "training_only": True,
    "promotion_authorized": False,
}


def specification(parent_plan, parent_result, parent_poll):
    from probe_c5_caption_timing import specification as source_specification

    plan = source_specification(parent_plan, parent_result, parent_poll)
    sources = dict(plan["source_sha256"])
    for name in (
        "c5_temporal_memory_adapter.py",
        "c5_temporal_memory_model.py",
        "c5_temporal_memory_risk.py",
        "probe_c5_temporal_memory.py",
    ):
        sources["tools/" + name] = sha(ROOT / "tools" / name)
    # The reused metadata authenticator is also frozen explicitly.
    sources["tools/probe_c5_caption_timing.py"] = sha(ROOT / "tools/probe_c5_caption_timing.py")
    return {
        **plan,
        "schema": "umi-studio-temporal-memory-gradient-plan/1",
        "policy": POLICY,
        "adapter_configuration": CONFIGURATION,
        "parameter_shapes": {name: list(shape) for name, shape in SHAPES.items()},
        "source_sha256": sources,
    }


def checked(path):
    from terra_training_recovery import verify_snapshot

    plan = read(path)
    require(
        plan
        == specification(
            *(bound(plan[k]) for k in ("parent_plan", "parent_result", "parent_terminal_poll"))
        ),
        "temporal memory proof changed exact source grades, configuration or objective",
    )
    verify_snapshot(path, plan)
    return plan


def freeze(path, parent_plan, parent_result, parent_poll):
    from c5_experiment_sources import archive_sources

    owned(path, "runs")
    require(not path.parent.exists(), "temporal memory proof needs a fresh plan")
    plan = specification(parent_plan, parent_result, parent_poll)
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))


def reference_backward(model, clip, mask, labels, quality):
    """Direct objective with a complete graph for each candidate, independent of risk helper."""
    import torch
    from terra_quality_training import sequence_log_probability

    outputs = [model(clip, mask, value) for value in labels]
    logs = torch.stack(
        [
            sequence_log_probability(output["language"].logits, value)
            for output, value in zip(outputs, labels, strict=True)
        ]
    )
    counts = logs.new_tensor([int((value != -100).sum()) for value in labels])
    probabilities = torch.softmax(POLICY["risk_alpha"] * logs / counts, dim=0)
    error = (probabilities * (10 - logs.new_tensor(quality))).sum()
    penalty = outputs[0]["normalized_residual_penalty"]
    caption = outputs[0]["language"].loss
    loss = error + POLICY["caption_weight"] * caption + POLICY["residual_weight"] * penalty
    loss.backward()
    return {
        "loss": float(loss.detach()),
        "expected_terra_error": float(error.detach()),
        "caption_loss": float(caption.detach()),
        "normalized_residual_penalty": float(penalty.detach()),
        "candidate_probabilities": probabilities.detach().tolist(),
        "candidate_token_counts": counts.long().tolist(),
        "original_encoder_calls": len(labels),
        "differentiable_encoder_calls": 0,
    }


def compare_gradients(reference, optimized):
    difference = 0.0
    for report in (reference, optimized):
        require(
            report["parameters"] == 24600
            and report["frozen_backbone_gradients_absent"] is True
            and set(report["gradient_values"]) == set(SHAPES),
            "temporal memory proof changed its complete frozen/trainable gradient scope",
        )
        squares = 0.0
        for name, shape in SHAPES.items():
            values = np.asarray(report["gradient_values"][name], dtype=np.float32)
            require(
                values.shape == shape and np.isfinite(values).all(),
                "temporal memory proof omitted finite shaped parameter gradients",
            )
            squares += float(np.square(values.astype(np.float64)).sum())
        norm = math.sqrt(squares)
        require(
            math.isfinite(report["gradient_norm"])
            and norm > 0
            and math.isclose(norm, report["gradient_norm"], rel_tol=1e-5, abs_tol=1e-8),
            "temporal memory proof lacks a finite nonzero gradient signal",
        )
    for name in SHAPES:
        a = np.asarray(reference["gradient_values"][name], dtype=np.float32)
        b = np.asarray(optimized["gradient_values"][name], dtype=np.float32)
        require(
            np.allclose(
                b,
                a,
                rtol=POLICY["gradient_relative_tolerance"],
                atol=POLICY["gradient_absolute_tolerance"],
            ),
            "temporal memory optimized derivative differs from full autograd reference",
        )
        difference = max(difference, float(np.max(np.abs(b - a))))
    return difference


def compare_losses(reference, optimized):
    require(
        all(
            abs(reference[key] - optimized[key]) <= POLICY["loss_absolute_tolerance"]
            for key in (
                "loss",
                "expected_terra_error",
                "caption_loss",
                "normalized_residual_penalty",
            )
        )
        and reference["candidate_token_counts"] == optimized["candidate_token_counts"]
        and len(reference["candidate_probabilities"]) == len(optimized["candidate_probabilities"])
        and np.allclose(
            reference["candidate_probabilities"],
            optimized["candidate_probabilities"],
            rtol=0,
            atol=POLICY["probability_absolute_tolerance"],
        )
        and optimized["original_encoder_calls"] == optimized["adapter_backward_calls"] == 1
        and reference["differentiable_encoder_calls"]
        == optimized["differentiable_encoder_calls"]
        == 0
        and all(
            optimized[k] == POLICY[k]
            for k in (
                "risk_alpha",
                "caption_weight",
                "residual_weight",
                "sequence_score_normalization",
            )
        ),
        "temporal memory changed objective, probabilities or single adapter backward",
    )


def initial_identities(model, dataset):
    import torch
    from c5_timing_semantic_model import clip_input

    require(len(dataset) == 532, "temporal memory proof requires all training clips")
    for index in range(532):
        clip = clip_input(dataset.load(index))
        mask = torch.ones(1, len(clip.streams["pose_features"]), dtype=torch.bool)
        with torch.no_grad():
            details = model.encode_details(clip, mask)
        require(
            torch.equal(
                details["memory"].view(torch.int32), details["base_memory"].view(torch.int32)
            )
            and details["normalized_residual_penalty"].item() == 0,
            "temporal memory initializer changed original memory bits",
        )
        if (index + 1) % 50 == 0:
            print(
                json.dumps({"stage": "memory_initial_identity", "clips": index + 1, "of": 532}),
                flush=True,
            )
    return {"clips": 532, "bitwise_identity_including_signed_zero": True}


def set_state(model, initial, state):
    import torch

    model.memory_adapter.load_state_dict(initial, strict=True)
    require(state in (0, 1), "unknown temporal memory proof state")
    if state:
        with torch.no_grad():
            output = model.memory_adapter.output_projection.weight
            output.copy_(
                0.03
                * torch.sin(torch.arange(output.numel(), dtype=torch.float32).reshape_as(output))
            )
            model.memory_adapter.temporal_weights[:, 0].fill_(0.2)
            model.memory_adapter.temporal_weights[:, 2].fill_(-0.1)


def qualify(path, output, plan):
    import probe_c5_timing_semantic as original
    import torch
    from c5_temporal_memory_model import load_student, require_model
    from c5_temporal_memory_risk import backward_memory_risk, gradient_report
    from c5_terra_tail import candidate_labels
    from c5_timing_semantic_model import clip_input

    dataset = TimingFeatureDataset(bound(plan["data_audit"]))
    model, tokenizer = load_student(plan["model_plan"], plan["adapter_configuration"])
    initial = {name: value.clone() for name, value in model.memory_adapter.state_dict().items()}
    require(
        torch.count_nonzero(initial["output_projection.weight"]).item() == 0,
        "memory output initializer is not zero",
    )
    write(output / "memory-identity.json", initial_identities(model, dataset))
    values = original.selected_values(dataset, plan)
    pools = read(bound(plan["pools"]))
    original.validate_pools(values, pools, tokenizer)
    require(
        original.paired_inputs(values, pools) == read(bound(plan["paired_inputs"])),
        "memory proof changed authentic candidate tokens",
    )
    files = []
    for state in range(2):
        set_state(model, initial, state)
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
                optimized = backward_memory_risk(
                    model,
                    clip,
                    mask,
                    labels,
                    quality,
                    **{k: POLICY[k] for k in ("caption_weight", "residual_weight")},
                    alpha=POLICY["risk_alpha"],
                )
                optimized_gradient = gradient_report(model)
                difference = compare_gradients(reference_gradient, optimized_gradient)
                compare_losses(reference, optimized)
                target = output / f"case-{state}-{grading['pass']}-{case}.json"
                write(
                    target,
                    {
                        "state": state,
                        "state_name": POLICY["states"][state],
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
                    json.dumps({"stage": "memory_gradient", "comparisons": len(files), "of": 16}),
                    flush=True,
                )
    set_state(model, initial, 0)
    model.zero_grad(set_to_none=True)
    require_model(model, plan["model_plan"], plan["adapter_configuration"])
    require(
        all(
            torch.equal(value, initial[name])
            for name, value in model.memory_adapter.state_dict().items()
        )
        and len(files) == 16
        and plan == checked(path),
        "temporal memory proof changed its inputs or initial state",
    )
    write(
        output / "qualification.json",
        {
            "schema": "umi-studio-temporal-memory-gradient-qualification/1",
            "complete": True,
            "plan": artifact(path),
            "worker": artifact(output / "worker.json"),
            "memory_identity": artifact(output / "memory-identity.json"),
            "comparisons": files,
            "grading_provenance": plan["grading_provenance"],
            "original_model_values_unchanged": True,
            "adapter_initializer_restored": True,
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
    from probe_c5_timing_semantic import guard
    from train_terra_c5_tail import gpu_lease

    require_environment()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    plan = checked(path)
    owned(output, "checkpoints")
    require(
        not output.exists() and not path.with_name("execution-claim.json").exists(),
        "memory proof needs one fresh execution",
    )
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
    plan, result, poll = checked(plan_path), read(result_path), read(poll_path)
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
        and Path(argv[1]).absolute() == ROOT / "tools/probe_c5_temporal_memory.py"
        and Path(argv[argv.index(flag) + 1]).absolute() == plan_path
        and Path(argv[argv.index("--output") + 1]).absolute() == directory,
        "memory proof needs actual terminal worker/program/plan evidence",
    )
    claim_path = plan_path.with_name("execution-claim.json")
    claim = read(claim_path)
    require(
        claim["plan"] == artifact(plan_path) and claim["worker"] == authority["worker.json"],
        "memory proof lost its execution claim",
    )
    require(
        result_path.name == "qualification.json"
        and result["schema"] == "umi-studio-temporal-memory-gradient-qualification/1"
        and result["complete"] is True
        and result["plan"] == artifact(plan_path)
        and result["worker"] == authority["worker.json"]
        and result["grading_provenance"] == plan["grading_provenance"]
        and result["original_model_values_unchanged"]
        is result["adapter_initializer_restored"]
        is result["training_only"]
        is True
        and result["new_Terra_requests"] == result["optimizer_updates"] == 0
        and result["quality_gain_established"] is result["promotion_authorized"] is False,
        "memory proof changed complete original-model or grading binding",
    )
    identity_path = bound(result["memory_identity"])
    require(
        identity_path == directory / "memory-identity.json"
        and read(identity_path) == {"clips": 532, "bitwise_identity_including_signed_zero": True}
        and len(result["comparisons"]) == 16,
        "memory proof omitted complete identity or comparisons",
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
        path, value = bound(descriptor), read(bound(descriptor))
        require(
            path == directory / f"case-{state}-{grading['pass']}-{case}.json"
            and value["state"] == state
            and value["state_name"] == POLICY["states"][state]
            and value["pass"] == grading["pass"]
            and value["case"] == case
            and value["index"] == POLICY["case_indices"][case]
            and value["row"] == plan["records"][case]["row"]
            and value["record"] == plan["records"][case]
            and value["actual_grades"] == grading["qualities"][case],
            "memory proof changed actual case/state/grade coverage",
        )
        require(
            compare_gradients(value["reference_gradient"], value["optimized_gradient"])
            == value["maximum_absolute_gradient_difference"],
            "memory proof changed its gradient difference",
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
            and abs(optimized["caption_loss"] + float(logs[0]) / counts[0]) <= 1e-5
            and value["reference"]["original_encoder_calls"] == len(grading["qualities"][case]),
            "memory proof changed complete token normalization or reference scope",
        )
        expected.add(path)
    paths = list(directory.rglob("*"))
    require(
        all(not p.is_symlink() and (not p.is_file() or p.stat().st_nlink == 1) for p in paths)
        and {p for p in paths if p.is_file()} == expected,
        "memory proof contains partial, extra or aliased output evidence",
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
        parser.error("freeze from complete authentic training grades, or run one frozen proof")


if __name__ == "__main__":
    main()
