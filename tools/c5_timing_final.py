"""Authenticate a complete timing final, replaying only its four-value optimizer.

No C5 weights, neural forwards, new scores, development inputs or confirmation
clips are loaded. The audit cannot establish held-out quality or native runtime.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from c5_timing_feature_data import ROOT, artifact, bound, owned, read, require, terminal
from launch_terra_visual_job import write


def replay_steps(plan, result, directory, rewards):
    import torch
    from terra_quality_training import minimum_risk
    from train_c5_timing import RECIPE

    require(plan["recipe"] == RECIPE, "timing final changed its exact optimizer recipe")
    parameter = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=RECIPE["learning_rate"],
        betas=tuple(RECIPE["betas"]),
        eps=RECIPE["epsilon"],
        weight_decay=RECIPE["weight_decay"],
    )
    require(len(result["steps"]) == len(plan["batches"]) == 134, "timing final has partial epochs")
    exposures = 0
    epoch_signal = {1: False, 2: False}
    for number, (descriptor, batch) in enumerate(
        zip(result["steps"], plan["batches"], strict=True), start=1
    ):
        path = bound(descriptor)
        require(path == directory / f"step-{number:04d}.json", "timing final changed step order")
        step = read(path)
        exposures += len(batch["indices"])
        require(
            step["step"] == number
            and step["batch"] == batch
            and step["exposures"] == exposures
            and [s["index"] for s in step["samples"]] == batch["indices"]
            and step["before"] == parameter.detach().tolist(),
            "timing final changed input order, exposure count or pre-update values",
        )
        for sample in step["samples"]:
            risk = sample["risk"]
            quality = rewards["qualities"][sample["index"]]
            logs = torch.tensor(risk["candidate_sequence_log_probabilities"], dtype=torch.float32)
            expected = minimum_risk(logs, quality, alpha=RECIPE["risk_alpha"])
            require(
                risk["caption_weight"] == 0
                and risk["differentiable_encoder_calls"] == 1
                and math.isfinite(risk["caption_loss"])
                and risk["caption_loss"] >= 0
                and math.isfinite(risk["memory_gradient_norm"])
                and risk["memory_gradient_norm"] >= 0
                and type(risk["timing_memory_has_gradient_path"]) is bool
                and math.isfinite(sample["regularizer"])
                and sample["regularizer"] >= 0
                and math.isclose(risk["loss"], float(expected["loss"]), rel_tol=1e-6, abs_tol=1e-7)
                and risk["loss"] == risk["expected_terra_error"]
                and torch.allclose(
                    torch.tensor(risk["candidate_probabilities"]),
                    expected["probabilities"],
                    rtol=1e-6,
                    atol=1e-7,
                )
                and math.isclose(
                    risk["terra_gradient_coefficient_l1"],
                    float(expected["coefficients"].abs().sum()),
                    rel_tol=1e-6,
                    abs_tol=1e-7,
                ),
                "timing final loss differs from actual bound candidate grades and probabilities",
            )
        gradient = step["gradient"]
        values = torch.tensor(gradient["gradient_values"], dtype=torch.float32)
        require(
            gradient["parameters"] == 4
            and values.shape == (4,)
            and torch.isfinite(values).all()
            and gradient["frozen_backbone_gradients_absent"] is True
            and math.isfinite(gradient["gradient_norm"])
            and math.isclose(
                gradient["gradient_norm"], float(values.norm()), rel_tol=1e-6, abs_tol=1e-8
            ),
            "timing final lacks complete finite four-parameter gradients",
        )
        epoch_signal[batch["epoch"]] |= gradient["gradient_norm"] > 0
        optimizer.zero_grad(set_to_none=True)
        parameter.grad = values
        torch.nn.utils.clip_grad_norm_(
            [parameter], RECIPE["gradient_norm_limit"], error_if_nonfinite=True
        )
        optimizer.step()
        with torch.no_grad():
            parameter.clamp_(
                -RECIPE["absolute_parameter_limit"], RECIPE["absolute_parameter_limit"]
            )
        require(
            step["after"] == parameter.detach().tolist(), "timing final optimizer replay differs"
        )
    require(
        exposures == result["exposures"] == 1064
        and result["optimizer_updates"] == 134
        and result["completed_epochs"] == 2
        and all(epoch_signal.values())
        and result["weights"] == parameter.detach().tolist()
        and torch.count_nonzero(parameter) > 0,
        "timing final did not complete its whole fixed training recipe",
    )
    return parameter.detach(), optimizer.state_dict()


def same_state(left, right):
    import torch

    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and torch.equal(left, right)
        )
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(same_state(v, right[k]) for k, v in left.items())
        )
    if isinstance(left, list | tuple):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(same_state(a, b) for a, b in zip(left, right, strict=True))
        )
    return type(left) is type(right) and left == right


def audit(plan_path, result_path, poll_path):
    import torch
    from train_c5_timing import (
        authenticated_rewards,
        checked,
        comparison,
        donor_indices,
        validate_pool,
    )
    from transformers import ByT5Tokenizer

    owned(plan_path, "runs")
    owned(poll_path, "runs")
    plan = checked(plan_path)
    result = read(result_path)
    directory = result_path.parent
    owned(result_path, "checkpoints")
    authority = terminal(directory)
    worker = read(bound(authority["worker.json"]))
    poll = read(poll_path)
    argv = worker["argv"]
    plan_flag = "--plan" if "--plan" in argv else "--freeze"
    require(
        poll.get("tool") == "write_stdin"
        and type(poll.get("requested_session_id")) is int
        and poll.get("actual_owned_pid") == worker["pid"]
        and type(poll["tool_result"].get("exit_code")) is int
        and poll["tool_result"]["exit_code"] == 0
        and poll["tool_result"].get("session_id") is None
        and Path(argv[1]).absolute() == ROOT / "tools/train_c5_timing.py"
        and Path(argv[argv.index(plan_flag) + 1]).absolute() == plan_path
        and Path(argv[argv.index("--output") + 1]).absolute() == directory,
        "timing final requires actual terminal tool/worker/program/plan evidence",
    )
    require(
        result_path.name == "result.json"
        and result["schema"] == "umi-studio-four-parameter-timing-training/1"
        and result["complete"] is True
        and result["plan"] == artifact(plan_path)
        and result["worker"] == authority["worker.json"]
        and result["original_backbones_unchanged"] is True
        and result["training_clips"] == 532
        and result["CPU_threads"] == 2
        and result["GPU_use"] is False
        and result["quality_gain_established"] is False
        and result["promotion_authorized"] is False,
        "timing final changed training coverage, initialization or quality scope",
    )
    claims = [
        plan_path.with_name("execution-claim.json"),
        bound(plan["qualification_plan"]).with_name("optimizer-child-claim.json"),
    ]
    claim = read(claims[0])
    require(
        claim == read(claims[1])
        and claim["plan"] == artifact(plan_path)
        and claim["worker"] == authority["worker.json"],
        "timing final lost its single actual optimizer execution claim",
    )
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / plan["model_plan"]["reference"]["bundle_root"] / "model/models/byt5_base",
        local_files_only=True,
    )
    expected_files = {
        directory / name
        for name in (
            "worker.json",
            "exit.json",
            "result.json",
            "comparison.json",
            "rewards.json",
            "final-adapter.pt",
        )
    }
    donors = donor_indices(plan["records"])
    pools = []
    for index, record in enumerate(plan["records"]):
        path = directory / f"pools/clip-{index:04d}.json"
        pool = read(path)
        validate_pool(record, plan["records"][donors[index]], pool, tokenizer)
        pools.append(pool)
        expected_files.add(path)
    paired = comparison(plan["records"], pools)
    require(read(bound(result["comparison"])) == paired, "timing final comparison changed")
    rewards = authenticated_rewards(paired, directory / "semantic")
    require(
        read(bound(result["rewards"])) == rewards
        and result["new_Terra_requests"] == len(paired["requests"]),
        "timing final rewards differ from complete actual calibrated scores",
    )
    weights, optimizer = replay_steps(plan, result, directory, rewards)
    adapter_path = bound(result["adapter"])
    require(adapter_path == directory / "final-adapter.pt", "timing final selected another adapter")
    saved = torch.load(adapter_path, map_location="cpu", weights_only=True)
    require(
        set(saved) == {"timing", "optimizer", "epochs", "updates", "exposures", "plan"}
        and set(saved["timing"]) == {"rate_weights"}
        and saved["timing"]["rate_weights"].shape == (4,)
        and saved["timing"]["rate_weights"].dtype == torch.float32
        and torch.equal(saved["timing"]["rate_weights"], weights)
        and same_state(saved["optimizer"], optimizer)
        and saved["epochs"] == 2
        and saved["updates"] == 134
        and saved["exposures"] == 1064
        and saved["plan"] == artifact(plan_path),
        "timing final serialized values or AdamW state differ from exact replay",
    )
    expected_files.update(bound(v) for v in result["steps"])
    expected_files.update(bound(v) for v in rewards["raw"])
    actual = list(directory.rglob("*"))
    require(
        all(not p.is_symlink() and (not p.is_file() or p.stat().st_nlink == 1) for p in actual)
        and {p for p in actual if p.is_file()} == expected_files,
        "timing final contains aliases, partials or unbound output files",
    )
    return {
        "schema": "umi-studio-complete-timing-final-audit/1",
        "complete": True,
        "plan": artifact(plan_path),
        "result": artifact(result_path),
        "terminal_poll": artifact(poll_path),
        "authority": authority,
        "execution_claims": [artifact(p) for p in claims],
        "adapter": result["adapter"],
        "weights": weights.tolist(),
        "optimizer_updates_replayed": 134,
        "complete_training_exposures": 1064,
        "full_epochs": 2,
        "training_clips": 532,
        "all_actual_rewards_authenticated": True,
        "model_plan": plan["model_plan"],
        "maximum_log_rate": plan["maximum_log_rate"],
        "source_sha256": plan["source_sha256"],
        "audit_program": artifact(Path(__file__)),
        "C5_model_or_judge_executed": False,
        "only_four_value_optimizer_replayed": True,
        "native_trained_runtime_qualified": False,
        "quality_gain_established": False,
        "promotion_authorized": False,
    }


def checked_final(path):
    saved = read(path)
    require(saved["audit_program"] == artifact(Path(__file__)), "timing final auditor changed")
    require(
        saved == audit(bound(saved["plan"]), bound(saved["result"]), bound(saved["terminal_poll"])),
        "timing final audit changed",
    )
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "result", "terminal-poll", "output"):
        parser.add_argument("--" + name, type=lambda p: Path(p).absolute(), required=True)
    args = parser.parse_args()
    owned(args.output, "runs")
    require(not args.output.exists(), "timing final audit requires a fresh output")
    require(
        all(
            os.environ.get(name) == "1"
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "PYTHONDONTWRITEBYTECODE",
            )
        ),
        "timing final audit requires CPU1 and disabled bytecode",
    )
    value = audit(args.plan, args.result, args.terminal_poll)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, value)


if __name__ == "__main__":
    main()
