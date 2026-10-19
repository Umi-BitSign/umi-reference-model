"""Authenticate caption-timing training by replaying only four optimizer values."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from c5_timing_feature_data import ROOT, artifact, bound, owned, read, require, terminal
from launch_terra_visual_job import write


def replay_steps(plan, result, directory, rewards, pools):
    import torch
    from terra_quality_training import minimum_risk
    from train_c5_caption_timing import RECIPE

    require(plan["recipe"] == RECIPE, "caption final changed the qualified optimizer objective")
    parameter = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=RECIPE["learning_rate"],
        betas=tuple(RECIPE["betas"]),
        eps=RECIPE["epsilon"],
        weight_decay=RECIPE["weight_decay"],
    )
    require(len(result["steps"]) == len(plan["batches"]) == 134, "caption final has partial epochs")
    exposures, epoch_signal = 0, {1: False, 2: False}
    for number, (descriptor, batch) in enumerate(
        zip(result["steps"], plan["batches"], strict=True), start=1
    ):
        path = bound(descriptor)
        require(
            path == directory / f"step-{number:04d}.json",
            "caption final changed optimizer step order",
        )
        step = read(path)
        exposures += len(batch["indices"])
        require(
            step["step"] == number
            and step["batch"] == batch
            and step["exposures"] == exposures
            and [s["index"] for s in step["samples"]] == batch["indices"]
            and step["before"] == parameter.detach().tolist(),
            "caption final changed source order, exposure counters or initial update values",
        )
        for sample in step["samples"]:
            index, risk = sample["index"], sample["risk"]
            counts = [len(c["tokens"]) for c in pools[index]["candidates"]]
            logs = torch.tensor(risk["candidate_sequence_log_probabilities"], dtype=torch.float32)
            require(
                logs.shape == (len(counts),) and torch.isfinite(logs).all(),
                "caption final changed complete candidate likelihoods",
            )
            normalized = logs / logs.new_tensor(counts)
            expected = minimum_risk(
                normalized, rewards["qualities"][index], alpha=RECIPE["risk_alpha"]
            )
            require(
                risk["candidate_token_counts"] == counts
                and risk["candidate_normalized_log_probabilities"] == normalized.tolist()
                and risk["caption_weight"] == RECIPE["caption_weight"]
                and risk["risk_alpha"] == RECIPE["risk_alpha"]
                and risk["sequence_score_normalization"] == RECIPE["sequence_score_normalization"]
                and risk["differentiable_encoder_calls"] == 1
                and math.isfinite(risk["caption_loss"])
                and risk["caption_loss"] >= 0
                and math.isclose(
                    risk["caption_loss"], -float(logs[0]) / counts[0], rel_tol=1e-6, abs_tol=1e-5
                )
                and math.isclose(
                    risk["expected_terra_error"],
                    float(expected["loss"]),
                    rel_tol=1e-6,
                    abs_tol=1e-7,
                )
                and math.isclose(
                    risk["loss"],
                    float(expected["loss"]) + RECIPE["caption_weight"] * risk["caption_loss"],
                    rel_tol=1e-6,
                    abs_tol=1e-7,
                )
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
                )
                and math.isfinite(risk["memory_gradient_norm"])
                and risk["memory_gradient_norm"] >= 0
                and type(risk["timing_memory_has_gradient_path"]) is bool
                and math.isfinite(sample["regularizer"])
                and sample["regularizer"] >= 0,
                "caption final loss differs from bound complete token counts, grades or objective",
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
            "caption final lacks complete finite four-value gradients",
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
            step["after"] == parameter.detach().tolist(), "caption final optimizer replay differs"
        )
    require(
        exposures == result["exposures"] == 1064
        and result["optimizer_updates"] == 134
        and result["completed_epochs"] == 2
        and all(epoch_signal.values())
        and result["weights"] == parameter.detach().tolist()
        and torch.count_nonzero(parameter).item() > 0,
        "caption final did not complete both full epochs or exact final values",
    )
    return parameter.detach(), optimizer.state_dict()


def audit(plan_path, result_path, poll_path):
    import torch
    from c5_timing_final import same_state
    from train_c5_caption_timing import checked, materials
    from transformers import ByT5Tokenizer

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
        and Path(argv[1]).absolute() == ROOT / "tools/train_c5_caption_timing.py"
        and Path(argv[argv.index(flag) + 1]).absolute() == plan_path
        and Path(argv[argv.index("--output") + 1]).absolute() == directory,
        "caption final needs actual successful terminal worker/program/plan evidence",
    )
    require(
        result_path.name == "result.json"
        and result["schema"] == "umi-studio-caption-supervised-timing-training/1"
        and result["complete"] is True
        and result["plan"] == artifact(plan_path)
        and result["worker"] == authority["worker.json"]
        and result["original_backbones_unchanged"] is True
        and result["grading_provenance"] == plan["grading_provenance"]
        and result["new_Terra_requests"] == 0
        and result["training_clips"] == 532
        and result["CPU_threads"] == 2
        and result["GPU_use"] is False
        and result["quality_gain_established"] is result["promotion_authorized"] is False,
        "caption final changed whole-training scope, public initializer or request lineage",
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
        "caption final lost its single qualified optimizer execution",
    )
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / plan["model_plan"]["reference"]["bundle_root"] / "model/models/byt5_base",
        local_files_only=True,
    )
    pools, rewards = materials(plan, tokenizer)
    weights, state = replay_steps(plan, result, directory, rewards, pools)
    final = bound(result["adapter"])
    require(final == directory / "final-adapter.pt", "caption final selected another checkpoint")
    saved = torch.load(final, map_location="cpu", weights_only=True)
    require(
        set(saved) == {"timing", "optimizer", "epochs", "updates", "exposures", "plan"}
        and set(saved["timing"]) == {"rate_weights"}
        and same_state(saved["timing"]["rate_weights"], weights)
        and same_state(saved["optimizer"], state)
        and saved["epochs"] == 2
        and saved["updates"] == 134
        and saved["exposures"] == 1064
        and saved["plan"] == artifact(plan_path),
        "caption final serialized adapter or AdamW state differs from exact replay",
    )
    expected = {
        directory / name for name in ("worker.json", "exit.json", "result.json", "final-adapter.pt")
    }
    expected.update(bound(descriptor) for descriptor in result["steps"])
    paths = list(directory.rglob("*"))
    require(
        all(not p.is_symlink() and (not p.is_file() or p.stat().st_nlink == 1) for p in paths)
        and {p for p in paths if p.is_file()} == expected,
        "caption final contains partial, extra or aliased output evidence",
    )
    return {
        "schema": "umi-studio-complete-caption-timing-final-audit/1",
        "complete": True,
        "plan": artifact(plan_path),
        "result": artifact(result_path),
        "terminal_poll": artifact(poll_path),
        "authority": authority,
        "execution_claims": [artifact(path) for path in claims],
        "adapter": result["adapter"],
        "weights": weights.tolist(),
        "maximum_log_rate": plan["maximum_log_rate"],
        "optimizer_updates_replayed": 134,
        "complete_training_exposures": 1064,
        "full_epochs": 2,
        "training_clips": 532,
        "all_actual_rewards_authenticated": True,
        "model_plan": plan["model_plan"],
        "source_sha256": plan["source_sha256"],
        "grading_provenance": plan["grading_provenance"],
        "audit_program": artifact(Path(__file__)),
        "C5_model_or_judge_executed": False,
        "only_four_value_optimizer_replayed": True,
        "native_trained_runtime_qualified": False,
        "quality_gain_established": False,
        "promotion_authorized": False,
    }


def checked_final(path):
    saved = read(path)
    require(saved["audit_program"] == artifact(Path(__file__)), "caption final auditor changed")
    require(
        saved == audit(*(bound(saved[k]) for k in ("plan", "result", "terminal_poll"))),
        "caption final audit changed",
    )
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "result", "terminal-poll", "output"):
        parser.add_argument("--" + name, type=lambda p: Path(p).absolute(), required=True)
    args = parser.parse_args()
    owned(args.output, "runs")
    require(not args.output.exists(), "caption final audit needs a fresh output")
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
        "caption final audit requires offline CPU1",
    )
    value = audit(args.plan, args.result, args.terminal_poll)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, value)


if __name__ == "__main__":
    main()
