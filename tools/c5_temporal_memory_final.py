"""Authenticate complete temporal-memory training by exact adapter AdamW replay."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from c5_timing_feature_data import ROOT, artifact, bound, owned, read, require, terminal
from launch_terra_visual_job import write


def replay_steps(plan, result, directory, rewards, pools):
    import torch
    from c5_temporal_memory_adapter import TemporalMemoryAdapter
    from probe_c5_temporal_memory import SHAPES
    from terra_quality_training import minimum_risk
    from train_c5_temporal_memory import RECIPE, state_hashes

    require(plan["recipe"] == RECIPE, "memory final changed the qualified optimizer objective")
    adapter = TemporalMemoryAdapter(
        rank=plan["adapter_configuration"]["rank"],
        maximum_relative_residual=plan["adapter_configuration"]["maximum_relative_residual"],
        seed=plan["adapter_configuration"]["seed"],
    )
    require(
        adapter.configuration() == plan["adapter_configuration"],
        "memory replay configuration changed",
    )
    selected = {
        "memory_adapter." + name: parameter
        for name, parameter in sorted(adapter.named_parameters())
    }
    optimizer = torch.optim.AdamW(
        selected.values(),
        lr=RECIPE["learning_rate"],
        betas=tuple(RECIPE["betas"]),
        eps=RECIPE["epsilon"],
        weight_decay=RECIPE["weight_decay"],
    )
    require(len(result["steps"]) == len(plan["batches"]) == 268, "memory final has partial epochs")
    exposures, epoch_signal = 0, {epoch: False for epoch in range(1, RECIPE["epochs"] + 1)}
    for number, (descriptor, batch) in enumerate(
        zip(result["steps"], plan["batches"], strict=True), start=1
    ):
        path = bound(descriptor)
        require(
            path == directory / f"step-{number:04d}.json",
            "memory final changed optimizer step order",
        )
        step = read(path)
        exposures += len(batch["indices"])
        require(
            step["step"] == number
            and step["batch"] == batch
            and step["exposures"] == exposures
            and [s["index"] for s in step["samples"]] == batch["indices"]
            and step["before"] == state_hashes(adapter),
            "memory final changed source order, exposure counters or initial update values",
        )
        for sample in step["samples"]:
            index, risk = sample["index"], sample["risk"]
            counts = [len(c["tokens"]) for c in pools[index]["candidates"]]
            logs = torch.tensor(risk["candidate_sequence_log_probabilities"], dtype=torch.float32)
            require(
                logs.shape == (len(counts),) and torch.isfinite(logs).all(),
                "memory final changed complete candidate likelihoods",
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
                and risk["differentiable_encoder_calls"] == 0
                and risk["original_encoder_calls"] == risk["adapter_backward_calls"] == 1
                and risk["residual_weight"] == RECIPE["residual_weight"]
                and math.isfinite(risk["normalized_residual_penalty"])
                and 0
                <= risk["normalized_residual_penalty"]
                <= plan["adapter_configuration"]["maximum_relative_residual"] ** 2 + 1e-8
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
                    float(expected["loss"])
                    + RECIPE["caption_weight"] * risk["caption_loss"]
                    + RECIPE["residual_weight"] * risk["normalized_residual_penalty"],
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
                and risk["memory_gradient_norm"] >= 0,
                "memory final loss differs from bound complete token counts, grades or objective",
            )
        gradient = step["gradient"]
        require(
            gradient["parameters"] == 24600
            and set(gradient["gradient_values"]) == set(SHAPES)
            and gradient["frozen_backbone_gradients_absent"] is True,
            "memory final changed complete gradient scope",
        )
        values = {
            name: torch.tensor(gradient["gradient_values"][name], dtype=torch.float32)
            for name in SHAPES
        }
        require(
            all(
                tuple(value.shape) == SHAPES[name] and torch.isfinite(value).all()
                for name, value in values.items()
            ),
            "memory final omitted finite shaped gradients",
        )
        norm = float(torch.cat([values[name].flatten() for name in selected]).norm())
        require(
            math.isfinite(gradient["gradient_norm"])
            and math.isclose(gradient["gradient_norm"], norm, rel_tol=1e-6, abs_tol=1e-8),
            "memory final changed gradient norm",
        )
        epoch_signal[batch["epoch"]] |= norm > 0
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in selected.items():
            parameter.grad = values[name]
        torch.nn.utils.clip_grad_norm_(
            selected.values(), RECIPE["gradient_norm_limit"], error_if_nonfinite=True
        )
        optimizer.step()
        require(step["after"] == state_hashes(adapter), "memory final optimizer replay differs")
    require(
        exposures == result["exposures"] == 2128
        and result["optimizer_updates"] == 268
        and result["completed_epochs"] == 4
        and all(epoch_signal.values())
        and result["state_hashes"] == state_hashes(adapter)
        and torch.count_nonzero(adapter.output_projection.weight).item() > 0,
        "memory final did not complete four full epochs or exact final values",
    )
    return adapter.state_dict(), optimizer.state_dict()


def audit(plan_path, result_path, poll_path):
    import torch
    from c5_timing_final import same_state
    from train_c5_temporal_memory import checked, materials
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
        and Path(argv[1]).absolute() == ROOT / "tools/train_c5_temporal_memory.py"
        and Path(argv[argv.index(flag) + 1]).absolute() == plan_path
        and Path(argv[argv.index("--output") + 1]).absolute() == directory,
        "memory final needs actual successful terminal worker/program/plan evidence",
    )
    require(
        result_path.name == "result.json"
        and result["schema"] == "umi-studio-temporal-memory-training/1"
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
        "memory final changed whole-training scope, public initializer or request lineage",
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
        "memory final lost its single qualified optimizer execution",
    )
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / plan["model_plan"]["reference"]["bundle_root"] / "model/models/byt5_base",
        local_files_only=True,
    )
    pools, rewards = materials(plan, tokenizer)
    adapter, state = replay_steps(plan, result, directory, rewards, pools)
    final = bound(result["adapter"])
    require(final == directory / "final-adapter.pt", "memory final selected another checkpoint")
    saved = torch.load(final, map_location="cpu", weights_only=True)
    require(
        set(saved)
        == {
            "memory_adapter",
            "configuration",
            "optimizer",
            "epochs",
            "updates",
            "exposures",
            "plan",
        }
        and saved["configuration"] == plan["adapter_configuration"]
        and same_state(saved["memory_adapter"], adapter)
        and same_state(saved["optimizer"], state)
        and saved["epochs"] == 4
        and saved["updates"] == 268
        and saved["exposures"] == 2128
        and saved["plan"] == artifact(plan_path),
        "memory final serialized adapter or AdamW state differs from exact replay",
    )
    expected = {
        directory / name for name in ("worker.json", "exit.json", "result.json", "final-adapter.pt")
    }
    expected.update(bound(descriptor) for descriptor in result["steps"])
    paths = list(directory.rglob("*"))
    require(
        all(not p.is_symlink() and (not p.is_file() or p.stat().st_nlink == 1) for p in paths)
        and {p for p in paths if p.is_file()} == expected,
        "memory final contains partial, extra or aliased output evidence",
    )
    return {
        "schema": "umi-studio-complete-temporal-memory-final-audit/1",
        "complete": True,
        "plan": artifact(plan_path),
        "result": artifact(result_path),
        "terminal_poll": artifact(poll_path),
        "authority": authority,
        "execution_claims": [artifact(path) for path in claims],
        "adapter": result["adapter"],
        "adapter_configuration": plan["adapter_configuration"],
        "state_hashes": result["state_hashes"],
        "optimizer_updates_replayed": 268,
        "complete_training_exposures": 2128,
        "full_epochs": 4,
        "training_clips": 532,
        "all_actual_rewards_authenticated": True,
        "model_plan": plan["model_plan"],
        "source_sha256": plan["source_sha256"],
        "grading_provenance": plan["grading_provenance"],
        "audit_program": artifact(Path(__file__)),
        "C5_model_or_judge_executed": False,
        "only_declared24600_parameter_optimizer_replayed": True,
        "native_trained_runtime_qualified": False,
        "quality_gain_established": False,
        "promotion_authorized": False,
    }


def checked_final(path):
    saved = read(path)
    require(saved["audit_program"] == artifact(Path(__file__)), "memory final auditor changed")
    require(
        saved == audit(*(bound(saved[k]) for k in ("plan", "result", "terminal_poll"))),
        "memory final audit changed",
    )
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "result", "terminal-poll", "output"):
        parser.add_argument("--" + name, type=lambda p: Path(p).absolute(), required=True)
    args = parser.parse_args()
    owned(args.output, "runs")
    require(not args.output.exists(), "memory final audit needs a fresh output")
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
        "memory final audit requires offline CPU1",
    )
    value = audit(args.plan, args.result, args.terminal_poll)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, value)


if __name__ == "__main__":
    main()
