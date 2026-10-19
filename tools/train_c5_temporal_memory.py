"""Train a qualified local temporal-memory adapter on authenticated caption/risk data.

Reuse the complete authenticated original-public training pools and grades as
data. Fresh public initialization, four whole epochs, one final adapter, no API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from c5_timing_feature_data import (
    ROOT,
    TimingFeatureDataset,
    artifact,
    bound,
    owned,
    read,
    require,
    sha,
)
from launch_terra_visual_job import identity, now, write

RECIPE = {
    "epochs": 4,
    "clips_per_epoch": 532,
    "batch_size": 8,
    "planned_exposures": 2128,
    "planned_optimizer_updates": 268,
    "shuffle_seed": 20261019,
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "betas": [0.9, 0.999],
    "epsilon": 1e-8,
    "weight_decay": 0.01,
    "gradient_norm_limit": 1.0,
    "risk_alpha": 1.0,
    "caption_weight": 0.1,
    "residual_weight": 0.01,
    "sequence_score_normalization": "mean_valid_token_log_probability",
    "candidate_pool_policy": "reuse all exact original-public pools from completed source training",
    "candidate_generation": "none; retain source human/greedy/sample/distinct-caption donor tokens",
    "grading": "reuse complete authenticated source rewards and controls; no new judge calls",
    "new_Terra_requests": 0,
    "parent_model_or_optimizer_inheritance": False,
    "CPU_threads": 2,
    "GPU_use": False,
    "zero_support_clips": "retain full captions and center context; block added neighbor mixing",
    "selection": "one fixed final after four complete epochs; no development checkpoint selection",
    "maximum_output_bytes": 256 * 1024**2,
    "physical_free_floor_bytes": 84 * 1024**3,
    "promotion_authorized": False,
}


def batches():
    result = []
    for epoch in range(RECIPE["epochs"]):
        order = list(range(RECIPE["clips_per_epoch"]))
        random.Random(RECIPE["shuffle_seed"] + epoch).shuffle(order)
        for offset in range(0, len(order), RECIPE["batch_size"]):
            result.append(
                {"epoch": epoch + 1, "indices": order[offset : offset + RECIPE["batch_size"]]}
            )
    return result


def state_hashes(adapter):
    return {
        name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, value in sorted(adapter.state_dict().items())
    }


def guard(output):
    import shutil

    from evaluate_tiny_rgb_feature_winner import peak_bytes

    files = list(output.rglob("*")) if output.exists() else []
    require(all(not p.is_symlink() for p in files), "memory training output contains aliases")
    remaining = RECIPE["maximum_output_bytes"] - sum(p.stat().st_size for p in files if p.is_file())
    require(
        remaining >= 0
        and shutil.disk_usage(ROOT).free
        >= RECIPE["physical_free_floor_bytes"] + remaining + 128 * 1024**2
        and peak_bytes() <= 32 * 1024**3,
        "memory training exceeds storage or process memory allowance",
    )


def specification(qualification_plan, qualification_result, qualification_poll, parent_final_audit):
    from c5_timing_final import checked_final
    from probe_c5_temporal_memory import checked_qualification

    qualified, _, authority = checked_qualification(
        qualification_plan, qualification_result, qualification_poll
    )
    parent = checked_final(parent_final_audit)
    source_plan = read(bound(parent["plan"]))
    source_result = read(bound(parent["result"]))
    require(
        qualified["model_plan"] == parent["model_plan"] == source_plan["model_plan"]
        and qualified["data_audit"] == source_plan["data_audit"]
        and qualified["parent_plan"] == source_plan["qualification_plan"]
        and qualified["parent_result"] == source_plan["qualification_result"]
        and qualified["parent_terminal_poll"] == source_plan["qualification_terminal_poll"]
        and [source_plan["records"][i] for i in qualified["policy"]["case_indices"]]
        == qualified["records"]
        and qualified["policy"]["risk_alpha"] == RECIPE["risk_alpha"]
        and qualified["policy"]["caption_weight"] == RECIPE["caption_weight"]
        and qualified["policy"]["residual_weight"] == RECIPE["residual_weight"]
        and qualified["policy"]["sequence_score_normalization"]
        == RECIPE["sequence_score_normalization"],
        "temporal memory training changed its qualified objective, public model or training data",
    )
    directory = bound(parent["result"]).parent
    pools = [artifact(directory / f"pools/clip-{i:04d}.json") for i in range(532)]
    paired = read(bound(source_result["comparison"]))
    rewards = read(bound(source_result["rewards"]))
    require(
        len(source_plan["records"]) == 532
        and len(rewards["qualities"]) == 532
        and source_result["new_Terra_requests"] == len(paired["requests"]),
        "temporal memory training source is not the complete authenticated training inventory",
    )
    sources = dict(qualified["source_sha256"])
    for name, digest in parent["source_sha256"].items():
        require(
            name not in sources or sources[name] == digest,
            "temporal memory training source ancestry conflicts",
        )
        sources[name] = digest
    for name in (
        "train_c5_temporal_memory.py",
        "c5_temporal_memory_final.py",
        "train_c5_caption_timing.py",
    ):
        sources["tools/" + name] = sha(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-temporal-memory-training-plan/1",
        "qualification_plan": artifact(qualification_plan),
        "qualification_result": artifact(qualification_result),
        "qualification_terminal_poll": artifact(qualification_poll),
        "qualification_evidence": authority,
        "parent_final_audit": artifact(parent_final_audit),
        "parent_training_plan": parent["plan"],
        "parent_training_result": parent["result"],
        "data_audit": source_plan["data_audit"],
        "model_plan": parent["model_plan"],
        "adapter_configuration": qualified["adapter_configuration"],
        "records": source_plan["records"],
        "pool_assets": pools,
        "rewards": source_result["rewards"],
        "comparison": source_result["comparison"],
        "raw_grading_evidence": rewards["raw"],
        "grading_provenance": {
            "reused_training_requests": source_result["new_Terra_requests"],
            "qualification_source": qualified["grading_provenance"],
            "unique_historical_API_attempts_including_failed_predecessor": source_result[
                "new_Terra_requests"
            ]
            + qualified["grading_provenance"]["inherited_API_attempts_including_predecessor"],
            "new_Terra_requests": 0,
        },
        "recipe": RECIPE,
        "batches": batches(),
        "source_sha256": sources,
        "initialization": (
            "public C5 plus seeded zero-output temporal memory adapter; no trained inheritance"
        ),
        "quality_gain_established": False,
        "promotion_authorized": False,
    }


def checked(path):
    from terra_training_recovery import verify_snapshot

    plan = read(path)
    require(
        plan
        == specification(
            *(
                bound(plan[k])
                for k in (
                    "qualification_plan",
                    "qualification_result",
                    "qualification_terminal_poll",
                    "parent_final_audit",
                )
            )
        ),
        "temporal memory training changed qualification, rewards or exact full recipe",
    )
    verify_snapshot(path, plan)
    return plan


def freeze(path, qualification_plan, qualification_result, qualification_poll, parent_final_audit):
    from c5_experiment_sources import archive_sources

    owned(path, "runs")
    require(not path.parent.exists(), "temporal memory training needs a fresh plan")
    plan = specification(
        qualification_plan, qualification_result, qualification_poll, parent_final_audit
    )
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))


def materials(plan, tokenizer):
    from train_c5_caption_timing import materials as authenticated_materials

    return authenticated_materials(plan, tokenizer)


def train(path, output, plan):
    import torch
    from c5_temporal_memory_model import load_student, require_model
    from c5_temporal_memory_risk import backward_memory_risk, gradient_report
    from c5_terra_tail import candidate_labels
    from c5_timing_semantic_model import clip_input

    dataset = TimingFeatureDataset(bound(plan["data_audit"]))
    model, tokenizer = load_student(plan["model_plan"], plan["adapter_configuration"])
    require(
        torch.count_nonzero(model.memory_adapter.output_projection.weight).item() == 0,
        "temporal memory training did not start from original zero-output memory",
    )
    pools, rewards = materials(plan, tokenizer)
    require_model(model, plan["model_plan"], plan["adapter_configuration"])
    require(
        plan == checked(path),
        "temporal memory training changed its inputs before optimizer creation",
    )
    selected = model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        selected.values(),
        lr=RECIPE["learning_rate"],
        betas=tuple(RECIPE["betas"]),
        eps=RECIPE["epsilon"],
        weight_decay=RECIPE["weight_decay"],
    )
    steps, exposures, epoch_signal = (
        [],
        0,
        {epoch: False for epoch in range(1, RECIPE["epochs"] + 1)},
    )
    for number, batch in enumerate(plan["batches"], start=1):
        guard(output)
        optimizer.zero_grad(set_to_none=True)
        samples = []
        for index in batch["indices"]:
            values = dataset.load(index)
            clip = clip_input(values)
            mask = torch.ones(1, len(clip.streams["pose_features"]), dtype=torch.bool)
            measured = backward_memory_risk(
                model,
                clip,
                mask,
                candidate_labels(pools[index]["candidates"]),
                rewards["qualities"][index],
                alpha=RECIPE["risk_alpha"],
                caption_weight=RECIPE["caption_weight"],
                residual_weight=RECIPE["residual_weight"],
                divisor=len(batch["indices"]),
            )
            samples.append({"index": index, "risk": measured})
            exposures += 1
        gradient = gradient_report(model, require_signal=False)
        epoch_signal[batch["epoch"]] |= gradient["gradient_norm"] > 0
        torch.nn.utils.clip_grad_norm_(
            selected.values(), RECIPE["gradient_norm_limit"], error_if_nonfinite=True
        )
        before = state_hashes(model.memory_adapter)
        optimizer.step()
        model.require_fixed()
        target = output / f"step-{number:04d}.json"
        write(
            target,
            {
                "step": number,
                "batch": batch,
                "exposures": exposures,
                "samples": samples,
                "gradient": gradient,
                "before": before,
                "after": state_hashes(model.memory_adapter),
            },
        )
        steps.append(artifact(target))
        print(
            json.dumps(
                {
                    "stage": "temporal_memory_optimizer",
                    "step": number,
                    "of": 268,
                    "epoch": batch["epoch"],
                    "exposures": exposures,
                    "state_hashes": state_hashes(model.memory_adapter),
                }
            ),
            flush=True,
        )
    require(
        exposures == 2128
        and len(steps) == 268
        and all(epoch_signal.values())
        and torch.count_nonzero(model.memory_adapter.output_projection.weight).item() > 0,
        "temporal memory did not complete four full epochs with actual learning signal",
    )
    require_model(model, plan["model_plan"], plan["adapter_configuration"])
    require(
        materials(plan, tokenizer) == (pools, rewards) and plan == checked(path),
        "temporal memory training source evidence changed before the final checkpoint",
    )
    optimizer.zero_grad(set_to_none=True)
    guard(output)
    final = output / "final-adapter.pt"
    with final.open("xb") as stream:
        torch.save(
            {
                "memory_adapter": model.memory_adapter.state_dict(),
                "configuration": plan["adapter_configuration"],
                "optimizer": optimizer.state_dict(),
                "epochs": 4,
                "updates": 268,
                "exposures": 2128,
                "plan": artifact(path),
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    write(
        output / "result.json",
        {
            "schema": "umi-studio-temporal-memory-training/1",
            "complete": True,
            "plan": artifact(path),
            "worker": artifact(output / "worker.json"),
            "adapter": artifact(final),
            "steps": steps,
            "exposures": 2128,
            "optimizer_updates": 268,
            "completed_epochs": 4,
            "state_hashes": state_hashes(model.memory_adapter),
            "original_backbones_unchanged": True,
            "training_clips": 532,
            "grading_provenance": plan["grading_provenance"],
            "new_Terra_requests": 0,
            "CPU_threads": 2,
            "GPU_use": False,
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
    child = bound(plan["qualification_plan"]).with_name("optimizer-child-claim.json")
    claim_path = path.with_name("execution-claim.json")
    require(
        not output.exists() and not child.exists() and not claim_path.exists(),
        "temporal memory training permits only one fresh complete optimizer execution",
    )
    guard(output)
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        claim = {
            "plan": artifact(path),
            "worker": artifact(output / "worker.json"),
            "claimed_at": now(),
        }
        write(child, claim)
        write(claim_path, claim)
        with gpu_lease():
            train(path, output, plan)
        code = 0
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "freeze",
        "plan",
        "output",
        "qualification-plan",
        "qualification-result",
        "qualification-terminal-poll",
        "parent-final-audit",
    ):
        parser.add_argument("--" + name, type=lambda p: Path(p).absolute())
    args = parser.parse_args()
    if (
        args.freeze
        and all(
            (
                args.qualification_plan,
                args.qualification_result,
                args.qualification_terminal_poll,
                args.parent_final_audit,
            )
        )
        and not args.plan
    ):
        freeze(
            args.freeze,
            args.qualification_plan,
            args.qualification_result,
            args.qualification_terminal_poll,
            args.parent_final_audit,
        )
        if args.output:
            run(args.freeze, args.output)
    elif args.plan and args.output and not args.freeze:
        run(args.plan, args.output)
    else:
        parser.error(
            "freeze from actual caption qualification and complete source rewards, or run one plan"
        )


if __name__ == "__main__":
    main()
