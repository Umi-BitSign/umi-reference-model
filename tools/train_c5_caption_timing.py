"""Train four timing values with qualified caption/token-normalized risk.

Reuse the complete authenticated original-public training pools and grades as
data. Fresh public initialization, two whole epochs, one final adapter, no API.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import train_c5_timing as original
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
    **original.RECIPE,
    "risk_alpha": 1.0,
    "caption_weight": 0.1,
    "sequence_score_normalization": "mean_valid_token_log_probability",
    "candidate_pool_policy": "reuse all exact original-public pools from completed source training",
    "candidate_generation": "none; retain source human/greedy/sample/distinct-caption donor tokens",
    "grading": "reuse complete authenticated source rewards and controls; no new judge calls",
    "new_Terra_requests": 0,
    "parent_model_or_optimizer_inheritance": False,
}


def specification(qualification_plan, qualification_result, qualification_poll, parent_final_audit):
    from c5_timing_final import checked_final
    from probe_c5_caption_timing import checked_qualification

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
        and qualified["policy"]["sequence_score_normalization"]
        == RECIPE["sequence_score_normalization"],
        "caption training changed its qualified objective, public model or training data",
    )
    directory = bound(parent["result"]).parent
    pools = [artifact(directory / f"pools/clip-{i:04d}.json") for i in range(532)]
    paired = read(bound(source_result["comparison"]))
    rewards = read(bound(source_result["rewards"]))
    require(
        len(source_plan["records"]) == 532
        and len(rewards["qualities"]) == 532
        and source_result["new_Terra_requests"] == len(paired["requests"]),
        "caption training source is not the complete authenticated training inventory",
    )
    sources = dict(qualified["source_sha256"])
    for name, digest in parent["source_sha256"].items():
        require(
            name not in sources or sources[name] == digest,
            "caption training source ancestry conflicts",
        )
        sources[name] = digest
    for name in ("train_c5_caption_timing.py", "c5_caption_timing_final.py"):
        sources["tools/" + name] = sha(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-caption-supervised-timing-training-plan/1",
        "qualification_plan": artifact(qualification_plan),
        "qualification_result": artifact(qualification_result),
        "qualification_terminal_poll": artifact(qualification_poll),
        "qualification_evidence": authority,
        "parent_final_audit": artifact(parent_final_audit),
        "parent_training_plan": parent["plan"],
        "parent_training_result": parent["result"],
        "data_audit": source_plan["data_audit"],
        "model_plan": parent["model_plan"],
        "maximum_log_rate": parent["maximum_log_rate"],
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
        "batches": original.batches(),
        "source_sha256": sources,
        "initialization": "public C5 plus zero timing; no adapter or optimizer inheritance",
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
        "caption training changed its actual qualification, source rewards or exact full recipe",
    )
    verify_snapshot(path, plan)
    return plan


def freeze(path, qualification_plan, qualification_result, qualification_poll, parent_final_audit):
    from c5_experiment_sources import archive_sources

    owned(path, "runs")
    require(not path.parent.exists(), "caption training needs a fresh plan")
    plan = specification(
        qualification_plan, qualification_result, qualification_poll, parent_final_audit
    )
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))


def materials(plan, tokenizer):
    pools = [read(bound(descriptor)) for descriptor in plan["pool_assets"]]
    require(len(pools) == len(plan["records"]) == 532, "caption training omitted source pools")
    donors = original.donor_indices(plan["records"])
    for index, pool in enumerate(pools):
        original.validate_pool(
            plan["records"][index], plan["records"][donors[index]], pool, tokenizer
        )
    paired = original.comparison(plan["records"], pools)
    require(
        paired == read(bound(plan["comparison"])), "caption training changed original graded pairs"
    )
    source = bound(plan["rewards"])
    rewards = original.authenticated_rewards(paired, source.parent / "semantic")
    require(
        rewards == read(source) and rewards["raw"] == plan["raw_grading_evidence"],
        "caption training changed complete authenticated rewards or true request lineage",
    )
    return pools, rewards


def train(path, output, plan):
    import torch
    from c5_caption_timing_risk import backward_caption_timing_risk
    from c5_terra_tail import candidate_labels
    from c5_timing_semantic_model import clip_input, gradient_report, load_student, require_model

    dataset = TimingFeatureDataset(bound(plan["data_audit"]))
    model, tokenizer = load_student(plan["model_plan"])
    require(
        torch.count_nonzero(model.timing.rate_weights).item() == 0
        and model.timing.maximum_log_rate == plan["maximum_log_rate"],
        "caption training did not start from original zero timing",
    )
    pools, rewards = materials(plan, tokenizer)
    require_model(model, plan["model_plan"])
    require(plan == checked(path), "caption training changed its inputs before optimizer creation")
    selected = model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        selected.values(),
        lr=RECIPE["learning_rate"],
        betas=tuple(RECIPE["betas"]),
        eps=RECIPE["epsilon"],
        weight_decay=RECIPE["weight_decay"],
    )
    steps, exposures, epoch_signal = [], 0, {1: False, 2: False}
    for number, batch in enumerate(plan["batches"], start=1):
        original.guard(output)
        optimizer.zero_grad(set_to_none=True)
        samples = []
        for index in batch["indices"]:
            values = dataset.load(index)
            clip = clip_input(values)
            mask = torch.ones(1, len(clip.streams["pose_features"]), dtype=torch.bool)
            measured = backward_caption_timing_risk(
                model,
                clip,
                mask,
                candidate_labels(pools[index]["candidates"]),
                rewards["qualities"][index],
                alpha=RECIPE["risk_alpha"],
                caption_weight=RECIPE["caption_weight"],
                divisor=len(batch["indices"]),
            )
            timing = model.timing(clip.streams, valid_intervals=clip.valid_intervals)
            regularizer = (
                RECIPE["normalized_displacement_weight"] * timing["normalized_displacement_penalty"]
                + RECIPE["log_rate_smoothness_weight"] * timing["log_rate_smoothness_penalty"]
                + model.timing.rate_weights.sum() * 0
            )
            (regularizer / len(batch["indices"])).backward()
            samples.append(
                {"index": index, "risk": measured, "regularizer": float(regularizer.detach())}
            )
            exposures += 1
        gradient = gradient_report(model, require_signal=False)
        epoch_signal[batch["epoch"]] |= gradient["gradient_norm"] > 0
        torch.nn.utils.clip_grad_norm_(
            selected.values(), RECIPE["gradient_norm_limit"], error_if_nonfinite=True
        )
        before = model.timing.rate_weights.detach().tolist()
        optimizer.step()
        with torch.no_grad():
            model.timing.rate_weights.clamp_(
                -RECIPE["absolute_parameter_limit"], RECIPE["absolute_parameter_limit"]
            )
        model.require_fixed()
        require(
            torch.isfinite(model.timing.rate_weights).all(),
            "caption timing produced nonfinite values",
        )
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
                "after": model.timing.rate_weights.detach().tolist(),
            },
        )
        steps.append(artifact(target))
        print(
            json.dumps(
                {
                    "stage": "caption_timing_optimizer",
                    "step": number,
                    "of": 134,
                    "epoch": batch["epoch"],
                    "exposures": exposures,
                    "weights": model.timing.rate_weights.detach().tolist(),
                }
            ),
            flush=True,
        )
    require(
        exposures == 1064
        and len(steps) == 134
        and all(epoch_signal.values())
        and torch.count_nonzero(model.timing.rate_weights).item() > 0,
        "caption timing did not complete both full epochs with actual learning signal",
    )
    require_model(model, plan["model_plan"])
    require(
        materials(plan, tokenizer) == (pools, rewards) and plan == checked(path),
        "caption training source evidence changed before the final checkpoint",
    )
    optimizer.zero_grad(set_to_none=True)
    original.guard(output)
    final = output / "final-adapter.pt"
    with final.open("xb") as stream:
        torch.save(
            {
                "timing": model.timing.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epochs": 2,
                "updates": 134,
                "exposures": 1064,
                "plan": artifact(path),
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    write(
        output / "result.json",
        {
            "schema": "umi-studio-caption-supervised-timing-training/1",
            "complete": True,
            "plan": artifact(path),
            "worker": artifact(output / "worker.json"),
            "adapter": artifact(final),
            "steps": steps,
            "exposures": 1064,
            "optimizer_updates": 134,
            "completed_epochs": 2,
            "weights": model.timing.rate_weights.detach().tolist(),
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
        "caption training permits only one fresh complete optimizer execution",
    )
    original.guard(output)
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
