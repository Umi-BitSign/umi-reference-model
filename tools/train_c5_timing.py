"""Train only four timing weights after actual native C5 risk qualification.

One fixed public-initialized pool per complete training clip, fresh calibrated
grades, two full epochs and one final adapter. No development selection or API
retry. Native appearance, sequence and language values remain frozen.
"""

from __future__ import annotations

import argparse
import json
import math
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
    terminal,
)
from launch_terra_visual_job import identity, now, write

RECIPE = {
    "epochs": 2,
    "clips_per_epoch": 532,
    "batch_size": 8,
    "planned_exposures": 1064,
    "planned_optimizer_updates": 134,
    "shuffle_seed": 20261019,
    "optimizer": "AdamW",
    "learning_rate": 0.005,
    "betas": [0.9, 0.999],
    "epsilon": 1e-8,
    "weight_decay": 0.01,
    "gradient_norm_limit": 1.0,
    "absolute_parameter_limit": 0.5,
    "risk_alpha": 0.005,
    "caption_weight": 0,
    "normalized_displacement_weight": 1.0,
    "log_rate_smoothness_weight": 0.01,
    "candidate_horizon": 512,
    "candidate_pool_policy": "one fixed original-public pool per clip before any update",
    "candidate_generation": "human anchor, reference-blind greedy/sample, next distinct caption",
    "donor_collision_policy": "deduplicate tokens; preserve donor provenance and actual grade",
    "rollout_seed": 20261019,
    "grading": "one fresh grade per distinct pair; unchanged four controls in every request",
    "maximum_noncontrol_pairs_per_request": 12,
    "automatic_retries": 0,
    "CPU_threads": 2,
    "GPU_use": False,
    "zero_support_clips": "retain complete cases; zero timing derivative without visible support",
    "selection": "fixed final after both complete epochs; no development checkpoint selection",
    "maximum_output_bytes": 64 * 1024**2,
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


def qualification(plan_path, result_path, poll_path):
    import probe_c5_timing_semantic as proof

    plan = proof.checked(plan_path)
    result = read(result_path)
    authority = terminal(result_path.parent)
    poll, worker = read(poll_path), read(bound(authority["worker.json"]))
    argv = worker["argv"]
    plan_flag = "--plan" if "--plan" in argv else "--freeze"
    require(
        Path(argv[1]).absolute() == ROOT / "tools/probe_c5_timing_semantic.py"
        and Path(argv[argv.index(plan_flag) + 1]).absolute() == plan_path
        and Path(argv[argv.index("--output") + 1]).absolute() == result_path.parent,
        "timing training qualification worker changed its actual program or paths",
    )
    require(
        poll.get("tool") == "write_stdin"
        and type(poll.get("requested_session_id")) is int
        and poll.get("actual_owned_pid") == worker["pid"]
        and type(poll["tool_result"].get("exit_code")) is int
        and poll["tool_result"]["exit_code"] == 0
        and poll["tool_result"].get("session_id") is None,
        "timing training requires actual successful terminal qualification tool evidence",
    )
    require(
        result_path.name == "qualification.json"
        and result["schema"] == "umi-studio-native224-timing-semantic-qualification/1"
        and result["complete"] is True
        and result["plan"] == artifact(plan_path)
        and result["worker"] == authority["worker.json"]
        and result["complete_grading_passes"] == 2
        and result["optimizer_updates"] == 0
        and result["original_model_values_unchanged"] is True
        and result["training_only"] is True
        and result["quality_gain_established"] is False
        and result["promotion_authorized"] is False,
        "timing training requires complete actual zero-update C5 qualification",
    )
    pairs = read(bound(result["comparison"]))
    judged, raw = proof.authenticated_passes(pairs, result_path.parent / "semantic")
    require(
        judged == result["grading"] and raw == result["raw_grading_evidence"], "risk grades changed"
    )
    require(
        read(bound(result["feature_identity"]))
        == {"clips": 532, "bitwise_identity_including_signed_zero": True}
        and len(result["native_identities"]) == 4
        and len(result["gradients"]) == 8,
        "timing qualification omitted complete feature identity or actual fixed cases",
    )
    bound(result["pools"])
    for index, descriptor in enumerate(result["native_identities"]):
        value = read(bound(descriptor))
        require(
            value["row"] == plan["records"][index]["row"]
            and value["memory_bitwise_equal"] is True
            and value["teacher_forcing_logits_and_loss_bitwise_equal"] is True
            and value["native_generation_tokens_equal"] is True
            and value["generation"] == plan["model_plan"]["reference"]["generation"],
            "timing qualification lost actual native identity",
        )
    for number in (1, 2):
        signal = {"word": False, "phrase": False}
        for index in range(4):
            value = result["gradients"][(number - 1) * 4 + index]
            require(
                value["pass"] == number
                and value["case"] == index
                and value["row"] == plan["records"][index]["row"]
                and value["actual_grades"] == judged[number - 1]["qualities"][index],
                "timing qualification gradient/input/reward join changed",
            )
            gradient = value["single_encoder_gradients"]
            original = value["gradients"]
            require(
                gradient["parameters"] == 4
                and len(gradient["gradient_values"]) == 4
                and all(math.isfinite(x) for x in gradient["gradient_values"])
                and math.isfinite(gradient["gradient_norm"])
                and gradient["gradient_norm"] >= 0
                and gradient["frozen_backbone_gradients_absent"] is True
                and value["single_encoder_risk"]["differentiable_encoder_calls"] == 1
                and math.isfinite(value["maximum_absolute_gradient_difference"]),
                "timing qualification lost finite actual single-encoder gradients",
            )
            require(
                len(original["gradient_values"]) == 4
                and all(math.isfinite(v) for v in original["gradient_values"])
                and original["frozen_backbone_gradients_absent"] is True
                and all(
                    abs(a - b)
                    <= proof.POLICY["gradient_absolute_tolerance"]
                    + proof.POLICY["gradient_relative_tolerance"] * abs(b)
                    for a, b in zip(
                        gradient["gradient_values"], original["gradient_values"], strict=True
                    )
                )
                and math.isclose(
                    gradient["gradient_norm"],
                    math.sqrt(sum(v * v for v in gradient["gradient_values"])),
                    rel_tol=1e-5,
                    abs_tol=1e-8,
                )
                and abs(value["single_encoder_risk"]["loss"] - value["risk"]["loss"]) <= 1e-5,
                "timing qualification single-encoder comparison failed numerical recheck",
            )
            signal["word" if index < 2 else "phrase"] |= gradient["gradient_norm"] > 0
        require(all(signal.values()), "timing qualification lacks word or phrase learning signal")
    return plan, result, authority


def specification(qualification_plan, qualification_result, poll):
    import probe_c5_timing_semantic as proof
    from c5_timing_feature_data import checked as checked_data

    plan, _, authority = qualification(qualification_plan, qualification_result, poll)
    data = checked_data(bound(plan["data_audit"]))
    schedule = batches()
    require(
        len(schedule) == RECIPE["planned_optimizer_updates"]
        and sum(len(b["indices"]) for b in schedule) == RECIPE["planned_exposures"],
        "timing recipe no longer covers both complete epochs",
    )
    sources = dict(plan["source_sha256"])
    sources["tools/train_c5_timing.py"] = sha(Path(__file__))
    return {
        "schema": "umi-studio-four-parameter-timing-training-plan/1",
        "qualification_plan": artifact(qualification_plan),
        "qualification_result": artifact(qualification_result),
        "qualification_terminal_poll": artifact(poll),
        "qualification_authority": authority,
        "data_audit": plan["data_audit"],
        "model_plan": plan["model_plan"],
        "judge_protocol": plan["judge_protocol"],
        "recipe": RECIPE,
        "maximum_log_rate": proof.POLICY["maximum_log_rate"],
        "records": data["records"],
        "batches": schedule,
        "source_sha256": sources,
        "initialization": "original C5 plus exactly zero timing weights; no optimizer inheritance",
        "promotion_authorized": False,
    }


def checked(path):
    from terra_training_recovery import verify_snapshot

    plan = read(path)
    require(
        plan
        == specification(
            bound(plan["qualification_plan"]),
            bound(plan["qualification_result"]),
            bound(plan["qualification_terminal_poll"]),
        ),
        "timing training plan changed its actual qualification, data or full recipe",
    )
    verify_snapshot(path, plan)
    return plan


def freeze(path, qualification_plan, qualification_result, poll):
    from c5_experiment_sources import archive_sources

    owned(path, "runs")
    require(not path.parent.exists(), "timing training requires a fresh plan directory")
    plan = specification(qualification_plan, qualification_result, poll)
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    return plan


def guard(output):
    import shutil

    from evaluate_tiny_rgb_feature_winner import peak_bytes

    paths = list(output.rglob("*")) if output.exists() else []
    require(all(not p.is_symlink() for p in paths), "timing training output contains aliases")
    used = sum(p.stat().st_size for p in paths if p.is_file())
    remaining = RECIPE["maximum_output_bytes"] - used
    require(
        remaining >= 0
        and shutil.disk_usage(ROOT).free
        >= RECIPE["physical_free_floor_bytes"] + 128 * 1024**2 + remaining
        and peak_bytes() <= 32 * 1024**3,
        "timing training storage or process memory allowance exceeded",
    )


def donor_indices(records):
    references = [record["row"]["reference"] for record in records]
    require(len(records) == RECIPE["clips_per_epoch"], "timing training cannot omit cases")
    result = []
    for index, reference in enumerate(references):
        selected = next(
            (
                (index + offset) % len(records)
                for offset in range(1, len(records))
                if references[(index + offset) % len(records)] != reference
            ),
            None,
        )
        require(selected is not None, "timing training lacks a distinct training-caption donor")
        result.append(selected)
    return result


def validate_pool(record, donor_record, saved, tokenizer):
    from c5_terra_tail import candidate_labels

    row, other = record["row"], donor_record["row"]
    pool, donor = saved["candidates"], saved["donor"]
    require(
        row["role"] == other["role"] == "training"
        and row["training_allowed"] is other["training_allowed"] is True
        and row["reference"] != other["reference"]
        and saved["row"] == row
        and donor
        == {
            "tokens": tokenizer(other["reference"], truncation=False).input_ids,
            "text": other["reference"],
            "kind": "other_training_caption",
            "truncated": False,
            "termination": "eos",
            "donor_features": donor_record["features"],
        }
        and 2 <= len(pool) <= 4
        and pool[0]
        == {
            "tokens": tokenizer(row["reference"], truncation=False).input_ids,
            "text": row["reference"],
            "kind": "human_anchor",
            "truncated": False,
            "termination": "eos",
        },
        "timing training pool lost its exact approved anchor or predeclared donor",
    )
    candidate_labels(pool)
    duplicate = saved["donor_duplicate_index"]
    if duplicate is None:
        require(pool[-1] == donor, "missing distinct timing donor")
        rollouts = pool[1:-1]
    else:
        require(
            type(duplicate) is int
            and 0 < duplicate < len(pool)
            and pool[duplicate]["tokens"] == donor["tokens"]
            and pool[duplicate]["text"] == donor["text"],
            "timing donor deduplication changed observed tokens or source caption",
        )
        rollouts = pool[1:]
    require(
        all(c["kind"] in ("greedy", "sampled") for c in rollouts)
        and len({c["kind"] for c in rollouts}) == len(rollouts),
        "timing training changed observed rollout provenance",
    )
    for candidate in pool:
        require(
            tokenizer.decode(
                candidate["tokens"], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            == candidate["text"],
            "timing training judged text differs from likelihood tokens",
        )


def prepare_pools(model, tokenizer, dataset, plan, output):
    import torch
    from c5_timing_semantic_model import clip_input
    from terra_teacher_candidates import TrainingRolloutBudget
    from terra_visual_learning import candidate_pool

    donors = donor_indices(plan["records"])
    directory = output / "pools"
    directory.mkdir()
    pools = []
    for index, record in enumerate(plan["records"]):
        guard(output)
        values = dataset.load(index)
        clip = clip_input(values)
        mask = torch.ones(1, len(clip.streams["pose_features"]), dtype=torch.bool)
        pool = candidate_pool(
            model,
            tokenizer,
            clip,
            mask,
            values["row"]["reference"],
            seed=RECIPE["rollout_seed"] + index,
            maximum_tokens=RECIPE["candidate_horizon"],
            budget_factory=TrainingRolloutBudget,
        )
        other = plan["records"][donors[index]]
        donor = {
            "tokens": tokenizer(other["row"]["reference"], truncation=False).input_ids,
            "text": other["row"]["reference"],
            "kind": "other_training_caption",
            "truncated": False,
            "termination": "eos",
            "donor_features": other["features"],
        }
        duplicate = next((i for i, c in enumerate(pool) if c["tokens"] == donor["tokens"]), None)
        saved = {
            "row": values["row"],
            "candidates": pool if duplicate is not None else [*pool, donor],
            "donor": donor,
            "donor_duplicate_index": duplicate,
        }
        validate_pool(record, other, saved, tokenizer)
        write(directory / f"clip-{index:04d}.json", saved)
        pools.append(saved)
        print(
            json.dumps({"stage": "timing_training_pools", "clips": index + 1, "of": 532}),
            flush=True,
        )
    return pools


def comparison(records, pools):
    from semantic_quality_judge import calibration_controls

    require(len(records) == len(pools) == 532, "timing rewards omitted an approved training clip")
    items, known, mappings = [], {}, []
    for record, pool in zip(records, pools, strict=True):
        indices = []
        for candidate in pool["candidates"]:
            key = record["row"]["reference"], candidate["text"]
            if key not in known:
                known[key] = len(items)
                items.append({"reference": key[0], "candidate": key[1]})
            indices.append(known[key])
        mappings.append({"features": record["features"], "indices": indices})
    controls = calibration_controls()
    order = list(range(len(items)))
    random.Random(RECIPE["shuffle_seed"]).shuffle(order)
    requests = []
    for offset in range(0, len(order), 12):
        indices = order[offset : offset + 12]
        requests.append(
            {
                "indices": indices,
                "rows": [items[i] for i in indices]
                + [{"reference": c["reference"], "candidate": c["candidate"]} for c in controls],
            }
        )
    return {"items": items, "cases": mappings, "controls": controls, "requests": requests}


def authenticated_rewards(paired, directory):
    from terra_polarity_judge import authenticated_scores

    scores, controls = [None] * len(paired["items"]), []
    raw = []
    for index, request in enumerate(paired["requests"]):
        sub = directory / f"request-{index:03d}"
        actual = authenticated_scores(request["rows"], sub)
        control_values = actual[-4:]
        require(
            all(
                c["range"][0] <= value <= c["range"][1]
                for c, value in zip(paired["controls"], control_values, strict=True)
            ),
            "timing training judge failed the original controls; no update permitted",
        )
        controls.append(control_values)
        for position, value in zip(request["indices"], actual[:-4], strict=True):
            require(scores[position] is None, "timing training score repeated a noncontrol pair")
            scores[position] = value
        raw.extend(
            artifact(sub / name)
            for name in ("request.json", "started.json", "response.json", "scores.json")
        )
    require(
        all(type(value) is int and 0 <= value <= 10 for value in scores)
        and not list(directory.rglob("failure.json"))
        and {p for p in directory.rglob("*") if p.is_file()} == {ROOT / v["path"] for v in raw},
        "timing training scoring is incomplete or has unbound evidence",
    )
    qualities = [[scores[i] for i in case["indices"]] for case in paired["cases"]]
    require(
        all(q and q[0] == 10 for q in qualities), "timing training judge rejected an exact anchor"
    )
    return {"scores": scores, "qualities": qualities, "controls": controls, "raw": raw}


def train(path, output, plan):
    import torch
    from c5_terra_tail import candidate_labels
    from c5_timing_risk import backward_timing_risk
    from c5_timing_semantic_model import clip_input, gradient_report, load_student, require_model
    from terra_polarity_judge import Pacer

    dataset = TimingFeatureDataset(bound(plan["data_audit"]))
    model, tokenizer = load_student(plan["model_plan"])
    require(
        torch.count_nonzero(model.timing.rate_weights) == 0,
        "timing training did not start at identity",
    )
    pools = prepare_pools(model, tokenizer, dataset, plan, output)
    paired = comparison(plan["records"], pools)
    write(output / "comparison.json", paired)
    semantic = output / "semantic"
    semantic.mkdir()
    pacer = Pacer()
    for index, request in enumerate(paired["requests"]):
        guard(output)
        pacer.score(request["rows"], semantic / f"request-{index:03d}")
    rewards = authenticated_rewards(paired, semantic)
    write(output / "rewards.json", rewards)
    donors = donor_indices(plan["records"])
    for index, pool in enumerate(pools):
        require(
            pool == read(output / f"pools/clip-{index:04d}.json"), "actual rollout receipt changed"
        )
        validate_pool(plan["records"][index], plan["records"][donors[index]], pool, tokenizer)
    require_model(model, plan["model_plan"])
    require(read(output / "comparison.json") == paired, "timing training comparison bytes changed")
    require(
        plan == checked(path), "timing training changed frozen inputs before optimizer creation"
    )
    optimizer = torch.optim.AdamW(
        model.trainable_parameters().values(),
        lr=RECIPE["learning_rate"],
        betas=tuple(RECIPE["betas"]),
        eps=RECIPE["epsilon"],
        weight_decay=RECIPE["weight_decay"],
    )
    steps, exposures, epoch_signal = [], 0, {1: False, 2: False}
    for number, batch in enumerate(plan["batches"], start=1):
        guard(output)
        optimizer.zero_grad(set_to_none=True)
        samples = []
        for index in batch["indices"]:
            values = dataset.load(index)
            clip = clip_input(values)
            mask = torch.ones(1, len(clip.streams["pose_features"]), dtype=torch.bool)
            measured = backward_timing_risk(
                model,
                clip,
                mask,
                candidate_labels(pools[index]["candidates"]),
                rewards["qualities"][index],
                alpha=RECIPE["risk_alpha"],
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
            model.trainable_parameters().values(),
            RECIPE["gradient_norm_limit"],
            error_if_nonfinite=True,
        )
        before = model.timing.rate_weights.detach().clone()
        optimizer.step()
        with torch.no_grad():
            model.timing.rate_weights.clamp_(
                -RECIPE["absolute_parameter_limit"], RECIPE["absolute_parameter_limit"]
            )
        model.require_fixed()
        require(
            torch.isfinite(model.timing.rate_weights).all(),
            "timing update produced nonfinite weights",
        )
        receipt = output / f"step-{number:04d}.json"
        write(
            receipt,
            {
                "step": number,
                "batch": batch,
                "exposures": exposures,
                "samples": samples,
                "gradient": gradient,
                "before": before.tolist(),
                "after": model.timing.rate_weights.detach().tolist(),
            },
        )
        steps.append(artifact(receipt))
        print(
            json.dumps(
                {
                    "stage": "timing_optimizer",
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
        exposures == RECIPE["planned_exposures"]
        and len(steps) == RECIPE["planned_optimizer_updates"]
        and all(epoch_signal.values())
        and torch.count_nonzero(model.timing.rate_weights) > 0,
        "timing training did not complete both full epochs with actual learning signal",
    )
    require_model(model, plan["model_plan"])
    require(
        read(output / "comparison.json") == paired
        and read(output / "rewards.json") == rewards == authenticated_rewards(paired, semantic)
        and all(
            pool == read(output / f"pools/clip-{index:04d}.json")
            for index, pool in enumerate(pools)
        ),
        "timing training final evidence changed its actual rollouts or rewards",
    )
    require(
        plan == checked(path), "timing training inputs or sources changed during optimizer work"
    )
    optimizer.zero_grad(set_to_none=True)
    guard(output)
    final = output / "final-adapter.pt"
    with final.open("xb") as stream:
        torch.save(
            {
                "timing": model.timing.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epochs": 2,
                "updates": len(steps),
                "exposures": exposures,
                "plan": artifact(path),
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    write(
        output / "result.json",
        {
            "schema": "umi-studio-four-parameter-timing-training/1",
            "complete": True,
            "plan": artifact(path),
            "worker": artifact(output / "worker.json"),
            "adapter": artifact(final),
            "steps": steps,
            "exposures": exposures,
            "optimizer_updates": len(steps),
            "completed_epochs": 2,
            "weights": model.timing.rate_weights.detach().tolist(),
            "rewards": artifact(output / "rewards.json"),
            "comparison": artifact(output / "comparison.json"),
            "original_backbones_unchanged": True,
            "training_clips": 532,
            "new_Terra_requests": len(paired["requests"]),
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
    require(not output.exists(), "timing training needs a fresh output")
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
        write(bound(plan["qualification_plan"]).with_name("optimizer-child-claim.json"), claim)
        write(path.with_name("execution-claim.json"), claim)
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
    ):
        parser.add_argument("--" + name, type=lambda p: Path(p).absolute())
    args = parser.parse_args()
    if (
        args.freeze
        and all(
            (args.qualification_plan, args.qualification_result, args.qualification_terminal_poll)
        )
        and not args.plan
    ):
        freeze(
            args.freeze,
            args.qualification_plan,
            args.qualification_result,
            args.qualification_terminal_poll,
        )
        if args.output:
            run(args.freeze, args.output)
    elif args.plan and args.output and not args.freeze:
        run(args.plan, args.output)
    else:
        parser.error("freeze after complete actual timing qualification, or run a frozen plan once")


if __name__ == "__main__":
    main()
