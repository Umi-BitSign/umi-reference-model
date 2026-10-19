"""Qualify native identity and actual graded-risk gradients of the timing adapter.

Requires the completed532-input audit before freezing or loading any C5 model.
Uses four fixed training cases and original C5 generation. No optimizer runs.
"""

from __future__ import annotations

import argparse
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
from c5_timing_feature_data import (
    checked as checked_data,
)
from launch_terra_visual_job import identity, now, write

POLICY = {
    "case_indices": [0, 1, 214, 215],
    "case_selection": "fixed full-native-reference training cases; no score-based selection",
    "all532_initial_feature_identities_required": True,
    "actual_memory_logits_and_native_generation_identity_cases": 4,
    "trainable_parameters": 4,
    "maximum_log_rate": 0.6931471805599453,
    "candidate_generation": "human anchor, reference-blind greedy and sampled, cyclic donor",
    "donor_offset": 2,
    "duplicate_donor_policy": "stop; no replacement or skipped case",
    "candidate_horizon": 512,
    "rollout_seed": 20261019,
    "risk_alpha": 0.005,
    "caption_weight": 0,
    "complete_grading_passes": 2,
    "gradient_scope": "every case in each complete authentic grading pass",
    "compare_single_encoder_against_original_streamed_risk": True,
    "gradient_relative_tolerance": 0.005,
    "gradient_absolute_tolerance": 1e-7,
    "CPU_threads": 2,
    "GPU_use": False,
    "optimizer_updates": 0,
    "maximum_output_bytes": 32 * 1024**2,
    "physical_free_floor_bytes": 84 * 1024**3,
    "quality_gain_established": False,
    "promotion_authorized": False,
}
SOURCES = (
    "probe_c5_timing_semantic.py",
    "c5_timing_feature_data.py",
    "c5_timing_semantic_model.py",
    "c5_timing_risk.py",
    "c5_monotone_feature_timing.py",
    "terra_visual_learning.py",
    "terra_teacher_candidates.py",
    "terra_quality_training.py",
    "terra_polarity_judge.py",
    "probe_roi_semantic_learning.py",
    "tiny_rgb_feature_winner_student.py",
    "evaluate_c5_words.py",
    "evaluate_roi_short_translation.py",
    "evaluate_tiny_rgb_feature_winner.py",
    "c5_terra_tail.py",
    "c5_winner_language_decoder.py",
    "tiny_rgb_winner_student.py",
    "launch_terra_visual_job.py",
    "train_terra_c5_tail.py",
    "c5_experiment_sources.py",
    "prepare_c5_timing_features.py",
)


def specification(audit_path):
    from terra_polarity_judge import protocol

    data = checked_data(audit_path)
    sources = dict(data["source_sha256"])
    for name in SOURCES:
        path = "tools/" + name
        require(path not in sources or sources[path] == sha(ROOT / path), "source conflict")
        sources[path] = sha(ROOT / path)
    return {
        "schema": "umi-studio-native224-timing-semantic-qualification-plan/1",
        "data_audit": artifact(audit_path),
        "model_plan": data["model_plan"],
        "policy": POLICY,
        "judge_protocol": protocol(),
        "records": [data["records"][i] for i in POLICY["case_indices"]],
        "source_sha256": sources,
        "optimizer_training_authorized": False,
        "quality_gain_established": False,
        "promotion_authorized": False,
    }


def checked(path):
    from terra_training_recovery import verify_snapshot

    plan = read(path)
    require(plan == specification(bound(plan["data_audit"])), "timing qualification plan changed")
    verify_snapshot(path, plan)
    return plan


def freeze(path, audit_path):
    from c5_experiment_sources import archive_sources

    owned(path, "runs")
    require(not path.parent.exists(), "timing qualification requires a fresh plan directory")
    plan = specification(audit_path)
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    return plan


def guard(output):
    import shutil

    from evaluate_tiny_rgb_feature_winner import peak_bytes

    files = list(output.rglob("*")) if output.exists() else []
    require(all(not p.is_symlink() for p in files), "timing qualification contains aliases")
    used = sum(p.stat().st_size for p in files if p.is_file())
    remaining = POLICY["maximum_output_bytes"] - used
    require(
        remaining >= 0
        and shutil.disk_usage(ROOT).free
        >= POLICY["physical_free_floor_bytes"] + 128 * 1024**2 + remaining
        and peak_bytes() <= 32 * 1024**3,
        "timing qualification exceeds storage or process memory allowance",
    )


def selected_values(dataset, plan):
    values = [dataset.load(i) for i in POLICY["case_indices"]]
    require(
        [v["row"] for v in values] == [r["row"] for r in plan["records"]]
        and len({(v["row"]["source"], v["row"]["example_id"]) for v in values}) == 4,
        "timing qualification lost its four complete training identities",
    )
    return values


def require_zero(model):
    import torch

    require(
        torch.count_nonzero(model.timing.rate_weights).item() == 0
        and model.timing.maximum_log_rate == POLICY["maximum_log_rate"],
        "zero-update timing qualification changed its identity initializer",
    )


def all_feature_identities(model, dataset):
    import torch
    from c5_timing_semantic_model import clip_input

    require_zero(model)
    with torch.no_grad():
        for index in range(len(dataset)):
            clip = clip_input(dataset.load(index))
            frames = len(clip.streams["pose_features"])
            features, _, _ = model.timed_features(clip, torch.ones(1, frames, dtype=torch.bool))
            require(
                all(
                    torch.equal(features[name][0].view(torch.int32), value.view(torch.int32))
                    for name, value in clip.streams.items()
                ),
                "actual original features lost bitwise timing identity",
            )
    return {"clips": len(dataset), "bitwise_identity_including_signed_zero": True}


def identity_check(model, tokenizer, values, model_plan):
    import torch
    from c5_terra_tail import candidate_labels
    from c5_timing_semantic_model import clip_input
    from evaluate_roi_short_translation import decode
    from evaluate_tiny_rgb_feature_winner import CpuBudget
    from transformers import StoppingCriteriaList
    from transformers.modeling_outputs import BaseModelOutput

    require_zero(model)
    clip = clip_input(values)
    frames = len(clip.streams["pose_features"])
    mask = torch.ones(1, frames, dtype=torch.bool)
    gold = tokenizer(values["row"]["reference"], truncation=False).input_ids
    anchor = {"tokens": gold, "truncated": False, "termination": "eos"}
    labels = candidate_labels([anchor])[0]
    with torch.no_grad():
        direct = model.semantic_encoder(
            **{k: v[None] for k, v in clip.streams.items()}, return_dict=True
        ).last_hidden_state
        adapted, actual_mask = model.encode(clip, mask)
        require(
            direct.shape == (1, frames, 1536)
            and torch.isfinite(direct).all()
            and torch.equal(direct.view(torch.int32), adapted.view(torch.int32))
            and torch.equal(actual_mask, mask),
            "actual C5 memory changed at initial timing identity",
        )
        expected = model.language(
            encoder_outputs=BaseModelOutput(last_hidden_state=direct),
            attention_mask=mask.long(),
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        actual = model(clip, mask, labels)["language"]
        require(
            torch.equal(expected.logits.view(torch.int32), actual.logits.view(torch.int32))
            and torch.equal(expected.loss.view(torch.int32), actual.loss.view(torch.int32)),
            "actual C5 teacher-forcing identity failed",
        )
        _, tokens = decode(model.language, tokenizer, model_plan["reference"]["generation"], direct)
        adapted_tokens = model.generate(
            clip,
            mask,
            **model_plan["reference"]["generation"],
            early_stopping=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([CpuBudget()]),
        )[0].tolist()
        require(tokens == adapted_tokens, "actual timing model changed native generation tokens")
    return {
        "row": values["row"],
        "frames": frames,
        "memory_bitwise_equal": True,
        "teacher_forcing_logits_and_loss_bitwise_equal": True,
        "native_generation_tokens_equal": True,
        "tokens": tokens,
        "generation": model_plan["reference"]["generation"],
    }


def donor_candidate(values, donor, tokenizer):
    require(values["row"]["reference"] != donor["row"]["reference"], "donor equals anchor")
    return {
        "tokens": tokenizer(donor["row"]["reference"], truncation=False).input_ids,
        "text": donor["row"]["reference"],
        "kind": "other_training_caption",
        "truncated": False,
        "termination": "eos",
        "donor_identity": {k: donor["row"][k] for k in ("source", "example_id")},
    }


def validate_pools(values, pools, tokenizer):
    from c5_terra_tail import candidate_labels

    require(len(values) == len(pools) == 4, "timing proof omitted a fixed candidate pool")
    for index, (value, pool) in enumerate(zip(values, pools, strict=True)):
        candidate_labels(pool)
        anchor = value["row"]["reference"]
        require(
            2 <= len(pool) <= 4
            and pool[0]
            == {
                "tokens": tokenizer(anchor, truncation=False).input_ids,
                "text": anchor,
                "kind": "human_anchor",
                "truncated": False,
                "termination": "eos",
            }
            and pool[-1]
            == donor_candidate(
                value, values[(index + POLICY["donor_offset"]) % len(values)], tokenizer
            )
            and all(c["kind"] in ("greedy", "sampled") for c in pool[1:-1])
            and len({c["kind"] for c in pool}) == len(pool),
            "timing proof changed actual anchor, observed rollout or fixed donor",
        )
        for candidate in pool:
            require(
                tokenizer.decode(
                    candidate["tokens"],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                == candidate["text"],
                "timing likelihood tokens differ from judged text",
            )


def prepare_pools(model, tokenizer, values):
    import torch
    from c5_timing_semantic_model import clip_input
    from terra_teacher_candidates import TrainingRolloutBudget
    from terra_visual_learning import candidate_pool

    pools = []
    for index, value in enumerate(values):
        clip = clip_input(value)
        mask = torch.ones(1, len(clip.streams["pose_features"]), dtype=torch.bool)
        pool = candidate_pool(
            model,
            tokenizer,
            clip,
            mask,
            value["row"]["reference"],
            seed=POLICY["rollout_seed"] + index,
            maximum_tokens=POLICY["candidate_horizon"],
            budget_factory=TrainingRolloutBudget,
        )
        donor = donor_candidate(
            value, values[(index + POLICY["donor_offset"]) % len(values)], tokenizer
        )
        require(
            all(c["tokens"] != donor["tokens"] for c in pool),
            "fixed timing donor duplicates rollout; no automatic substitution",
        )
        pools.append([*pool, donor])
    validate_pools(values, pools, tokenizer)
    return pools


def paired_inputs(values, pools):
    from semantic_quality_judge import calibration_controls

    items, known, mappings = [], {}, []
    for value, pool in zip(values, pools, strict=True):
        indices = []
        for candidate in pool:
            key = value["row"]["reference"], candidate["text"]
            if key not in known:
                known[key] = len(items)
                items.append({"reference": key[0], "candidate": key[1]})
            indices.append(known[key])
        mappings.append({"row": value["row"], "indices": indices})
    controls = []
    for control in calibration_controls():
        controls.append({"index": len(items), "range": control["range"]})
        items.append({"reference": control["reference"], "candidate": control["candidate"]})
    return {"items": items, "cases": mappings, "controls": controls}


def grading_requests(paired):
    requests = []
    for number in (1, 2):
        indices = list(range(len(paired["items"])))
        random.Random(POLICY["rollout_seed"] + number).shuffle(indices)
        for offset in range(0, len(indices), 16):
            requests.append({"pass": number, "indices": indices[offset : offset + 16]})
    return requests


def authenticated_passes(paired, directory):
    from probe_roi_semantic_learning import qualities
    from terra_polarity_judge import authenticated_scores

    request_path = directory / "requests.json"
    requests = grading_requests(paired)
    require(read(request_path) == requests, "timing proof changed blinded grading order")
    passes = {1: [None] * len(paired["items"]), 2: [None] * len(paired["items"])}
    raw = [artifact(request_path)]
    for index, request in enumerate(requests):
        selected = [paired["items"][i] for i in request["indices"]]
        sub = directory / f"request-{index:03d}"
        scores = authenticated_scores(selected, sub)
        for position, score in zip(request["indices"], scores, strict=True):
            require(passes[request["pass"]][position] is None, "duplicate timing grade")
            passes[request["pass"]][position] = score
        raw.extend(
            artifact(sub / name)
            for name in ("request.json", "started.json", "response.json", "scores.json")
        )
    require(not list(directory.rglob("failure.json")), "timing proof has failed grading")
    expected = {request_path} | {ROOT / item["path"] for item in raw}
    require(
        {p for p in directory.rglob("*") if p.is_file()} == expected,
        "timing proof grading has extra or incomplete requests",
    )
    result = []
    for number, scores in passes.items():
        result.append({"pass": number, "scores": scores, "qualities": qualities(paired, scores)})
    return result, raw


def actual_gradients(model, values, pools, judged, *, output=None):
    import torch
    from c5_terra_tail import candidate_labels
    from c5_timing_risk import backward_timing_risk
    from c5_timing_semantic_model import clip_input, gradient_report
    from terra_visual_learning import backward_candidate_risk

    result = []
    for grading in judged:
        family_signal = {"word": False, "phrase": False}
        for index, (value, pool, quality) in enumerate(
            zip(values, pools, grading["qualities"], strict=True)
        ):
            if output is not None:
                guard(output)
            clip = clip_input(value)
            mask = torch.ones(1, len(clip.streams["pose_features"]), dtype=torch.bool)
            model.zero_grad(set_to_none=True)
            measured = backward_candidate_risk(
                model,
                clip,
                mask,
                candidate_labels(pool),
                quality,
                alpha=POLICY["risk_alpha"],
                caption_weight=0,
            )
            gradients = gradient_report(model, require_signal=False)
            original_gradient = model.timing.rate_weights.grad.detach().clone()
            model.zero_grad(set_to_none=True)
            single_encoder = backward_timing_risk(
                model, clip, mask, candidate_labels(pool), quality, alpha=POLICY["risk_alpha"]
            )
            single_gradient = gradient_report(model, require_signal=False)
            require(
                torch.allclose(
                    model.timing.rate_weights.grad,
                    original_gradient,
                    rtol=POLICY["gradient_relative_tolerance"],
                    atol=POLICY["gradient_absolute_tolerance"],
                )
                and abs(single_encoder["loss"] - measured["loss"]) <= 1e-5,
                "actual single-encoder timing risk disagrees with original streamed gradient",
            )
            difference = model.timing.rate_weights.grad - original_gradient
            family = "word" if index < 2 else "phrase"
            family_signal[family] |= gradients["gradient_norm"] > 0
            require(measured["terra_gradient_coefficient_l1"] > 0, "no graded-risk contrast")
            result.append(
                {
                    "pass": grading["pass"],
                    "case": index,
                    "row": value["row"],
                    "actual_grades": quality,
                    "gradients": gradients,
                    "single_encoder_gradients": single_gradient,
                    "single_encoder_risk": single_encoder,
                    "maximum_absolute_gradient_difference": float(difference.abs().max()),
                    "risk": {k: v for k, v in measured.items() if k != "gold_logits"},
                }
            )
            print(json.dumps({"stage": "actual_timing_risk", **result[-1]}), flush=True)
        require(all(family_signal.values()), "timing proof lacks word or phrase risk gradients")
    model.zero_grad(set_to_none=True)
    require_zero(model)
    return result


def qualify(path, output, plan):
    from c5_timing_semantic_model import load_student, require_model
    from terra_polarity_judge import Pacer

    dataset = TimingFeatureDataset(bound(plan["data_audit"]))
    values = selected_values(dataset, plan)
    model, tokenizer = load_student(plan["model_plan"])
    guard(output)
    feature_identity = all_feature_identities(model, dataset)
    write(output / "feature-identity.json", feature_identity)
    identities = []
    for index, value in enumerate(values):
        guard(output)
        report = identity_check(model, tokenizer, value, plan["model_plan"])
        write(output / f"identity-{index:02d}.json", report)
        identities.append(artifact(output / f"identity-{index:02d}.json"))
        print(json.dumps({"stage": "actual_timing_native_identity", "case": index}), flush=True)
    pools = prepare_pools(model, tokenizer, values)
    write(output / "pools.json", pools)
    paired = paired_inputs(values, pools)
    write(output / "comparison.json", paired)
    semantic = output / "semantic"
    semantic.mkdir()
    requests = grading_requests(paired)
    write(semantic / "requests.json", requests)
    pacer = Pacer()
    for index, request in enumerate(requests):
        guard(output)
        pacer.score(
            [paired["items"][i] for i in request["indices"]], semantic / f"request-{index:03d}"
        )
    judged, raw = authenticated_passes(paired, semantic)
    validate_pools(values, read(output / "pools.json"), tokenizer)
    gradients = actual_gradients(model, values, pools, judged, output=output)
    require_model(model, plan["model_plan"])
    require(plan == checked(path), "timing proof sources/data changed during execution")
    guard(output)
    write(
        output / "qualification.json",
        {
            "schema": "umi-studio-native224-timing-semantic-qualification/1",
            "complete": True,
            "plan": artifact(path),
            "worker": artifact(output / "worker.json"),
            "feature_identity": artifact(output / "feature-identity.json"),
            "native_identities": identities,
            "pools": artifact(output / "pools.json"),
            "comparison": artifact(output / "comparison.json"),
            "grading": judged,
            "raw_grading_evidence": raw,
            "gradients": gradients,
            "complete_grading_passes": 2,
            "new_Terra_requests": len(requests),
            "original_model_values_unchanged": True,
            "optimizer_updates": 0,
            "CPU_threads": 2,
            "GPU_use": False,
            "training_only": True,
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
    require(not output.exists(), "timing qualification requires a fresh output")
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
                "automatic_retries": 0,
            },
        )
        with gpu_lease():
            qualify(path, output, plan)
        code = 0
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "audit", "plan", "output"):
        parser.add_argument("--" + name, type=lambda value: Path(value).absolute())
    args = parser.parse_args()
    if args.freeze and args.audit and not args.plan:
        freeze(args.freeze, args.audit)
        if args.output:
            run(args.freeze, args.output)
    elif args.plan and args.output and not args.freeze and not args.audit:
        run(args.plan, args.output)
    else:
        parser.error("freeze from complete actual data audit, or run a frozen plan once")


if __name__ == "__main__":
    main()
