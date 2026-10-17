"""Complete word-training schedules, authentic Terra rewards and bounded checkpoints."""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import torch
from benchmark_fleurs_dev import sha256
from c5_terra_tail import candidate_labels
from c5_word_semantic_model import PARAMETERS, TENSORS
from launch_terra_visual_job import write
from probe_jeff_terra_quality import terra_payload
from semantic_quality_judge import calibration_controls, parse_terra_scores
from terra_training_recovery import artifact
from terra_word_semantic_protocol import training_case, training_negative
from terra_word_verified_inputs import expected_header

ROOT = Path(__file__).resolve().parents[1]
RECIPE = {
    "epochs": 4,
    "batch_clips": 4,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "risk_alpha": 0.005,
    "caption_weight": 0,
    "seed": 20261017,
    "candidate_horizon": 512,
    "candidate_generation": "exact human anchor plus reference-blind greedy and sampled policy",
    "donor": "first cyclic different-label training identity in complete original inventory",
    "duplicate_donor": "retain existing identical rollout tokens; no replacement or duplicate",
    "reward": "actual calibrated Terra-medium integer0-10; minimize expected10-quality",
    "zero_contrast": "record true expected error; skip optimizer only if whole batch is tied",
    "augmentation": "none",
    "checkpoint_selection": "fixed final word-stage checkpoint; epoch checkpoints recovery only",
    "later_curriculum": "retain all214 easier word clips in separately qualified phrase stages",
    "maximum_pairs_per_request": 16,
    "minimum_request_start_gap_seconds": 60,
    "own_requests_in_flight": 1,
    "automatic_retries": 0,
    "automatic_continuations": 0,
    "WER_is_diagnostic_only": True,
    "new_downloads": 0,
    "promotion_authorized": False,
}
STORAGE = {
    "physical_floor_bytes": 80 * 1024**3,
    "output_budget_bytes": 4 * 1024**3,
    "headroom_bytes": 128 * 1024**2,
    "checkpoint_pair_reserve_bytes": PARAMETERS * 12 + 1024**2,
}


def training_records(binding):
    records = [
        case
        for case in binding["records"]
        if case["record"]["original"]["row"]["role"] == "training"
    ]
    identities, signers = set(), defaultdict(set)
    for case in records:
        row = training_case(case)
        if row["example_id"] in identities:
            raise ValueError("word training cannot repeat a complete input identity")
        identities.add(row["example_id"])
        signers[row["label"]].add(row["participant"])
    if (
        len(records) != 214
        or len(signers) != 60
        or min(map(len, signers.values())) < 2
        or sum(expected_header(case["record"])["sampled_frames"] for case in records) != 12147
    ):
        raise ValueError("word training changed the complete214-clip/60-label input inventory")
    return records


def donor_indices(records):
    rows = [training_case(case) for case in records]
    result = []
    for index, row in enumerate(rows):
        for offset in range(1, len(rows)):
            other = (index + offset) % len(rows)
            if rows[other]["label"] != row["label"]:
                result.append(other)
                break
        else:
            raise ValueError("word training lacks a different-label donor")
    return result


def batches_for(records):
    if not records:
        raise ValueError("word training needs a complete nonempty inventory")
    for case in records:
        training_case(case)
    batches = []
    for epoch in range(RECIPE["epochs"]):
        order = list(range(len(records)))
        random.Random(RECIPE["seed"] + epoch).shuffle(order)
        for offset in range(0, len(order), RECIPE["batch_clips"]):
            batches.append(
                {
                    "epoch": epoch,
                    "training_record_indices": order[offset : offset + RECIPE["batch_clips"]],
                }
            )
    return batches


def verify_coverage(plan, observed_order, examples, saved):
    expected = [i for batch in plan["batches"] for i in batch["training_record_indices"]]
    if (
        observed_order != expected
        or examples != plan["planned_training_examples"]
        or Counter(observed_order)
        != Counter({i: RECIPE["epochs"] for i in range(len(plan["records"]))})
        or not saved
    ):
        raise ValueError("word final checkpoint lacks the complete fixed curriculum")


def validate_pool(case, donor, pool, tokenizer):
    row = training_case(case)
    candidate_labels(pool)
    negative = training_negative(case, donor, tokenizer)
    if (
        not 2 <= len(pool) <= 4
        or pool[0]["kind"] != "human_anchor"
        or pool[0]["text"] != row["reference"]
        or pool[0]["tokens"] != tokenizer(row["reference"], truncation=False).input_ids
        or negative["tokens"] not in [candidate["tokens"] for candidate in pool]
        or len({candidate["kind"] for candidate in pool}) != len(pool)
        or any(
            candidate["kind"] not in ("greedy", "sampled", "other_training_caption")
            for candidate in pool[1:]
        )
    ):
        raise ValueError("word training pool changed its anchor, rollouts or fixed donor")
    for candidate in pool:
        if (
            candidate["kind"] == "other_training_caption" and candidate != negative
        ) or tokenizer.decode(
            candidate["tokens"], skip_special_tokens=True, clean_up_tokenization_spaces=False
        ) != candidate["text"]:
            raise ValueError("word reward text or donor differs from actual likelihood tokens")


def paired_inputs(records, donors, pools, tokenizer):
    if not records or not len(records) == len(donors) == len(pools):
        raise ValueError("word rewards need complete actual batch/donor/pool coverage")
    items, known, cases = [], {}, []
    for case, donor, pool in zip(records, donors, pools, strict=True):
        validate_pool(case, donor, pool, tokenizer)
        indices = []
        for candidate in pool:
            key = training_case(case)["reference"], candidate["text"]
            if key not in known:
                known[key] = len(items)
                items.append({"reference": key[0], "candidate": key[1]})
            indices.append(known[key])
        cases.append(
            {
                "word_input_receipt": case["receipt"],
                "donor_input_receipt": donor["receipt"],
                "indices": indices,
            }
        )
    controls = []
    for control in calibration_controls():
        controls.append({"index": len(items), "range": control["range"]})
        items.append({"reference": control["reference"], "candidate": control["candidate"]})
    return {"items": items, "cases": cases, "controls": controls}


def rewards_for(paired, scores):
    if (
        len(scores) != len(paired["items"])
        or any(type(value) is not int or not 0 <= value <= 10 for value in scores)
        or any(
            not control["range"][0] <= scores[control["index"]] <= control["range"][1]
            for control in paired["controls"]
        )
    ):
        raise ValueError("word judge failed complete integer rewards or original calibration")
    qualities = [[scores[i] for i in case["indices"]] for case in paired["cases"]]
    if any(quality[0] != 10 for quality in qualities):
        raise ValueError("word judge rejected an exact complete human anchor")
    return qualities


def shuffled_indices(paired, seed):
    indices = list(range(len(paired["items"])))
    random.Random(seed).shuffle(indices)
    return indices


def verified_rewards(paired, directory, seed):
    def read(name):
        return json.loads(name.read_text())

    if read(directory / "blinded-inputs.json") != paired:
        raise ValueError("word rewards changed actual candidate/receipt pairings")
    indices, scores = shuffled_indices(paired, seed), [None] * len(paired["items"])
    expected_dirs = {f"request-{offset // 16:03d}" for offset in range(0, len(indices), 16)}
    if {path.name for path in directory.glob("request-*")} != expected_dirs:
        raise ValueError("word rewards changed request coverage")
    for offset in range(0, len(indices), 16):
        chosen = indices[offset : offset + 16]
        path = directory / f"request-{offset // 16:03d}"
        request, started, response, scored = [
            read(path / name)
            for name in ("request.json", "started.json", "response.json", "scores.json")
        ]
        if (
            request != terra_payload([paired["items"][i] for i in chosen])
            or started["request_sha256"] != sha256(path / "request.json")
            or started["model"] != "gpt-5.6-terra"
            or started["reasoning_effort"] != "medium"
            or started["pairs"] != len(chosen)
            or started["automatic_retries"] != 0
            or response.get("model") != "gpt-5.6-terra"
            or response.get("reasoning", {}).get("effort") != "medium"
            or response.get("status") != "completed"
            or scored["model"] != "gpt-5.6-terra"
            or scored["reasoning_effort"] != "medium"
            or scored["response_sha256"] != sha256(path / "response.json")
        ):
            raise ValueError(
                "word rewards lack authentic model, rubric or request/response binding"
            )
        parsed = parse_terra_scores(response, [f"pair_{i:02d}" for i in range(len(chosen))])
        actual = [parsed[f"pair_{i:02d}"] for i in range(len(chosen))]
        if scored["scores"] != actual:
            raise ValueError("word rewards differ from the raw response scores")
        for index, value in zip(chosen, actual, strict=True):
            scores[index] = value
    return {"scores": scores, "qualities": rewards_for(paired, scores)}


def score_batch(records, donors, pools, tokenizer, directory, pacer, seed):
    paired = paired_inputs(records, donors, pools, tokenizer)
    write(directory / "blinded-inputs.json", paired)
    indices = shuffled_indices(paired, seed)
    for offset in range(0, len(indices), 16):
        chosen = indices[offset : offset + 16]
        pacer.score([paired["items"][i] for i in chosen], directory / f"request-{offset // 16:03d}")
    result = verified_rewards(paired, directory, seed)
    write(directory / "rewards.json", result)
    return result["qualities"]


def storage(prospective=0, *, output=None, root=ROOT, disk_usage=shutil.disk_usage):
    if type(prospective) is not int or prospective < 0:
        raise ValueError("word storage needs a nonnegative byte reservation")
    used = 0
    if output is not None:
        if not output.resolve().is_relative_to(root / "checkpoints"):
            raise ValueError("word training artifacts must remain in checkpoints/")
        for path in output.rglob("*"):
            if path.is_symlink():
                raise ValueError("word output cannot follow aliases")
            if path.is_file():
                used += path.stat().st_size
    remaining = STORAGE["output_budget_bytes"] - used
    free = disk_usage(root).free
    if (
        prospective > remaining
        or free < STORAGE["physical_floor_bytes"] + STORAGE["headroom_bytes"] + remaining
    ):
        raise RuntimeError("word training exceeded its4GiB reserve or unchanged80GiB floor")
    return {
        "output_bytes": used,
        "remaining_output_reserve_bytes": remaining,
        "physical_free_bytes": free,
    }


@torch.no_grad()
def synchronize_visual(gpu, cpu):
    source, destination = gpu.vision.state_dict(), cpu.vision.state_dict()
    if source.keys() != destination.keys():
        raise ValueError("word rollout/trainer visual state topology differs")
    for name, value in source.items():
        if value.shape != destination[name].shape or value.dtype != destination[name].dtype:
            raise ValueError("word rollout/trainer visual state shape or dtype differs")
        destination[name].copy_(value.detach().cpu())


def update_visual(model, optimizer, expected_names, *, before_mutation=None):
    from roi_semantic_student import visual_gradients

    gradients = visual_gradients(model, expected_tensors=TENSORS)
    if set(gradients) != set(expected_names):
        raise ValueError("word update changed original350-tensor visual gradient scope")
    del gradients
    selected = model.trainable_parameters()
    norm = torch.nn.utils.clip_grad_norm_(
        selected.values(), RECIPE["gradient_clip"], error_if_nonfinite=True
    )
    if before_mutation is not None:
        before_mutation()
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in selected.values()):
        raise ValueError("word optimizer produced nonfinite visual values")
    model.zero_grad(set_to_none=True)
    return float(norm)


def save_checkpoint(
    model, optimizer, output, path, reference_sha, examples, updates, rng, *, recovery=False
):
    from safetensors.torch import save_file
    from train_tiny_rgb_feature_winner_full import optimizer_payload

    storage(STORAGE["checkpoint_pair_reserve_bytes"], output=output)
    prefix = "visual-recovery-after" if recovery else "visual-after"
    checkpoint = output / f"{prefix}-{examples:06d}.safetensors"
    optimizer_path = checkpoint.with_name(checkpoint.stem + "-optimizer.safetensors")
    receipt = checkpoint.with_suffix(".json")
    if any(target.exists() for target in (checkpoint, optimizer_path, receipt)):
        raise ValueError("word checkpoint never overwrites evidence")
    parameters = model.trainable_parameters()
    values, metadata = optimizer_payload(optimizer, parameters, rng)
    provenance = {"plan_sha256": sha256(path), "update": str(examples)}
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in parameters.items()},
        str(checkpoint),
        metadata={
            **provenance,
            "optimizer_updates": str(updates),
            "fixed_winner_checkpoint": reference_sha,
        },
    )
    save_file(values, str(optimizer_path), metadata={**provenance, **metadata})
    write(
        receipt,
        {
            "complete": True,
            "training_examples": examples,
            "optimizer_updates": updates,
            "visual": artifact(checkpoint),
            "optimizer": artifact(optimizer_path),
            "committed_batch_RNG": True,
        },
    )
    storage(output=output)
    return checkpoint
