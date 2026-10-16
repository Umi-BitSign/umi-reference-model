"""Deterministic short-phrase sampling and blinded semantic-only checkpoint decisions."""

from __future__ import annotations

import hashlib
import random
import statistics
import time

from audit_native_terra_semantics import cluster_interval
from launch_terra_visual_job import write
from prepare_terra_short_curriculum import training_pairs
from semantic_quality_judge import calibration_controls
from terra_semantic_selection import decision

RECIPE = {
    "seed": 20261016,
    "epochs": 4,
    "epochs_are_maximum": True,
    "semantic_stage_mastery_quality": 9.0,
    "batch_clips": 8,
    "learning_rate": 1e-5,
    "learning_rate_schedule": "32-example linear warmup times cosine decay over all exposures",
    "warmup_examples": 32,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "risk_alpha": 0.005,
    "caption_weight": 0.1,
    "objective": "actual Terra expected semantic error, plus auxiliary human-caption CE",
    "judge_model": "gpt-5.6-terra",
    "judge_reasoning_effort": "medium",
    "model": "original C5; same qualified eight decoder cross-attention tensors only",
    "trainable_parameters": 9437184,
    "trainable_tensors": 8,
    "check_after_epochs": [1, 2, 4],
    "maximum_scoring_pairs": 16,
    "minimum_seconds_between_request_starts": 30,
    "own_requests_in_flight": 1,
    "automatic_retries": 0,
    "WER_is_diagnostic_only": True,
    "new_data_downloads": False,
    "semantic_evaluation_repeats": 2,
    "significant_semantic_decline_stops_training": True,
}


def order(records):
    pairs = training_pairs(records)
    result = []
    for epoch in range(RECIPE["epochs"]):
        result.extend(
            sorted(
                pairs,
                key=lambda p, epoch=epoch: hashlib.sha256(
                    f"short-terra:{RECIPE['seed']}:{epoch}:{p['raw']['source']}:{p['raw']['example_id']}".encode()
                ).hexdigest(),
            )
        )
    return result


def comparison_items(rows, baseline, predictions):
    expected = [(r["source"], r["example_id"]) for r in rows]
    if (
        not expected
        or len(set(expected)) != len(expected)
        or any(r["role"] != "development" or r["training_allowed"] is not False for r in rows)
    ):
        raise ValueError("semantic curriculum decisions require reserved development identities")
    indexed = []
    for values in (baseline, predictions):
        keyed = {(r["source"], r["example_id"]): r for r in values}
        if len(keyed) != len(values) or set(keyed) != set(expected):
            raise ValueError("semantic curriculum comparison coverage differs")
        if any(
            keyed[k]["reference"] != r["reference"]
            or keyed[k]["source_group"] != r["source_group"]
            or keyed[k]["signer"] != r["signer"]
            or keyed[k]["role"] != "development"
            or keyed[k]["training_allowed"] is not False
            for k, r in zip(expected, rows, strict=True)
        ):
            raise ValueError("semantic curriculum comparison labels/roles changed")
        indexed.append(keyed)
    items, indices, cases = [], {}, []
    for key, row in zip(expected, rows, strict=True):
        pair = []
        for values in indexed:
            identity = (row["reference"], values[key]["hypothesis"])
            if identity not in indices:
                indices[identity] = len(items)
                items.append({"reference": identity[0], "candidate": identity[1]})
            pair.append(indices[identity])
        cases.append(
            {
                "source": key[0],
                "example_id": key[1],
                "source_group": row["source_group"],
                "indices": pair,
            }
        )
    controls = []
    for row in calibration_controls():
        controls.append({"index": len(items), "range": row["range"]})
        items.append({"reference": row["reference"], "candidate": row["candidate"]})
    return {"items": items, "cases": cases, "controls": controls}


def semantic_decision(comparison, scores, baseline, predictions):
    if len(scores) != 2 or any(len(s) != len(comparison["items"]) for s in scores):
        raise ValueError("short semantic decision needs two complete blinded passes")
    if any(type(v) is not int or not 0 <= v <= 10 for s in scores for v in s):
        raise ValueError("short semantic score outside the owner rubric")
    calibrated = all(
        c["range"][0] <= s[c["index"]] <= c["range"][1]
        for s in scores
        for c in comparison["controls"]
    )
    paired = [
        [statistics.mean(s[i] for s in scores) for i in case["indices"]]
        for case in comparison["cases"]
    ]
    deltas = [b - a for a, b in paired]
    interval = cluster_interval(comparison["cases"], deltas)
    # The shared selector's historical audit input name is internal to this adapter.
    result = decision(
        {
            "complete": True,
            "calibration_controls_passed_both_passes": calibrated,
            "C5_mean_quality": statistics.mean(a for a, _ in paired),
            "native4096_mean_quality": statistics.mean(b for _, b in paired),
            "source_group_bootstrap95_interval": interval,
            "clips": len(paired),
        },
        {
            "C5_mean_WER": statistics.mean(r["WER_diagnostic"] for r in baseline),
            "candidate_mean_WER": statistics.mean(r["WER_diagnostic"] for r in predictions),
        },
    )
    result.update(
        {
            "scope": "reserved_STEM_short_phrases_only",
            "curriculum_stage_only": True,
            "broader_semantic_and_native_composite_confirmation_required": True,
            "stop_for_semantic_decline": calibrated
            and result["absolute_gain"] <= -0.1
            and interval[1] < 0,
            "stop_for_failed_calibration": not calibrated,
            "control_stage_mastered": calibrated
            and result["C5_mean_quality"] >= RECIPE["semantic_stage_mastery_quality"],
            "candidate_stage_mastered": calibrated
            and result["candidate_mean_quality"] >= RECIPE["semantic_stage_mastery_quality"],
        }
    )
    return result


class JudgePacer:
    def __init__(self, *, clock=time.monotonic, sleep=time.sleep):
        self.clock, self.sleep, self.previous_start = clock, sleep, None

    def score(self, rows, output):
        from terra_quality_training import score_with_terra

        if not 1 <= len(rows) <= RECIPE["maximum_scoring_pairs"]:
            raise ValueError("short curriculum scorer needs 1-16 pairs")
        if self.previous_start is not None:
            remaining = RECIPE["minimum_seconds_between_request_starts"] - (
                self.clock() - self.previous_start
            )
            if remaining > 0:
                self.sleep(remaining)
        self.previous_start = self.clock()
        return score_with_terra(rows, output)


def score_comparison(comparison, output, pacer):
    output.mkdir()
    write(output / "blinded-inputs.json", comparison)
    result = []
    for repeat in range(2):
        indices = list(range(len(comparison["items"])))
        random.Random(RECIPE["seed"] + repeat).shuffle(indices)
        scores = [None] * len(indices)
        for offset in range(0, len(indices), 16):
            chosen = indices[offset : offset + 16]
            response = pacer.score(
                [comparison["items"][i] for i in chosen],
                output / f"pass-{repeat + 1}-batch-{offset // 16:03d}",
            )
            for i, value in zip(chosen, response["scores"], strict=True):
                scores[i] = value
        write(output / f"pass-{repeat + 1}-scores.json", scores)
        result.append(scores)
    return result
