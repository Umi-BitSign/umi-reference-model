"""Deterministic geometry metrics; no predictor fitting or semantic promotion.

Primary errors use normalized image XY axes. Detector-relative wrist depth is
excluded. Paired comparisons require identical targets and scoring support.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np

METRIC = "detector_xy_coordinate_mae_v1"
JOINTS = ("wrist", "index_mcp", "middle_mcp", "ring_mcp", "little_mcp")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clip_xy(predictions, targets, target_valid, input_tip_valid):
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    target_valid, input_tip_valid = np.asarray(target_valid), np.asarray(input_tip_valid)
    require(
        targets.ndim == 4 and targets.shape[2:] == (5, 2) and predictions.shape == targets.shape,
        "expected matching frame/hand/five-target/XY arrays",
    )
    require(targets.shape[0] > 0 and targets.shape[1] > 0, "empty target layout")
    require(
        target_valid.shape == input_tip_valid.shape == targets.shape[:-1]
        and target_valid.dtype == input_tip_valid.dtype == np.bool_,
        "invalid target/input masks",
    )
    require(
        np.array_equal(target_valid, np.isfinite(targets).all(axis=-1)),
        "target mask must preserve every known target",
    )
    available = input_tip_valid.all(axis=-1)
    eligible = target_valid & available[..., None]
    require(np.isfinite(predictions[eligible]).all(), "nonfinite prediction on declared support")
    safe_predictions = np.where(eligible[..., None], predictions, 0).astype(np.float64)
    safe_targets = np.where(eligible[..., None], targets, 0).astype(np.float64)
    errors = np.abs(safe_predictions - safe_targets).mean(axis=-1)
    require(np.isfinite(errors).all(), "geometry error overflow")
    per_hand_count = eligible.sum(axis=-1)
    scored_hands = per_hand_count > 0
    per_hand = np.divide(
        errors.sum(axis=-1),
        per_hand_count,
        out=np.zeros(per_hand_count.shape, dtype=np.float64),
        where=scored_hands,
    )
    support = hashlib.sha256(METRIC.encode())
    support.update(json.dumps(list(targets.shape)).encode())
    support.update(np.ascontiguousarray(eligible).tobytes())
    support.update(np.ascontiguousarray(safe_targets.astype("<f8")).tobytes())
    return {
        "metric": METRIC,
        "score": float(per_hand[scored_hands].mean()) if scored_hands.any() else None,
        "support_sha256": support.hexdigest(),
        "scored_hands": int(scored_hands.sum()),
        "scored_target_points": int(eligible.sum()),
        "input_complete_hands": int(available.sum()),
        "known_target_hands": int(target_valid.any(axis=-1).sum()),
        "per_joint": {
            name: {
                "count": int(eligible[..., index].sum()),
                "mae": float(errors[..., index][eligible[..., index]].mean())
                if eligible[..., index].any()
                else None,
            }
            for index, name in enumerate(JOINTS)
        },
    }


def paired_group_comparison(baseline, candidate, *, seed=20261017, bootstraps=4000):
    """Average clips within groups, groups within families, then families equally.

    The caller supplies the frozen split's group IDs. Bootstrap samples whole
    groups within each family; it never treats individual frames as independent.
    """
    require(type(seed) is int and type(bootstraps) is int and bootstraps >= 16, "invalid bootstrap")

    def index(rows):
        result = {row["clip_id"]: row for row in rows}
        require(len(result) == len(rows) > 0, "duplicate or empty clip inventory")
        return result

    first, second = index(baseline), index(candidate)
    require(set(first) == set(second), "paired clip inventory differs")
    values = defaultdict(lambda: defaultdict(list))
    families, unscored, group_families = set(), [], {}
    for clip in sorted(first):
        left, right = first[clip], second[clip]
        require(left["metric"] == right["metric"] == METRIC, "different metric convention")
        require(
            (left["family"], left["group"]) == (right["family"], right["group"]),
            "paired identity groups differ",
        )
        family, group = left["family"], left["group"]
        require(
            isinstance(family, str) and family and isinstance(group, str) and group, "missing group"
        )
        families.add(family)
        require(
            group not in group_families or group_families[group] == family,
            "cross-family group requires a shared cluster bootstrap",
        )
        group_families[group] = family
        require(
            left["support_sha256"] == right["support_sha256"]
            and left["scored_hands"] == right["scored_hands"],
            "paired target or scoring support differs",
        )
        if left["score"] is None or right["score"] is None:
            require(left["score"] is right["score"] is None, "one model omitted a score")
            require(left["scored_hands"] == 0, "missing score on known support")
            unscored.append(clip)
            continue
        require(left["scored_hands"] > 0, "score without target support")
        require(
            all(np.isfinite(row["score"]) and row["score"] >= 0 for row in (left, right)),
            "invalid geometry score",
        )
        values[family][group].append((left["score"], right["score"]))
    require(set(values) == families, "every family needs at least one scored group")
    groups = {
        family: {group: np.mean(values[family][group], axis=0) for group in sorted(values[family])}
        for family in sorted(values)
    }
    baseline_mean = np.mean(
        [np.mean([v[0] for v in grouped.values()]) for grouped in groups.values()]
    )
    candidate_mean = np.mean(
        [np.mean([v[1] for v in grouped.values()]) for grouped in groups.values()]
    )
    rng = np.random.default_rng(seed)
    draws = np.zeros(bootstraps)
    for grouped in groups.values():
        gains = np.array([value[0] - value[1] for value in grouped.values()])
        draws += rng.choice(gains, size=(bootstraps, len(gains)), replace=True).mean(axis=1) / len(
            groups
        )
    return {
        "metric": METRIC,
        "baseline_mean": float(baseline_mean),
        "candidate_mean": float(candidate_mean),
        "mean_error_reduction": float(baseline_mean - candidate_mean),
        "paired_group_bootstrap95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "bootstrap_seed": seed,
        "bootstrap_samples": bootstraps,
        "family_group_counts": {family: len(grouped) for family, grouped in groups.items()},
        "families_with_only_one_scored_group": [
            family for family, grouped in groups.items() if len(grouped) == 1
        ],
        "paired_clips": len(first),
        "unscored_clips": unscored,
        "aggregation": "clips within groups; groups within families; equal family weights",
        "not_a_semantic_quality_gate": True,
    }
