"""Training-only checks of semantic gradient balance and caption/video conditioning."""

from __future__ import annotations

import hashlib
import math
import statistics


def stratum(row):
    if len(row["quality"]) == 1:
        return "singleton"
    return "contrast" if len(set(row["quality"])) > 1 else "tied_pair"


def retrieval_cases(records):
    """Select before inference: six contrasts, three ties and three singletons."""
    result, groups, captions = [], set(), set()
    ordered = sorted(
        records,
        key=lambda r: hashlib.sha256(
            ("grounding-v1:" + r["data_row"]["source"] + ":" + r["data_row"]["example_id"]).encode()
        ).hexdigest(),
    )
    for kind, count in (("contrast", 6), ("tied_pair", 3), ("singleton", 3)):
        chosen = 0
        for row in ordered:
            data = row["data_row"]
            caption = " ".join(data["reference"].lower().split())
            if (
                stratum(row) == kind
                and data["source_group"] not in groups
                and caption not in captions
            ):
                if data["role"] != "training" or data["training_allowed"] is not True:
                    raise ValueError("conditioning diagnostics cannot use development labels")
                result.append(row)
                groups.add(data["source_group"])
                captions.add(caption)
                chosen += 1
                if chosen == count:
                    break
        if chosen != count:
            raise ValueError("insufficient source-distinct training cases for conditioning probe")
    return result


def gradient_balance(semantic, auxiliary):
    import torch

    if not semantic or semantic.keys() != auxiliary.keys():
        raise ValueError("gradient components have different parameter scopes")
    sem_sq = aux_sq = dot = 0.0
    for name in semantic:
        first, second = semantic[name].double(), auxiliary[name].double()
        if (
            first.shape != second.shape
            or not torch.isfinite(first).all()
            or not torch.isfinite(second).all()
        ):
            raise ValueError("gradient components have nonfinite or mismatched values")
        sem_sq += float(first.square().sum())
        aux_sq += float(second.square().sum())
        dot += float((first * second).sum())
    sem, aux = math.sqrt(sem_sq), math.sqrt(aux_sq)
    return {
        "semantic_gradient_norm": sem,
        "weighted_auxiliary_gradient_norm": aux,
        "semantic_to_weighted_auxiliary_norm_ratio": sem / aux if aux else None,
        "gradient_cosine": dot / (sem * aux) if sem and aux else None,
        "combined_gradient_norm": math.sqrt(max(0, sem_sq + aux_sq + 2 * dot)),
        "semantic_gradient_is_zero": sem == 0,
    }


def sum_equivalence(semantic, auxiliary, combined):
    import torch

    if semantic.keys() != auxiliary.keys() or semantic.keys() != combined.keys():
        raise ValueError("combined gradient parameter scope differs")
    errors = []
    for name in semantic:
        expected = semantic[name].double() + auxiliary[name].double()
        actual = combined[name].double()
        if not torch.isfinite(actual).all() or expected.shape != actual.shape:
            raise ValueError("combined gradient has nonfinite or mismatched values")
        errors.append(float((actual - expected).norm()) / max(float(expected.norm()), 1e-12))
    worst = max(errors)
    if not math.isfinite(worst) or worst > 0.005:
        raise ValueError("semantic plus auxiliary gradients do not match the actual full loss")
    return worst


def conditioning(matrix):
    n = len(matrix)
    if (
        n < 2
        or any(len(row) != n for row in matrix)
        or any(not math.isfinite(v) or v < 0 for row in matrix for v in row)
    ):
        raise ValueError("conditioning matrix must be square, finite and nonnegative")
    gaps = [
        statistics.mean(v for j, v in enumerate(row) if j != i) - row[i]
        for i, row in enumerate(matrix)
    ]
    return {
        "caption_rows_video_columns_CE": matrix,
        "mean_correct_video_CE": statistics.mean(matrix[i][i] for i in range(n)),
        "mean_other_video_minus_correct_CE": statistics.mean(gaps),
        "per_caption_other_video_minus_correct_CE": gaps,
        "strict_correct_video_top1": sum(
            all(row[i] < v for j, v in enumerate(row) if j != i) for i, row in enumerate(matrix)
        ),
        "training_cases": n,
        "other_videos_are_not_human_verified_semantic_negatives": True,
        "sign_cue_reliance_or_generalization_proven": False,
    }
