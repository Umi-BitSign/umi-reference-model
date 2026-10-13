"""A frozen-teacher retention objective for the audited C5 corpus."""

from __future__ import annotations

import copy
import math

from c5_all_source_training import RECIPE as CORPUS_RECIPE

RECIPE = {
    **copy.deepcopy(CORPUS_RECIPE),
    "id": "c5-all-source-fused-lora-retention-v1",
    "retention": {
        "coefficient": 1.0,
        "temperature": 2.0,
        "teacher": "frozen_starting_adapter",
        "prefixes": "training_reference_only",
        "reduction": "mean_nonignored_tokens",
    },
}
COVERAGE_RECIPE = {
    **copy.deepcopy(RECIPE),
    "id": "c5-all-source-fused-lora-retention-coverage-v1",
    "schedule_policy": {
        "id": "weighted_fair_unique_group_coverage/1",
        "within_group": "matched_control_order",
        "exhausted_groups": "remove_without_resampling",
        "loss_weights_unchanged": True,
    },
}
RETENTION_RECIPES = {recipe["id"]: recipe for recipe in (RECIPE, COVERAGE_RECIPE)}
ALL_SOURCE_IDS = {CORPUS_RECIPE["id"], *RETENTION_RECIPES}


def teacher_kl(student_logits, teacher_logits, labels, temperature: float):
    """Mean KL(teacher || student), temperature-scaled, with no teacher gradient."""
    import torch

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("retention temperature must be finite and positive")
    if student_logits.shape != teacher_logits.shape or student_logits.shape[:-1] != labels.shape:
        raise ValueError("retention logits and labels differ in shape")
    valid = labels.ne(-100)
    if not valid.any():
        raise ValueError("retention requires at least one nonignored token")
    student_log = torch.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_log = torch.log_softmax(teacher_logits.detach().float() / temperature, dim=-1)
    tokens = (teacher_log.exp() * (teacher_log - student_log)).sum(dim=-1)
    return tokens[valid].mean() * temperature**2
