"""Training-only word proof cases and actual candidate-set bindings; no API execution."""

from __future__ import annotations

import torch
from c5_small_input_semantic import clip_input
from c5_terra_tail import candidate_labels
from probe_roi_semantic_learning import qualities as qualities
from semantic_quality_judge import calibration_controls
from terra_teacher_candidates import TrainingRolloutBudget
from terra_visual_learning import candidate_pool
from terra_word_native_input import require_role
from terra_word_verified_inputs import expected_header, load_verified_input

POLICY = {
    "selection": (
        "longest training timeline, longest remaining byte caption, then shortest remaining "
        "timeline; distinct labels, lexicographic identity ties; no development text or scores"
    ),
    "case_count": 3,
    "candidate_generation": "exact human anchor plus reference-blind greedy and sampled student",
    "additional_candidate": (
        "cyclic next selected training caption; actual Terra grade, never assumed0"
    ),
    "duplicate_donor_policy": "stop if predeclared donor tokens duplicate a rollout",
    "training_rollout_seed": 20261017,
    "candidate_horizon": 512,
    "risk_alpha": 0.005,
    "caption_weight": 0,
    "judge": "gpt-5.6-terra",
    "reasoning_effort": "medium",
    "maximum_pairs_per_request": 16,
    "minimum_request_start_gap_seconds": 60,
    "automatic_retries": 0,
    "calibration": "original exact, paraphrase, generic and negation controls; no relaxation",
    "CPU_threads": 2,
    "optimizer_updates": 0,
    "model_training_qualification_complete": False,
    "promotion_authorized": False,
}


def training_case(case):
    require_role(case["record"], "training")
    return case["record"]["original"]["row"]


def select_cases(bound):
    training, known = [], set()
    for case in bound:
        record = case["record"]
        row = record["original"]["row"]
        require_role(record, row["role"])
        identity = row["example_id"]
        if identity in known:
            raise ValueError("word semantic proof cannot select from repeated identities")
        known.add(identity)
        if row["role"] == "training":
            training.append(case)
    selected, labels = [], set()
    for rank in (
        lambda case: (
            -expected_header(case["record"])["sampled_frames"],
            training_case(case)["example_id"],
        ),
        lambda case: (
            -len(training_case(case)["reference"].encode()),
            training_case(case)["example_id"],
        ),
        lambda case: (
            expected_header(case["record"])["sampled_frames"],
            training_case(case)["example_id"],
        ),
    ):
        eligible = [case for case in training if training_case(case)["label"] not in labels]
        if not eligible:
            raise ValueError("word semantic proof lacks three eligible distinct training labels")
        case = min(eligible, key=rank)
        selected.append(case)
        labels.add(training_case(case)["label"])
    require_cases(selected)
    return selected


def require_cases(cases):
    rows = [training_case(case) for case in cases]
    if (
        len(rows) != POLICY["case_count"]
        or len({row["example_id"] for row in rows}) != len(rows)
        or len({row["label"] for row in rows}) != len(rows)
        or len({(case["receipt"]["path"], case["receipt"]["sha256"]) for case in cases})
        != len(cases)
    ):
        raise ValueError("word semantic proof requires three distinct predeclared training inputs")


def training_negative(case, donor, tokenizer):
    row, other = training_case(case), training_case(donor)
    if row["example_id"] == other["example_id"] or row["label"] == other["label"]:
        raise ValueError("word contrastive caption needs a different approved training label")
    tokens = tokenizer(other["reference"], truncation=False).input_ids
    candidate = {
        "tokens": tokens,
        "text": other["reference"],
        "kind": "other_training_caption",
        "truncated": False,
        "termination": "eos",
        "donor_input_receipt": donor["receipt"],
    }
    candidate_labels([candidate])
    return candidate


def validate_pools(cases, pools, tokenizer):
    require_cases(cases)
    if len(pools) != len(cases):
        raise ValueError("word semantic pools lost complete input coverage")
    for index, (case, pool) in enumerate(zip(cases, pools, strict=True)):
        row = training_case(case)
        candidate_labels(pool)
        expected_donor = training_negative(case, cases[(index + 1) % len(cases)], tokenizer)
        if (
            not 2 <= len(pool) <= 4
            or pool[0]["kind"] != "human_anchor"
            or pool[0]["text"] != row["reference"]
            or pool[0]["tokens"] != tokenizer(row["reference"], truncation=False).input_ids
            or pool[-1] != expected_donor
            or any(candidate["kind"] not in ("greedy", "sampled") for candidate in pool[1:-1])
            or len({candidate["kind"] for candidate in pool}) != len(pool)
        ):
            raise ValueError("word semantic pool changed its exact anchor, rollouts or fixed donor")
        for candidate in pool:
            text = tokenizer.decode(
                candidate["tokens"], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if text != candidate["text"]:
                raise ValueError("word semantic score text differs from actual likelihood tokens")


def prepare_pools(student, tokenizer, cases):
    require_cases(cases)
    if any(parameter.device.type != "cpu" for parameter in student.parameters()):
        raise ValueError("word qualification candidate rollouts must remain CPU-only")
    pools = []
    for index, case in enumerate(cases):
        values = load_verified_input(case, purpose="training")
        clip = clip_input(values)
        mask = torch.ones(1, len(clip.pixels), dtype=torch.bool)
        pool = candidate_pool(
            student,
            tokenizer,
            clip,
            mask,
            training_case(case)["reference"],
            seed=POLICY["training_rollout_seed"] + index,
            maximum_tokens=POLICY["candidate_horizon"],
            budget_factory=TrainingRolloutBudget,
        )
        negative = training_negative(case, cases[(index + 1) % len(cases)], tokenizer)
        if any(candidate["tokens"] == negative["tokens"] for candidate in pool):
            raise ValueError(
                "predeclared word donor duplicates an observed rollout; no substitution"
            )
        pools.append([*pool, negative])
    validate_pools(cases, pools, tokenizer)
    return pools


def paired_inputs(cases, pools, tokenizer):
    validate_pools(cases, pools, tokenizer)
    items, known, mappings = [], {}, []
    for case, pool in zip(cases, pools, strict=True):
        row, indices = training_case(case), []
        for candidate in pool:
            key = row["reference"], candidate["text"]
            if key not in known:
                known[key] = len(items)
                items.append({"reference": key[0], "candidate": key[1]})
            indices.append(known[key])
        mappings.append({"word_input_receipt": case["receipt"], "indices": indices})
    controls = []
    for control in calibration_controls():
        controls.append({"index": len(items), "range": control["range"]})
        items.append({"reference": control["reference"], "candidate": control["candidate"]})
    return {"items": items, "cases": mappings, "controls": controls}
