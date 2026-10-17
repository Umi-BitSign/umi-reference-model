"""Training-only mixed proof candidates and authentic score mappings; no API execution."""

from __future__ import annotations

from c5_mixed_semantic_model import load_clip
from c5_terra_tail import candidate_labels
from c5_word_training import shuffled_indices as shuffled_indices
from c5_word_training import verified_rewards as verified_rewards
from probe_roi_semantic_learning import qualities as qualities
from semantic_quality_judge import calibration_controls
from terra_mixed_native_input import require_receipt, training_row
from terra_mixed_semantic_cases import POLICY as CASE_POLICY
from terra_mixed_semantic_cases import donor_indices, reference_key, require_cases
from terra_teacher_candidates import TrainingRolloutBudget
from terra_visual_learning import candidate_pool

POLICY = {
    "case_selection": CASE_POLICY,
    "scope": "three predeclared complete training cases; no development inputs",
    "candidate_generation": "exact human anchor plus reference-blind CPU greedy and sampled policy",
    "additional_candidate": (
        "fixed cyclic selected training caption; actual Terra grade, never assumed0"
    ),
    "duplicate_donor_policy": (
        "retain identical observed rollout tokens without replacement or duplication"
    ),
    "training_rollout_seed": 20261017,
    "candidate_horizon": 512,
    "risk_alpha": 0.005,
    "caption_weight": 0,
    "judge": "gpt-5.6-terra",
    "reasoning_effort": "medium",
    "maximum_pairs_per_request": 16,
    "minimum_request_start_gap_seconds": 60,
    "own_requests_in_flight": 1,
    "automatic_retries": 0,
    "automatic_continuations": 0,
    "CPU_threads": 2,
    "optimizer_updates": 0,
    "model_training_qualification_complete": False,
    "promotion_authorized": False,
}


def require_bound_cases(cases):
    require_cases([bound["case"] for bound in cases])
    for bound in cases:
        require_receipt(bound["case"], bound["verified_input"])
    if len({(bound["receipt"]["path"], bound["receipt"]["sha256"]) for bound in cases}) != 3:
        raise ValueError("mixed proof candidates require three distinct verified training receipts")


def training_negative(case, donor, tokenizer):
    row = training_row(case["case"])
    other = training_row(donor["case"])
    if (row["source"], row["example_id"]) == (
        other["source"],
        other["example_id"],
    ) or reference_key(case["case"]) == reference_key(donor["case"]):
        raise ValueError("mixed proof donor needs a different approved training identity/reference")
    candidate = {
        "tokens": tokenizer(other["reference"], truncation=False).input_ids,
        "text": other["reference"],
        "kind": "other_training_caption",
        "truncated": False,
        "termination": "eos",
        "donor_input_receipt": donor["receipt"],
    }
    candidate_labels([candidate])
    if (
        tokenizer.decode(
            candidate["tokens"], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        != candidate["text"]
    ):
        raise ValueError("mixed proof donor cannot alter or truncate its complete training caption")
    return candidate


def validate_pools(cases, pools, tokenizer):
    require_bound_cases(cases)
    if len(pools) != len(cases):
        raise ValueError("mixed proof pool coverage differs from complete selected inputs")
    donors = donor_indices([bound["case"] for bound in cases])
    for case, donor_index, pool in zip(cases, donors, pools, strict=True):
        row = training_row(case["case"])
        candidate_labels(pool)
        negative = training_negative(case, cases[donor_index], tokenizer)
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
            raise ValueError("mixed proof pool changed its exact anchor, rollouts or fixed donor")
        for candidate in pool:
            if (
                candidate["kind"] == "other_training_caption" and candidate != negative
            ) or tokenizer.decode(
                candidate["tokens"], skip_special_tokens=True, clean_up_tokenization_spaces=False
            ) != candidate["text"]:
                raise ValueError(
                    "mixed proof score text or donor differs from actual likelihood tokens"
                )


def prepare_pools(student, tokenizer, cases):
    require_bound_cases(cases)
    if any(parameter.device.type != "cpu" for parameter in student.parameters()):
        raise ValueError("mixed proof candidate rollouts must remain CPU-only")
    donors = donor_indices([bound["case"] for bound in cases])
    pools = []
    for index, (case, donor_index) in enumerate(zip(cases, donors, strict=True)):
        clip, mask = load_clip(case)
        pool = candidate_pool(
            student,
            tokenizer,
            clip,
            mask,
            training_row(case["case"])["reference"],
            seed=POLICY["training_rollout_seed"] + index,
            maximum_tokens=POLICY["candidate_horizon"],
            budget_factory=TrainingRolloutBudget,
        )
        del clip, mask
        negative = training_negative(case, cases[donor_index], tokenizer)
        if negative["tokens"] not in [candidate["tokens"] for candidate in pool]:
            pool.append(negative)
        pools.append(pool)
    validate_pools(cases, pools, tokenizer)
    return pools


def paired_inputs(cases, pools, tokenizer):
    validate_pools(cases, pools, tokenizer)
    donors = donor_indices([bound["case"] for bound in cases])
    items, known, mappings = [], {}, []
    for case, donor_index, pool in zip(cases, donors, pools, strict=True):
        row, indices = training_row(case["case"]), []
        for candidate in pool:
            key = row["reference"], candidate["text"]
            if key not in known:
                known[key] = len(items)
                items.append({"reference": key[0], "candidate": key[1]})
            indices.append(known[key])
        mappings.append(
            {
                "mixed_input_receipt": case["receipt"],
                "donor_input_receipt": cases[donor_index]["receipt"],
                "indices": indices,
            }
        )
    controls = []
    for control in calibration_controls():
        controls.append({"index": len(items), "range": control["range"]})
        items.append({"reference": control["reference"], "candidate": control["candidate"]})
    return {"items": items, "cases": mappings, "controls": controls}


def verified_proof_rewards(paired, directory, seed):
    """Authenticate actual response files and reject tied/failed-control proof rewards."""
    measured = verified_rewards(paired, directory, seed)
    if qualities(paired, measured["scores"]) != measured["qualities"]:
        raise ValueError("mixed proof reward mapping differs from actual calibrated grades")
    return measured
