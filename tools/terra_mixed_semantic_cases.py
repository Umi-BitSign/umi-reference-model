"""Predeclare complete mixed-curriculum risk cases without choosing model weights."""

from __future__ import annotations

from collections import Counter

from benchmark_fleurs_dev import normalize
from terra_mixed_native_input import expected_header, require_receipt, training_row

POLICY = {
    "selection": (
        "longest complete phrase timeline, longest remaining phrase UTF8-byte caption, "
        "then shortest retained word timeline; distinct normalized references and identity ties"
    ),
    "case_count": 3,
    "case_kinds": ["phrase", "phrase", "word"],
    "tie_order": "source, then example_id, ascending",
    "selection_uses_model_outputs_or_grades": False,
    "selection_uses_development_text": False,
    "donors": "cyclic next selected complete training caption; actual Terra grade, never assumed0",
    "no_invented_word_boundaries": True,
    "whole_original_timelines_required": True,
    "model_initializer_selected": False,
    "model_training_qualification_complete": False,
    "optimizer_recipe_selected": False,
    "optimizer_updates": 0,
    "new_Terra_requests": 0,
    "promotion_authorized": False,
}


def reference_key(case):
    words = tuple(normalize(training_row(case)["reference"]))
    if not words:
        raise ValueError("mixed risk cases require an intact nonempty training reference")
    return words


def identity(case):
    row = training_row(case)
    return row["source"], row["example_id"]


def require_cases(cases):
    if (
        len(cases) != POLICY["case_count"]
        or [case["kind"] for case in cases] != POLICY["case_kinds"]
        or len({identity(case) for case in cases}) != len(cases)
        or len({reference_key(case) for case in cases}) != len(cases)
    ):
        raise ValueError("mixed risk proof requires two distinct phrases and one retained word")
    for case in cases:
        expected_header(case)


def select_cases(cases):
    # Validate every role before selecting or reading references for a rank.
    keys = [identity(case) for case in cases]
    if len(set(keys)) != len(keys) or len({key[1] for key in keys}) != len(keys):
        raise ValueError("mixed risk selection cannot repeat a training identity")
    for case in cases:
        expected_header(case)
    selected, references = [], set()
    for kind, rank in (
        ("phrase", lambda case: -expected_header(case)["sampled_frames"]),
        ("phrase", lambda case: -len(training_row(case)["reference"].encode("utf-8"))),
        ("word", lambda case: expected_header(case)["sampled_frames"]),
    ):
        eligible = [
            case for case in cases if case["kind"] == kind and reference_key(case) not in references
        ]
        if not eligible:
            raise ValueError("mixed risk selection lacks distinct complete training cases")
        chosen = min(eligible, key=lambda case: (rank(case), *identity(case)))
        selected.append(chosen)
        references.add(reference_key(chosen))
    require_cases(selected)
    return selected


def summary(cases):
    require_cases(cases)
    return [
        {
            "kind": case["kind"],
            "source": training_row(case)["source"],
            "example_id": training_row(case)["example_id"],
            "sampled_frames": expected_header(case)["sampled_frames"],
            "complete_reference_UTF8_bytes": len(training_row(case)["reference"].encode("utf-8")),
        }
        for case in cases
    ]


def bind_cases(selected, verified):
    """Rejoin a proposal to the complete checked mixed preflight, without inference.

    The caller must obtain ``verified`` from prepare_terra_mixed_native.checked_result
    and pin that actual binding in its consuming immutable proof plan.
    """
    if (
        verified.get("counts") != {"word": 214, "phrase": 318}
        or verified.get("model_training_qualification_still_required") is not True
        or len(verified["records"]) != 532
        or Counter(record["case"]["kind"] for record in verified["records"])
        != {"word": 214, "phrase": 318}
    ):
        raise ValueError("mixed risk binding requires the complete532-input verified preflight")
    actual = select_cases([record["case"] for record in verified["records"]])
    require_cases(selected)
    if selected != actual:
        raise ValueError("mixed risk binding cannot change the predeclared training cases")
    indexed = {identity(record["case"]): record for record in verified["records"]}
    bound = [indexed[identity(case)] for case in selected]
    for record in bound:
        require_receipt(record["case"], record["verified_input"])
    if len({(record["receipt"]["path"], record["receipt"]["sha256"]) for record in bound}) != 3:
        raise ValueError("mixed risk cases require three distinct actual input receipts")
    return bound


def donor_indices(cases):
    require_cases(cases)
    return [(index + 1) % len(cases) for index in range(len(cases))]
