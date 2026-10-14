from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from tiny_rgb_capacity_protocol import STAGES, select_cases, verified_comparison


def data():
    rows = [
        {"source": source, "example_id": str(i), "role": "training", "training_allowed": True}
        for i, source in enumerate(("how2sign", "fleurs", "asl_stem_original", "openasl_scale"))
    ]
    headers = [
        {**row, "sampled_frames": frames, "target_tokens": target}
        for row, frames, target in zip(rows, (197, 80, 1, 70), (10, 80, 1, 20), strict=True)
    ]
    return rows, headers


def comparison():
    return {
        "loss_absolute_difference": 1e-6,
        "logit_maximum_absolute_difference": 1e-5,
        "total_gradient_norm_relative_error": 1e-5,
        "gradient_differences": [
            {"parameter": "vision.weight", "finite": True, "relative_l2_error": 1e-5}
        ],
    }


def test_cases_cover_real_extrema_and_fleurs_without_evaluation_headers():
    rows, headers = data()
    headers.append(
        {
            "source": "private_eval",
            "example_id": "never",
            "role": "open_development",
            "sampled_frames": 999,
            "target_tokens": 999,
        }
    )
    selected = select_cases(rows, headers)
    assert {r["training"]["source"] for r in selected} == {
        "how2sign",
        "fleurs",
        "asl_stem_original",
        "openasl_scale",
    }
    assert all(r["training"]["training_allowed"] is True for r in selected)


def test_protected_training_identity_and_missing_header_are_rejected():
    rows, headers = data()
    rows[0]["training_allowed"] = False
    with pytest.raises(ValueError, match="training identities"):
        select_cases(rows, headers)
    rows, headers = data()
    with pytest.raises(ValueError, match="headers"):
        select_cases(rows, headers[:-1])


def test_unchanged_numeric_limits_accept_good_and_reject_wrong_topology():
    assert verified_comparison(comparison(), 1) == 1e-5
    with pytest.raises(ValueError, match="coverage"):
        verified_comparison(comparison(), 2)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.006])
def test_nonfinite_or_excessive_gradient_error_is_rejected(value):
    record = copy.deepcopy(comparison())
    record["gradient_differences"][0]["relative_l2_error"] = value
    with pytest.raises(ValueError, match="per-parameter"):
        verified_comparison(record, 1)


def test_cpu_equivalence_uses_stricter_forward_limit():
    record = comparison()
    record["logit_maximum_absolute_difference"] = 0.0005
    verified_comparison(record, 1)
    with pytest.raises(ValueError, match="parity"):
        verified_comparison(record, 1, cpu_equivalence=True)


def test_padding_and_activation_are_distinct_existing_tolerance_stages():
    assert set(STAGES) == {"cpu_equivalence", "cpu_padding", "cpu_mps_comparison"}
    record = comparison()
    record["logit_maximum_absolute_difference"] = 0.00015831
    verified_comparison(record, 1)
    with pytest.raises(ValueError, match="parity"):
        verified_comparison(record, 1, cpu_equivalence=True)
