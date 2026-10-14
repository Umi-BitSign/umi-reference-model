from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from tiny_rgb_winner_student_protocol import RECIPE, select_cases


def pairs():
    return [
        {
            "raw": {
                "source": source,
                "example_id": str(i),
                "reference": reference,
                "role": "training",
                "training_allowed": True,
            },
            "header": {"sampled_frames": frames},
            "feature_pretraining_allowed": True,
        }
        for i, (source, frames, reference) in enumerate(
            [
                ("how2sign_new", 197, "hello"),
                ("how2sign_raw", 20, "é" * 100),
                ("asl_stem_original", 2, "hi"),
                ("fleurs", 112, "hey"),
                ("openasl_scale", 80, "yo"),
                ("asl_stem_supplemental", 12, "word"),
            ]
        )
    ]


def test_proof_selects_native_frame_extrema_and_byte_extrema_and_every_family():
    chosen = select_cases(pairs())
    assert {p["raw"]["example_id"] for p in chosen} == {"0", "1", "2", "3", "4"}
    assert RECIPE["trainable_parameter_tensors"] == 53
    assert RECIPE["trainable_parameters"] == 6474334
    assert not RECIPE["initial_diagnostic_weights_reused"]


@pytest.mark.parametrize("key", ["training_allowed", "feature_pretraining_allowed"])
def test_proof_rejects_protected_or_unavailable_feature_pairs(key):
    rows = copy.deepcopy(pairs())
    if key == "training_allowed":
        rows[0]["raw"][key] = False
    else:
        rows[0][key] = False
    with pytest.raises(ValueError, match="approved training"):
        select_cases(rows)
