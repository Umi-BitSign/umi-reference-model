from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from freeze_tiny_rgb_visual_teacher_corpus import usable_feature_signal


def values():
    return {
        name: np.zeros((2, width), dtype=np.float32)
        for name, width in (
            ("face_features", 384),
            ("left_hand_features", 384),
            ("right_hand_features", 384),
            ("pose_features", 14),
        )
    }


def test_zero_hand_face_targets_remain_unavailable_even_with_pose():
    targets = values()
    targets["pose_features"].fill(1)
    assert not usable_feature_signal(targets)


def test_one_observed_hand_is_sufficient_without_face_or_other_hand():
    targets = values()
    targets["left_hand_features"][1, 1] = 1
    assert usable_feature_signal(targets)


def test_feature_signal_uses_vector_norm_not_an_elementwise_cutoff():
    targets = values()
    targets["face_features"].fill(1e-7)
    assert usable_feature_signal(targets)
