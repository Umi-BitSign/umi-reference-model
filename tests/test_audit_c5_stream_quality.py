import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_c5_stream_quality import sample_training, stream_statistics


def test_measures_blank_templates_repeats_and_missing_pose():
    values = np.array([[1., 2.], [1., 2.], [3., 4.]], dtype=np.float32)
    streams = {name: values for name in ("face_features", "left_hand_features", "right_hand_features")}
    streams["pose_features"] = np.array([[-9999., -9999.], [0., 0.], [0., 0.]], dtype=np.float32)
    blanks = {name: np.array([1., 2.]) for name in streams if name != "pose_features"}
    stats = stream_statistics(streams, blanks)
    assert stats["face_features"]["blank_template_match_fraction"] == pytest.approx(2/3)
    assert stats["face_features"]["exact_adjacent_repeat_fraction"] == .5
    assert stats["pose_features"]["all_missing_pose_fraction"] == pytest.approx(1/3)


def test_sampling_is_deterministic_and_balanced_without_result_information():
    rows = [{"source": "how2sign_original", "id": f"{s}-{i}", "signer": s} for s in (1, 8) for i in range(10)]
    selected = sample_training(rows, 3)
    assert selected == sample_training(list(reversed(rows)), 3)
    assert sum(row["signer"] == 1 for row in selected) == 3
    assert sum(row["signer"] == 8 for row in selected) == 3
