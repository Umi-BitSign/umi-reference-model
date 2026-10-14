from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_tiny_rgb_duration import findings


def test_truncated_clips_quarantined_by_annotation_without_reference_or_model_scores():
    rows = [
        {
            "source": "how2sign_new",
            "example_id": str(i),
            "video": f"data/{i}.mp4",
            "video_sha256": str(i),
        }
        for i in range(3)
    ]
    headers = [
        {**row, "role": "training", "source_frames": frames, "source_fps": 25}
        for row, frames in zip(rows, (1, 90, 25), strict=True)
    ]
    annotations = [
        {"SENTENCE_NAME": str(i), "START_REALIGNED": "1", "END_REALIGNED": end}
        for i, end in enumerate(("4.3", "4.3", "2"))
    ]
    rejected = findings(rows, headers, annotations)
    assert len(rejected) == 1 and rejected[0]["example_id"] == "0"
    assert rejected[0]["training_allowed"] is False
    assert rejected[0]["original_bytes_preserved"] is True
    assert rejected[0]["recovery_pending"] is True
