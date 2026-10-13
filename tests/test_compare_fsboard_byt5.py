from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from compare_fsboard_byt5 import checked_rows, paired_metrics


def test_paired_metrics_require_exact_clip_and_source_match() -> None:
    candidate = {}
    comparator = {}
    for index in range(60):
        name = f"clip-{index:02d}.mp4"
        source = {
            "clip_filename": name,
            "signer_id": f"signer-{index // 4}",
            "selection_sha256": "selection",
            "landmarks_sha256": f"landmarks-{index}",
            "reference": "a",
        }
        candidate[name] = source | {
            "hypothesis": "a" if index < 30 else "b",
            "single_reference_cer": 0.0 if index < 30 else 1.0,
            "hand_dominance": "right" if index < 56 else "left",
        }
        comparator[name] = source | {
            "hypothesis": "b" if index < 30 else "a",
            "single_reference_cer": 1.0 if index < 30 else 0.0,
        }
    metrics = paired_metrics(candidate, comparator)
    assert metrics["candidate_cer_wins"] == 30
    assert metrics["tflite_cer_wins"] == 30
    assert metrics["cer_ties"] == 0
    assert metrics["candidate_mean_cer"] == 0.5
    assert metrics["tflite_mean_cer"] == 0.5
    assert metrics["by_hand_dominance"]["left"]["clips"] == 4
    comparator["clip-00.mp4"]["landmarks_sha256"] = "wrong"
    with pytest.raises(ValueError, match="differ in source"):
        paired_metrics(candidate, comparator)


def test_pinned_tflite_comparator_rows_recompute_cer() -> None:
    pilot = checked_rows(
        ROOT / "runs/fsboard-tflite-training-layout-probe-v1/results.jsonl",
        "umi-studio-fsboard-tflite-diagnostic/2",
    )
    followup = checked_rows(
        ROOT / "runs/fsboard-tflite-followup-v1/results.jsonl",
        "umi-studio-fsboard-tflite-diagnostic/2",
    )
    assert len(pilot) == len(followup) == 30
    assert not pilot.keys() & followup.keys()
