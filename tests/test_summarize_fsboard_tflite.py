from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from benchmark_fleurs_dev import sha256
from summarize_fsboard_tflite import load_rows, summarize_sample


def pilot_inputs() -> tuple[dict, str, list[dict], list[dict], list[dict]]:
    base = ROOT / "runs/fsboard-validation-pilot-v1"
    selection_path = base / "selection.json"
    return (
        json.loads(selection_path.read_text()),
        sha256(selection_path),
        load_rows(ROOT / "runs/fsboard-tflite-training-layout-probe-v1/results.jsonl"),
        load_rows(base / "coreml-umi-normalized-v2.jsonl"),
        load_rows(base / "coreml-mirrored-left-v2.jsonl"),
    )


def test_pilot_parity_summary_matches_verified_results() -> None:
    result = summarize_sample(*pilot_inputs())
    assert result["clips"] == 30
    assert result["signers"] == 15
    assert result["exact_transcription_matches"] == 19
    assert result["tflite_mean_cer"] == pytest.approx(0.15002096286303448)
    assert result["coreml_mean_cer"] == pytest.approx(0.1449919640840357)


def test_parity_summary_rejects_changed_hypothesis_or_layout() -> None:
    selection, selection_hash, lite, right, left = pilot_inputs()
    lite[0] = {**lite[0], "hypothesis": "changed"}
    with pytest.raises(ValueError, match="CER differs"):
        summarize_sample(selection, selection_hash, lite, right, left)
    lite[0] = {**lite[0], "hypothesis": "lnmigan", "input_layout": "submission_columns"}
    with pytest.raises(ValueError, match="provenance differs"):
        summarize_sample(selection, selection_hash, lite, right, left)
