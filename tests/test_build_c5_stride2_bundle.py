"""Candidate bundle inventory assigns stable artifact roles."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_c5_stride2_bundle import role


def test_bundle_roles_cover_weights_code_and_provenance():
    assert role("models/model.safetensors") == "weights"
    assert role("umi_inference.py") == "inference"
    assert role("MODEL_IDENTITY.json") == "provenance"
    assert role("licenses/fairseq-MIT.txt") == "license"
