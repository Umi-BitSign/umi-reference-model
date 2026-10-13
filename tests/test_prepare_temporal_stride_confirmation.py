"""Confirmation clip preparation keeps domain-specific boundaries fixed."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_temporal_stride_confirmation import prepare_clip


def test_how2sign_resamples_the_complete_publisher_clip(tmp_path):
    record = {
        "domain": "how2sign",
        "source_path": "source.mp4",
        "start_seconds": 0.0,
        "end_seconds": None,
    }
    destination = tmp_path / "clip.mp4"
    with patch("prepare_temporal_stride_confirmation.resample") as operation:
        prepare_clip(record, destination)
    operation.assert_called_once()
    assert operation.call_args.args[1] == destination


def test_fleurs_extracts_only_the_frozen_interval(tmp_path):
    record = {
        "domain": "fleurs",
        "source_path": "source.mp4",
        "start_seconds": 1.25,
        "end_seconds": 4.5,
    }
    destination = tmp_path / "clip.mp4"
    with patch("prepare_temporal_stride_confirmation.extract_clip") as operation:
        prepare_clip(record, destination)
    assert operation.call_args.args[1:] == (destination, 1.25, 4.5, "25")


def test_rejects_changed_boundaries_or_domain(tmp_path):
    with pytest.raises(ValueError, match="boundaries"):
        prepare_clip(
            {
                "domain": "how2sign",
                "source_path": "source.mp4",
                "start_seconds": 1.0,
                "end_seconds": None,
            },
            tmp_path / "clip.mp4",
        )
    with pytest.raises(ValueError, match="another domain"):
        prepare_clip(
            {
                "domain": "other",
                "source_path": "source.mp4",
                "start_seconds": 0.0,
                "end_seconds": None,
            },
            tmp_path / "clip.mp4",
        )
