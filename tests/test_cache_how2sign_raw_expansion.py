"""Raw expansion encoding reuses only indexes bound to each video selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_fleurs_dev import sha256
from cache_how2sign_raw_expansion import existing_indexes


def test_maps_existing_encoder_index_to_its_frozen_selection(tmp_path):
    selection = tmp_path / "source-selection.json"
    selection.write_text(json.dumps({"video_name": "source"}))
    index = tmp_path / "pilot-encoder-cache.jsonl"
    index.write_text(json.dumps({"selection_sha256": sha256(selection)}) + "\n")
    selections = {"source": (selection, tmp_path / "crops.jsonl")}
    assert existing_indexes(tmp_path, selections) == {"source": index}
    duplicate = tmp_path / "duplicate-encoder-cache.jsonl"
    duplicate.write_text(index.read_text())
    with pytest.raises(ValueError, match="multiple"):
        existing_indexes(tmp_path, selections)


def test_rejects_index_for_unknown_selection(tmp_path):
    selection = tmp_path / "source-selection.json"
    selection.write_text(json.dumps({"video_name": "source"}))
    index = tmp_path / "foreign-encoder-cache.jsonl"
    index.write_text(json.dumps({"selection_sha256": "foreign"}) + "\n")
    with pytest.raises(ValueError, match="unknown"):
        existing_indexes(tmp_path, {"source": (selection, tmp_path / "crops.jsonl")})
