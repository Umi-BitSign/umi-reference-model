"""ASL STEM Wiki encoder indexes stay bound to cache bytes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_fleurs_dev import sha256
from cache_asl_stem_wiki_encoder import checked_completed


def test_checked_completed_validates_cache_and_provenance(tmp_path):
    cache = tmp_path / "clip.npz"
    cache.write_bytes(b"cache")
    index = tmp_path / "index.jsonl"
    row = {
        "schema": "umi-studio-asl-stem-wiki-encoder-cache-example/1",
        "example_id": "clip",
        "split": "training",
        "selection_sha256": "selection",
        "source_receipts_sha256": "receipts",
        "clip_fps": 25,
        "encoder_dtype": "float32",
        "stream_feature_dtype": "float32",
        "encoder_cache_sha256": sha256(cache),
    }
    index.write_text(json.dumps(row) + "\n")
    assert checked_completed(
        index, "selection", "receipts", tmp_path
    ).keys() == {"clip"}
    cache.write_bytes(b"changed")
    with pytest.raises(ValueError, match="cache differs"):
        checked_completed(index, "selection", "receipts", tmp_path)
