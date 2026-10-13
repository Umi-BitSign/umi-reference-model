"""Encoder-index reuse checks for an expanded aligned selection."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from rebind_how2sign_clip_receipts import SOURCE_FIELDS
from rebind_how2sign_encoder_index import rebound_rows


def _fixture():
    common = {key: f"same-{key}" for key in SOURCE_FIELDS}
    rows = [
        {"sentence_name": f"sentence{number}", "signer": 1, "reference": f"reference{number}"}
        for number in range(3)
    ]
    old = {"schema": "umi-studio-how2sign-training-aligned/1", **common, "rows": rows[:2]}
    expanded = {"schema": "umi-studio-how2sign-training-aligned/1", **common, "rows": rows}
    index = [
        {
            "sentence_name": row["sentence_name"],
            "selection_sha256": "prior-selection-hash",
            "clip_receipts_sha256": None,
            "source_clip_sha256": f"clip-hash-{number}",
            "encoder_cache_sha256": f"cache-hash-{number}",
            "signer": row["signer"],
            "reference": row["reference"],
            "reuse_source": {"pilot_index_sha256": "original-pilot"},
        }
        for number, row in enumerate(rows[:2])
    ]
    receipts = {
        row["sentence_name"]: {
            "selection_sha256": "expanded-selection-hash",
            "clip_sha256": f"clip-hash-{number}",
        }
        for number, row in enumerate(rows[:2])
    }
    return old, expanded, index, receipts


def _rebind(old, expanded, index, receipts):
    return rebound_rows(
        old,
        expanded,
        index,
        receipts,
        "prior-selection-hash",
        "expanded-selection-hash",
        "prior-index-hash",
        "miner-revision",
    )


def test_rebinds_complete_prior_index_and_keeps_pilot_provenance():
    old, expanded, index, receipts = _fixture()
    rebound = _rebind(old, expanded, index, receipts)
    assert [row["sentence_name"] for row in rebound] == ["sentence0", "sentence1"]
    assert {row["selection_sha256"] for row in rebound} == {"expanded-selection-hash"}
    assert {row["rebind_source"]["prior_index_sha256"] for row in rebound} == {"prior-index-hash"}
    assert rebound[0]["reuse_source"] == {"pilot_index_sha256": "original-pilot"}
    assert index[0]["selection_sha256"] == "prior-selection-hash"


def test_rejects_changed_expansion_or_receipt():
    old, expanded, index, receipts = _fixture()
    expanded["rows"][0] = {**expanded["rows"][0], "reference": "changed"}
    with pytest.raises(ValueError, match="exact expansion subset"):
        _rebind(old, expanded, index, receipts)
    expanded["rows"][0] = old["rows"][0]
    receipts["sentence0"]["clip_sha256"] = "wrong"
    with pytest.raises(ValueError, match="differs"):
        _rebind(old, expanded, index, receipts)
    receipts["sentence0"]["clip_sha256"] = "clip-hash-0"
    with pytest.raises(ValueError, match="exact expansion subset"):
        _rebind(old, expanded, index[:-1], receipts)
