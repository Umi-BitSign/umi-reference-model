from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import prepare_roi_short_corpus as data


def row(number, signer, *, source="asl_stem_original", training=True):
    return {
        "example_id": str(number),
        "source": source,
        "signer": signer,
        "role": "training" if training else "development",
        "training_allowed": training,
        "original_pair": {"raw": {"example_id": str(number)}},
    }


def inventory():
    records = [row(i, f"P{i % 25 + 100}") for i in range(88)]
    records.extend([row(88, "P0"), row(89, "P19"), row(90, "P4", training=False)])
    records.append(row(91, "external", source="how2sign_new"))
    parent = {"records": records, "reserved_STEM_participants": ["P4"]}
    pilot = {
        "ROI_diagnostic_participants": ["P19", "P0"],
        "records": [
            {
                "original_row": row(200, signer),
                "roi_role": "feature_diagnostic",
                "roi_training_allowed": False,
            }
            for signer in ("P19", "P0")
        ],
    }
    return parent, pilot


def test_broader_selection_keeps_all_training_signers_disjoint_from_both_reservations():
    parent, pilot = inventory()
    selected = data.select(parent, pilot)
    training = [r for r in selected if r["roi_training_allowed"]]
    assert len(training) == 88
    assert len({r["original_row"]["signer"] for r in training}) == 25
    assert not {r["original_row"]["signer"] for r in training} & {"P4", "P19", "P0"}
    assert [r["original_row"] for r in selected[-2:]] == [
        r["original_row"] for r in pilot["records"]
    ]
    assert not any(r["original_row"]["source"] == "how2sign_new" for r in selected)


def test_selection_rejects_changed_inventory_or_outer_permissions():
    parent, pilot = inventory()
    bad = copy.deepcopy(parent)
    bad["records"].pop(0)
    with pytest.raises(ValueError):
        data.select(bad, pilot)
    bad = copy.deepcopy(parent)
    bad["records"][0]["training_allowed"] = False
    with pytest.raises(ValueError):
        data.select(bad, pilot)


def test_selection_cannot_replace_a_diagnostic_signer():
    parent, pilot = inventory()
    pilot["records"][0]["original_row"]["signer"] = "P100"
    with pytest.raises(ValueError):
        data.select(parent, pilot)


class Native:
    def __init__(self):
        self.indices = []
        self.dtype = np.uint8

    def __len__(self):
        return 7

    def get_avg_fps(self):
        return 25

    def get_batch(self, indices):
        self.indices.append(indices)
        return SimpleNamespace(asnumpy=lambda: np.full((1, 2, 3, 3), indices[0], dtype=self.dtype))


def header():
    return {"source_frames": 7, "sampled_frames": 4, "source_shape": [2, 3]}


def test_native_reference_decodes_one_frame_per_index_with_complete_stride2_coverage():
    native = Native()
    reader = data.NativeSampledReader(native, header())
    assert not hasattr(reader, "seek")
    assert [int(reader[i][0, 0, 0]) for i in range(len(reader))] == [0, 2, 4, 6]
    assert native.indices == [[0], [2], [4], [6]]


def test_native_reference_rejects_reordering_and_wrong_sample_count():
    reader = data.NativeSampledReader(Native(), header())
    with pytest.raises(ValueError):
        reader[1]
    reader[0]
    with pytest.raises(ValueError):
        reader[0]
    with pytest.raises(ValueError):
        data.NativeSampledReader(Native(), {**header(), "sampled_frames": 3})


def test_native_reference_rejects_wrong_dtype_and_shape():
    native = Native()
    native.dtype = np.float32
    with pytest.raises(ValueError):
        data.NativeSampledReader(native, header())[0]
    with pytest.raises(ValueError):
        data.NativeSampledReader(Native(), {**header(), "source_shape": [3, 2]})[0]


def test_native_reference_enforces_its_byte_bound(monkeypatch):
    monkeypatch.setitem(data.POLICY, "maximum_reference_frame_bytes", 1)
    with pytest.raises(ValueError, match="128MiB"):
        data.NativeSampledReader(Native(), header())[0]


def test_supplemental_landmark_layout_uses_existing_receipted_tree():
    assert data.LAYOUT["asl_stem_supplemental"][1] == (
        "data/asl-stem-wiki/supplemental-landmarks-25fps"
    )
