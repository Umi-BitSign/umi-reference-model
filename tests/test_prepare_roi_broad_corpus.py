from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import prepare_roi_broad_corpus as data


def pair(source="asl_stem_original", ident="training-1", signer="P100", reference="short phrase"):
    indices = np.arange(0, 7, 2, dtype=np.int64)
    digest = hashlib.sha256(indices.tobytes()).hexdigest()
    raw = {
        "source": source,
        "example_id": ident,
        "reference": reference,
        "signer": signer,
        "role": "training",
        "training_allowed": True,
        "video_fps": 25,
        "video_sha256": "clip-sha",
        "video": f"data/clips/{ident}.mp4",
    }
    return {
        "raw": raw,
        "teacher": {
            "source": source,
            "id": ident,
            "reference": reference,
            "role": "training",
            "training_allowed": True,
            "cache_sha256": "teacher-sha",
        },
        "header": {
            "source_frames": 7,
            "sampled_frames": 4,
            "source_fps": 25,
            "source_shape": [8, 12],
            "frame_indices_sha256": digest,
            "video_sha256": "clip-sha",
        },
        "alignment": {"sampled_frames": 4, "frame_indices_sha256": digest},
        "native_teacher_stride": 2,
        "feature_pretraining_allowed": True,
    }


def receipts(p):
    lm = {"clip_sha256": "clip-sha", "frames": 7, "landmarks_sha256": "lm-sha"}
    stream = {
        "example_id": p["raw"]["example_id"],
        "split": "training",
        "clip_sha256": "clip-sha",
        "encoder_cache_sha256": "teacher-sha",
        "landmarks_sha256": "lm-sha",
        "landmark_receipts_sha256": "index-sha",
        "temporal_stride": 2,
    }
    p["native_receipt_row_sha256"] = data.row_digest(stream)
    return lm, stream


def arrays(p):
    from tiny_rgb_visual_teacher import STREAM_WIDTHS

    return {
        "pixels": np.zeros((4, 3, 64, 64, 3), dtype=np.uint8),
        "frame_indices": np.arange(0, 7, 2, dtype=np.int64),
        **{name: np.ones((4, width), dtype=np.float32) for name, width in STREAM_WIDTHS.items()},
    }


def test_selection_reserves_STEM_signers_but_not_names_in_other_sources():
    inventory = [
        pair(ident="reserved", signer="P4"),
        pair(ident="diagnostic", signer="P19"),
        pair(ident="short"),
        pair("openasl_scale", "other", "P4", "one two three four five six"),
        pair("how2sign_precut_expansion", "short-how2", 8),
        pair("how2sign_new", "missing-landmarks", 8),
        pair(ident="long-caption", reference="one two three four five six seven"),
        pair(ident="single-word", reference="word"),
    ]
    selected, counts = data.select(inventory, ["P4", "P19"])
    assert {p["raw"]["example_id"] for p in selected} == {"short", "other", "short-how2"}
    assert counts["eligible_after_reserved_signers"] == 6
    assert counts["metadata_pairs_in_existing_landmark_families"] == 5
    assert counts["eligible_pairs_awaiting_landmark_binding"] == 1
    assert counts["selected_by_word_band"] == {"2-3": 2, "4-6": 1}


@pytest.mark.parametrize("mutation", ["role", "permission", "teacher", "duplicate"])
def test_selection_rejects_unauthorized_or_duplicate_rows(mutation):
    p = pair()
    inventory = [p]
    if mutation == "role":
        p["raw"]["role"] = "development"
    elif mutation == "permission":
        p["raw"]["training_allowed"] = False
    elif mutation == "teacher":
        p["teacher"]["reference"] = "other caption"
    else:
        inventory.append(copy.deepcopy(p))
    with pytest.raises(ValueError):
        data.select(inventory, [])


def test_long_selected_clips_stop_instead_of_being_cropped_or_dropped():
    p = pair()
    p["header"]["sampled_frames"] = 150
    with pytest.raises(ValueError, match="truncate"):
        data.select([p], [])


def test_indexes_reject_duplicate_or_missing_identity():
    assert data.indexed([{"sentence_name": "how2"}]) == {"how2": {"sentence_name": "how2"}}
    for values in ([{}, {}], [{"example_id": "x"}, {"example_id": "x"}]):
        with pytest.raises(ValueError):
            data.indexed(values)


def test_binding_accepts_the_canonical25fps_derivative_and_preserves_openasl_names():
    p = pair("how2sign_precut_expansion")
    lm, stream = receipts(p)
    stream["resampled_clip_sha256"] = stream.pop("clip_sha256")
    p["native_receipt_row_sha256"] = data.row_digest(stream)
    assert data.binding(p, lm, stream, "index-sha") == "training-1"
    p = pair("openasl_scale", "source-00:01:20.000")
    p["raw"]["video"] = "data/openasl-scale/clips-25fps/openasl-scale-0012-source.mp4"
    lm, stream = receipts(p)
    lm["clip_id"] = "openasl-scale-0012-source"
    assert data.binding(p, lm, stream, "index-sha") == "openasl-scale-0012-source"


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "development"),
        ("temporal_stride", 1),
        ("clip_sha256", "wrong"),
        ("landmarks_sha256", "wrong"),
        ("encoder_cache_sha256", "wrong"),
        ("landmark_receipts_sha256", "wrong"),
    ],
)
def test_binding_checks_receipt_fields_even_if_row_digest_is_changed(field, value):
    p = pair()
    lm, stream = receipts(p)
    stream[field] = value
    p["native_receipt_row_sha256"] = data.row_digest(stream)
    with pytest.raises(ValueError):
        data.binding(p, lm, stream, "index-sha")


def test_binding_rejects_an_equal_length_but_different_source_or_receipt():
    p = pair()
    lm, stream = receipts(p)
    p["native_receipt_row_sha256"] = "different"
    with pytest.raises(ValueError):
        data.binding(p, lm, stream, "index-sha")
    p["native_receipt_row_sha256"] = data.row_digest(stream)
    lm["clip_sha256"] = "original24fps-sha"
    with pytest.raises(ValueError):
        data.binding(p, lm, stream, "index-sha")


def test_arrays_use_complete_exact_timeline_and_finite_native_targets():
    p = pair()
    values = arrays(p)
    assert data.validate_arrays(values, p) == sum(v.nbytes for v in values.values())


@pytest.mark.parametrize(
    "mutation",
    [
        "pixels-shape",
        "pixels-dtype",
        "reordered",
        "missing-frame",
        "wrong-stride",
        "nan-target",
        "target-width",
        "index-dtype",
        "alignment",
        "header-digest",
        "extra-array",
    ],
)
def test_array_validation_rejects_silent_alignment_or_target_changes(mutation):
    p, values = pair(), arrays(pair())
    if mutation == "pixels-shape":
        values["pixels"] = values["pixels"][:-1]
    elif mutation == "pixels-dtype":
        values["pixels"] = values["pixels"].astype(np.float32)
    elif mutation == "reordered":
        values["frame_indices"] = values["frame_indices"][::-1]
    elif mutation == "missing-frame":
        values["frame_indices"] = values["frame_indices"][:-1]
    elif mutation == "wrong-stride":
        values["frame_indices"] = np.arange(4, dtype=np.int64)
    elif mutation == "nan-target":
        values["pose_features"][0, 0] = np.nan
    elif mutation == "target-width":
        values["face_features"] = values["face_features"][:, :-1]
    elif mutation == "index-dtype":
        values["frame_indices"] = values["frame_indices"].astype(np.int32)
    elif mutation == "alignment":
        p["alignment"]["sampled_frames"] = 3
    elif mutation == "header-digest":
        p["header"]["frame_indices_sha256"] = "wrong"
    else:
        values["unknown"] = np.zeros(1)
    with pytest.raises(ValueError):
        data.validate_arrays(values, p)


def test_array_residency_bound_is_enforced(monkeypatch):
    monkeypatch.setitem(data.POLICY, "maximum_clip_arrays_bytes", 1)
    with pytest.raises(ValueError, match="resident"):
        data.validate_arrays(arrays(pair()), pair())


def test_checked_file_cannot_resolve_private_data_outside_data_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="private"):
        data.checked_file("../outside.mp4", "wrong", private=True)


def test_no_inference_scoring_or_model_training_is_claimed():
    for key in (
        "GPU_use",
        "selection_uses_development_text_or_scores",
        "translation_promotion_authorized",
    ):
        assert data.POLICY[key] is False
    for key in ("new_downloads", "new_Terra_requests", "optimizer_updates"):
        assert data.POLICY[key] == 0
    assert data.POLICY["maximum_caption_words"] == 6
    assert data.POLICY["maximum_sampled_frames"] == 149


def test_openasl_uses_its_actual_landmark_tree_not_a_guessed25fps_suffix():
    assert data.LAYOUT["openasl_scale"][1] == "data/openasl-scale/landmarks"
