from __future__ import annotations

import copy
import hashlib
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import terra_mixed_native_input as mixed
from extract_mediapipe_landmarks import write_landmarks
from training_media_artifact import media_artifact


def phrase_record(ident="phrase", frames=5, signer="P2", source="asl_stem_original"):
    indices = np.arange(0, frames, 2, dtype=np.int64)
    digest = hashlib.sha256(indices.tobytes()).hexdigest()
    raw = {
        "source": source,
        "example_id": ident,
        "source_group": f"{source}:{signer}:{ident}",
        "signer": signer,
        "reference": "Complete phrase",
        "role": "training",
        "training_allowed": True,
        "video": f"data/clips/{ident}.mp4",
        "video_sha256": "0" * 64,
        "video_fps": 25,
    }
    header = {
        **{key: raw[key] for key in ("source", "example_id", "role")},
        "source_fps": 25,
        "source_frames": frames,
        "source_shape": [480, 640],
        "sampled_frames": len(indices),
        "frame_indices_sha256": digest,
        "video_sha256": raw["video_sha256"],
    }
    pair = {
        "raw": raw,
        "header": header,
        "alignment": {
            **{
                key: raw[key]
                for key in ("source", "example_id", "source_group", "role", "training_allowed")
            },
            "sampled_frames": len(indices),
            "frame_indices_sha256": digest,
        },
        "teacher": {
            **{
                key: raw[key]
                for key in ("source", "signer", "reference", "role", "training_allowed")
            },
            "frames": len(indices),
            "stored_temporal_stride": 2,
            "cache": f"data/teacher/{ident}.npz",
            "cache_sha256": "1" * 64,
        },
        "feature_pretraining_allowed": True,
        "native_teacher_stride": 2,
    }
    return {
        "original_row": {**raw, "original_pair": pair},
        "roi_role": "training",
        "roi_training_allowed": True,
        "landmarks": {"path": f"data/landmarks/{ident}.npz", "sha256": "2" * 64},
    }


def word_case(ident="word", frames=5, signer="P2", role="training", label="word"):
    return {
        "record": {
            "original": {
                "row": {
                    "example_id": ident,
                    "source_group": f"asl_stem:{signer}:{ident}",
                    "participant": signer,
                    "reference": label,
                    "label": label,
                    "role": role,
                    "training_allowed": role == "training",
                }
            },
            "ROI_lineage_training_allowed": role == "training" and signer not in mixed.DIAGNOSTIC,
            "normalized_probe": {"decoded_frames": frames, "height": 480, "width": 640},
        }
    }


@pytest.fixture
def full_inventory():
    words = {
        "records": [
            word_case(
                f"word{i}",
                frames=(57 if i < 213 else 6) * 2,
                signer=("P1", "P2", "P3")[(i // 60) % 3],
                label=f"Word{i % 60}",
            )
            for i in range(214)
        ]
    }
    words["records"].extend(
        word_case(
            f"held-word{i}",
            signer=sorted(mixed.RESERVED)[i % 8],
            role="development",
            label=f"Word{i % 60}",
        )
        for i in range(96)
    )
    records = [phrase_record(f"phrase{i}", frames=(84 if i < 317 else 101) * 2) for i in range(318)]
    for signer in sorted(mixed.DIAGNOSTIC):
        record = phrase_record("diagnostic-" + signer, signer=signer)
        record.update(roi_role="feature_diagnostic", roi_training_allowed=False)
        records.append(record)
    phrase_plan = {
        "records": records,
        "reserved_STEM_participants": sorted(mixed.RESERVED),
        "ROI_diagnostic_participants": sorted(mixed.DIAGNOSTIC),
    }
    phrase_data = {
        "complete": True,
        "counts": {"training": 318, "feature_diagnostic": 2},
        "records": copy.deepcopy(records),
    }
    sentences = {
        "records": [
            {
                "development": {
                    "example_id": f"sentence{i}",
                    "source_group": f"sentence:{i}",
                    "role": "development",
                    "training_allowed": False,
                }
            }
            for i in range(34)
        ]
    }
    native = [
        {
            "example_id": f"native{i}",
            "source_group": f"native:{i}",
            "role": "open_development",
            "training_allowed": False,
        }
        for i in range(142)
    ]
    return words, phrase_plan, phrase_data, sentences, native


def test_full_mixed_inventory_retains_every_easier_word_and_original_phrase(full_inventory):
    original = copy.deepcopy(full_inventory)
    cases, boundaries = mixed.select_inputs(*full_inventory)
    assert len(cases) == 532 and sum(c["kind"] == "word" for c in cases) == 214
    assert sum(mixed.expected_header(case)["sampled_frames"] for case in cases) == 38876
    assert len({mixed.training_row(case)["example_id"] for case in cases}) == 532
    assert not any(any(value.values()) for value in boundaries.values())
    assert full_inventory == original


def test_mixed_inventory_rejects_repeated_identity_across_word_and_phrase_stages(full_inventory):
    _, plan, data, _, _ = full_inventory
    plan["records"][0] = phrase_record("word0", frames=168)
    data["records"][0] = copy.deepcopy(plan["records"][0])
    with pytest.raises(ValueError, match="retain all214"):
        mixed.select_inputs(*full_inventory)


@pytest.mark.parametrize("cohort", ["word", "sentence", "native", "diagnostic"])
@pytest.mark.parametrize("field", ["example_id", "source_group"])
def test_mixed_inventory_rejects_every_protected_id_and_source_group(full_inventory, cohort, field):
    words, plan, data, sentences, native = full_inventory
    source = plan["records"][0]["original_row"][field]
    if cohort == "word":
        target = words["records"][-1]["record"]["original"]["row"]
    elif cohort == "sentence":
        target = sentences["records"][0]["development"]
    elif cohort == "native":
        target = native[0]
    else:
        target = plan["records"][-1]["original_row"]
    target[field] = source
    if cohort == "diagnostic":
        data["records"][-1] = copy.deepcopy(plan["records"][-1])
    with pytest.raises(ValueError, match="boundar"):
        mixed.select_inputs(*full_inventory)


@pytest.mark.parametrize("mutation", ["word", "phrase", "diagnostic", "binding", "timeline"])
def test_mixed_inventory_rejects_omission_or_changed_original_preparation(full_inventory, mutation):
    words, plan, data, _, _ = full_inventory
    if mutation == "word":
        words["records"].pop(0)
    elif mutation == "phrase":
        plan["records"].pop(0)
        data["records"].pop(0)
    elif mutation == "diagnostic":
        plan["records"][-1]["roi_training_allowed"] = True
        data["records"][-1] = copy.deepcopy(plan["records"][-1])
    elif mutation == "binding":
        data["records"][0]["original_row"]["reference"] = "Changed caption"
    else:
        pair = plan["records"][0]["original_row"]["original_pair"]
        pair["header"]["source_frames"] += 2
        data["records"][0] = copy.deepcopy(plan["records"][0])
    with pytest.raises(ValueError):
        mixed.select_inputs(*full_inventory)


@pytest.mark.parametrize("signer", sorted(mixed.RESERVED | mixed.DIAGNOSTIC))
def test_all_reserved_and_diagnostic_STEM_signers_fail_before_media_reads(monkeypatch, signer):
    monkeypatch.setattr(mixed, "media_artifact", lambda *a: pytest.fail("read forbidden media"))
    with pytest.raises(ValueError, match="reserved roles"):
        mixed.read_phrase(phrase_record(signer=signer))


def test_signer_namespaces_remain_separate_across_source_families():
    record = phrase_record(signer="P4", source="how2sign_precut_expansion")
    assert mixed.phrase_row(record)["signer"] == "P4"


@pytest.mark.parametrize("frames", [5, 6])
@pytest.mark.parametrize("mutation", [None, "pixels", "pose"])
def test_native_phrase_RGB_pose_and_complete_odd_even_timeline(
    tmp_path, monkeypatch, frames, mutation
):
    record = phrase_record(frames=frames)
    monkeypatch.setattr(mixed, "ROOT", tmp_path)
    monkeypatch.setattr(mixed, "media_artifact", partial(media_artifact, root=tmp_path))
    pair = record["original_row"]["original_pair"]
    video, landmark, teacher = (
        tmp_path / name
        for name in (pair["raw"]["video"], record["landmarks"]["path"], pair["teacher"]["cache"])
    )
    for path in (video, landmark, teacher):
        path.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"synthetic source; readers below are fixtures")
    digest = media_artifact(video, root=tmp_path)["sha256"]
    pair["raw"]["video_sha256"] = pair["header"]["video_sha256"] = digest
    record["original_row"]["video_sha256"] = digest
    landmarks = {index: None for index in range(frames)}
    landmarks[2] = {"pose_landmarks": None}
    write_landmarks(
        landmark,
        landmarks,
        source=video,
        width=640,
        height=480,
        fps=25,
        mediapipe_version="0.10.14",
    )
    record["landmarks"] = media_artifact(landmark, root=tmp_path)
    pose = np.full((3, 14), -9999, dtype=np.float32)
    np.savez(teacher, pose_features=pose)
    pair["teacher"]["cache_sha256"] = media_artifact(teacher, root=tmp_path)["sha256"]
    sampled_dictionaries = []

    class Reader:
        def __init__(self, path, header):
            assert path == video and header["source_frames"] == frames

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Extractor:
        def __init__(self, output_size):
            assert output_size == (112, 112)

        def values(self, reader, sampled, base):
            assert list(sampled) == [0, 1, 2]
            assert sampled[0] is None and sampled[2] is None
            assert sampled[1]["pose_landmarks"] is None
            sampled_dictionaries.append(sampled)
            offset = int(reader == "native" and mutation == "pixels")
            return [
                np.full((112, 112, 3), [base + index + offset, 20, 30], dtype=np.uint8)
                for index in (0, 2, 4)
            ]

        def extract_hand_frames(self, reader, sampled):
            return self.values(reader, sampled, 10), self.values(reader, sampled, 20)

        def extract_face_frames(self, reader, sampled):
            return self.values(reader, sampled, 30)

    class Pose:
        def process_landmarks_sequence(self, sampled):
            assert all(value is sampled for value in sampled_dictionaries)
            return pose.astype(np.float64) + int(mutation == "pose")

    monkeypatch.setattr(mixed, "SampledRgbReader", Reader)
    monkeypatch.setattr(mixed, "native_helpers", lambda: (Extractor, Extractor, lambda p: "native"))
    monkeypatch.setattr(mixed, "NativeSampledReader", lambda reader, header: reader)
    monkeypatch.setattr(mixed, "pose_processor", Pose)
    if mutation:
        with pytest.raises(ValueError, match="crop pixels|recomputed pose"):
            mixed.read_phrase(record)
        return
    result = mixed.read_phrase(record)
    assert result["pixels"].shape == (3, 3, 112, 112, 3)
    np.testing.assert_array_equal(result["frame_indices"], [0, 2, 4])
    np.testing.assert_array_equal(result["pose_features"], pose)
    np.testing.assert_array_equal(
        result["pixels"][:, 0, 0, 0], [[30, 20, 30], [30, 20, 32], [30, 20, 34]]
    )
    np.testing.assert_array_equal(
        result["pixels"][0, :, 0, 0], [[30, 20, 30], [30, 20, 10], [30, 20, 20]]
    )
    mixed.require_receipt(
        {"kind": "phrase", "phrase": record},
        mixed.input_receipt({"kind": "phrase", "phrase": record}, result),
    )


def test_word_dispatch_requires_the_original_role_and_verified_array_bytes(monkeypatch):
    case = {"kind": "word", "word": word_case()}
    header = mixed.expected_header(case)
    values = {
        "pixels": np.zeros((3, 3, 112, 112, 3), dtype=np.uint8),
        "pose_features": np.full((3, 14), -9999, dtype=np.float32),
        "frame_indices": np.asarray([0, 2, 4], dtype=np.int64),
        "header": header,
    }
    case["word"]["verified_input"] = {
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
            for name, value in values.items()
            if name != "header"
        }
    }
    monkeypatch.setattr(mixed, "load_word", lambda bound, purpose: values)
    assert mixed.read_input(case) is values
    receipt = mixed.input_receipt(case, values)
    mixed.require_receipt(case, receipt)
    values["pixels"][0, 0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="easier word"):
        mixed.input_receipt(case, values)
    receipt["arrays"]["pixels"]["sha256"] = "f" * 64
    with pytest.raises(ValueError):
        mixed.require_receipt(case, receipt)
    case["word"]["record"]["original"]["row"]["role"] = "development"
    monkeypatch.setattr(mixed, "load_word", lambda *a, **k: pytest.fail("read forbidden word"))
    with pytest.raises(ValueError):
        mixed.read_input(case)
