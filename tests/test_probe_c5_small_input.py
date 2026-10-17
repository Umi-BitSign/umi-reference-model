from __future__ import annotations

import copy
import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import probe_c5_small_input as probe


def parent():
    return {
        "reserved_STEM_participants": ["P4"],
        "ROI_diagnostic_participants": ["P19", "P0"],
        "records": [
            {
                "roi_role": "training",
                "roi_training_allowed": True,
                "original_row": {
                    "example_id": str(i),
                    "signer": f"P{i + 1}",
                    "role": "training",
                    "training_allowed": True,
                    "original_pair": {"header": {"sampled_frames": 20}},
                },
            }
            for i in range(2)
        ],
    }


@pytest.mark.parametrize("field", ["roi_role", "roi_training_allowed", "role", "training_allowed"])
def test_training_cases_require_both_real_training_permissions(field):
    plan = parent()
    assert probe.training_cases(plan) == plan["records"]
    row = plan["records"][0]
    target = row if field.startswith("roi_") else row["original_row"]
    target[field] = False if field.endswith("allowed") else "development"
    with pytest.raises(ValueError, match="training"):
        probe.training_cases(plan)


@pytest.mark.parametrize("signer", ["P4", "P19", "P0"])
def test_training_cases_exclude_all_reserved_and_auxiliary_signers(signer):
    plan = parent()
    plan["records"][0]["original_row"]["signer"] = signer
    with pytest.raises(ValueError, match="nonreserved"):
        probe.training_cases(plan)


def test_training_cases_reject_duplicate_media_and_single_frame_cases():
    plan = parent()
    plan["records"][1] = copy.deepcopy(plan["records"][0])
    with pytest.raises(ValueError, match="duplicated"):
        probe.training_cases(plan)
    plan = parent()
    plan["records"][0]["original_row"]["original_pair"]["header"]["sampled_frames"] = 1
    with pytest.raises(ValueError):
        probe.training_cases(plan)


def test_native_packing_preserves_the_declared_surface_count_and_BGR_to_RGB_order():
    pixels = np.zeros((112, 112, 3), dtype=np.uint8)
    pixels[..., 2] = 255
    result = probe.pack_rgb([pixels, pixels], 2, 112)
    assert result.shape == (2, 112, 112, 3)
    assert (result[..., 0] == 255).all() and (result[..., 2] == 0).all()
    assert result.flags.c_contiguous and result.strides[-1] == 1
    for values, frames, side in [([pixels], 2, 112), ([pixels], 1, 224)]:
        with pytest.raises(ValueError):
            probe.pack_rgb(values, frames, side)


class LinearVisual(nn.Module):
    """Analytical gradient fixture, never a real DINO numerical qualification."""

    def __init__(self):
        super().__init__()
        self.hands, self.face = nn.Module(), nn.Module()
        for model in (self.hands, self.face):
            model.mask_token = nn.Parameter(torch.zeros(1, 384), requires_grad=False)
        self.weight = nn.Parameter(torch.linspace(0.1, 0.5, 384))

    def trainable_parameters(self):
        return {"weight": self.weight}

    def forward(self, pixels):
        return {
            name: pixels[:, i, :1].expand(-1, 384) * self.weight
            for i, name in enumerate(probe.STREAMS)
        }


def test_actual_feature_probe_matches_independent_closed_form_gradient_without_updates():
    model = LinearVisual()
    pixels = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
    before = model.weight.detach().clone()
    loss, output, gradients = probe.feature_probe(model, pixels)
    expected_gradient = torch.linspace(-1, 1, 384) * pixels.mean() / 384
    torch.testing.assert_close(gradients["weight"], expected_gradient)
    torch.testing.assert_close(model.weight, before)
    assert output.shape == (6, 384) and np.isfinite(loss)
    assert model.hands.mask_token.grad is None and model.face.mask_token.grad is None
    # A prior unrelated gradient cannot contaminate a subsequent comparison.
    model.weight.grad.fill_(999)
    _, _, repeated = probe.feature_probe(model, pixels)
    torch.testing.assert_close(repeated["weight"], expected_gradient)


def test_frozen_policy_does_not_claim_training_or_translation_qualification():
    assert probe.POLICY["training_qualification_complete"] is False
    assert probe.POLICY["translation_quality_evaluated"] is False
    assert probe.POLICY["new_Terra_requests"] == probe.POLICY["optimizer_updates"] == 0
    assert probe.POLICY["native_side"] == 112 and probe.POLICY["control_side"] == 224
    assert probe.verified_comparison.__module__ == "tiny_rgb_capacity_protocol"


def test_native_crop_path_binds_real_data_files_and_keeps_complete_RGB_geometry(
    tmp_path, monkeypatch
):
    import extract_mediapipe_landmarks
    import prepare_roi_appearance
    from training_media_artifact import media_artifact

    video, landmark = tmp_path / "data/clip.mp4", tmp_path / "data/landmarks.npz"
    video.parent.mkdir()
    video.write_bytes(b"bound video fixture")
    landmark.write_bytes(b"bound landmark fixture")
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(
        probe, "media_artifact", partial(media_artifact, root=tmp_path), raising=False
    )
    frames = np.stack([np.full((4, 4, 3), i, dtype=np.uint8) for i in range(3)])
    observed = []

    class Sampled:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return frames[::2]

        def __exit__(self, *args):
            pass

    class NativeReader:
        def __init__(self, path):
            pass

        def __len__(self):
            return len(frames)

        def get_batch(self, indices):
            return SimpleNamespace(asnumpy=lambda: frames[list(indices)])

    class Extractor:
        def __init__(self, output_size):
            self.side = output_size[0]

        def values(self, reader):
            values = []
            for i in range(len(reader)):
                source_index = int(reader[i][0, 0, 0])
                observed.append(source_index)
                values.append(
                    np.full((self.side, self.side, 3), [10 + source_index, 20, 30], dtype=np.uint8)
                )
            return values

        def extract_hand_frames(self, reader, sampled):
            values = self.values(reader)
            return values, values

        def extract_face_frames(self, reader, sampled):
            return self.values(reader)

    monkeypatch.setattr(prepare_roi_appearance, "SampledRgbReader", Sampled)
    monkeypatch.setattr(
        prepare_roi_appearance, "native_helpers", lambda: (Extractor, Extractor, NativeReader)
    )
    monkeypatch.setattr(
        extract_mediapipe_landmarks,
        "unpack_landmarks",
        lambda path: (
            {i: None for i in range(3)},
            {"source_sha256": probe.sha256(video), "frames": 3, "fps": 25},
        ),
    )
    row = {
        "landmarks": media_artifact(landmark, root=tmp_path),
        "original_row": {
            "original_pair": {
                "raw": {"video": "data/clip.mp4", "video_sha256": probe.sha256(video)},
                "header": {"source_frames": 3, "sampled_frames": 2, "source_shape": [4, 4]},
            }
        },
    }
    result = probe.crop_case(row, 112)
    assert result.shape == (2, 3, 112, 112, 3)
    assert observed == [0, 2] * 4
    np.testing.assert_array_equal(result[:, 0, 0, 0], [[30, 20, 10], [30, 20, 12]])
    landmark.write_bytes(b"changed landmark bytes")
    with pytest.raises(ValueError, match="bytes changed"):
        probe.crop_case(row, 112)


def test_legacy_loader_preloads_canonical_reader_in_a_fresh_process_without_loading_weights(
    tmp_path,
):
    code = r"""
import importlib.abc, importlib.util, json, sys
from pathlib import Path
sys.path.insert(0, "tools")
import probe_c5_small_input as probe
import prepare_roi_appearance as roi
root = Path(sys.argv[1])
probe.ROOT = root
roi.DEMO = root / "artifacts/baseline/vendor/demo"
winner = root / "artifacts/winner/model/vendor/demo"
for directory in (roi.DEMO, winner):
    directory.mkdir(parents=True)
    (directory / "video_reader.py").write_text("class VideoReader: pass\n")
(roi.DEMO / "crop_hands.py").write_text(
    "from video_reader import VideoReader\nclass HandExtractor: pass\n"
)
(roi.DEMO / "crop_face.py").write_text(
    "from video_reader import VideoReader\nclass FaceExtractor: pass\n"
)
original_spec = importlib.util.spec_from_file_location
class Loader(importlib.abc.Loader):
    def create_module(self, spec): return None
    def exec_module(self, module):
        import video_reader
        assert Path(video_reader.__file__).resolve() == (roi.DEMO / "video_reader.py").resolve()
        class Embedder:
            def __init__(self, path, batch_size, device):
                assert batch_size == 2 and device == "cpu"
                self.path = path
        module.DINOEmbedder = Embedder
def spec(name, path, *args, **kwargs):
    if name == "small_dino_legacy":
        return importlib.util.spec_from_loader(name, Loader())
    return original_spec(name, path, *args, **kwargs)
importlib.util.spec_from_file_location = spec
hands, face = probe.legacy_models({"reference": {"bundle_root": "artifacts/winner"}})
assert hands.path.endswith("dinov2hand.safetensors")
assert face.path.endswith("dinov2face.safetensors")
print("canonical reader loaded first; no model weights loaded")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "no model weights loaded" in result.stdout
