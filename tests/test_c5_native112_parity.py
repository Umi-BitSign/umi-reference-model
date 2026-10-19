from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_native112_parity as parity
import compare_c5_native112_export as command


def values():
    pixels = np.zeros((3, 3, 112, 112, 3), np.uint8)
    for stream in range(3):
        pixels[:, stream] = np.arange(3, dtype=np.uint8)[:, None, None, None] + stream * 10
    return {
        "pixels": pixels,
        "pose_features": np.arange(42, dtype=np.float32).reshape(3, 14),
        "frame_indices": np.array([0, 2, 4], np.int64),
        "header": {"source_frames": 5, "source_fps": 25},
    }


class Embedder:
    def __init__(self):
        self.seen = []

    def extract_embeddings_from_frames(self, frames):
        self.seen.append(frames.copy())
        return np.repeat(frames[:, 0, 0, 0, None].astype(np.float32), 384, axis=1)


def streams():
    result = {name: np.full((3, 384), i, np.float32) for i, name in enumerate(parity.STREAMS)}
    result["pose_features"] = np.arange(42, dtype=np.float32).reshape(3, 14)
    return result


def test_native_branch_order_and_complete_grid_retain_pose_and_pixels():
    data = values()
    before = copy.deepcopy(data)
    hands, face = Embedder(), Embedder()
    actual = parity.native_streams({"hands": hands, "face": face}, data)
    for i, name in enumerate(parity.STREAMS):
        np.testing.assert_array_equal(actual[name][:, 0], np.arange(3) + i * 10)
    np.testing.assert_array_equal(actual["pose_features"], data["pose_features"])
    assert actual["pose_features"] is not data["pose_features"]
    assert len(face.seen) == 1 and len(hands.seen) == 2
    assert parity.check_input(data) == parity.check_input(before)


@pytest.mark.parametrize(
    "fault",
    [
        "interior_gap",
        "tail_drop",
        "offset",
        "duplicate",
        "clock",
        "shape",
        "pose",
        "pixel_dtype",
        "index_dtype",
    ],
)
def test_invalid_timeline_or_surface_refuses_before_any_embedder(fault):
    data = values()
    if fault == "interior_gap":
        data["frame_indices"] = np.array([0, 4, 6], np.int64)
    elif fault == "tail_drop":
        data["pixels"] = data["pixels"][:-1]
    elif fault == "offset":
        data["frame_indices"] += 1
    elif fault == "duplicate":
        data["frame_indices"][1] = 0
    elif fault == "clock":
        data["header"]["source_fps"] = 30
    elif fault == "shape":
        data["pixels"] = np.zeros((3, 3, 224, 224, 3), np.uint8)
    elif fault == "pose":
        data["pose_features"][0, 0] = float("nan")
    elif fault == "pixel_dtype":
        data["pixels"] = data["pixels"].astype(np.float32)
    elif fault == "index_dtype":
        data["frame_indices"] = data["frame_indices"].astype(np.float32)
    embedders = {"hands": Embedder(), "face": Embedder()}
    with pytest.raises(ValueError):
        parity.native_streams(embedders, data)
    assert all(not e.seen for e in embedders.values())


@pytest.mark.parametrize(
    "fault", [None, "swap", "drop", "extra", "dtype", "nan", "pose", "pose_bits", "drift"]
)
def test_actual_feature_comparison_uses_all_frames_branches_and_pose_bits(fault):
    first = streams()
    second = copy.deepcopy(first)
    if fault == "swap":
        second["left_hand_features"], second["right_hand_features"] = (
            second["right_hand_features"],
            second["left_hand_features"],
        )
    elif fault == "drop":
        second["face_features"] = second["face_features"][:-1]
    elif fault == "extra":
        second["new"] = np.zeros((3, 384), np.float32)
    elif fault == "dtype":
        second["face_features"] = second["face_features"].astype(np.float64)
    elif fault == "nan":
        second["face_features"][-1, -1] = float("nan")
    elif fault == "pose":
        second["pose_features"][-1, -1] = np.nextafter(np.float32(41), np.float32(np.inf))
    elif fault == "pose_bits":
        second["pose_features"][0, 0] = -0.0
    elif fault == "drift":
        second["face_features"][-1, -1] += 0.01
    if fault is None:
        report = parity.compare_streams(first, second, 3)
        assert report["pose_bitwise_equal"] and report["frames"] == 3
        assert all(v["maximum_absolute_difference"] == 0 for v in report["streams"].values())
    else:
        with pytest.raises(ValueError):
            parity.compare_streams(first, second, 3)


def test_small_recorded_feature_error_passes_the_predeclared_tolerance():
    first = streams()
    second = copy.deepcopy(first)
    second["face_features"][-1, -1] = parity.ATOL / 2
    report = parity.compare_streams(first, second, 3)
    assert 0 < report["streams"]["face_features"]["maximum_absolute_difference"] < parity.ATOL


@pytest.mark.parametrize("tokens", [[0, 4, 1], [0, 5, 1], [0, 4], [], [True, 4, 1], [-1, 4, 1]])
def test_token_identity_includes_prefix_and_termination(tokens):
    if tokens == [0, 4, 1]:
        assert parity.compare_tokens([0, 4, 1], tokens)["exact_token_match"]
    else:
        with pytest.raises(ValueError):
            parity.compare_tokens([0, 4, 1], tokens)


def test_command_refuses_ineligible_export_before_models_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(command, "ROOT", tmp_path)
    monkeypatch.setattr(
        command,
        "authenticate_export",
        lambda _: (_ for _ in ()).throw(ValueError("ineligible final")),
    )
    output = tmp_path / "checkpoints/parity"
    with pytest.raises(ValueError, match="ineligible final"):
        command.run(tmp_path / "checkpoints/absent/EXPORT.json", output)
    assert not output.exists()


def test_runtime_comparison_requires_token_parity_after_real_feature_comparison(monkeypatch):
    data = values()
    embedders = {"hands": Embedder(), "face": Embedder()}
    expected = {
        "features": parity.native_streams(embedders, data),
        "tokens": [0, 4, 1],
        "text": "test",
    }
    monkeypatch.setattr(command, "decode_native", lambda *args: ("test", [0, 4, 1]))
    result = command.compare_runtime(embedders, None, None, {}, [data], [expected])
    assert result[0]["exact_token_match"] and result[0]["seconds"] >= 0
    monkeypatch.setattr(command, "decode_native", lambda *args: ("different", [0, 5, 1]))
    with pytest.raises(ValueError, match="token IDs differ"):
        command.compare_runtime(embedders, None, None, {}, [data], [expected])


@pytest.mark.parametrize("fault", [None, "token_mismatch", "load_failure", "dependency_changed"])
def test_complete_command_transaction_with_synthetic_model_and_Metal_admission(
    tmp_path, monkeypatch, fault
):
    """Exercise actual orchestration/serialization; no real model, device or data."""
    import json
    from contextlib import contextmanager
    from types import SimpleNamespace

    monkeypatch.setattr(command, "ROOT", tmp_path)
    monkeypatch.setattr(command.exporter, "ROOT", tmp_path)
    tool_names = (
        "compare_c5_native112_export.py",
        "c5_native112_parity.py",
        "c5_native112_visual_loader.py",
        "c5_native112_frame_embedder.py",
        "export_c5_native112_visual.py",
        "c5_native112_export_values.py",
    )
    (tmp_path / "tools").mkdir()
    for name in tool_names:
        (tmp_path / "tools" / name).write_text("# synthetic pinned source\n")
    monkeypatch.setattr(command, "__file__", str(tmp_path / "tools" / tool_names[0]))
    plan_path = tmp_path / "runs/training/plan.json"
    plan_path.parent.mkdir(parents=True)
    dependency = tmp_path / "tools/frozen_training_dependency.py"
    dependency.write_text("# original frozen dependency\n")
    plan_path.write_text(
        json.dumps(
            {
                "records": [{"synthetic_case": 0}, {"synthetic_case": 1}],
                "source_sha256": {
                    "tools/frozen_training_dependency.py": command.exporter.sha(dependency)
                },
            }
        )
    )
    record = {
        "evidence": {"training_plan": command.exporter.artifact(plan_path), "initialization": {}}
    }
    export = tmp_path / "checkpoints/export/EXPORT.json"
    export.parent.mkdir(parents=True)
    export.write_text(json.dumps(record))
    output = tmp_path / "checkpoints/parity"
    calls = []

    def admission(path):
        assert path == export and not output.exists()
        calls.append("admission")
        return record, {"reference": {"generation": {}}}, {}

    monkeypatch.setattr(command, "authenticate_export", admission)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        monkeypatch.setenv(key, "2")
    monkeypatch.setattr(
        command.exporter.shutil, "disk_usage", lambda _: SimpleNamespace(free=200 * 1024**3)
    )
    monkeypatch.setattr(command.time, "sleep", lambda _: None)
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        set_num_threads=lambda _: None,
        set_num_interop_threads=lambda _: None,
        mps=SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "terra_word_verified_inputs",
        SimpleNamespace(load_verified_input=lambda case, purpose: values()),
    )

    @contextmanager
    def lease():
        calls.append("lease_enter")
        try:
            yield
        finally:
            calls.append("lease_exit")

    monkeypatch.setitem(sys.modules, "train_terra_c5_tail", SimpleNamespace(gpu_lease=lease))

    class Model:
        def to(self, device):
            assert device == "mps"
            calls.append("move_mps")
            return self

    class Native(Embedder):
        def __init__(self, model=None, *, batch_context=None, batch_size=2):
            super().__init__()
            self.model = Model() if model is None else model
            self.context = batch_context

        def extract_embeddings_from_frames(self, frames):
            if self.context is None:
                return super().extract_embeddings_from_frames(frames)
            with self.context():
                return super().extract_embeddings_from_frames(frames)

    monkeypatch.setitem(
        sys.modules, "c5_native112_frame_embedder", SimpleNamespace(Native112FrameEmbedder=Native)
    )

    def load(*args, **kwargs):
        calls.append("load_export")
        if fault == "load_failure":
            raise ValueError("synthetic load failure")
        return {"hands": Native(), "face": Native()}

    monkeypatch.setattr(command, "load_visual_pair", load)

    def reference(plan, initialization, cases):
        calls.append("reference")
        return (
            [
                {
                    "features": parity.native_streams({"hands": Native(), "face": Native()}, case),
                    "tokens": [0, 4, 1],
                    "text": "test",
                    "seconds": 1.0,
                }
                for case in cases
            ],
            None,
        )

    monkeypatch.setattr(command, "reference_paths", reference)

    def translator(_):
        if fault == "dependency_changed":
            dependency.write_text("# altered dependency\n")
        return object()

    monkeypatch.setattr(command, "native_translator", translator)
    monkeypatch.setattr(
        command,
        "decode_native",
        lambda *args: ("test", [0, 5, 1] if fault == "token_mismatch" else [0, 4, 1]),
    )
    if fault is None:
        command.run(export, output)
        actual = json.loads((output / "result.json").read_text())
        assert actual["complete"] and len(actual["CPU"]) == len(actual["Metal"]) == 2
        assert all(row["exact_token_match"] for row in actual["CPU"] + actual["Metal"])
        assert actual["resources"]["mps_active_seconds"] == actual["resources"]["mps_idle_seconds"]
        assert actual["end_to_end_detector_or_serving_qualified"] is False
        assert not (output / "failure.json").exists()
        assert json.loads((output / "exit.json").read_text())["exit_code"] == 0
    else:
        with pytest.raises(ValueError):
            command.run(export, output)
        assert (output / "failure.json").exists() and not (output / "result.json").exists()
        assert json.loads((output / "exit.json").read_text())["exit_code"] == 1
    assert calls[:4] == ["admission", "lease_enter", "reference", "load_export"]
    assert calls[-1] == "lease_exit"
    assert (output / "reference-features-00.npz").exists()
    assert len(list((output / "source-snapshot/tools").glob("*.py"))) == 7


@pytest.mark.parametrize(
    "fault", [None, "foreign_module", "reference", "topology", "encoder", "dtype", "load_exception"]
)
def test_native_translator_uses_pinned_class_values_and_restores_import_state(
    tmp_path, monkeypatch, fault
):
    """Real tiny CPU Torch module; the published checkpoint loader is synthetic."""
    import hashlib
    import os
    from types import SimpleNamespace

    import torch
    from torch import nn

    torch.set_num_threads(1)
    monkeypatch.setattr(command, "ROOT", tmp_path)
    model = nn.Module()
    model.encoder = nn.Linear(2, 2)
    model.output = nn.Linear(2, 2)

    def digest(state):
        return hashlib.sha256(
            b"".join(
                k.encode() + v.detach().cpu().numpy().tobytes() for k, v in sorted(state.items())
            )
        ).hexdigest()

    expected = digest(model.encoder.state_dict())
    reference = {"bundle_root": "artifacts/original"}
    plan = {"reference": reference, "initial_reference_hashes": {"encoder": expected}}
    root = tmp_path / "artifacts/original/model"
    if fault == "encoder":
        with torch.no_grad():
            model.encoder.weight.add_(1)
    if fault == "dtype":
        model.output.double()
    calls = []

    class Factory:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert path == root / "models/checkpoint-11625"
            assert kwargs == {"local_files_only": True, "use_safetensors": True}
            calls.append("load")
            if fault == "load_exception":
                raise RuntimeError("synthetic native load failure")
            return model

    monkeypatch.setitem(
        sys.modules,
        "inference",
        SimpleNamespace(
            __file__=str(
                tmp_path / "foreign.py"
                if fault == "foreign_module"
                else root / "vendor/demo/inference.py"
            ),
            SignLanguageByT5ForConditionalGeneration=Factory,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "c5_winner_reference",
        SimpleNamespace(load_reference=lambda: {} if fault == "reference" else reference),
    )
    monkeypatch.setitem(
        sys.modules,
        "c5_winner_mps_safe",
        SimpleNamespace(
            replace_fairseq_exact_gelu=lambda _: 11 if fault == "topology" else 12,
            replace_native_layer_norms=lambda _: 30,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "tiny_rgb_mps_safe", SimpleNamespace(replace_native_gelu=lambda _: 1)
    )
    monkeypatch.setitem(
        sys.modules, "probe_tiny_rgb_capacity", SimpleNamespace(state_digest=digest)
    )
    monkeypatch.setenv("SHUBERT_DINOV2_SOURCE", "/synthetic/previous-source")
    before = list(sys.path)
    if fault is None:
        assert command.native_translator(plan) is model
        assert all(not p.requires_grad for p in model.parameters())
        assert all(not m.training for m in model.modules())
    else:
        with pytest.raises((ValueError, RuntimeError)):
            command.native_translator(plan)
    assert sys.path == before
    assert os.environ["SHUBERT_DINOV2_SOURCE"] == "/synthetic/previous-source"
    assert calls == ([] if fault in ("foreign_module", "reference") else ["load"])
