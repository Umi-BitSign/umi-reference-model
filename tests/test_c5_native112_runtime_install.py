"""Private runtime wiring only; no actual model loading or inference."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_native112_runtime_install as adapter


class Translator:
    def __init__(self):
        self.calls = []
        self.encoder = object()

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return object()

    def parameters(self):
        return iter([SimpleNamespace(requires_grad=False, device="cpu", dtype="torch.float32")])


def wrapper():
    base = Translator()
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    observed, budget = [], object()
    model = adapter.FixedGeneration(base, tokenizer, stopping_factory=lambda: budget,
                                    observe_tokens=observed.append)
    inputs = {name: object() for name in ("face_features", "left_hand_features",
                                          "right_hand_features", "pose_features")}
    return model, base, observed, budget, inputs


def test_generation_adds_all_frozen_options_keeps_stream_identity_and_observes_tokens():
    model, base, observed, budget, inputs = wrapper()
    result = model.generate(**inputs, max_length=2048, num_beams=5,
                            early_stopping=True, pad_token_id=0, eos_token_id=1)
    assert result is observed[0]
    assert model.encoder is base.encoder
    assert len(base.calls) == 1
    assert all(base.calls[0][k] == v for k, v in adapter.GENERATION.items())
    assert all(base.calls[0][k] is v for k, v in inputs.items())
    assert base.calls[0]["do_sample"] is False
    assert base.calls[0]["stopping_criteria"] is budget


@pytest.mark.parametrize("name,value", [
    ("length_penalty", 1.0), ("no_repeat_ngram_size", 0), ("max_length", 512),
    ("num_beams", 1), ("do_sample", True), ("early_stopping", False),
    ("pad_token_id", 9), ("eos_token_id", 9), ("max_new_tokens", 32),
    ("generation_config", {}), ("stopping_criteria", []),
])
def test_changed_or_extra_generation_options_are_rejected_before_model_call(name, value):
    model, base, _, _, inputs = wrapper()
    with pytest.raises(ValueError):
        model.generate(**inputs, **{name: value})
    assert base.calls == []


def test_missing_stream_is_rejected_before_model_call():
    model, base, _, _, inputs = wrapper()
    inputs.pop("pose_features")
    with pytest.raises(ValueError, match="complete stream"):
        model.generate(**inputs)
    assert base.calls == []


def test_installer_validates_before_changing_runtime_and_installs_only_once(monkeypatch):
    events = []
    runtime = SimpleNamespace(generation_max_length=2048, generation_num_beams=5,
                              model_execution_concurrency=1, _translator=Translator(),
                              device="cpu", component_devices={"existing": "preserved"},
                              _tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=1))
    branches = {name: SimpleNamespace(device="cpu", _check_model=lambda: events.append("validated"))
                for name in ("hands", "face")}

    def crop(runtime, side):
        assert events == ["validated", "validated"]
        events.append(("crop", side))
        return {"visual_crop_size": side}

    def stride(runtime):
        assert events[-1] == ("crop", 112)
        runtime._c5_temporal_stride = 2

    monkeypatch.setattr(adapter, "set_visual_crop_size", crop)
    monkeypatch.setattr(adapter, "install_post_mediapipe_stride", stride)
    changed = copy.deepcopy(adapter.GENERATION)
    changed["length_penalty"] = 1.0
    with pytest.raises(ValueError, match="original generation"):
        adapter.install(runtime, branches, changed, translator=Translator())
    assert events == [] and not hasattr(runtime, "_c5_native112_installed")
    translator = Translator()
    report = adapter.install(runtime, branches, adapter.GENERATION, translator=translator, stopping_factory=list)
    assert runtime._hand_model is branches["hands"] and runtime._face_model is branches["face"]
    assert isinstance(runtime._translator, adapter.FixedGeneration)
    assert runtime._translator.base is translator
    assert runtime.component_devices["existing"] == "preserved"
    assert runtime.component_devices["shubert_byt5"] == "cpu"
    assert report["actual_video_detector_feature_token_or_latency_proof"] is False
    with pytest.raises(ValueError, match="reinstall"):
        adapter.install(runtime, branches, adapter.GENERATION, translator=translator)


@pytest.mark.parametrize("field,value", [("device", "mps"), ("dtype", "torch.float16"), ("requires_grad", True)])
def test_non_cpu_or_trainable_translator_is_rejected_before_crop_changes(monkeypatch, field, value):
    parameter = SimpleNamespace(device="cpu", dtype="torch.float32", requires_grad=False)
    setattr(parameter, field, value)
    translator = Translator()
    translator.parameters = lambda: iter([parameter])
    runtime = SimpleNamespace(generation_max_length=2048, generation_num_beams=5,
                              model_execution_concurrency=1, device="cpu")
    branches = {name: object() for name in ("hands", "face")}
    with pytest.raises(ValueError, match="frozen CPU FP32 translator"):
        adapter.install(runtime, branches, adapter.GENERATION, translator=translator)
    assert not hasattr(runtime, "_c5_native112_installed")


def test_runtime_device_cannot_move_language_and_tensor_construction_to_metal():
    runtime = SimpleNamespace(generation_max_length=2048, generation_num_beams=5,
                              model_execution_concurrency=1, device="mps")
    with pytest.raises(ValueError, match="execution concurrency"):
        adapter.install(runtime, {"hands": object(), "face": object()}, adapter.GENERATION,
                        translator=Translator())


