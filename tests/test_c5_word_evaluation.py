from __future__ import annotations

import copy
import hashlib
import json
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_word_evaluation_io as word_io
import c5_word_final as final
import evaluate_c5_word_sentences as sentences
import test_c5_word_training as training_fixtures
from terra_training_recovery import artifact


@pytest.fixture
def tiny_training(tmp_path, monkeypatch):
    return training_fixtures.tiny_training.__wrapped__(tmp_path, monkeypatch)


@pytest.fixture
def completed_word_run(tiny_training, monkeypatch):
    env = tiny_training
    trainer = training_fixtures.trainer
    plan = env.plan
    plan["recipe"] = copy.deepcopy(training_fixtures.word.RECIPE)
    plan["unique_training_clips"] = len(plan["records"])
    plan["maximum_optimizer_updates"] = len(plan["batches"])
    plan["model_plan"]["reference"]["bundle_root"] = "artifacts/original"
    scope = env.path.parent / "scope.json"
    scope.write_text(
        json.dumps(
            {
                "parameters": [
                    {"parameter": name.removeprefix("vision."), "shape": list(value.shape)}
                    for name, value in env.template.trainable_parameters().items()
                ]
            }
        )
    )
    descriptor = partial(artifact, root=env.path.parents[2])
    plan["model_plan"]["visual_scope"] = descriptor(scope)
    env.path.write_text(json.dumps(plan))
    resources = {
        "current_allowed_gpu_duty_fraction": 0.5,
        "current_cpu_threads": 2,
        "memory_policy": "framework_default_no_artificial_six_GiB_cap",
        "mps_active_seconds": 1.0,
        "mps_idle_seconds": 1.0,
        "maximum_allocated_bytes": 1,
        "maximum_driver_allocated_bytes": 1,
    }
    monkeypatch.setattr(training_fixtures.FakeDuty, "receipt", lambda self: resources.copy())
    trainer.fit(env.path, env.output, plan)
    training_fixtures.word.write(env.output / "worker.json", {"pid": 123})
    training_fixtures.word.write(env.output / "exit.json", {"pid": 123, "exit_code": 0})
    root = env.path.parents[2]
    monkeypatch.setattr(final, "ROOT", root)
    monkeypatch.setattr(final, "artifact", descriptor)
    monkeypatch.setattr(final, "checked_training", lambda path: plan)
    from transformers import ByT5Tokenizer

    monkeypatch.setattr(
        ByT5Tokenizer, "from_pretrained", lambda *a, **k: training_fixtures.Tokenizer()
    )
    return env


def test_final_verifier_rejoins_all_rewards_and_actual_checkpoints(completed_word_run):
    env = completed_word_run
    plan, result, evidence = final.checked_final(env.path, env.output / "result.json")
    assert plan == env.plan and result["training_examples"] == 24
    assert len([value for value in evidence if value["path"].endswith("/response.json")]) == 8
    assert len([value for value in evidence if value["path"].endswith("/rewards.json")]) == 8
    assert (
        len([value for value in evidence if value["path"].endswith("-optimizer.safetensors")]) == 4
    )


@pytest.mark.parametrize(
    "file,change",
    [
        ("result.json", lambda value: value.update(training_examples=18)),
        ("result.json", lambda value: value.update(optimizer_updates=7)),
        (
            "result.json",
            lambda value: value.update(all_fixed_backbones_and_unused_masks_unchanged=False),
        ),
        ("result.json", lambda value: value.update(visual_final_sha256="wrong")),
        ("run.json", lambda value: value.update(initial_visual_sha256="other")),
        ("exit.json", lambda value: value.update(exit_code=1)),
        ("visual-after-000024.json", lambda value: value.update(complete=False)),
        ("visual-after-000024.json", lambda value: value.update(committed_batch_RNG=False)),
        ("batch-0000/pool-00.json", lambda value: value.update(seed=-1)),
        ("batch-0000/rewards.json", lambda value: value["qualities"][0].__setitem__(1, 4)),
        ("batch-0000/request-000/request.json", lambda value: value.update(model="fallback")),
        ("batch-0000/request-000/scores.json", lambda value: value["scores"].__setitem__(0, 8)),
    ],
)
def test_final_verifier_rejects_partial_changed_or_unauthenticated_training(
    completed_word_run, file, change
):
    env = completed_word_run
    training_fixtures.overwrite(env.output / file, change)
    with pytest.raises((ValueError, RuntimeError)):
        final.checked_final(env.path, env.output / "result.json")


@pytest.mark.parametrize("mutation", ["missing_batch", "extra_batch", "missing_pool", "stopped"])
def test_final_verifier_rejects_incomplete_or_stopped_actual_inventory(
    completed_word_run, mutation
):
    env = completed_word_run
    if mutation == "missing_batch":
        (env.output / "batch-0000").rename(env.output / "removed-batch")
    elif mutation == "extra_batch":
        (env.output / "batch-9999").mkdir()
    elif mutation == "missing_pool":
        (env.output / "batch-0000/pool-00.json").unlink()
    else:
        (env.output / "stopped.json").write_text("{}")
    with pytest.raises(ValueError):
        final.checked_final(env.path, env.output / "result.json")


@pytest.mark.parametrize("mutation", ["missing_step", "order", "risk", "tied", "norm", "resource"])
def test_final_verifier_rejects_counter_risk_gradient_or_resource_changes(
    completed_word_run, mutation
):
    env = completed_word_run
    path = env.output / "steps.jsonl"
    steps = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "missing_step":
        steps.pop()
    elif mutation == "order":
        steps[0]["training_record_indices"].reverse()
    elif mutation == "risk":
        steps[0]["before_update_risk"][0]["expected_terra_error"] = float("nan")
    elif mutation == "tied":
        steps[0]["before_update_risk"][0]["zero_reward_contrast"] = True
    elif mutation == "norm":
        steps[0]["gradient_norm_before_clip"] = 0
    else:
        steps[0]["resources"]["current_allowed_gpu_duty_fraction"] = 1.0
    path.write_text("\n".join(json.dumps(step) for step in steps) + "\n")
    with pytest.raises(ValueError):
        final.checked_final(env.path, env.output / "result.json")


def test_fixed_final_cannot_substitute_recovery_or_earlier_epoch(completed_word_run):
    env = completed_word_run
    result_path = env.output / "result.json"
    result = json.loads(result_path.read_text())
    result["visual"] = result["checkpoints"][-2]
    result["checkpoints"][-1] = result["visual"]
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        final.checked_final(env.path, result_path)


class TinyAppearance(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        self.bias = nn.Parameter(torch.ones(2))
        self.mask_token = nn.Parameter(torch.full((1, 2), 7.0), requires_grad=False)


@pytest.fixture
def appearance_case(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    target = tmp_path / "checkpoints/final.safetensors"
    target.parent.mkdir()
    initial = TinyAppearance()
    values = {"vision.weight": torch.full((2, 2), 3.0), "vision.bias": torch.full((2,), 2.0)}
    save_file(values, str(target))
    expected = TinyAppearance()
    expected.weight.data.copy_(values["vision.weight"])
    expected.bias.data.copy_(values["vision.bias"])
    descriptor = partial(artifact, root=tmp_path)
    monkeypatch.setattr(sentences, "ROOT", tmp_path)
    monkeypatch.setattr(sentences, "artifact", descriptor)
    monkeypatch.setattr(sentences.native, "appearance_model", lambda *args: copy.deepcopy(initial))
    plan = {
        "candidate_visual": descriptor(target),
        "candidate_visual_state_sha256": sentences.state_digest(expected.state_dict()),
        "model_plan": {"expected_visual_gradient_names": list(values)},
    }
    return plan, target, initial


def test_sentence_loader_installs_final_values_and_preserves_original_masks(appearance_case):
    plan, _, initial = appearance_case
    model = sentences.candidate_appearance(plan, object(), object())
    assert (model.weight == 3).all() and (model.bias == 2).all()
    torch.testing.assert_close(model.mask_token, initial.mask_token, rtol=0, atol=0)
    assert model.mask_token.requires_grad is False


@pytest.mark.parametrize("mutation", ["missing", "extra", "dtype", "shape", "nan", "hash"])
def test_sentence_loader_rejects_incomplete_or_changed_final_state(appearance_case, mutation):
    from safetensors.torch import load_file, save_file

    plan, target, _ = appearance_case
    values = load_file(str(target))
    if mutation == "missing":
        values.pop("vision.bias")
    elif mutation == "extra":
        values["encoder.weight"] = torch.ones(1)
    elif mutation == "dtype":
        values["vision.bias"] = values["vision.bias"].double()
    elif mutation == "shape":
        values["vision.bias"] = torch.ones(3)
    elif mutation == "nan":
        values["vision.bias"][0] = float("nan")
    else:
        plan["candidate_visual_state_sha256"] = "different"
    if mutation != "hash":
        save_file(values, str(target))
        plan["candidate_visual"] = sentences.artifact(target)
    with pytest.raises(ValueError):
        sentences.candidate_appearance(plan, object(), object())


def test_sentence_inference_keeps_original_scientific_bytecode_and_globals():
    original = sentences.native.infer
    bound = sentences.bound_inference()
    assert bound.__code__ is original.__code__
    assert (
        bound.__defaults__ == original.__defaults__
        and bound.__kwdefaults__ == original.__kwdefaults__
    )
    replacements = {"appearance_model", "checked", "storage"}
    assert bound.__globals__.keys() == original.__globals__.keys()
    assert all(
        bound.__globals__[name] is value
        for name, value in original.__globals__.items()
        if name not in replacements
    )
    assert bound.__globals__["appearance_model"] is sentences.candidate_appearance
    assert bound.__globals__["storage"] is word_io.storage


def test_sentence_plan_adapter_does_not_relabel_original_initializer(monkeypatch):
    seen = []
    monkeypatch.setattr(sentences, "bound_inference", lambda: lambda *args: seen.append(args[2]))
    plan = {
        "candidate_visual_state_sha256": "trained",
        "model_plan": {"pretrained_state_sha256": "original"},
    }
    sentences.infer(Path("plan"), Path("output"), plan)
    assert "pretrained_state_sha256" not in plan
    assert seen[0]["pretrained_state_sha256"] == "trained"
    assert seen[0]["model_plan"]["pretrained_state_sha256"] == "original"


def test_sentence_freeze_waits_for_complete_training_without_creating_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(sentences, "ROOT", tmp_path)
    monkeypatch.setattr(
        sentences,
        "checked_final",
        lambda *args: (_ for _ in ()).throw(ValueError("training incomplete")),
    )
    target = tmp_path / "runs/evaluation/plan.json"
    with pytest.raises(ValueError, match="incomplete"):
        sentences.freeze(target, Path("training"), Path("result"))
    assert not target.parent.exists()


@pytest.fixture
def stream_case(tmp_path, monkeypatch):
    record = {
        "original": {
            "row": {
                "example_id": "word",
                "participant": "P4",
                "role": "development",
                "training_allowed": False,
                "source_group": "asl_stem:P4:article",
                "reference": "Word",
                "label": "word",
            }
        },
        "ROI_lineage_training_allowed": False,
        "normalized_probe": {"decoded_frames": 5, "width": 640, "height": 480},
    }
    arrays = {
        **{name: np.ones((3, 384), dtype=np.float32) for name in word_io.STREAMS},
        "pose_features": np.full((3, 14), -9999, dtype=np.float32),
        "frame_indices": np.asarray([0, 2, 4], dtype=np.int64),
    }
    case = {
        "record": record,
        "verified_input": {
            "arrays": {
                name: {"sha256": hashlib.sha256(value.tobytes()).hexdigest()}
                for name, value in arrays.items()
                if name in ("pose_features", "frame_indices")
            }
        },
    }
    monkeypatch.setattr(word_io, "ROOT", tmp_path)
    monkeypatch.setattr(word_io, "artifact", partial(artifact, root=tmp_path))
    monkeypatch.setattr(word_io, "storage", lambda *a, **k: {})
    path = tmp_path / "checkpoints/controls/streams.npz"
    path.parent.mkdir(parents=True)
    return case, arrays, path


def test_original_word_stream_cache_preserves_all_verified_pose_and_indices(stream_case):
    case, arrays, path = stream_case
    descriptor = word_io.save_streams(path, arrays, case)
    actual = word_io.checked_streams(descriptor, case)
    for name, value in arrays.items():
        np.testing.assert_array_equal(actual[name], value)
    with pytest.raises(FileExistsError):
        word_io.save_streams(path, arrays, case)


@pytest.mark.parametrize(
    "mutation", ["pose", "indices", "dtype", "frames", "nan", "missing", "role"]
)
def test_original_word_stream_cache_rejects_changed_timeline_pose_or_scope(stream_case, mutation):
    case, arrays, _ = stream_case
    arrays = copy.deepcopy(arrays)
    if mutation == "pose":
        arrays["pose_features"][0, 0] = 0
    elif mutation == "indices":
        arrays["frame_indices"][1] = 1
    elif mutation == "dtype":
        arrays["face_features"] = arrays["face_features"].astype(np.float64)
    elif mutation == "frames":
        arrays["face_features"] = arrays["face_features"][:-1]
    elif mutation == "nan":
        arrays["face_features"][0, 0] = float("nan")
    elif mutation == "missing":
        del arrays["left_hand_features"]
    else:
        case["record"]["original"]["row"].update(role="training")
    with pytest.raises(ValueError):
        word_io.validate_streams(arrays, case)


def test_original_word_cache_changed_bytes_fail_before_array_loading(stream_case, monkeypatch):
    case, arrays, path = stream_case
    descriptor = word_io.save_streams(path, arrays, case)
    path.write_bytes(b"changed")
    monkeypatch.setattr(
        word_io.np, "load", lambda *a, **k: pytest.fail("read unauthenticated cache")
    )
    with pytest.raises(ValueError):
        word_io.checked_streams(descriptor, case)


@pytest.mark.parametrize("native_matches", [True, False])
def test_word_224_controls_decode_original_frames_and_preserve_verified_pose(
    stream_case, monkeypatch, native_matches
):
    from training_media_artifact import media_artifact

    case, arrays, path = stream_case
    root = path.parents[2]
    video, landmark = root / "data/word.mp4", root / "data/word.npz"
    video.parent.mkdir()
    video.write_bytes(b"original complete video fixture")
    landmark.write_bytes(b"landmark fixture")
    record = case["record"]
    record["clip"] = media_artifact(video, root=root)
    record["landmarks"] = media_artifact(landmark, root=root)
    monkeypatch.setattr(word_io, "media_artifact", partial(media_artifact, root=root))
    header = word_io.expected_header(record)
    verified = {
        "header": header,
        "pixels": np.zeros((3, 3, 112, 112, 3), dtype=np.uint8),
        "pose_features": arrays["pose_features"],
        "frame_indices": arrays["frame_indices"],
    }
    monkeypatch.setattr(
        word_io, "load_verified_input", lambda case, purpose: copy.deepcopy(verified)
    )
    frames = {i: None for i in range(5)}
    frames[2] = {"pose_landmarks": None}
    metadata = {
        "source_sha256": record["clip"]["sha256"],
        "frames": 5,
        "fps": 25,
        "height": 480,
        "width": 640,
    }
    monkeypatch.setattr(word_io, "unpack_landmarks", lambda path: (frames, metadata))

    class Reader:
        def __init__(self, filename, actual):
            assert filename == video and actual == header

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Extractor:
        def __init__(self, output_size):
            assert output_size == (224, 224)

        def values(self, reader, sampled, base):
            assert sampled == {0: None, 1: {"pose_landmarks": None}, 2: None}
            offset = int(reader == "native" and not native_matches)
            return [
                np.full((224, 224, 3), [i + base + offset, 20, 30], dtype=np.uint8)
                for i in (0, 2, 4)
            ]

        def extract_hand_frames(self, reader, sampled):
            return self.values(reader, sampled, 10), self.values(reader, sampled, 20)

        def extract_face_frames(self, reader, sampled):
            return self.values(reader, sampled, 30)

    monkeypatch.setattr(word_io, "SampledRgbReader", Reader)
    monkeypatch.setattr(word_io, "NativeSampledReader", lambda reader, header: "native")
    monkeypatch.setattr(
        word_io, "native_helpers", lambda: (Extractor, Extractor, lambda path: "native")
    )
    if not native_matches:
        with pytest.raises(ValueError, match="streamed/reference"):
            word_io.original_control_input(case)
        return
    result = word_io.original_control_input(case)
    assert result["pixels"].shape == (3, 3, 224, 224, 3)
    np.testing.assert_array_equal(
        result["pixels"][:, 0, 0, 0], [[30, 20, 30], [30, 20, 32], [30, 20, 34]]
    )
    np.testing.assert_array_equal(
        result["pixels"][0, :, 0, 0], [[30, 20, 30], [30, 20, 10], [30, 20, 20]]
    )
    np.testing.assert_array_equal(result["pose_features"], arrays["pose_features"])
    np.testing.assert_array_equal(result["frame_indices"], [0, 2, 4])
    assert not np.any(verified["pixels"])  # Control pixels came from original-frame crops.


def test_word_control_rejects_training_roles_before_verified_input_access(stream_case, monkeypatch):
    case, _, _ = stream_case
    case["record"]["original"]["row"]["role"] = "training"
    monkeypatch.setattr(
        word_io, "load_verified_input", lambda *a, **k: pytest.fail("read forbidden role")
    )
    with pytest.raises(ValueError):
        word_io.original_control_input(case)


def test_word_evaluation_storage_retains_eighty_GiB_floor(tmp_path):
    exact = sum(word_io.STORAGE.values())
    word_io.storage(root=tmp_path, disk_usage=lambda root: SimpleNamespace(free=exact))
    with pytest.raises(RuntimeError):
        word_io.storage(root=tmp_path, disk_usage=lambda root: SimpleNamespace(free=exact - 1))
    with pytest.raises(RuntimeError):
        word_io.storage(
            word_io.STORAGE["output_budget_bytes"] + 1,
            root=tmp_path,
            disk_usage=lambda root: SimpleNamespace(free=exact),
        )


@pytest.mark.parametrize(
    "function,original,replacements",
    [
        ("run_inference", "native.run", {"checked", "infer", "storage"}),
        ("run_scores", "scoring.run", {"checked_scores", "storage"}),
        ("score_specification", "scoring.score_specification", {"checked"}),
    ],
)
def test_sentence_rebinding_keeps_generation_rubric_and_metrics(
    monkeypatch, function, original, replacements
):
    actual_factory = sentences.FunctionType
    captured = []

    def factory(*args):
        result = actual_factory(*args)
        captured.append(result)
        return lambda *args, **kwargs: None

    monkeypatch.setattr(sentences, "FunctionType", factory)
    getattr(sentences, function)(Path("plan"), Path("output"))
    module, name = original.split(".")
    expected = getattr(getattr(sentences, module), name)
    actual = captured[0]
    assert actual.__code__ is expected.__code__
    assert all(
        actual.__globals__[key] is value
        for key, value in expected.__globals__.items()
        if key not in replacements
    )


@pytest.mark.parametrize("failure", ["inference", "scores", None])
def test_sentence_pipeline_serializes_children_and_never_retries(tmp_path, monkeypatch, failure):
    import probe_roi_seen_translation

    monkeypatch.setattr(sentences, "ROOT", tmp_path)
    monkeypatch.setattr(sentences, "artifact", partial(artifact, root=tmp_path))
    monkeypatch.setattr(sentences, "storage", lambda *a, **k: {})
    plan = tmp_path / "runs/evaluation/plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text("{}")
    output = tmp_path / "checkpoints/evaluation"
    training_result = {"path": "checkpoints/train/result.json", "sha256": "fixed"}
    monkeypatch.setattr(sentences, "checked", lambda path: {"training_result": training_result})
    calls, freezes = [], []

    def child(command, check):
        mode = command[command.index("--mode") + 1]
        calls.append(mode)
        directory = Path(command[command.index("--output") + 1])
        directory.mkdir()
        if mode == failure:
            return SimpleNamespace(returncode=1)
        filename = "receipt.json" if mode == "inference" else "decision.json"
        (directory / filename).write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sentences, "subprocess", SimpleNamespace(run=child))

    def score_specification(path, decoded):
        assert calls == ["inference"] and decoded.exists()
        return {"actual_decoded": True}

    monkeypatch.setattr(sentences, "score_specification", score_specification)

    def freeze(path, spec):
        assert spec == {"actual_decoded": True}
        freezes.append(path)
        path.parent.mkdir()
        path.write_text("{}")

    monkeypatch.setattr(probe_roi_seen_translation, "freeze", freeze)
    if failure:
        with pytest.raises(RuntimeError):
            sentences.pipeline(plan, output)
        assert json.loads((output / "exit.json").read_text())["exit_code"] == 1
        assert not (output / "complete.json").exists()
    else:
        sentences.pipeline(plan, output)
        result = json.loads((output / "complete.json").read_text())
        assert result["training_result"] == training_result and result["optimizer_updates"] == 0
    assert calls == (["inference"] if failure == "inference" else ["inference", "scores"])
    assert len(freezes) == (0 if failure == "inference" else 1)


def test_sentence_scoring_cannot_start_from_exit_zero_without_actual_inference_receipt(
    tmp_path, monkeypatch
):
    import probe_roi_seen_translation

    monkeypatch.setattr(sentences, "ROOT", tmp_path)
    monkeypatch.setattr(sentences, "checked", lambda path: {})
    monkeypatch.setattr(sentences, "storage", lambda *a, **k: {})
    plan = tmp_path / "runs/evaluation/plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text("{}")
    output = tmp_path / "checkpoints/evaluation"
    calls = []
    monkeypatch.setattr(
        sentences,
        "subprocess",
        SimpleNamespace(run=lambda *a, **k: calls.append(a[0]) or SimpleNamespace(returncode=0)),
    )
    monkeypatch.setattr(
        probe_roi_seen_translation,
        "freeze",
        lambda *a, **k: pytest.fail("froze scores without actual tokens"),
    )
    with pytest.raises(FileNotFoundError):
        sentences.pipeline(plan, output)
    assert len(calls) == 1
    assert not (output / "complete.json").exists()
