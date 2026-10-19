"""Full-video hybrid native compatibility on two fixed training clips.

Execute only after eligible final export and complete crop-to-token parity.
Fresh offline Linux detection is included in each CPU/Metal request time.
This does not test native macOS detector portability or production deadlines.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np

import export_c5_native112_visual as exporter
from c5_native112_parity import check_input, compare_streams, compare_tokens, require
from c5_native112_runtime_install import install
from c5_native112_visual_loader import authenticate_export, load_visual_pair
from extract_mediapipe_landmarks import packed_landmarks, unpack_landmarks
from prepare_asl_stem_wiki_landmarks import DOCKER_SOCKET, IMAGE, container_command, docker_image_id

ROOT = Path(__file__).resolve().parents[1]
INDICES = (0, 1)
REPEATS = 2
BUDGET = 6 * 1024**3
MAX_REFERENCE_COPY = 4 * 1024**3
PREPARATION = ROOT / "runs/c5-native112-full-runtime-preparation-v2/validation.json"


def fingerprint(rows):
    digest = hashlib.sha256()
    for name, array in sorted(packed_landmarks(rows).items()):
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def terminal(directory):
    worker, exit_record = [exporter.read(directory / name) for name in ("worker.json", "exit.json")]
    require(type(worker.get("pid")) is int and worker["pid"] > 0
            and exit_record.get("pid") == worker["pid"]
            and type(exit_record.get("exit_code")) is int and exit_record["exit_code"] == 0
            and not list(directory.rglob("failure.json")) and not list(directory.rglob("stopped.json")),
            "full runtime requires a complete same-worker predecessor")
    return {name: exporter.artifact(directory / (name + ".json")) for name in ("worker", "exit")}


def admit(export_path, parity_path):
    record, model_plan, _ = authenticate_export(export_path)
    authority = terminal(parity_path.parent)
    parity = exporter.read(parity_path)
    ppath = exporter.bound(parity["plan"])
    p = exporter.read(ppath)
    training = exporter.read(exporter.bound(record["evidence"]["training_plan"]))
    require(parity_path.name == "result.json" and ppath == parity_path.with_name("plan.json")
            and parity.get("complete") is True and p["export"] == exporter.artifact(export_path)
            and p["case_indices"] == list(INDICES)
            and p["records"] == [training["records"][i] for i in INDICES]
            and len(parity["CPU"]) == len(parity["Metal"]) == len(INDICES)
            and parity["worker"] == authority["worker"],
            "full runtime lost the same-export fixed training parity evidence")
    require(p["CPU_threads"] == 2 and p["GPU_duty_fraction"] == .5
            and p["optimizer_updates"] == p["new_Terra_requests"] == 0,
            "predecessor resource or model scope changed")
    for name, digest in p["source_sha256"].items():
        require(exporter.sha(ROOT / name) == digest
                and exporter.sha(parity_path.parent / "source-snapshot" / name) == digest,
                "preprocessed parity sources changed")
    references = []
    for i, case in enumerate(p["records"]):
        row = case["record"]["original"]["row"]
        require(row["role"] == "training" and row["training_allowed"] is True,
                "runtime proof cannot decode reserved confirmation or development clips")
        reference_path = parity_path.with_name(f"reference-{i:02d}.json")
        reference = exporter.read(reference_path)
        exporter.bound(reference["features"])
        for kind in ("CPU", "Metal"):
            compare_tokens(reference["tokens"], parity[kind][i]["tokens"])
            require(parity[kind][i]["text"] == reference["text"]
                    and parity[kind][i]["exact_token_match"] is True,
                    "predecessor native token parity changed")
        references.append({"receipt": exporter.artifact(reference_path), **reference})
    return record, model_plan, p["records"], references, authority


def private_copy(source, target, guard):
    require(not target.exists() and source.is_dir(), "runtime copy needs a fresh private tree")
    originals = {str(p.relative_to(source)): p for p in source.rglob("*") if p.is_file()}
    require(all(not p.is_symlink() for p in source.rglob("*")), "runtime source contains aliases")
    require(sum(p.stat().st_size for p in originals.values()) <= MAX_REFERENCE_COPY,
            "runtime reference copy exceeds its four-GiB allowance")
    guard(sum(p.stat().st_size for p in originals.values()))
    shutil.copytree(source, target)
    # Only the private root may need a cache directory during local loading.
    target.chmod(0o700)
    for name, original in originals.items():
        copied = target / name
        before, after = original.stat(), copied.stat()
        require((before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
                and before.st_nlink == 1 and after.st_nlink == 1
                and exporter.sha(copied) == exporter.sha(original),
                "runtime private copy aliased or changed a reference asset")
    return {name: exporter.sha(path) for name, path in originals.items()}


def clean_timed_out_container(name, digest, image, environment):
    fields = '{"image":{{json .Image}},"label":{{json (index .Config.Labels "umi.native112.plan")}}}'
    inspected = subprocess.run(["docker", "inspect", "--format", fields, name], env=environment,
                               capture_output=True, text=True)
    if inspected.returncode:
        require("No such" in inspected.stderr, "cannot authenticate timed-out container ownership")
        return
    record = json.loads(inspected.stdout)
    require(record["label"] == digest and record["image"] == image,
            "refusing cleanup of an unrelated container")
    subprocess.run(["docker", "rm", "--force", name], env=environment,
                   capture_output=True, text=True, check=True)


class FreshDetector:
    def __init__(self, *, clip, source, output, image, digest, number, guard):
        self.clip, self.source, self.output = clip, source, output
        self.image, self.digest, self.number, self.guard = image, digest, number, guard
        self.receipt = None

    def __call__(self, frames, face_asset, hand_asset):
        require(self.receipt is None and not self.output.exists(), "fresh detector cannot replay a request")
        expected = self.source["normalized_probe"]
        require(isinstance(frames, np.ndarray)
                and frames.shape == (expected["decoded_frames"], expected["height"], expected["width"], 3)
                and frames.dtype == np.uint8, "runtime decoded timeline or frame geometry changed")
        require(exporter.sha(Path(face_asset)) == exporter.sha(ROOT / "artifacts/baseline-6fe8df59/models/face_landmarker_v2_with_blendshapes.task")
                and exporter.sha(Path(hand_asset)) == exporter.sha(ROOT / "artifacts/baseline-6fe8df59/models/hand_landmarker.task"),
                "runtime detector assets differ from the authenticated training inputs")
        self.guard(16 * 1024**2)
        self.output.mkdir(parents=True)
        command = container_command(docker_socket=DOCKER_SOCKET, clip=self.clip,
                                    output_name="landmarks.npz", landmark_dir=self.output)
        command[command.index(IMAGE)] = self.image
        name = f"umi-native112-{self.digest[:12]}-{self.number:04d}"
        position = command.index(self.image)
        command[position:position] = ["--pull", "never", "--name", name,
                                     "--label", "umi.native112.plan=" + self.digest]
        environment = {**os.environ, "DOCKER_HOST": DOCKER_SOCKET}
        with (self.output / "command.json").open("x") as f:
            json.dump({"argv": command, "timeout_seconds": 600, "automatic_retries": 0}, f, indent=2)
        try:
            completed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            clean_timed_out_container(name, self.digest, self.image, environment)
            raise
        (self.output / "container.log").write_text(completed.stdout + completed.stderr)
        require(completed.returncode == 0, "fresh offline detector failed; no retry")
        sidecar = self.output / "landmarks.npz"
        require(sidecar.stat().st_size <= 16 * 1024**2, "fresh detector sidecar exceeds per-request bound")
        from prepare_terra_word_landmarks import checked_landmarks
        metadata = checked_landmarks(sidecar, self.source)
        rows, _ = unpack_landmarks(sidecar)
        original = exporter.bound(self.source["landmarks"])
        old_rows, _ = unpack_landmarks(original)
        current, previous = fingerprint(rows), fingerprint(old_rows)
        require(current == previous, "fresh detector changed the complete legacy landmark graph")
        self.receipt = {"sidecar": exporter.artifact(sidecar), "metadata": metadata,
                        "complete_legacy_landmark_fingerprint": current,
                        "original_landmarks": self.source["landmarks"],
                        "fresh_detection": True, "cached_landmark_substitution": False}
        self.guard()
        return rows


class ObservedEmbedder:
    def __init__(self, base, expected):
        self.base, self.expected, self.calls = base, expected, []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def extract_embeddings_from_frames(self, frames):
        require(len(self.calls) < len(self.expected), "runtime repeated a visual stream")
        incoming = np.asarray(frames)
        expected = self.expected[len(self.calls)]
        require(incoming.dtype == np.uint8 and np.array_equal(incoming, expected),
                "complete runtime RGB crop pixels differ from the verified checkpoint input")
        features = self.base.extract_embeddings_from_frames(frames)
        self.calls.append({"pixels_sha256": hashlib.sha256(incoming.tobytes()).hexdigest(),
                           "features": features.copy()})
        return features


def load_private_class(project, filename, name):
    specification = importlib.util.spec_from_file_location(name, project / filename)
    require(specification is not None and specification.loader is not None and name not in sys.modules,
            "private runtime module name is unavailable or already loaded")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def checked_preparation():
    p = exporter.read(PREPARATION)
    require(p.get("schema") == "umi-studio-native112-full-runtime-preparation/1"
            and p.get("implementation_validated") is True and p.get("tests_passed", 0) >= 22
            and p.get("actual_runtime_model_or_detector_executed") is False,
            "full runtime requires the reviewed implementation preparation")
    for name, digest in p["source_sha256"].items():
        path = ROOT / name
        require(path.resolve() == path and path.is_relative_to(ROOT) and path.is_file()
                and exporter.sha(path) == digest, "full runtime preparation source changed")
    return exporter.artifact(PREPARATION)


def run(export_path, parity_path, output):
    output = output.absolute()
    data = ROOT / "data" / output.name
    require(output.resolve() == output and output.is_relative_to(ROOT / "checkpoints")
            and output != ROOT / "checkpoints" and not output.exists() and not data.exists(),
            "full runtime proof needs fresh owned output and data paths")
    record, model_plan, cases, references, authority = admit(export_path, parity_path)
    prepared = checked_preparation()
    # Heavy imports/model construction follow complete same-final eligibility.
    import torch
    from c5_native112_frame_embedder import Native112FrameEmbedder
    from compare_c5_native112_export import native_translator, reference_paths
    from launch_terra_visual_job import identity, now
    from terra_word_verified_inputs import load_verified_input
    from train_terra_c5_tail import gpu_lease

    require(all(os.environ.get(k) == "2" for k in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")),
            "full runtime requires the pinned CPU2 environment")
    require(all(os.environ.get(k) == "1" for k in
                ("PYTHONDONTWRITEBYTECODE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")),
            "full runtime requires offline loading and disabled bytecode writes")
    require(torch.backends.mps.is_available(), "full runtime requires both CPU and Metal")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    source_model = ROOT / model_plan["reference"]["bundle_root"] / "model"
    training = exporter.read(exporter.bound(record["evidence"]["training_plan"]))
    sources = dict(training["source_sha256"])
    for name in ("probe_c5_native112_full_runtime.py", "c5_native112_runtime_install.py",
                 "compare_c5_native112_export.py", "c5_native112_visual_loader.py",
                 "c5_native112_parity.py", "c5_native112_frame_embedder.py",
                 "export_c5_native112_visual.py", "c5_native112_export_values.py"):
        sources["tools/" + name] = exporter.sha(ROOT / "tools" / name)

    def guard(extra=0):
        paths = [p for folder in (output, data) if folder.exists() for p in folder.rglob("*")]
        require(all(not p.is_symlink() for p in paths), "runtime output cannot contain aliases")
        used = sum(p.stat().st_size for p in paths if p.is_file())
        require(used + extra <= BUDGET
                and shutil.disk_usage(ROOT).free >= 84 * 1024**3 + 128 * 1024**2 + BUDGET - used,
                "full runtime output cap, physical floor or remaining reserve failed")

    guard(BUDGET)
    output.mkdir(parents=True)
    data.mkdir()
    from launch_terra_visual_job import write

    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        image = docker_image_id(DOCKER_SOCKET)
        detector_plans = [exporter.read(exporter.bound(case["record"]["landmark_plan"])) for case in cases]
        require(all(p["docker_image_id"] == image
                    and all(exporter.sha(ROOT / name) == digest for name, digest in p["model_assets"].items())
                    for p in detector_plans), "fresh runtime detector image/assets changed from training")
        plan = {"schema": "umi-studio-native112-full-video-compatibility/1",
                "implementation_preparation": prepared,
                "export": exporter.artifact(export_path), "preprocessed_parity": exporter.artifact(parity_path),
                "predecessor_authority": authority, "case_indices": list(INDICES),
                "cases": cases, "references": references, "repeats": REPEATS,
                "training_detector_plans": [case["record"]["landmark_plan"] for case in cases],
                "data_dir": str(data.relative_to(ROOT)), "source_sha256": sources, "image": image,
                "CPU_threads": 2, "GPU_duty_fraction": .5, "network": "none",
                "data_budget_bytes": BUDGET, "optimizer_updates": 0, "new_Terra_requests": 0,
                "scope": "two fixed training clips; private original runtime with fresh Linux CPU detection",
                "native_macos_detector_portability_proven": False, "production_deadline_qualified": False}
        write(output / "plan.json", plan)
        digest = exporter.sha(output / "plan.json")
        for name, expected in sources.items():
            require(exporter.sha(ROOT / name) == expected, "full runtime source changed")
            target = output / "source-snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        copied = private_copy(source_model, data / "private-model", guard)
        write(output / "private-model-copy.json", {"source": str(source_model.relative_to(ROOT)), "files": copied,
                                                 "hardlinks_or_reference_modifications": False})
        project = data / "private-model"
        for key, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                           "HF_HOME": str(data / "run/hf"), "TORCH_HOME": str(data / "run/torch"),
                           "MPLCONFIGDIR": str(data / "run/matplotlib"),
                           "SHUBERT_DINOV2_SOURCE": str(source_model / "vendor/dinov2-source")}.items():
            os.environ[key] = value
        with gpu_lease():
            values = [load_verified_input(case, purpose="training") for case in cases]
            require([check_input(v) for v in values] == exporter.read(parity_path)["input_arrays"],
                    "full runtime checkpoint inputs differ from the preprocessed proof")
            expected_features = []
            for reference in references:
                with np.load(exporter.bound(reference["features"]), allow_pickle=False) as f:
                    expected_features.append({name: f[name].copy() for name in f.files})
            computed, _ = reference_paths(model_plan, record["evidence"]["initialization"], values)
            for i, fresh in enumerate(computed):
                compare_streams(expected_features[i], fresh["features"], len(values[i]["pixels"]))
                compare_tokens(references[i]["tokens"], fresh["tokens"])
                require(references[i]["text"] == fresh["text"], "trained reference decode changed")
                target = output / f"reproduced-reference-{i:02d}.npz"
                guard(sum(a.nbytes for a in fresh["features"].values()) + 65536)
                np.savez(target, **fresh["features"])
                write(output / f"reproduced-reference-{i:02d}.json", {
                    "features": exporter.artifact(target), "tokens": fresh["tokens"],
                    "text": fresh["text"], "seconds": fresh["seconds"],
                    "original_reference": references[i]["receipt"]})
            del computed
            gc.collect()
            translator = native_translator(model_plan)
            base = load_visual_pair(export_path, batch_context=nullcontext)
            module = load_private_class(project, "runtime.py", "private_native112_runtime")
            reader = load_private_class(project, "umi_video_reader.py", "private_native112_video_reader")
            runtime = module.SHuBERTInferenceRuntime(project, device="cpu", generation_max_length=2048,
                                                     generation_num_beams=5, dino_batch_size=2,
                                                     model_execution_concurrency=1, verify_assets=True)
            runtime._video_reader = reader.VideoReader
            install(runtime, base, model_plan["reference"]["generation"], translator=translator)
            gc.collect()
            strided_pose = runtime._process_pose_landmarks
            requests = []
            resources = {"mps_active_seconds": 0.0, "mps_idle_seconds": 0.0}

            @contextmanager
            def metal_context():
                torch.mps.synchronize()
                started = time.perf_counter()
                try:
                    yield
                finally:
                    torch.mps.synchronize()
                    elapsed = time.perf_counter() - started
                    resources["mps_active_seconds"] += elapsed
                    time.sleep(elapsed)
                    resources["mps_idle_seconds"] += elapsed

            for device in ("CPU", "Metal"):
                if device == "Metal":
                    with metal_context():
                        for embedder in base.values():
                            embedder.model.to("mps")
                    embedders = {key: Native112FrameEmbedder(e.model, batch_context=metal_context, batch_size=2)
                                 for key, e in base.items()}
                else:
                    embedders = base
                for repeat in range(REPEATS):
                    for index, (case, expected) in enumerate(zip(cases, values, strict=True)):
                        number = len(requests)
                        detector = FreshDetector(clip=exporter.bound(case["record"]["clip"]), source=case["record"],
                                                 output=data / f"request-{number:03d}", image=image,
                                                 digest=digest, number=number, guard=guard)
                        hands = ObservedEmbedder(embedders["hands"],
                                                 [expected["pixels"][:, 1], expected["pixels"][:, 2]])
                        face = ObservedEmbedder(embedders["face"], [expected["pixels"][:, 0]])
                        tokens, poses = [], []

                        def observed_pose(landmarks):
                            result = strided_pose(landmarks)
                            poses.append(np.asarray(result, dtype=np.float32).copy())
                            return result

                        runtime._hand_model, runtime._face_model = hands, face
                        runtime._video_holistic, runtime._process_pose_landmarks = detector, observed_pose

                        def observed_tokens(value):
                            require(value.ndim == 2 and value.shape[0] == 1,
                                    "runtime generation changed its one-sequence token contract")
                            tokens.append(value[0].detach().cpu().tolist())

                        runtime._translator.observe_tokens = observed_tokens
                        runtime.component_devices = {"dino_hand": str(hands.device), "dino_face": str(face.device),
                                                     "shubert_byt5": "cpu"}
                        started = time.perf_counter()
                        text = runtime.translate_path(ROOT / case["record"]["clip"]["path"])
                        elapsed = time.perf_counter() - started
                        require(len(hands.calls) == 2 and len(face.calls) == len(tokens) == len(poses) == 1
                                and detector.receipt is not None, "native runtime request coverage changed")
                        actual = {"face_features": face.calls[0]["features"],
                                  "left_hand_features": hands.calls[0]["features"],
                                  "right_hand_features": hands.calls[1]["features"], "pose_features": poses[0]}
                        features = compare_streams(expected_features[index], actual, len(expected["pixels"]))
                        token_report = compare_tokens(references[index]["tokens"], tokens[0])
                        require(text == references[index]["text"], "native full-video decoded text changed")
                        feature_path = output / f"features-{number:03d}.npz"
                        guard(sum(a.nbytes for a in actual.values()) + 65536)
                        np.savez(feature_path, **actual)
                        result = {"device": device, "repeat": repeat, "case_index": index,
                                  "case": case, "seconds": elapsed, "fresh_detector": detector.receipt,
                                  "crop_pixels_sha256": {"left": hands.calls[0]["pixels_sha256"],
                                                        "right": hands.calls[1]["pixels_sha256"],
                                                        "face": face.calls[0]["pixels_sha256"]},
                                  "feature_arrays": exporter.artifact(feature_path), "feature_comparison": features,
                                  "tokens": tokens[0], "text": text, **token_report,
                                  "runtime_diagnostics": runtime.last_diagnostics}
                        write(output / f"request-{number:03d}.json", result)
                        requests.append(result)
                        guard()
                        print(json.dumps({"requests": len(requests), "of": 8, "device": device,
                                          "case": index, "seconds": elapsed}), flush=True)
            require(len(requests) == 8 and all(exporter.sha(ROOT / n) == h for n, h in sources.items()),
                    "runtime proof lost complete request or frozen source coverage")
            require(all(exporter.sha(project / name) == h and exporter.sha(source_model / name) == h
                        for name, h in copied.items()), "runtime copy or immutable reference changed")
            require(exporter.artifact(export_path) == plan["export"]
                    and exporter.artifact(parity_path) == plan["preprocessed_parity"],
                    "runtime export or predecessor bytes changed")
            exporter.verify_evidence(record)
            write(output / "result.json", {
                "schema": plan["schema"], "complete": True, "plan": exporter.artifact(output / "plan.json"),
                "requests": [exporter.artifact(output / f"request-{i:03d}.json") for i in range(8)],
                "full_video_crop_pose_feature_token_parity": True, "fresh_linux_detection_per_request": True,
                "trained_checkpoint_references_independently_reproduced": [
                    exporter.artifact(output / f"reproduced-reference-{i:02d}.json") for i in INDICES],
                "mean_request_seconds": {device: sum(r["seconds"] for r in requests if r["device"] == device) / 4
                                         for device in ("CPU", "Metal")},
                "resources": resources, "latency_scope": "includes fresh container startup/detection, host decode/crops, DINO, CPU translation, integrity checks and Metal duty idle; excludes model loading; no warmup requests excluded",
                "native_macos_detector_portability_proven": False, "production_deadline_qualified": False,
                "independent_semantic_confirmation_completed": False, "promotion_authorized": False,
                "optimizer_updates": 0, "new_Terra_requests": 0})
        code = 0
    except BaseException as error:
        write(output / "failure.json", {"error": str(error), "automatic_retries": 0})
        raise
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("export", "preprocessed-parity", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.export.absolute(), args.preprocessed_parity.absolute(), args.output)


if __name__ == "__main__":
    main()

