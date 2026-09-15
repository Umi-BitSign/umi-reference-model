"""Reusable, warm SHuBERT + ByT5 inference runtime for the host worker."""

from __future__ import annotations

import hashlib
import os
import resource
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_GENERATION_NUM_BEAMS = 5

EXPECTED_ASSET_SHA256 = {
    "checkpoint-11625/model.safetensors": (
        "92c6c77ed7f2343f346c2a0cc7c025d733d47fcb1adc55150939f934d5f3b3b9"
    ),
    "dinov2hand.safetensors": "630a11c0aef6bca4f27bcb2f68dbf35ad33d365af877c1fca7c42e1e748064ac",
    "dinov2face.safetensors": "61e96b40c9b3b841520fb2fcf89b2a2fdda706d13723669d11304fc2064d295e",
}


def _rgb_crop_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Convert the crop helpers' OpenCV BGR output to DINO's RGB input."""
    output = []
    for frame in frames:
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
            or not frame.shape[0]
            or not frame.shape[1]
            or frame.dtype != np.uint8
        ):
            raise ValueError("expected a nonempty uint8 BGR crop")
        # The training pipeline wrote these BGR crops with OpenCV, then read
        # the cropped videos back as RGB for DINO. Direct inference omits that
        # video round trip, so it must restore channel order here explicitly.
        output.append(frame[..., ::-1].copy())
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SHuBERTInferenceRuntime:
    """One warm stack with parallel CPU preparation and explicitly bounded MPS use."""

    def __init__(
        self,
        project_root: Path,
        *,
        device: str = "mps",
        generation_max_length: int = 2048,
        generation_num_beams: int = DEFAULT_GENERATION_NUM_BEAMS,
        dino_batch_size: int = 128,
        model_execution_concurrency: int = 1,
        verify_assets: bool = True,
    ) -> None:
        if (
            isinstance(dino_batch_size, bool)
            or not isinstance(dino_batch_size, int)
            or dino_batch_size <= 0
        ):
            raise ValueError("dino_batch_size must be a positive integer")
        if (
            isinstance(model_execution_concurrency, bool)
            or not isinstance(model_execution_concurrency, int)
            or not 1 <= model_execution_concurrency <= 3
        ):
            raise ValueError("model_execution_concurrency must be in [1, 3]")
        self.project_root = project_root.resolve()
        self.bundle = self.project_root / "."
        self.models = self.bundle / "models"
        self.device = torch.device(device)
        self.generation_max_length = generation_max_length
        self.generation_num_beams = generation_num_beams
        self.dino_batch_size = dino_batch_size
        self._model_execution_concurrency = model_execution_concurrency
        self._thread_state = threading.local()
        self._model_execution_slots = threading.BoundedSemaphore(model_execution_concurrency)
        self._install_author_paths()
        if verify_assets:
            self._verify_assets()
        self._load_models()

    @property
    def last_diagnostics(self) -> dict[str, Any]:
        """Return diagnostics for the calling request thread only."""

        return getattr(self._thread_state, "last_diagnostics", {})

    @property
    def model_execution_concurrency(self) -> int:
        """Return the explicit bounded width for the one shared model stack."""

        return self._model_execution_concurrency

    @staticmethod
    def _check_deadline(deadline_unix_ns: int | None, stage: str) -> None:
        if deadline_unix_ns is not None and time.time_ns() >= deadline_unix_ns:
            raise TimeoutError(f"SHuBERT request deadline elapsed {stage}")

    def _install_author_paths(self) -> None:
        demo = self.project_root / "vendor/demo"
        fairseq = self.project_root / "vendor"
        dataset = self.project_root / "vendor/demo"
        for path in reversed((demo, fairseq, dataset)):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        os.environ.setdefault(
            "SHUBERT_DINOV2_SOURCE",
            str(self.project_root / "vendor/dinov2-source"),
        )

    def _verify_assets(self) -> None:
        for relative, expected in EXPECTED_ASSET_SHA256.items():
            path = self.models / relative
            if not path.is_file():
                raise FileNotFoundError(f"required SHuBERT worker asset is missing: {path}")
            actual = _sha256(path)
            if actual != expected:
                raise RuntimeError(f"SHuBERT worker asset hash mismatch: {path}")
        for relative in (
            "face_landmarker_v2_with_blendshapes.task",
            "hand_landmarker.task",
            "byt5_base/tokenizer_config.json",
            "checkpoint-11625/config.json",
        ):
            path = self.models / relative
            if not path.is_file():
                raise FileNotFoundError(f"required SHuBERT worker asset is missing: {path}")

    def _load_models(self) -> None:
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS explicitly requested by SHuBERT worker but unavailable")

        from body_features import process_pose_landmarks
        from crop_face import FaceExtractor
        from crop_hands import HandExtractor
        from dinov2_features import DINOEmbedder
        from inference import (
            ByT5Tokenizer,
            SignLanguageByT5ForConditionalGeneration,
        )
        from kpe_mediapipe import video_holistic
        from video_reader import VideoReader

        self._process_pose_landmarks = process_pose_landmarks
        self._video_holistic = video_holistic
        self._video_reader = VideoReader
        self._hand_extractor = HandExtractor()
        self._face_extractor = FaceExtractor()
        self._hand_model = DINOEmbedder(
            str(self.models / "dinov2hand.safetensors"),
            batch_size=self.dino_batch_size,
            device=str(self.device),
        )
        self._face_model = DINOEmbedder(
            str(self.models / "dinov2face.safetensors"),
            batch_size=self.dino_batch_size,
            device=str(self.device),
        )
        self._translator = SignLanguageByT5ForConditionalGeneration.from_pretrained(
            str(self.models / "checkpoint-11625"),
            cache_dir=str(self.project_root / ".cache/shubert-worker"),
            local_files_only=True,
        )
        self._tokenizer = ByT5Tokenizer.from_pretrained(
            str(self.models / "byt5_base"), local_files_only=True
        )
        self._translator.to(self.device)
        self._translator.eval()
        self.component_devices = {
            "dino_hand": str(self._hand_model.device),
            "dino_face": str(self._face_model.device),
            "shubert_byt5": str(next(self._translator.parameters()).device),
        }

    def translate_bytes(
        self,
        video: bytes,
        *,
        deadline_unix_ns: int | None = None,
    ) -> str:
        if not video:
            raise ValueError("cannot translate an empty video")
        self._check_deadline(deadline_unix_ns, "before temporary-file creation")
        temporary_root = self.project_root / "run/tmp"
        temporary_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, filename = tempfile.mkstemp(suffix=".mp4", dir=temporary_root)
        path = Path(filename)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(video)
                handle.flush()
            return self.translate_path(path, deadline_unix_ns=deadline_unix_ns)
        finally:
            path.unlink(missing_ok=True)

    def translate_path(
        self,
        path: Path,
        *,
        deadline_unix_ns: int | None = None,
    ) -> str:
        timings: dict[str, float] = {}

        def timed(name: str, operation):
            self._check_deadline(deadline_unix_ns, f"before {name}")
            started = time.perf_counter()
            result = operation()
            timings[name] = time.perf_counter() - started
            self._check_deadline(deadline_unix_ns, f"after {name}")
            return result

        total_started = time.perf_counter()
        preprocessing_started = total_started
        preprocessing_started_unix_ns = time.time_ns()
        reader = timed("video_open", lambda: self._video_reader(path))
        frames = timed("video_decode", lambda: reader.get_batch(range(len(reader))).asnumpy())
        landmarks = timed(
            "mediapipe_landmarks",
            lambda: self._video_holistic(
                frames,
                str(self.models / "face_landmarker_v2_with_blendshapes.task"),
                str(self.models / "hand_landmarker.task"),
            ),
        )
        left_frames, right_frames = timed(
            "hand_crops",
            lambda: self._hand_extractor.extract_hand_frames(frames, landmarks),
        )
        face_frames = timed(
            "face_crops", lambda: self._face_extractor.extract_face_frames(frames, landmarks)
        )
        pose = timed("body_pose", lambda: self._process_pose_landmarks(landmarks))
        timings["preprocessing_total"] = time.perf_counter() - preprocessing_started
        preprocessing_completed_unix_ns = time.time_ns()

        model_queue_started = time.perf_counter()
        with self._model_execution_slots:
            timings["model_queue_wait"] = time.perf_counter() - model_queue_started
            self._check_deadline(deadline_unix_ns, "after model queue")
            model_started = time.perf_counter()
            model_started_unix_ns = time.time_ns()
            left = timed(
                "dino_left_hand",
                lambda: self._hand_model.extract_embeddings_from_frames(
                    _rgb_crop_frames(left_frames)
                ),
            )
            right = timed(
                "dino_right_hand",
                lambda: self._hand_model.extract_embeddings_from_frames(
                    _rgb_crop_frames(right_frames)
                ),
            )
            face = timed(
                "dino_face",
                lambda: self._face_model.extract_embeddings_from_frames(
                    _rgb_crop_frames(face_frames)
                ),
            )

            def construct_tensors():
                return (
                    torch.tensor(face, dtype=torch.float32).unsqueeze(0).to(self.device),
                    torch.tensor(left, dtype=torch.float32).unsqueeze(0).to(self.device),
                    torch.tensor(right, dtype=torch.float32).unsqueeze(0).to(self.device),
                    torch.tensor(pose, dtype=torch.float32).unsqueeze(0).to(self.device),
                )

            face_tensor, left_tensor, right_tensor, pose_tensor = timed(
                "tensor_construction", construct_tensors
            )

            def generate() -> str:
                with torch.no_grad():
                    generated_ids = self._translator.generate(
                        face_features=face_tensor,
                        left_hand_features=left_tensor,
                        right_hand_features=right_tensor,
                        pose_features=pose_tensor,
                        max_length=self.generation_max_length,
                        num_beams=self.generation_num_beams,
                        early_stopping=True,
                        pad_token_id=self._tokenizer.pad_token_id,
                        eos_token_id=self._tokenizer.eos_token_id,
                    )
                return self._tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            hypothesis = timed("translation", generate)
            timings["model_execution_total"] = time.perf_counter() - model_started
            model_completed_unix_ns = time.time_ns()
        if not hypothesis.strip():
            raise RuntimeError("SHuBERT worker generated an empty hypothesis")
        timings["total"] = time.perf_counter() - total_started
        maximum_resident_set = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_memory_mb = (
            maximum_resident_set / (1024 * 1024)
            if sys.platform == "darwin"
            else maximum_resident_set / 1024
        )
        self._thread_state.last_diagnostics = {
            "frames": len(reader),
            "landmark_frames": len(landmarks),
            "landmark_frames_detected": sum(value is not None for value in landmarks.values()),
            "stream_shapes": {
                "face": list(face.shape),
                "left_hand": list(left.shape),
                "right_hand": list(right.shape),
                "body_pose": list(pose.shape),
            },
            "component_devices": self.component_devices,
            "execution_policy": {
                "cpu_preprocessing_parallel_safe": True,
                "model_execution_concurrency": self.model_execution_concurrency,
                "dino_frame_batch_size": self.dino_batch_size,
                "shared_model_stack_count": 1,
            },
            "execution_windows_unix_ns": {
                "preprocessing_started": preprocessing_started_unix_ns,
                "preprocessing_completed": preprocessing_completed_unix_ns,
                "model_started": model_started_unix_ns,
                "model_completed": model_completed_unix_ns,
            },
            "timings_seconds": timings,
            "peak_process_memory_mb": peak_memory_mb,
        }
        return hypothesis
