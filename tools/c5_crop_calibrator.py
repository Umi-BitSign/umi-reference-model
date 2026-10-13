"""Runtime wrapper for a frozen C5 DINO crop-feature calibrator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file

SCHEMA = "umi-studio-c5-crop-calibrator/1"
LOW_RANK_SCHEMA = "umi-studio-c5-crop-calibrator-low-rank/1"


class CalibratedEmbedder:
    def __init__(self, base, weight: np.ndarray, bias: np.ndarray) -> None:
        if weight.shape != (384, 384) or bias.shape != (384,):
            raise ValueError("crop calibrator tensor shapes differ")
        self.base = base
        self.weight = weight.astype(np.float32)
        self.bias = bias.astype(np.float32)
        self._torch_weight = None
        self._torch_bias = None
        device = getattr(base, "device", None)
        if getattr(device, "type", None) == "mps":
            import torch

            self._torch_weight = torch.from_numpy(self.weight).to(device)
            self._torch_bias = torch.from_numpy(self.bias).to(device)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def extract_embeddings_from_frames(self, frames):
        values = self.base.extract_embeddings_from_frames(frames).astype(np.float32)
        if self._torch_weight is not None:
            import torch

            with torch.inference_mode():
                result = (
                    torch.from_numpy(values).to(self._torch_weight.device)
                    @ self._torch_weight
                    + self._torch_bias
                )
            return result.cpu().numpy().astype(np.float32)
        return (values @ self.weight + self.bias).astype(np.float32)


class LowRankCalibratedEmbedder:
    def __init__(self, base, left: np.ndarray, right: np.ndarray, bias: np.ndarray) -> None:
        if (
            left.ndim != 2
            or left.shape[0] != 384
            or right.shape != (left.shape[1], 384)
            or bias.shape != (384,)
        ):
            raise ValueError("low-rank crop calibrator tensor shapes differ")
        self.base = base
        self.left = left.astype(np.float32)
        self.right = right.astype(np.float32)
        self.bias = bias.astype(np.float32)
        self._torch_left = None
        self._torch_right = None
        self._torch_bias = None
        device = getattr(base, "device", None)
        if getattr(device, "type", None) == "mps":
            import torch

            self._torch_left = torch.from_numpy(self.left).to(device)
            self._torch_right = torch.from_numpy(self.right).to(device)
            self._torch_bias = torch.from_numpy(self.bias).to(device)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def extract_embeddings_from_frames(self, frames):
        values = self.base.extract_embeddings_from_frames(frames).astype(np.float32)
        if self._torch_left is not None:
            import torch

            with torch.inference_mode():
                tensor = torch.from_numpy(values).to(self._torch_left.device)
                result = tensor + (tensor @ self._torch_left) @ self._torch_right + self._torch_bias
            return result.cpu().numpy().astype(np.float32)
        return (values + (values @ self.left) @ self.right + self.bias).astype(np.float32)


def install_crop_calibrator(model, path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="numpy") as handle:
        metadata = handle.metadata()
    if (
        metadata.get("schema") not in {SCHEMA, LOW_RANK_SCHEMA}
        or metadata.get("source_crop_size") != "168"
        or metadata.get("target_crop_size") != "224"
    ):
        raise ValueError("crop calibrator metadata differs")
    tensors = load_file(str(path))
    if metadata["schema"] == SCHEMA:
        if set(tensors) != {"hand_weight", "hand_bias", "face_weight", "face_bias"}:
            raise ValueError("crop calibrator tensors differ")
        model._hand_model = CalibratedEmbedder(
            model._hand_model, tensors["hand_weight"], tensors["hand_bias"]
        )
        model._face_model = CalibratedEmbedder(
            model._face_model, tensors["face_weight"], tensors["face_bias"]
        )
    else:
        expected = {
            "hand_left", "hand_right", "hand_bias", "face_left", "face_right", "face_bias"
        }
        if set(tensors) != expected:
            raise ValueError("low-rank crop calibrator tensors differ")
        model._hand_model = LowRankCalibratedEmbedder(
            model._hand_model,
            tensors["hand_left"],
            tensors["hand_right"],
            tensors["hand_bias"],
        )
        model._face_model = LowRankCalibratedEmbedder(
            model._face_model,
            tensors["face_left"],
            tensors["face_right"],
            tensors["face_bias"],
        )
    return metadata
