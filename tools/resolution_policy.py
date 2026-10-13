"""Low-resolution inference helpers for aspect-preserving video ablations."""

from __future__ import annotations

from typing import Any

import numpy as np

DINO_PATCH_SIZE = 14
BASELINE_VISUAL_CROP_SIZE = 224


def validate_source_short_side(value: int | None) -> int | None:
    if value is not None and not 64 <= value <= 1_024:
        raise ValueError("source short side must be between 64 and 1024 pixels")
    return value


def validate_visual_crop_size(value: int) -> int:
    if not 4 * DINO_PATCH_SIZE <= value <= BASELINE_VISUAL_CROP_SIZE:
        raise ValueError("visual crop size must be between 56 and 224 pixels")
    if value % DINO_PATCH_SIZE:
        raise ValueError("visual crop size must be a multiple of the DINO patch size")
    return value


def aspect_preserving_downscale(frame: np.ndarray, short_side: int) -> np.ndarray:
    """Shrink an HWC frame to a bounded short side without changing aspect ratio."""
    import cv2

    validate_source_short_side(short_side)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("resolution policy expects an HWC uint8 RGB frame")
    height, width = frame.shape[:2]
    if min(height, width) <= short_side:
        return np.ascontiguousarray(frame)
    scale = short_side / min(height, width)
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))
    resized = cv2.resize(
        frame,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    return np.ascontiguousarray(resized, dtype=np.uint8)


def make_short_side_reader(base_reader_type: type, short_side: int) -> type:
    """Wrap a Decord-compatible reader with downscale-only RGB frame resizing."""
    validate_source_short_side(short_side)

    class ShortSideVideoReader(base_reader_type):
        source_short_side = short_side

        def _read(self, index: int) -> np.ndarray:
            return aspect_preserving_downscale(super()._read(index), short_side)

        def get_batch(self, indices: Any):
            batch = super().get_batch(indices)
            frames = batch.asnumpy()
            if len(frames) == 0:
                return batch
            resized = [aspect_preserving_downscale(frame, short_side) for frame in frames]
            return batch.__class__(np.stack(resized, axis=0))

    ShortSideVideoReader.__name__ = f"ShortSide{short_side}{base_reader_type.__name__}"
    return ShortSideVideoReader


def set_visual_crop_size(model: object, crop_size: int) -> dict[str, int]:
    """Set hand and face crop tensors to a DINO-compatible square size."""
    from torchvision import transforms

    validate_visual_crop_size(crop_size)
    for model_attribute, extractor_attribute in (
        ("_hand_model", "_hand_extractor"),
        ("_face_model", "_face_extractor"),
    ):
        embedder = getattr(model, model_attribute)
        extractor = getattr(model, extractor_attribute)
        patch_size = getattr(embedder.model, "patch_size", None)
        if patch_size != DINO_PATCH_SIZE:
            raise ValueError("resolution policy requires the pinned DINO patch size")
        stages = list(getattr(embedder.transform, "transforms", ()))
        if not stages or stages[0].__class__.__name__ != "Resize":
            raise ValueError("resolution policy requires the pinned DINO resize transform")
        stages[0] = transforms.Resize((crop_size, crop_size))
        embedder.transform = transforms.Compose(stages)
        extractor.output_size = (crop_size, crop_size)
    patches_per_side = crop_size // DINO_PATCH_SIZE
    return {
        "visual_crop_size": crop_size,
        "dino_patch_size": DINO_PATCH_SIZE,
        "patch_tokens_per_stream_frame": patches_per_side**2,
    }
