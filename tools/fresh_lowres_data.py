"""Raw-video loading for the fresh low-resolution model."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)
KINETICS400_MEAN = torch.tensor((0.43216, 0.394666, 0.37645), dtype=torch.float32).view(1, 3, 1, 1)
KINETICS400_STD = torch.tensor((0.22803, 0.22145, 0.216989), dtype=torch.float32).view(1, 3, 1, 1)
AUGMENTATION_PROFILES = ("none", "mild_v1")


def augment_video(
    video: torch.Tensor, *, profile: str = "none", seed: int | None = None
) -> torch.Tensor:
    """Apply one seeded, clip-consistent perturbation before normalization."""
    if video.ndim != 4 or video.shape[1] != 3 or not video.is_floating_point():
        raise ValueError("augmentation expects floating-point TCHW RGB video")
    if profile not in AUGMENTATION_PROFILES:
        raise ValueError("unknown video augmentation profile")
    if profile == "none":
        return video
    if seed is None or seed < 0:
        raise ValueError("augmented video needs a nonnegative seed")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    height, width = video.shape[-2:]
    pad_x = max(1, round(width * 0.04))
    pad_y = max(1, round(height * 0.04))
    shift_x = int(torch.randint(-pad_x, pad_x + 1, (), generator=generator))
    shift_y = int(torch.randint(-pad_y, pad_y + 1, (), generator=generator))
    padded = F.pad(video, (pad_x, pad_x, pad_y, pad_y), mode="reflect")
    start_x = pad_x - shift_x
    start_y = pad_y - shift_y
    video = padded[..., start_y : start_y + height, start_x : start_x + width]

    contrast = 0.9 + 0.2 * float(torch.rand((), generator=generator))
    brightness = -0.06 + 0.12 * float(torch.rand((), generator=generator))
    video = ((video - 0.5) * contrast + 0.5 + brightness).clamp_(0, 1)
    if float(torch.rand((), generator=generator)) < 0.35:
        sigma = 0.6 + 0.5 * float(torch.rand((), generator=generator))
        coordinates = torch.arange(-2, 3, dtype=video.dtype)
        kernel = torch.exp(-(coordinates**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel_2d = (kernel[:, None] * kernel[None, :]).expand(3, 1, 5, 5)
        video = F.conv2d(F.pad(video, (2, 2, 2, 2), mode="reflect"), kernel_2d, groups=3)
    return video.contiguous()


def sample_frame_indices(
    frame_count: int, source_fps: float, target_fps: float, max_frames: int
) -> list[int]:
    if (
        frame_count < 1
        or not math.isfinite(source_fps)
        or source_fps <= 0
        or not math.isfinite(target_fps)
        or target_fps <= 0
        or target_fps > source_fps * 1.01
        or max_frames < 1
    ):
        raise ValueError("invalid video sampling metadata")
    step = source_fps / target_fps
    sampled = np.floor(np.arange(0, frame_count, step)).astype(np.int64)
    sampled = np.unique(np.clip(sampled, 0, frame_count - 1))
    if len(sampled) > max_frames:
        positions = np.linspace(0, len(sampled) - 1, max_frames).round().astype(np.int64)
        sampled = sampled[positions]
    return sampled.tolist()


def _resized_dimensions(height: int, width: int, short_side: int) -> tuple[int, int]:
    if height < 1 or width < 1 or short_side < 64:
        raise ValueError("invalid frame dimensions")
    scale = short_side / min(height, width)
    return max(1, round(width * scale)), max(1, round(height * scale))


def decode_video(
    path: Path,
    *,
    short_side: int,
    target_fps: float,
    max_frames: int,
    normalization: str = "imagenet",
    augmentation_profile: str = "none",
    augmentation_seed: int | None = None,
) -> torch.Tensor:
    """Decode selected RGB frames sequentially and apply pinned normalization."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open training video: {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        selected = sample_frame_indices(frame_count, source_fps, target_fps, max_frames)
        selected_set = set(selected)
        frames = []
        output_size = None
        frame_index = 0
        while selected_set:
            success, bgr = capture.read()
            if not success or bgr is None:
                break
            if frame_index in selected_set:
                if output_size is None:
                    output_size = _resized_dimensions(
                        int(bgr.shape[0]), int(bgr.shape[1]), short_side
                    )
                resized = cv2.resize(bgr, output_size, interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                selected_set.remove(frame_index)
            frame_index += 1
        if selected_set or len(frames) != len(selected):
            raise ValueError(f"training video ended before selected frames: {path}")
    finally:
        capture.release()
    video = torch.from_numpy(np.ascontiguousarray(np.stack(frames))).permute(0, 3, 1, 2)
    video = video.to(dtype=torch.float32).div_(255)
    video = augment_video(
        video, profile=augmentation_profile, seed=augmentation_seed
    )
    if normalization == "imagenet":
        mean, standard_deviation = IMAGENET_MEAN, IMAGENET_STD
    elif normalization == "kinetics400":
        mean, standard_deviation = KINETICS400_MEAN, KINETICS400_STD
    else:
        raise ValueError("unknown video normalization")
    return video.sub_(mean).div_(standard_deviation)


def load_inventory(path: Path, workspace: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise ValueError("fresh training inventory is empty")
    seen = set()
    for row in rows:
        identity = (row.get("source"), row.get("example_id"))
        video = (workspace / row.get("video", "")).resolve()
        try:
            video.relative_to(workspace.resolve())
        except ValueError as error:
            raise ValueError("fresh training video is outside the workspace") from error
        if (
            row.get("schema") != "umi-studio-fresh-lowres-training-example/1"
            or identity in seen
            or any("cache" in key or "baseline" in key for key in row)
            or not video.is_file()
            or not isinstance(row.get("reference"), str)
            or not row["reference"].strip()
            or not isinstance(row.get("video_sha256"), str)
            or len(row["video_sha256"]) != 64
        ):
            raise ValueError(f"invalid fresh training inventory row: {identity}")
        row["video_path"] = video
        seen.add(identity)
    return rows
