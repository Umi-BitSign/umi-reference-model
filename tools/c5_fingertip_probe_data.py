"""Fingertip-only reconstruction inputs from the retained training packet.

This module performs no model fitting. Contexts are offline, bidirectional views
of original frames. Detector wrist/MCP values are targets only; they never set
input normalization, visibility or temporal continuity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKET = Path("checkpoints/c5-temporal-span-proposals-v1")
COMPLETE_SHA = "2b4cff666daa77ef683abc04ee1f07015c63f12b60cc8a86b0a56050a069bdd4"
MINER_SHA = "a9d4a8bf68de39faa0618dc8f1dd6bed14d9d713576214727b03cc0f85a15510"
EXCLUDED = {"P10", "P13", "P14", "P21", "P22", "P31", "P4", "P6", "P0", "P19"}
TARGET_JOINTS = (0, 5, 9, 13, 17)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tip_continuity(tips):
    """Reproduce packet continuity using only the two unnamed fingertip tracks."""
    xy = tips[..., :2].astype(np.float64)
    finite = np.isfinite(xy).all(axis=(2, 3))
    result = np.zeros(finite.shape, dtype=bool)
    distance = np.sqrt(np.mean(np.diff(xy, axis=0) ** 2, axis=(2, 3)))
    result[1:] = finite[1:] & finite[:-1] & (distance <= 0.35)
    return result


@dataclass(frozen=True)
class Sequence:
    metadata: dict
    tips: np.ndarray
    tip_valid: np.ndarray
    targets: np.ndarray
    target_valid: np.ndarray
    continuity: np.ndarray
    frame_indices: np.ndarray
    seconds: np.ndarray


def sequence_view(record, arrays, coordinates="xy"):
    require(coordinates in {"xy", "native_xyz"}, "unknown coordinate convention")
    frames = record["frames"]
    require(type(frames) is int and frames > 0 and record["fps"] == 25, "invalid timeline")
    require(arrays["fingertips"].shape == (frames, 10, 3), "invalid fingertip shape")
    require(arrays["full_hands"].shape == (frames, 42, 3), "invalid hand shape")
    require(arrays["hand_continuity"].shape == (frames, 2), "invalid continuity shape")
    require(
        np.array_equal(arrays["source_frame_indices"], np.arange(frames))
        and np.array_equal(arrays["canonical_seconds"], np.arange(frames) / 25),
        "all original frame indices and timestamps must be retained",
    )
    original_tips = arrays["fingertips"].reshape(frames, 2, 5, 3)
    continuity = tip_continuity(original_tips)
    require(
        np.array_equal(continuity, arrays["hand_continuity"]),
        "stored continuity disagrees with fingertip-only reconstruction",
    )
    dimensions = 2 if coordinates == "xy" else 3
    tips = original_tips[..., :dimensions].copy()
    hands = arrays["full_hands"].reshape(frames, 2, 21, 3)
    targets = hands[:, :, TARGET_JOINTS, :dimensions].copy()
    metadata = {
        key: record[key] for key in ("kind", "source", "signer", "source_group", "example_id")
    }
    metadata.update(coordinates=coordinates, fps=25, target_joints=list(TARGET_JOINTS))
    return Sequence(
        metadata=metadata,
        tips=tips,
        tip_valid=np.isfinite(tips).all(axis=-1),
        targets=targets,
        target_valid=np.isfinite(targets).all(axis=-1),
        continuity=continuity,
        frame_indices=arrays["source_frame_indices"].copy(),
        seconds=arrays["canonical_seconds"].copy(),
    )


def context(sequence, center, width):
    """Keep every center, padding clip edges and masking across track resets.

    Input and target masks remain separate. A missing target never changes which
    input points are returned. There is no requirement for both hands to exist.
    """
    require(type(width) is int and width in {1, 5, 13}, "width must be 1,5,13")
    frames = len(sequence.frame_indices)
    require(type(center) is int and 0 <= center < frames, "invalid center frame")
    radius = width // 2
    indices = np.arange(center - radius, center + radius + 1)
    inside = (indices >= 0) & (indices < frames)
    connected = np.zeros((width, 2), dtype=bool)
    connected[radius] = True
    for position in range(radius - 1, -1, -1):
        if inside[position]:
            connected[position] = (
                connected[position + 1] & sequence.continuity[indices[position] + 1]
            )
    for position in range(radius + 1, width):
        if inside[position]:
            connected[position] = connected[position - 1] & sequence.continuity[indices[position]]
    safe = np.clip(indices, 0, frames - 1)
    valid = sequence.tip_valid[safe] & connected[..., None] & inside[:, None, None]
    values = np.where(valid[..., None], sequence.tips[safe], 0)
    target_valid = sequence.target_valid[center].copy()
    return {
        "input_values": values,
        "input_valid": valid,
        "target_values": np.where(target_valid[..., None], sequence.targets[center], 0),
        "target_valid": target_valid,
        "context_frame_indices": np.where(inside, indices, -1),
        "context_seconds": np.where(inside, sequence.seconds[safe], np.nan),
        "center_frame": center,
        "center_seconds": float(sequence.seconds[center]),
    }


class Packet:
    """Load only the byte-bound training packet, without opening source video."""

    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.directory = self.root / PACKET
        path = self.directory / "complete.json"
        require(sha(path) == COMPLETE_SHA, "retained packet binding changed")
        require(sha(self.directory / "source.py") == MINER_SHA, "archived miner changed")
        manifest = json.loads(path.read_text())
        require(
            manifest["complete"] is True and manifest["clips"] == len(manifest["records"]) == 532,
            "incomplete training packet",
        )
        self.records = manifest["records"]

    def _bound(self, descriptor, expected):
        path = self.root / descriptor["path"]
        require(path.resolve() == (self.directory / expected).resolve(), "unexpected packet path")
        require(sha(path) == descriptor["sha256"], "packet artifact changed")
        return path

    def load(self, index, coordinates="xy"):
        require(type(index) is int and 0 <= index < len(self.records), "invalid clip index")
        record = json.loads(self._bound(self.records[index], f"clip-{index:04d}.json").read_text())
        require(record["kind"] in {"word", "phrase"}, "unknown training kind")
        require(
            not record["source"].startswith("asl_stem") or record["signer"] not in EXCLUDED,
            "reserved or diagnostic signer is not eligible",
        )
        path = self._bound(record["features"], f"clip-{index:04d}.npz")
        with np.load(path, allow_pickle=False) as arrays:
            return sequence_view(record, arrays, coordinates)
