"""Load frozen generic visual-stream examples without C5 encoder states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmark_fleurs_dev import sha256
from cache_fleurs_encoder import checked_stream_features

SCHEMA = "umi-studio-fresh-stream-example/1"
STREAM_NAMES = ("face_features", "left_hand_features", "right_hand_features", "pose_features")


def load_inventory(path: Path, workspace: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    seen = set()
    for row in rows:
        identity = (row.get("source"), row.get("example_id"))
        cache = (workspace / row.get("cache", "")).resolve()
        sidecar_value = row.get("sidecar")
        sidecar = (workspace / sidecar_value).resolve() if sidecar_value else None
        if (
            row.get("schema") != SCHEMA
            or row.get("split") != "training"
            or identity in seen
            or not row.get("reference", "").strip()
            or row.get("stored_temporal_stride") not in {1, 2}
            or not cache.is_file()
            or (sidecar is not None and not sidecar.is_file())
        ):
            raise ValueError(f"invalid fresh stream inventory row: {identity}")
        cache.relative_to(workspace.resolve())
        if sidecar is not None:
            sidecar.relative_to(workspace.resolve())
        row["cache_path"] = cache
        row["sidecar_path"] = sidecar
        seen.add(identity)
    if not rows:
        raise ValueError("fresh stream inventory is empty")
    return rows


def load_stream_tensor(row: dict, *, verify_hash: bool = True) -> np.ndarray:
    cache = row["cache_path"]
    sidecar = row["sidecar_path"]
    source = sidecar or cache
    expected_hash = row["sidecar_sha256"] if sidecar is not None else row["cache_sha256"]
    if verify_hash and sha256(source) != expected_hash:
        raise ValueError(f"fresh stream file changed: {row['example_id']}")
    with np.load(source, allow_pickle=False) as payload:
        names = set(payload.files)
        if not set(STREAM_NAMES).issubset(names):
            raise ValueError(f"fresh stream fields are missing: {row['example_id']}")
        streams = checked_stream_features(
            {name: payload[name] for name in STREAM_NAMES}, row["frames"]
        )
    values = np.concatenate([streams[name] for name in STREAM_NAMES], axis=1)
    if row["stored_temporal_stride"] == 1:
        values = values[::2]
    if values.shape[0] < 1 or values.shape[1] != 1166:
        raise ValueError("fresh stream tensor has invalid shape")
    return np.ascontiguousarray(values, dtype=np.float32)


def load_teacher_tensor(row: dict, *, verify_hash: bool = True) -> np.ndarray:
    cache = row["cache_path"]
    if verify_hash and sha256(cache) != row["cache_sha256"]:
        raise ValueError(f"fresh stream teacher cache changed: {row['example_id']}")
    with np.load(cache, allow_pickle=False) as payload:
        if "encoder_hidden_state" not in payload.files:
            raise ValueError(f"fresh stream teacher state is missing: {row['example_id']}")
        values = np.asarray(payload["encoder_hidden_state"], dtype=np.float32)
    if values.shape != (row["frames"], 1536) or not np.isfinite(values).all():
        raise ValueError(f"fresh stream teacher state is invalid: {row['example_id']}")
    if row["stored_temporal_stride"] == 1:
        values = values[::2]
    return np.ascontiguousarray(values, dtype=np.float32)

