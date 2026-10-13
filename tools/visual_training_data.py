"""Verify visual-stream training examples and build a signer-balanced schedule."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from benchmark_fleurs_dev import miner_revision, sha256
from cache_fleurs_encoder import checked_stream_features

STREAM_NAMES = ("face_features", "left_hand_features", "right_hand_features", "pose_features")
SIGNER_REPEATS = {1: 8, 2: 10, 3: 3, 4: 12, 5: 3, 8: 1, 9: 14, 11: 12}
FLEURS_REPEAT = 8


def load_visual_examples(
    index: Path,
    cache_dir: Path,
    baseline: Path,
    source: str,
    *,
    sidecar_index: Path | None = None,
    sidecar_dir: Path | None = None,
    miner_health_url: str | None = None,
) -> list[dict]:
    if (sidecar_index is None) != (sidecar_dir is None):
        raise ValueError("visual sidecar index and directory must be supplied together")
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    if not rows or source not in {"fleurs", "how2sign_original", "how2sign_new", "how2sign_raw"}:
        raise ValueError("visual training source or index is empty")
    index_hash = sha256(index)
    identity = sha256(baseline / "MODEL_IDENTITY.json")
    runtime = sha256(baseline / "runtime.py")
    sidecars = {}
    if sidecar_index is not None:
        sidecar_lines = sidecar_index.read_text().splitlines()
        sidecars = {
            record["sentence_name"]: record
            for record in map(json.loads, sidecar_lines)
        }
        if len(sidecars) != len(sidecar_lines):
            raise ValueError("visual sidecar index repeats a sentence")
    examples = []
    identifiers = set()
    used_sidecars = set()
    revision = miner_revision(miner_health_url, None) if miner_health_url else None
    for position, row in enumerate(rows):
        if miner_health_url and position % 16 == 0:
            revision = miner_revision(miner_health_url, revision)
        fleurs = source == "fleurs"
        identifier = row["example_id"] if fleurs else row["sentence_name"]
        if (
            identifier in identifiers
            or row.get("schema")
            != (
                "umi-studio-fleurs-dev-encoder-cache-example/1"
                if fleurs
                else "umi-studio-how2sign-encoder-cache-example/1"
            )
            or row.get("split") != "training"
            or row.get("signer") not in ({3} if fleurs else set(SIGNER_REPEATS))
            or Path(identifier).name != identifier
            or not isinstance(row.get("reference"), str)
            or not row["reference"].strip()
            or row.get("clip_fps") != 25
            or row.get("video_decode_strategy")
            != ("baseline_seek_per_frame" if fleurs else "opencv_sequential_full_range")
            or not isinstance(row.get("selection_sha256"), str)
            or len(row["selection_sha256"]) != 64
            or row.get("baseline_identity_sha256") != identity
            or row.get("baseline_runtime_sha256") != runtime
            or row.get("encoder_dtype") != "float32"
            or not isinstance(row.get("encoder_shape"), list)
            or len(row["encoder_shape"]) != 2
            or row["encoder_shape"][0] < 1
            or row["encoder_shape"][1] != 1536
        ):
            raise ValueError(f"visual index contains a non-training or invalid row: {identifier}")
        identifiers.add(identifier)
        cache = cache_dir / f"{identifier}.npz"
        if sha256(cache) != row["encoder_cache_sha256"]:
            raise ValueError(f"visual training cache checksum differs: {identifier}")
        sidecar = sidecars.get(identifier)
        embedded = row.get("stream_feature_dtype") == "float32"
        if embedded == (sidecar is not None):
            raise ValueError(f"visual streams are missing or duplicated: {identifier}")
        with np.load(cache, allow_pickle=False) as payload:
            expected = {"encoder_hidden_state", *STREAM_NAMES} if embedded else {
                "encoder_hidden_state"
            }
            if set(payload.files) != expected:
                raise ValueError(f"visual training cache fields differ: {identifier}")
            hidden = payload["encoder_hidden_state"]
            if (
                hidden.shape != tuple(row["encoder_shape"])
                or hidden.dtype != np.float32
                or not np.isfinite(hidden).all()
            ):
                raise ValueError(f"visual training encoder state is invalid: {identifier}")
            if embedded:
                stream_values = {name: payload[name] for name in STREAM_NAMES}
        if not embedded:
            if (
                sidecar["schema"] != "umi-studio-how2sign-visual-stream-sidecar/1"
                or sidecar["original_index_sha256"] != index_hash
                or sidecar["selection_sha256"] != row["selection_sha256"]
                or sidecar["encoder_cache_sha256"] != row["encoder_cache_sha256"]
                or sidecar["resampled_clip_sha256"] != row["resampled_clip_sha256"]
                or sidecar["baseline_identity_sha256"] != identity
                or sidecar["baseline_runtime_sha256"] != runtime
                or sidecar.get("stream_feature_dtype") != "float32"
            ):
                raise ValueError(f"visual sidecar source differs: {identifier}")
            sidecar_path = sidecar_dir / f"{identifier}.npz"
            if sha256(sidecar_path) != sidecar["stream_sha256"]:
                raise ValueError(f"visual sidecar checksum differs: {identifier}")
            with np.load(sidecar_path, allow_pickle=False) as payload:
                if set(payload.files) != set(STREAM_NAMES):
                    raise ValueError(f"visual sidecar fields differ: {identifier}")
                stream_values = {name: payload[name] for name in STREAM_NAMES}
            used_sidecars.add(identifier)
        else:
            sidecar_path = None
        if any(value.dtype != np.float32 for value in stream_values.values()):
            raise ValueError(f"visual streams have another dtype: {identifier}")
        streams = checked_stream_features(stream_values, row["encoder_shape"][0])
        metadata = row if embedded else sidecar
        if metadata.get("stream_feature_shapes") != {
            name: list(value.shape) for name, value in streams.items()
        }:
            raise ValueError(f"visual stream metadata differs: {identifier}")
        examples.append(
            {
                "id": identifier,
                "source": source,
                "signer": row["signer"],
                "reference": row["reference"],
                "cache": str(cache),
                "cache_sha256": row["encoder_cache_sha256"],
                "sidecar": str(sidecar_path) if sidecar_path else None,
                "sidecar_sha256": sidecar["stream_sha256"] if sidecar else None,
                "frames": row["encoder_shape"][0],
            }
        )
    if set(sidecars) != used_sidecars:
        raise ValueError("visual sidecar index contains an unselected or embedded sentence")
    if miner_health_url:
        miner_revision(miner_health_url, revision)
    return examples


def balanced_schedule(examples: list[dict], seed: int) -> list[dict]:
    if not examples or seed < 0:
        raise ValueError("visual schedule requires examples and a nonnegative seed")
    seen = set()
    how2sign_names = set()
    fleurs = []
    how2sign = []
    for row in examples:
        key = (row["source"], row["id"])
        if key in seen:
            raise ValueError("visual schedule repeats a source example")
        seen.add(key)
        if row["source"] == "fleurs":
            fleurs.append(row)
        elif row["source"] in {
            "how2sign_original",
            "how2sign_new",
            "how2sign_raw",
        }:
            if row["id"] in how2sign_names:
                raise ValueError("visual schedule repeats a How2Sign sentence across sources")
            how2sign_names.add(row["id"])
            how2sign.append(row)
        else:
            raise ValueError("visual schedule has an unknown source")
    counts = Counter(row["signer"] for row in how2sign)
    if not fleurs or set(counts) != set(SIGNER_REPEATS):
        raise ValueError("visual schedule lacks FLEURS or one How2Sign signer")
    scheduled = [row for row in fleurs for _ in range(FLEURS_REPEAT)]
    scheduled.extend(
        row for row in how2sign for _ in range(SIGNER_REPEATS[row["signer"]])
    )
    random.Random(seed).shuffle(scheduled)
    return scheduled
