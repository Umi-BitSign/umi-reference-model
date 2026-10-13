"""Owned, bounded, lossless raw-pixel cache. Evictions never remove original clips."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
from benchmark_fleurs_dev import sha256
from tiny_rgb_fit import ROOT, private_path
from tiny_rgb_full_training_tools import (
    decode_pixel_delta,
    decode_rgb_video_fast,
    encode_pixel_delta,
)


class PixelCache:
    def __init__(
        self,
        data_dir: Path,
        plan_sha: str,
        headers: list[dict],
        *,
        budget_bytes: int = 4 * 1024**3,
        minimum_free_bytes: int = 80 * 1024**3,
    ):
        self.directory = data_dir.resolve()
        if self.directory.exists() or not self.directory.is_relative_to(ROOT / "data"):
            raise ValueError("pixel cache requires a new owned private data/ directory")
        self.directory.mkdir(parents=True)
        self.payloads = self.directory / "pixels"
        self.payloads.mkdir()
        self.headers = {(r["source"], r["example_id"], r["role"]): r for r in headers}
        if len(self.headers) != len(headers):
            raise ValueError("pixel cache header identities repeat")
        self.limit = budget_bytes - 128 * 1024**2
        if self.limit <= 0:
            raise ValueError("pixel cache budget must include128MiB safety headroom")
        self.minimum_free = minimum_free_bytes
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = 0
        self.owner = self.directory / "CACHE_OWNERSHIP.json"
        self.owner.write_text(
            json.dumps(
                {
                    "schema": "umi-studio-owned-rgb-pixel-cache/1",
                    "plan_sha256": plan_sha,
                    "codec": "uint8_modulo_temporal_delta",
                    "budget_bytes": budget_bytes,
                    "source_implementation_sha256": sha256(Path(__file__)),
                    "original_source_clips_removed": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        self.owner_sha = sha256(self.owner)

    def check_storage(self, prospective: int = 0) -> None:
        if sha256(self.owner) != self.owner_sha:
            raise RuntimeError("pixel cache ownership marker changed")
        if shutil.disk_usage(ROOT).free < self.minimum_free + 128 * 1024**2 + prospective:
            raise RuntimeError("pixel cache reached physical disk floor")

    def event(self, record: dict) -> None:
        with self.directory.joinpath("cache-events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def evict(self) -> None:
        key, entry = next(iter(self.entries.items()))
        target = entry["path"]
        if (
            target.is_symlink()
            or target.resolve().parent != self.payloads
            or target.name != f"{key}.npz"
            or sha256(target) != entry["sha256"]
        ):
            raise RuntimeError("refusing to evict a changed or unowned pixel cache file")
        self.check_storage()
        target.unlink()
        self.entries.pop(key)
        self.bytes -= entry["bytes"]
        self.evictions += 1
        self.event(
            {
                "event": "evict_owned_derived_pixels",
                "key": key,
                "bytes": entry["bytes"],
                "recoverable_from_original_clip": True,
            }
        )

    def get(self, row: dict, side: int, *, expected_role: str, cache_allowed: bool = True):
        if (
            expected_role not in {"training", "open_development"}
            or row["role"] != expected_role
            or row["training_allowed"] is not (expected_role == "training")
            or side not in (128, 192)
        ):
            raise ValueError("pixel cache role or resolution is not authorized for this call")
        self.check_storage()
        source = private_path(row["video"], "data")
        if sha256(source) != row["video_sha256"]:
            raise ValueError("pixel cache original clip bytes changed")
        header = self.headers[(row["source"], row["example_id"], expected_role)]
        key = hashlib.sha256(
            json.dumps(
                {
                    "video_sha256": row["video_sha256"],
                    "side": side,
                    "role": expected_role,
                    "target_fps": 12.5,
                    "codec": "uint8_modulo_temporal_delta",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        hit = key in self.entries
        if hit:
            entry = self.entries.pop(key)
            self.entries[key] = entry
            if sha256(entry["path"]) != entry["sha256"]:
                raise ValueError("pixel cache payload bytes changed")
            with np.load(entry["path"], allow_pickle=False) as data:
                pixels, indices = decode_pixel_delta(data["pixel_delta"]), data["frame_indices"]
                metadata = json.loads(str(data["metadata"].item()))
            self.hits += 1
        else:
            pixels, indices, metadata = decode_rgb_video_fast(row["video"], side)
            self.misses += 1
        expected_shape = header["source_shape"]
        if (
            metadata["source_fps"] != header["source_fps"]
            or metadata["source_frames"] != header["source_frames"]
            or metadata["source_shape"] != expected_shape
            or len(pixels) != header["sampled_frames"]
            or indices.dtype != np.int64
            or hashlib.sha256(indices.tobytes()).hexdigest() != header["frame_indices_sha256"]
            or pixels.dtype != np.uint8
            or pixels.ndim != 4
            or pixels.shape[-1] != 3
            or min(pixels.shape[1:3]) > side
        ):
            raise ValueError("pixel cache frames differ from full source/header audit")
        if not hit and cache_allowed and pixels.nbytes + indices.nbytes < 128 * 1024**2:
            self.check_storage(pixels.nbytes + indices.nbytes + 4096)
            target = self.payloads / f"{key}.npz"
            temporary = self.payloads / f"{key}.partial.npz"
            if target.exists() or temporary.exists():
                raise ValueError("pixel cache output already exists without ownership")
            with temporary.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    pixel_delta=encode_pixel_delta(pixels),
                    frame_indices=indices,
                    metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                )
            size = temporary.stat().st_size
            while self.bytes + size > self.limit and self.entries:
                self.evict()
            if size > self.limit:
                raise RuntimeError("individual pixel payload exceeds the owned cache budget")
            temporary.rename(target)
            self.entries[key] = {"path": target, "sha256": sha256(target), "bytes": size}
            self.bytes += size
            self.event(
                {
                    "event": "store_lossless_pixels",
                    "key": key,
                    "bytes": size,
                    "video_sha256": row["video_sha256"],
                    "role": expected_role,
                    "side": side,
                }
            )
        return pixels, indices, metadata, hit

    def receipt(self):
        return {
            "pixel_cache_hits": self.hits,
            "pixel_cache_misses": self.misses,
            "pixel_cache_evictions": self.evictions,
            "pixel_cache_bytes": self.bytes,
            "original_source_clips_removed": False,
        }
