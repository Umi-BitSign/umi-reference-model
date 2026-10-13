from __future__ import annotations

import hashlib
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import tiny_rgb_fit
import tiny_rgb_pixel_cache
from audit_tiny_rgb_full import frame_indices
from benchmark_fleurs_dev import sha256
from tiny_rgb_full_run import lr_factor


def setup_cache(tmp_path, monkeypatch, *, small=False):
    monkeypatch.setattr(tiny_rgb_pixel_cache, "ROOT", tmp_path)
    monkeypatch.setattr(tiny_rgb_fit, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    pixels = np.random.default_rng(317).integers(0, 256, (4, 64, 112, 3), dtype=np.uint8)
    indices = np.array([0, 2, 4, 6], dtype=np.int64)
    rows, headers = [], []
    for identity in ("one", "two"):
        video = tmp_path / "data" / f"{identity}.mp4"
        video.write_bytes(identity.encode())
        rows.append(
            {
                "source": "fleurs",
                "example_id": identity,
                "role": "training",
                "training_allowed": True,
                "video": f"data/{identity}.mp4",
                "video_sha256": sha256(video),
            }
        )
        headers.append(
            {
                "source": "fleurs",
                "example_id": identity,
                "role": "training",
                "source_fps": 25.0,
                "source_frames": 8,
                "source_shape": [64, 112],
                "sampled_frames": 4,
                "frame_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
            }
        )

    def decode(_video, side):
        return (
            pixels.copy(),
            indices.copy(),
            {
                "source_fps": 25.0,
                "source_frames": 8,
                "source_shape": [64, 112],
                "short_side_cap": side,
                "target_fps": 12.5,
            },
        )

    monkeypatch.setattr(tiny_rgb_pixel_cache, "decode_rgb_video_fast", decode)
    cache = tiny_rgb_pixel_cache.PixelCache(
        tmp_path / "data/cache",
        "plan",
        headers,
        budget_bytes=128 * 1024**2 + 100000 if small else 4 * 1024**3,
        minimum_free_bytes=0,
    )
    return cache, rows, pixels


def test_owned_pixel_cache_roundtrips_exactly_and_rejects_role_confusion(tmp_path, monkeypatch):
    cache, rows, pixels = setup_cache(tmp_path, monkeypatch)
    first = cache.get(rows[0], 128, expected_role="training")
    second = cache.get(rows[0], 128, expected_role="training")
    assert np.array_equal(first[0], pixels) and np.array_equal(second[0], pixels)
    assert not first[3] and second[3]
    assert cache.receipt()["pixel_cache_hits"] == 1
    with pytest.raises(ValueError, match="role"):
        cache.get(rows[0], 128, expected_role="open_development")


def test_cache_evicts_only_its_owned_derived_pixels_and_preserves_originals(tmp_path, monkeypatch):
    cache, rows, _ = setup_cache(tmp_path, monkeypatch, small=True)
    cache.get(rows[0], 128, expected_role="training")
    first_path = next(iter(cache.entries.values()))["path"]
    cache.get(rows[1], 128, expected_role="training")
    assert not first_path.exists() and cache.evictions == 1
    assert all((tmp_path / row["video"]).is_file() for row in rows)
    assert cache.bytes <= cache.limit


def test_cache_refuses_to_delete_or_use_modified_payload_or_owner(tmp_path, monkeypatch):
    cache, rows, _ = setup_cache(tmp_path, monkeypatch)
    cache.get(rows[0], 128, expected_role="training")
    payload = next(iter(cache.entries.values()))["path"]
    payload.write_bytes(b"user modification")
    with pytest.raises(RuntimeError, match="unowned"):
        cache.evict()
    assert payload.exists()
    with pytest.raises(ValueError, match="payload"):
        cache.get(rows[0], 128, expected_role="training")
    cache.owner.write_text("changed")
    with pytest.raises(RuntimeError, match="ownership"):
        cache.check_storage()


def test_uncached_source_still_returns_all_frames_and_creates_no_payload(tmp_path, monkeypatch):
    cache, rows, pixels = setup_cache(tmp_path, monkeypatch)
    returned = cache.get(rows[0], 192, expected_role="training", cache_allowed=False)
    assert np.array_equal(returned[0], pixels)
    assert not cache.entries and cache.bytes == 0


def test_full_audit_temporal_policy_is_complete_and_never_truncates():
    assert frame_indices(20, 25, 12.5, 512).tolist() == list(range(0, 20, 2))
    assert frame_indices(1, 25, 12.5, 512).tolist() == [0]
    with pytest.raises(ValueError, match="truncate"):
        frame_indices(30, 25, 12.5, 4)
    with pytest.raises(ValueError, match="complete"):
        frame_indices(20, 10, 12.5, 512)


def test_full_learning_rate_schedule_has_warmup_and_fixed_cosine_floor():
    assert lr_factor(0, 12200) == pytest.approx(1 / 256)
    assert lr_factor(255, 12200) == 1.0
    assert lr_factor(256, 12200) == 1.0
    assert lr_factor(12200, 12200) == pytest.approx(0.1)


def test_full_launcher_sets_environment_before_any_trainer_import(monkeypatch):
    import launch_tiny_rgb_full

    called = []
    fake = types.ModuleType("train_tiny_rgb_full")

    def main():
        import os

        assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
        assert os.environ["OPENCV_FFMPEG_THREADS"] == "2"
        called.append(True)

    fake.main = main
    monkeypatch.setitem(sys.modules, "train_tiny_rgb_full", fake)
    importlib.reload(launch_tiny_rgb_full).main()
    assert called == [True]
