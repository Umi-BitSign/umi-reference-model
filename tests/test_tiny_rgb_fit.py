from __future__ import annotations

import importlib
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import tiny_rgb_fit
from benchmark_fleurs_dev import sha256
from tiny_rgb_fit import load_pixels, select_fit_cases, verify_backend_comparisons
from train_tiny_rgb_fit import fit_summary


def candidates():
    rows = []
    buckets = [("how2sign_raw", signer, 4) for signer in (1, 2, 3, 4, 5, 8, 9, 11)] + [
        ("fleurs", 3, 8),
        ("asl_stem_original", "P1", 3),
        ("asl_stem_supplemental", "P2", 3),
        ("openasl_scale", "source", 4),
    ]
    for source, signer, count in buckets:
        for index in range(count):
            identity = f"{source}-{signer}-{index}"
            rows.append(
                {
                    "source": source,
                    "signer": signer,
                    "example_id": identity,
                    "source_group": identity,
                    "reference": f"reference {identity}",
                    "role": "training",
                    "training_allowed": True,
                }
            )
    return rows


def test_fit_selection_is_training_only_independent_and_stable_with_full_source_coverage():
    rows = candidates()
    selected = select_fit_cases(rows)
    assert selected == select_fit_cases(list(reversed(rows)))
    assert len(selected) == 24
    assert len({r["source_group"] for r in selected}) == 24
    assert len({r["reference"] for r in selected}) == 24
    assert Counter(r["source"] for r in selected) == {
        "how2sign_raw": 16,
        "fleurs": 4,
        "asl_stem_original": 1,
        "asl_stem_supplemental": 1,
        "openasl_scale": 2,
    }
    assert set(
        Counter(r["signer"] for r in selected if r["source"].startswith("how2sign")).values()
    ) == {2}
    rows[0]["role"] = "heldout"
    with pytest.raises(ValueError, match="training identities"):
        select_fit_cases(rows)


def test_fit_selection_rejects_insufficient_distinct_content():
    rows = candidates()
    for row in rows:
        row["source_group"] = "same source"
    with pytest.raises(ValueError, match="insufficient independent"):
        select_fit_cases(rows)


@pytest.mark.parametrize("bad", ("hash", "float_pixels", "wrong_identity", "repeated_frame"))
def test_fit_pixel_cache_fails_closed(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(tiny_rgb_fit, "ROOT", tmp_path)
    target = tmp_path / "data/case.npz"
    target.parent.mkdir()
    pixels = np.zeros((3, 128, 224, 3), dtype=np.float32 if bad == "float_pixels" else np.uint8)
    indices = np.array([0, 2, 2] if bad == "repeated_frame" else [0, 2, 4], dtype=np.int64)
    np.savez(target, pixels=pixels, frame_indices=indices)
    row = {"source": "fleurs", "example_id": "one", "video_sha256": "video"}
    binding = {
        **row,
        "views": {
            "128": {"path": "data/case.npz", "sha256": sha256(target), "shape": [3, 128, 224, 3]}
        },
    }
    if bad == "hash":
        binding["views"]["128"]["sha256"] = "changed"
    if bad == "wrong_identity":
        binding["example_id"] = "two"
    with pytest.raises(ValueError, match="RGB"):
        load_pixels(row, 128, binding)


def test_fit_success_requires_visual_conditioning_and_cannot_authorize_quality_or_promotion():
    correct = [{"example_id": str(i), "wer": 0.0, "edits": 0} for i in range(24)]
    wrong = [{"example_id": str(i), "wer": 1.0, "edits": 5} for i in range(24)]
    summary = fit_summary(correct, wrong)
    assert summary["passes_training_fit_gate"]
    assert not summary["held_out_quality_verified"] and not summary["promotion_authorized"]
    assert not fit_summary(correct, correct)["passes_training_fit_gate"]
    with pytest.raises(ValueError, match="order differs"):
        fit_summary(correct, list(reversed(wrong)))


def test_launcher_sets_metal_fallback_before_importing_trainer(monkeypatch):
    import launch_tiny_rgb_fit

    called = []
    fake = types.ModuleType("train_tiny_rgb_fit")

    def main():
        import os

        assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
        assert os.environ["OMP_NUM_THREADS"] == "2"
        called.append(True)

    fake.main = main
    monkeypatch.setitem(sys.modules, "train_tiny_rgb_fit", fake)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    importlib.reload(launch_tiny_rgb_fit).main()
    assert called == [True]


@pytest.mark.parametrize("duty_fraction", (1.0, 0.5))
def test_rgb_gpu_guard_paces_expired_budget_without_assistant_wakeups(monkeypatch, duty_fraction):
    budget = types.SimpleNamespace(duty_fraction=lambda: duty_fraction)
    monkeypatch.setattr(tiny_rgb_fit, "ApprovedGpuWindow", lambda _: budget)
    clock = iter([1.0, 3.0])
    monkeypatch.setattr(tiny_rgb_fit.time, "perf_counter", lambda: next(clock))
    sleeps, synchronizations = [], []
    monkeypatch.setattr(tiny_rgb_fit.time, "sleep", sleeps.append)
    torch = types.SimpleNamespace(
        mps=types.SimpleNamespace(
            synchronize=lambda: synchronizations.append(True),
            current_allocated_memory=lambda: 100,
            driver_allocated_memory=lambda: 200,
        )
    )
    duty = tiny_rgb_fit.RgbDutyCycle(torch, Path("unused"))
    assert duty.run(lambda: "result") == "result"
    assert len(synchronizations) == 2
    assert sleeps == ([2.0] if duty_fraction == 0.5 else [])
    assert duty.active_seconds == 2.0
    assert duty.maximum_driver_allocated_bytes == 200


def test_rgb_gpu_guard_includes_driver_allocations(monkeypatch):
    monkeypatch.setattr(tiny_rgb_fit, "ApprovedGpuWindow", lambda _: None)
    torch = types.SimpleNamespace(
        mps=types.SimpleNamespace(
            synchronize=lambda: None,
            current_allocated_memory=lambda: 100,
            driver_allocated_memory=lambda: tiny_rgb_fit.RECIPE["maximum_process_mps_bytes"] + 1,
        )
    )
    duty = tiny_rgb_fit.RgbDutyCycle(torch, Path("unused"))
    with pytest.raises(RuntimeError, match="six-GiB"):
        duty.run(lambda: None)


@pytest.mark.parametrize("bad", (None, "gradient", "coverage", "loss", "nonfinite", "case"))
def test_backend_gate_requires_actual_full_parameter_cpu_metal_parity(bad):
    gradients = [
        {"parameter": f"parameter-{i}", "finite": True, "relative_l2_error": 1e-4}
        for i in range(126)
    ]
    record = {
        "short_side": 128,
        "example_id": "training",
        "optimizer_updates": 0,
        "dropout_enabled": False,
        "portable_gelu_replacements": 6,
        "gradient_differences": gradients,
        "cpu_loss": 10.0,
        "mps_loss": 10.00001,
        "logit_max_abs_difference": 1e-4,
        "cpu_total_gradient_norm": 100.0,
        "mps_total_gradient_norm": 100.001,
    }
    if bad == "gradient":
        gradients[0]["relative_l2_error"] = 1e5
    elif bad == "coverage":
        gradients.pop()
    elif bad == "loss":
        record["mps_loss"] = 11.0
    elif bad == "nonfinite":
        gradients[0]["relative_l2_error"] = float("nan")
    elif bad == "case":
        record["example_id"] = "heldout"
    expected = {(128, "training")}
    if bad is None:
        assert verify_backend_comparisons([record], expected)["comparisons"] == 1
    else:
        with pytest.raises(ValueError, match="RGB backend"):
            verify_backend_comparisons([record], expected)
