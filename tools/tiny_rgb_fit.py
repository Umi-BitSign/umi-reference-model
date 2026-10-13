"""Frozen, training-only RGB learnability experiment; no promotion or quality selection."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
from benchmark_fleurs_dev import normalize, sha256
from c5_gpu_budget import ApprovedGpuWindow

ROOT = Path(__file__).resolve().parents[1]
SIDES = (128, 192)
SEED = 317
RECIPE = {
    "activation_implementation": "portable_exact_gelu",
    "epochs": 256,
    "checks_at_epochs": [0, 1, 4, 8, 16, 32, 64, 128, 256],
    "checkpoints_at_epochs": [32, 128, 256],
    "visual_learning_rate": 3e-4,
    "language_learning_rate": 5e-5,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "gradient_accumulation": 1,
    "target_fps": 12.5,
    "max_frames": 512,
    "max_text_tokens": 128,
    "maximum_candidate_reference_bytes": 160,
    "augmentation": "none",
    "generation": {"num_beams": 1, "max_new_tokens": 128, "do_sample": False},
    "fit_gate_mean_wer": 0.25,
    "fit_gate_minimum_exact": 12,
    "fit_gate_minimum_swapped_wer_increase": 0.10,
    "maximum_process_mps_bytes": 6 * 1024**3,
    "minimum_free_bytes": 80 * 1024**3,
    "data_budget_bytes": 4 * 1024**3,
}


def rank(row: dict, salt: str = "selection") -> str:
    return hashlib.sha256(
        f"tiny-rgb-{SEED}:{salt}:{row['source']}:{row['example_id']}".encode()
    ).hexdigest()


def select_fit_cases(rows: list[dict]) -> list[dict]:
    """Two per How signer, four FLEURS, one per STEM tranche, two OpenASL sources."""
    if any(r.get("role") != "training" or r.get("training_allowed") is not True for r in rows):
        raise ValueError("RGB fit selection accepts approved training identities only")
    candidates = sorted([r for r in rows if 0 < len(r["reference"].encode()) <= 160], key=rank)
    buckets = [("how2sign", signer, 2) for signer in (1, 2, 3, 4, 5, 8, 9, 11)] + [
        ("fleurs", None, 4),
        ("asl_stem_original", None, 1),
        ("asl_stem_supplemental", None, 1),
        ("openasl_scale", None, 2),
    ]
    selected, groups, references = [], set(), set()
    for source, signer, count in buckets:
        bucket = []
        for row in candidates:
            matches = (
                row["source"].startswith("how2sign")
                if source == "how2sign"
                else row["source"] == source
            )
            reference = normalize(row["reference"])
            if (
                matches
                and (signer is None or row["signer"] == signer)
                and row["source_group"] not in groups
                and reference not in references
                and reference
            ):
                bucket.append(dict(row))
                groups.add(row["source_group"])
                references.add(reference)
            if len(bucket) == count:
                break
        if len(bucket) != count:
            raise ValueError(f"insufficient independent RGB fit cases in {source}:{signer}")
        selected.extend(bucket)
    return sorted(selected, key=lambda row: rank(row, "case-order"))


def private_path(relative: str, directory: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT / directory):
        raise ValueError(f"RGB experiment dependency must stay under {directory}/")
    return path


def sources() -> dict[str, str]:
    names = (
        "tiny_rgb_fit.py",
        "prepare_tiny_rgb_fit.py",
        "train_tiny_rgb_fit.py",
        "launch_tiny_rgb_fit.py",
        "tiny_rgb_language_model.py",
        "tiny_rgb_model.py",
        "tiny_rgb_data.py",
        "c5_gpu_budget.py",
        "benchmark_fleurs_dev.py",
        "fresh_lowres_model.py",
        "train_fresh_lowres.py",
        "audit_tiny_rgb_corpus.py",
        "c5_full_coverage_mps.py",
        "audit_c5_all_source_corpus.py",
        "c5_all_source_training.py",
        "c5_coverage_schedule.py",
        "c5_face_learning_pilot.py",
        "c5_experiment_sources.py",
        "tiny_rgb_mps_safe.py",
        "probe_tiny_rgb_numerics.py",
        "probe_tiny_rgb_operations.py",
        "freeze_tiny_rgb_long_fit.py",
    )
    return {f"tools/{name}": sha256(ROOT / "tools" / name) for name in names}


def storage(data_dir: Path, prospective_bytes: int = 0) -> None:
    headroom = 128 * 1024**2
    if shutil.disk_usage(ROOT).free < RECIPE["minimum_free_bytes"] + headroom + prospective_bytes:
        raise RuntimeError("RGB experiment reached physical disk floor")
    used = sum(p.stat().st_size for p in data_dir.rglob("*") if p.is_file())
    if used + prospective_bytes > RECIPE["data_budget_bytes"] - headroom:
        raise RuntimeError("RGB experiment reached transformed-pixel budget")


def verify_backend_comparisons(records: list[dict], expected: set[tuple[int, str]]) -> dict:
    identities = {(r["short_side"], r["example_id"]) for r in records}
    if len(records) != len(expected) or identities != expected:
        raise ValueError("RGB backend parity cases differ from matched eligible fit cases")
    worst = 0.0
    for record in records:
        differences = record["gradient_differences"]
        if (
            record["optimizer_updates"] != 0
            or record["dropout_enabled"]
            or record["portable_gelu_replacements"] != 6
            or len(differences) != 126
            or len({r["parameter"] for r in differences}) != 126
            or abs(record["cpu_loss"] - record["mps_loss"]) > 5e-4
            or record["logit_max_abs_difference"] > 1e-3
            or any(
                not math.isfinite(record[k])
                for k in (
                    "cpu_loss",
                    "mps_loss",
                    "cpu_total_gradient_norm",
                    "mps_total_gradient_norm",
                    "logit_max_abs_difference",
                )
            )
            or abs(record["mps_total_gradient_norm"] / record["cpu_total_gradient_norm"] - 1) > 5e-3
        ):
            raise ValueError("RGB backend loss or gradient coverage parity failed")
        for gradient in differences:
            error = gradient["relative_l2_error"]
            if not gradient["finite"] or not math.isfinite(error) or not 0 <= error <= 5e-3:
                raise ValueError("RGB backend per-parameter gradient parity failed")
            worst = max(worst, error)
    return {
        "comparisons": len(records),
        "maximum_relative_gradient_error": worst,
        "maximum_allowed_relative_gradient_error": 5e-3,
    }


def checked_backend(examples: list[dict], original_plan: Path) -> dict:
    """Bind the four actual repaired comparisons, without mutating the earlier plan."""
    old = json.loads(original_plan.read_text())
    if old["examples"] != examples or old["seed"] != SEED:
        raise ValueError("RGB repaired replay differs from original matched cases")
    directory = ROOT / "runs/tiny-rgb-numerics-portable-v1"
    receipt_path, comparisons_path = directory / "receipt.json", directory / "comparisons.json"
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt["plan_sha256"] != sha256(original_plan)
        or not receipt["portable_gelu_enabled"]
        or receipt["portable_gelu_source_sha256"] != sha256(ROOT / "tools/tiny_rgb_mps_safe.py")
        or receipt["comparisons_sha256"] != sha256(comparisons_path)
        or receipt["optimizer_updates"] != 0
        or receipt["weights_saved"]
        or not receipt["training_only"]
    ):
        raise ValueError("RGB repaired backend receipt changed")
    ledger = [
        json.loads(s) for s in original_plan.with_name("pixels.jsonl").read_text().splitlines()
    ]
    sizes = [(r["views"]["128"]["shape"][0], r["example_id"]) for r in ledger]
    extrema = {min(sizes)[1], max(sizes)[1]}
    expected = {(side, identity) for side in SIDES for identity in extrema}
    records = json.loads(comparisons_path.read_text())
    summary = verify_backend_comparisons(records, expected)
    digests = {r["visual_initialization_sha256"] for r in records}
    if len(digests) != 1:
        raise ValueError("RGB repaired backend did not use identical initial visual weights")
    return {
        **summary,
        "visual_initialization_sha256": next(iter(digests)),
        "original_plan": str(original_plan.relative_to(ROOT)),
        "original_plan_sha256": sha256(original_plan),
        "receipt_path": str(receipt_path.relative_to(ROOT)),
        "receipt_sha256": sha256(receipt_path),
        "comparisons_path": str(comparisons_path.relative_to(ROOT)),
        "comparisons_sha256": sha256(comparisons_path),
    }


def checked_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    if (
        plan["schema"] != "umi-studio-tiny-rgb-fit-plan/1"
        or plan["recipe"] != RECIPE
        or plan["short_sides"] != list(SIDES)
        or plan["seed"] != SEED
        or plan["source_sha256"] != sources()
    ):
        raise ValueError("frozen RGB fit implementation or recipe changed")
    snapshot = json.loads(path.with_name("source-snapshot").joinpath("manifest.json").read_text())
    if snapshot["plan_sha256"] != sha256(path):
        raise ValueError("RGB plan changed after source archive")
    dependencies = dict(plan["dependency_sha256"])
    dependencies.update(plan["language_asset_sha256"])
    if any(sha256(ROOT / name) != digest for name, digest in dependencies.items()):
        raise ValueError("frozen RGB fit dependency changed")
    # The original protected-set guard rechecks current protected identities, not just hashes.
    from c5_full_coverage_mps import load_training

    approved = load_training(
        ROOT / "runs/c5-all-source-corpus-v2/audit.json",
        ROOT / "runs/c5-all-source-corpus-v2/training-streams.jsonl",
    )
    allowed = {(r["source"], r["id"]): r for r in approved}
    inventory = [json.loads(s) for s in (ROOT / plan["raw_inventory"]).read_text().splitlines()]
    if select_fit_cases(inventory) != plan["examples"]:
        raise ValueError("RGB fit selection changed")
    if (
        checked_backend(plan["examples"], ROOT / plan["backend"]["original_plan"])
        != plan["backend"]
    ):
        raise ValueError("RGB repaired backend validation changed")
    for row in plan["examples"]:
        original = allowed[(row["source"], row["example_id"])]
        if normalize(original["reference"]) != normalize(row["reference"]):
            raise ValueError("RGB reference differs from protected training binding")
        if sha256(private_path(row["video"], "data")) != row["video_sha256"]:
            raise ValueError("RGB source bytes changed")
    ledger = path.with_name("pixels.jsonl")
    if sha256(ledger) != plan["pixel_ledger_sha256"]:
        raise ValueError("RGB pixel ledger changed")
    return plan


def load_pixels(row: dict, side: int, binding: dict) -> tuple[np.ndarray, np.ndarray]:
    view = binding["views"][str(side)]
    path = private_path(view["path"], "data")
    if (
        binding["source"] != row["source"]
        or binding["example_id"] != row["example_id"]
        or binding["video_sha256"] != row["video_sha256"]
        or sha256(path) != view["sha256"]
    ):
        raise ValueError("RGB pixel cache identity or bytes changed")
    with np.load(path, allow_pickle=False) as data:
        pixels, indices = data["pixels"], data["frame_indices"]
    if (
        pixels.dtype != np.uint8
        or pixels.ndim != 4
        or pixels.shape[-1] != 3
        or list(pixels.shape) != view["shape"]
        or len(pixels) != len(indices)
        or not 1 <= len(pixels) <= RECIPE["max_frames"]
        or indices.dtype != np.int64
        or not np.all(np.diff(indices) > 0)
        or min(pixels.shape[1:3]) > side
    ):
        raise ValueError("RGB cache is not complete downscaled uint8 pixels")
    return pixels, indices


class RgbDutyCycle:
    """Synchronized GPU work with expiring pacing and an independent allocation guard."""

    def __init__(self, torch, budget_path: Path):
        self.torch = torch
        self.budget = ApprovedGpuWindow(budget_path)
        self.active_seconds = 0.0
        self.idle_seconds = 0.0
        self.maximum_allocated_bytes = 0
        self.maximum_driver_allocated_bytes = 0

    def run(self, operation):
        self.torch.mps.synchronize()
        started = time.perf_counter()
        try:
            return operation()
        finally:
            self.torch.mps.synchronize()
            active = time.perf_counter() - started
            self.active_seconds += active
            allocated = self.torch.mps.current_allocated_memory()
            driver_allocated = self.torch.mps.driver_allocated_memory()
            self.maximum_allocated_bytes = max(self.maximum_allocated_bytes, allocated)
            self.maximum_driver_allocated_bytes = max(
                self.maximum_driver_allocated_bytes, driver_allocated
            )
            if max(allocated, driver_allocated) > RECIPE["maximum_process_mps_bytes"]:
                raise RuntimeError("RGB training process exceeded six-GiB MPS guard")
            if self.budget.duty_fraction() == 0.5:
                time.sleep(active)
                self.idle_seconds += active

    def receipt(self) -> dict:
        return {
            "mps_active_seconds": self.active_seconds,
            "mps_idle_seconds": self.idle_seconds,
            "maximum_allocated_bytes": self.maximum_allocated_bytes,
            "maximum_driver_allocated_bytes": self.maximum_driver_allocated_bytes,
            "current_allowed_gpu_duty_fraction": self.budget.duty_fraction(),
        }


def metal_working_set_bytes() -> int:
    """Read Metal's actual allocator denominator on the pinned older PyTorch runtime."""
    import ctypes

    metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    metal.MTLCreateSystemDefaultDevice.argtypes = []
    device = metal.MTLCreateSystemDefaultDevice()
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    selector = objc.sel_registerName(b"recommendedMaxWorkingSetSize")
    objc.objc_msgSend.restype = ctypes.c_uint64
    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    if not device or not selector:
        raise RuntimeError("could not read Metal working-set memory limit")
    result = int(objc.objc_msgSend(device, selector))
    if result < RECIPE["maximum_process_mps_bytes"]:
        raise RuntimeError("Metal working set cannot support the frozen RGB memory policy")
    return result


def gradient_receipt(model, torch) -> dict:
    groups = {"vision": [], "bridge": [], "language": []}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"RGB parameter lacks finite gradient: {name}")
        group = (
            "vision"
            if name.startswith("vision.")
            else ("language" if name.startswith("language_model.") else "bridge")
        )
        norm = float(parameter.grad.float().norm())
        if not math.isfinite(norm):
            raise RuntimeError("RGB gradient norm is nonfinite")
        groups[group].append(norm)
    if any(not values or not any(v > 0 for v in values) for values in groups.values()):
        raise RuntimeError("RGB vision, bridge or language gradient signal is absent")
    return {
        group: {
            "parameter_tensors": len(values),
            "nonzero_gradient_tensors": sum(v > 0 for v in values),
        }
        for group, values in groups.items()
    }
