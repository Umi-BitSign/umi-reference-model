"""Separate48-hour user-authorized compute policy; preserve frozen24-hour experiments."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path


class CertificationComputeWindow:
    def __init__(self, path: Path, *, wall_clock=time.time, monotonic=time.monotonic):
        self.record = json.loads(path.read_text())
        if (
            self.record.get("schema") != "umi-studio-certification-compute-window/1"
            or self.record.get("window_duty_fraction") != 1.0
            or self.record.get("after_window_duty_fraction") != 0.5
            or self.record.get("memory_policy") != "framework_default_no_artificial_six_GiB_cap"
            or self.record.get("window_cpu_threads") != 8
            or self.record.get("after_window_cpu_threads") != 2
            or self.record.get("miner_service_changes_authorized") is not False
            or self.record.get("protected_certification_material_access_authorized") is not False
        ):
            raise ValueError("unsupported certification compute policy")
        times = [
            datetime.fromisoformat(self.record[key].replace("Z", "+00:00"))
            for key in ("starts_at", "expires_at")
        ]
        if any(value.tzinfo is None for value in times):
            raise ValueError("certification compute timestamps require explicit time zones")
        self.starts_at, self.expires_at = (value.timestamp() for value in times)
        if self.expires_at - self.starts_at != 172800:
            raise ValueError("certification override must expire after exactly48 hours")
        self.wall_clock, self.monotonic = wall_clock, monotonic
        self.monotonic_deadline = monotonic() + max(
            0.0, min(172800.0, self.expires_at - wall_clock())
        )

    def duty_fraction(self):
        if (
            self.starts_at <= self.wall_clock() < self.expires_at
            and self.monotonic() < self.monotonic_deadline
        ):
            return 1.0
        return 0.5

    def cpu_threads(self):
        return (
            self.record["window_cpu_threads"]
            if self.duty_fraction() == 1.0
            else self.record["after_window_cpu_threads"]
        )


class WinnerDutyCycle:
    """Use normal Metal allocator safeguards and timed average-duty expiry."""

    def __init__(self, torch, budget):
        self.torch, self.budget = torch, budget
        self.active_seconds = self.idle_seconds = 0.0
        self.maximum_allocated_bytes = self.maximum_driver_allocated_bytes = 0

    def run(self, operation):
        self.torch.mps.synchronize()
        started = time.perf_counter()
        try:
            return operation()
        finally:
            self.torch.mps.synchronize()
            active = time.perf_counter() - started
            self.active_seconds += active
            self.maximum_allocated_bytes = max(
                self.maximum_allocated_bytes, self.torch.mps.current_allocated_memory()
            )
            self.maximum_driver_allocated_bytes = max(
                self.maximum_driver_allocated_bytes, self.torch.mps.driver_allocated_memory()
            )
            if self.budget.duty_fraction() == 0.5:
                time.sleep(active)
                self.idle_seconds += active
            threads = self.budget.cpu_threads()
            if self.torch.get_num_threads() != threads:
                self.torch.set_num_threads(threads)

    def receipt(self):
        return {
            "mps_active_seconds": self.active_seconds,
            "mps_idle_seconds": self.idle_seconds,
            "maximum_allocated_bytes": self.maximum_allocated_bytes,
            "maximum_driver_allocated_bytes": self.maximum_driver_allocated_bytes,
            "current_allowed_gpu_duty_fraction": self.budget.duty_fraction(),
            "memory_policy": self.budget.record["memory_policy"],
            "current_cpu_threads": self.budget.cpu_threads(),
        }
