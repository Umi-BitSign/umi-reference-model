"""Expire a user-approved GPU duty-cycle window without an assistant wake-up."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path


class ApprovedGpuWindow:
    def __init__(self, path: Path, *, wall_clock=time.time, monotonic=time.monotonic):
        self.record = json.loads(path.read_text())
        if (
            self.record.get("schema") != "umi-studio-gpu-budget-window/1"
            or self.record.get("window_duty_fraction") != 1.0
            or self.record.get("after_window_duty_fraction") != 0.5
        ):
            raise ValueError("unsupported GPU budget policy")
        times = [
            datetime.fromisoformat(self.record[key].replace("Z", "+00:00"))
            for key in ("starts_at", "expires_at")
        ]
        if any(value.tzinfo is None for value in times):
            raise ValueError("GPU budget timestamps require explicit time zones")
        self.starts_at, self.expires_at = (value.timestamp() for value in times)
        if self.expires_at - self.starts_at != 86400:
            raise ValueError("GPU override must expire after exactly 24 hours")
        self.wall_clock = wall_clock
        self.monotonic = monotonic
        self.monotonic_deadline = monotonic() + max(
            0.0, min(86400.0, self.expires_at - wall_clock())
        )

    def duty_fraction(self) -> float:
        now = self.wall_clock()
        if self.starts_at <= now < self.expires_at and self.monotonic() < self.monotonic_deadline:
            return 1.0
        return 0.5


def windowed_duty_cycle(base_class, budget: ApprovedGpuWindow):
    """Keep base synchronization and memory guards, recording actual idle time."""

    class WindowedDutyCycle(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            original_sleep = self.sleep
            self.actual_idle_seconds = 0.0

            def paced_sleep(active_seconds):
                idle = 0.0 if budget.duty_fraction() == 1.0 else active_seconds
                if idle:
                    original_sleep(idle)
                    self.actual_idle_seconds += idle

            self.sleep = paced_sleep

        def run(self, operation):
            try:
                return super().run(operation)
            finally:
                self.idle_seconds = self.actual_idle_seconds

    return WindowedDutyCycle
