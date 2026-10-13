from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_gpu_budget import ApprovedGpuWindow, windowed_duty_cycle
from train_c5_face_learning_pilot import DutyCycle


def policy(tmp_path):
    path = tmp_path / "budget.json"
    path.write_text(
        json.dumps(
            {
                "schema": "umi-studio-gpu-budget-window/1",
                "starts_at": "2026-10-13T00:48:21Z",
                "expires_at": "2026-10-14T00:48:21Z",
                "window_duty_fraction": 1.0,
                "after_window_duty_fraction": 0.5,
            }
        )
    )
    return path


def test_window_expires_by_wall_clock_and_survives_clock_rollback(tmp_path):
    now = [datetime.fromisoformat("2026-10-13T00:48:21+00:00").timestamp()]
    ticks = [100.0]
    budget = ApprovedGpuWindow(
        policy(tmp_path), wall_clock=lambda: now[0], monotonic=lambda: ticks[0]
    )
    assert budget.duty_fraction() == 1.0
    now[0] += 86400
    assert budget.duty_fraction() == 0.5
    now[0] -= 1000
    ticks[0] += 86400
    assert budget.duty_fraction() == 0.5


def test_late_launch_receives_only_remaining_window(tmp_path):
    now = [datetime.fromisoformat("2026-10-13T12:48:21+00:00").timestamp()]
    ticks = [100.0]
    budget = ApprovedGpuWindow(
        policy(tmp_path), wall_clock=lambda: now[0], monotonic=lambda: ticks[0]
    )
    assert budget.duty_fraction() == 1.0
    ticks[0] += 43200
    assert budget.duty_fraction() == 0.5


def test_invalid_policy_cannot_launch_unbounded_training(tmp_path):
    path = policy(tmp_path)
    value = json.loads(path.read_text())
    value["expires_at"] = "2026-10-15T00:48:21Z"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="24 hours"):
        ApprovedGpuWindow(path)


def test_timed_cycle_preserves_memory_guards_and_records_actual_idle():
    fractions = iter([1.0, 0.5, 0.5])
    budget = SimpleNamespace(duty_fraction=lambda: next(fractions))
    times = iter([0.0, 2.0, 3.0, 6.0, 7.0, 11.0])
    sleeps = []
    memory = [1024]
    torch = SimpleNamespace(
        mps=SimpleNamespace(synchronize=lambda: None, current_allocated_memory=lambda: memory[0])
    )
    duty = windowed_duty_cycle(DutyCycle, budget)(
        torch, clock=lambda: next(times), sleep=sleeps.append
    )
    assert duty.run(lambda: "first") == "first"
    assert duty.active_seconds == 2.0 and duty.idle_seconds == 0.0 and sleeps == []

    def fail():
        raise ValueError("operation failed")

    with pytest.raises(ValueError, match="operation failed"):
        duty.run(fail)
    assert duty.active_seconds == 5.0 and duty.idle_seconds == 3.0 and sleeps == [3.0]
    memory[0] = 7 * 1024**3
    with pytest.raises(RuntimeError, match="memory guard"):
        duty.run(lambda: None)
    assert duty.idle_seconds == 7.0 and sleeps == [3.0, 4.0]
