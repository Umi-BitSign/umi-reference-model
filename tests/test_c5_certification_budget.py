from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_certification_budget import CertificationComputeWindow


def policy(tmp_path):
    record = {
        "schema": "umi-studio-certification-compute-window/1",
        "starts_at": "2026-10-13T22:47:28Z",
        "expires_at": "2026-10-15T22:47:28Z",
        "window_duty_fraction": 1.0,
        "after_window_duty_fraction": 0.5,
        "window_cpu_threads": 8,
        "after_window_cpu_threads": 2,
        "memory_policy": "framework_default_no_artificial_six_GiB_cap",
        "miner_service_changes_authorized": False,
        "protected_certification_material_access_authorized": False,
    }
    target = tmp_path / "budget.json"
    target.write_text(json.dumps(record))
    return target, record


def test_48_hour_window_restores_50_percent_and_two_cpu_threads(tmp_path):
    target, record = policy(tmp_path)
    start = datetime.fromisoformat(record["starts_at"].replace("Z", "+00:00")).timestamp()
    clock = {"wall": start, "mono": 100.0}
    budget = CertificationComputeWindow(
        target, wall_clock=lambda: clock["wall"], monotonic=lambda: clock["mono"]
    )
    assert budget.duty_fraction() == 1 and budget.cpu_threads() == 8
    clock["wall"] += 172800
    assert budget.duty_fraction() == 0.5 and budget.cpu_threads() == 2


def test_wall_clock_rollback_cannot_extend_the_authorized_window(tmp_path):
    target, record = policy(tmp_path)
    start = datetime.fromisoformat(record["starts_at"].replace("Z", "+00:00")).timestamp()
    clock = {"wall": start, "mono": 100.0}
    budget = CertificationComputeWindow(
        target, wall_clock=lambda: clock["wall"], monotonic=lambda: clock["mono"]
    )
    clock["mono"] += 172800
    clock["wall"] = start + 1
    assert budget.duty_fraction() == 0.5


@pytest.mark.parametrize(
    "key,value",
    (
        ("expires_at", "2026-10-14T22:47:28Z"),
        ("after_window_duty_fraction", 1.0),
        ("miner_service_changes_authorized", True),
    ),
)
def test_changed_expiry_or_expanded_external_authority_is_rejected(tmp_path, key, value):
    target, record = policy(tmp_path)
    record[key] = value
    target.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        CertificationComputeWindow(target)
