"""Bounded training-only storage; bulk-ingestion policies remain unchanged."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = {
    "physical_floor_bytes": 60 * 1024**3,
    "output_budget_bytes": 2 * 1024**3,
    "headroom_bytes": 128 * 1024**2,
    "downloads_new_datasets": False,
}


def storage(prospective=0, *, output=None, root=ROOT, disk_usage=shutil.disk_usage):
    if prospective < 0:
        raise ValueError("negative training storage reservation")
    used = 0
    if output is not None:
        output = output.resolve()
        if not output.is_relative_to(root / "checkpoints"):
            raise ValueError("Terra output budget only applies to its checkpoint subtree")
        for path in output.rglob("*"):
            if path.is_symlink():
                raise ValueError("Terra output budget cannot follow links")
            if path.is_file():
                used += path.stat().st_size
    if used + prospective > POLICY["output_budget_bytes"]:
        raise RuntimeError("Terra run reached its explicit two-GiB output budget")
    remaining = POLICY["output_budget_bytes"] - used
    required = POLICY["physical_floor_bytes"] + POLICY["headroom_bytes"] + remaining
    if disk_usage(root).free < required:
        raise RuntimeError(
            "Terra run lacks its sixty-GiB physical floor and remaining output reserve"
        )
    return {
        "output_bytes": used,
        "remaining_output_reserve_bytes": remaining,
        "physical_free_bytes": disk_usage(root).free,
    }
