"""Bound the small clip-only OpenASL expansion independently of full-video downloads."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCALE_DATA = ROOT / "data/openasl-scale"
FREE_FLOOR_BYTES = 80 * 1024**3
DATA_CAP_BYTES = 4 * 1024**3
ITEM_MARGIN_BYTES = 128 * 1024**2


def checked_storage(*, reserve_item: bool = True) -> dict[str, int]:
    margin = ITEM_MARGIN_BYTES if reserve_item else 0
    free = shutil.disk_usage(ROOT).free
    used = 0
    if SCALE_DATA.exists():
        if SCALE_DATA.is_symlink():
            raise ValueError("OpenASL scale data directory must not be a symlink")
        for path in SCALE_DATA.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"OpenASL scale data contains a symlink: {path}")
            if path.is_file():
                used += path.stat().st_size
    if free < FREE_FLOOR_BYTES + margin:
        raise RuntimeError("OpenASL clip-only expansion reached its 80 GiB free-disk floor")
    if used + margin > DATA_CAP_BYTES:
        raise RuntimeError("OpenASL clip-only expansion reached its 4 GiB data budget")
    return {"free_bytes": free, "scale_data_bytes": used}
