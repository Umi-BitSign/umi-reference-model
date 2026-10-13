import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import openasl_scale_storage as storage


def test_rejects_insufficient_disk_or_data_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "SCALE_DATA", tmp_path)
    monkeypatch.setattr(storage, "FREE_FLOOR_BYTES", 100)
    monkeypatch.setattr(storage, "ITEM_MARGIN_BYTES", 10)
    monkeypatch.setattr(storage, "DATA_CAP_BYTES", 40)
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda _: SimpleNamespace(free=109))
    with pytest.raises(RuntimeError, match="free-disk floor"):
        storage.checked_storage()
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda _: SimpleNamespace(free=200))
    (tmp_path / "clip").write_bytes(b"x" * 31)
    with pytest.raises(RuntimeError, match="data budget"):
        storage.checked_storage()
    assert storage.checked_storage(reserve_item=False)["scale_data_bytes"] == 31
