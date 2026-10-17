"""Hash an existing canonical data file without widening recovery-artifact permissions."""

from __future__ import annotations

from pathlib import Path

from benchmark_fleurs_dev import sha256

ROOT = Path(__file__).resolve().parents[1]


def media_artifact(path, *, root=ROOT):
    original, resolved = path.absolute(), path.resolve()
    if (
        original != resolved
        or not resolved.is_relative_to(root.resolve() / "data")
        or not resolved.is_file()
    ):
        raise ValueError("training media must be an existing canonical file under data/")
    return {"path": str(resolved.relative_to(root.resolve())), "sha256": sha256(resolved)}
