"""Verify separately delivered inference assets without importing model code."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify():
    manifest = json.loads((ROOT / "MODEL_ARTIFACT.json").read_text())
    records = manifest["model_files"]
    if not isinstance(records, list) or not 1 <= len(records) <= 512:
        raise ValueError("Model asset inventory is empty or too large")
    seen = set()
    for item in records:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "bytes", "sha256"}
            or not isinstance(item["path"], str)
            or type(item["bytes"]) is not int
            or not 0 < item["bytes"] <= 4 * 1024**3
            or not isinstance(item["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        ):
            raise ValueError("Invalid model asset record")
        relative = Path(item["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != item["path"]
            or not relative.parts
            or relative.parts[0] != "models"
            or item["path"] in seen
        ):
            raise ValueError("Invalid or duplicate model asset path")
        seen.add(item["path"])
        path = ROOT / item["path"]
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError("Model asset escapes candidate directory")
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError("Missing or wrong-size model asset: " + item["path"])
        digest = hashlib.sha256()
        total = 0
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                total += len(chunk)
                if total > item["bytes"]:
                    raise ValueError("Model asset grew during verification: " + item["path"])
                digest.update(chunk)
        if total != item["bytes"] or digest.hexdigest() != item["sha256"]:
            raise ValueError("Model hash mismatch: " + item["path"])
    print("All model assets verified")


if __name__ == "__main__":
    verify()
