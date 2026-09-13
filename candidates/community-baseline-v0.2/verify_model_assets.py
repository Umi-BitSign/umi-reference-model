"""Verify separately delivered inference assets without importing model code."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def verify():
    manifest = json.loads((ROOT / "MODEL_ARTIFACT.json").read_text())
    for item in manifest["model_files"]:
        path = ROOT / item["path"]
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError("Model asset escapes candidate directory")
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError("Missing or wrong-size model asset: " + item["path"])
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError("Model hash mismatch: " + item["path"])
    print("All model assets verified")

if __name__ == "__main__":
    verify()
