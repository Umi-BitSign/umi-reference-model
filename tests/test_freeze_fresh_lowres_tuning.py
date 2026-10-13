from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from freeze_fresh_lowres_tuning import how2sign_rows


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n")


def test_how2sign_tuning_binds_resampled_clip_hash(tmp_path: Path) -> None:
    names = [f"clip-{index}" for index in range(16)]
    selection_path = tmp_path / "selection.json"
    _write_json(
        selection_path,
        {
            "schema": "umi-studio-how2sign-validation-selection/1",
            "training_excluded": True,
            "rows": [
                {
                    "sentence_name": name,
                    "video_id": f"video-{index}",
                    "signer": index % 4,
                    "reference": f"reference {index}",
                }
                for index, name in enumerate(names)
            ],
        },
    )
    selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    partition_path = tmp_path / "partition.json"
    _write_json(
        partition_path,
        {
            "schema": "umi-studio-how2sign-validation-partition/1",
            "selection_sha256": selection_sha,
            "confirmation_predictions_inspected": False,
            "tuning_sentence_names": names,
        },
    )
    index_path = tmp_path / "index.jsonl"
    index_path.write_text(
        "".join(
            json.dumps(
                {
                    "schema": "umi-studio-how2sign-encoder-cache-example/1",
                    "split": "validation",
                    "clip_fps": 25,
                    "selection_sha256": selection_sha,
                    "sentence_name": name,
                    "reference": f"reference {index}",
                    "signer": index % 4,
                    "source_clip_sha256": "a" * 64,
                    "resampled_clip_sha256": f"{index:064x}",
                }
            )
            + "\n"
            for index, name in enumerate(names)
        )
    )
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    for name in names:
        (video_dir / f"{name}.mp4").write_bytes(b"video")
    rows = how2sign_rows(
        tmp_path, selection_path, partition_path, index_path, video_dir
    )
    assert len(rows) == 16
    assert rows[1]["video_sha256"] == f"{1:064x}"
    assert rows[1]["video_sha256"] != "a" * 64
