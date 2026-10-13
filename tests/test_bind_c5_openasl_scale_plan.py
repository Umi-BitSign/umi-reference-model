import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from benchmark_fleurs_dev import sha256
from bind_c5_openasl_scale_plan import load_scale_examples


def test_loader_binds_only_training_partition(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    training, confirmation, reserve, index = [], [], [], []
    for source in range(180):
        target = training if source < 120 else confirmation
        second_target = training if source < 120 else reserve
        for clip, destination in ((0, target), (1, second_target)):
            example_id = f"source-{source:03d}-clip-{clip}"
            clip_id = f"cached-{source:03d}-{clip}"
            record = {
                "example_id": example_id,
                "youtube_id": f"source-{source:03d}",
                "source_group": f"openasl:source-{source:03d}",
                "reference": f"reference {source} {clip}",
            }
            destination.append(record)
            cache = cache_dir / f"{clip_id}.npz"
            cache.write_bytes(example_id.encode())
            index.append(
                {
                    "schema": "umi-studio-openasl-stream-cache/1",
                    "clip_id": clip_id,
                    **record,
                    "split": "training",
                    "temporal_stride": 2,
                    "spatial_policy": "full_source_frame_preserve_aspect_ratio",
                    "frames": 3,
                    "encoder_shape": [3, 1536],
                    "encoder_cache_sha256": sha256(cache),
                }
            )
    index_path = tmp_path / "index.jsonl"
    index_path.write_text("".join(json.dumps(row) + "\n" for row in index))
    partition = {
        "schema": "umi-studio-openasl-scale-partition/1",
        "training_records": training,
        "sealed_confirmation_records": confirmation,
        "unused_reserve_records": reserve,
    }
    examples, hashes = load_scale_examples(index_path, cache_dir, partition)
    assert len(examples) == 240
    assert len({row["source_group"] for row in examples}) == 120
    assert not {row["id"] for row in examples} & {
        row["example_id"] for row in confirmation + reserve
    }
    assert hashes == {"openasl_scale_index_sha256": sha256(index_path)}
