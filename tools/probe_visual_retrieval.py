"""Test frozen encoder mean-pool nearest-neighbor captions on training clips."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from pathlib import Path

import numpy as np

from benchmark_fleurs_dev import distance, miner_revision, normalize, sha256
from freeze_visual_lora_plan import SOURCE_LAYOUT

ROOT = Path(__file__).resolve().parents[1]
STAGE_PLAN = ROOT / "runs/visual-lora-stage1-v1/plan.json"
TRAINING_FIT = ROOT / "runs/visual-training-fit-probe-v1/result.json"
SOURCES = ("how2sign_original", "how2sign_new", "how2sign_raw")
VIDEO_ID = re.compile(r"^(.+)_\d+-\d+-rgb_front$")


def source_video_id(example_id: str) -> str:
    match = VIDEO_ID.fullmatch(example_id)
    if match is None:
        raise ValueError(f"How2Sign example ID has an unknown format: {example_id}")
    return match.group(1)


def normalized_mean(state: np.ndarray, frames: int) -> np.ndarray:
    if state.shape != (frames, 1536) or state.dtype != np.float32:
        raise ValueError("frozen encoder state shape or dtype differs")
    mean = state.mean(axis=0, dtype=np.float64).astype(np.float32)
    norm = np.linalg.norm(mean)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("frozen encoder mean has zero or nonfinite norm")
    return mean / norm


def nearest_neighbor(
    query_index: int, vectors: np.ndarray, video_ids: list[str]
) -> tuple[int, float]:
    scores = vectors @ vectors[query_index]
    eligible = [
        i for i, video_id in enumerate(video_ids)
        if video_id != video_ids[query_index]
    ]
    if not eligible:
        raise ValueError("no source-video-disjoint retrieval candidate")
    # Source order is fixed by the frozen indexes, so lower index breaks ties.
    best = max(eligible, key=lambda i: (float(scores[i]), -i))
    return best, float(scores[best])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--miner-health-url", required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs"):
        parser.error("retrieval probe needs a new run-ledger output")
    stage = json.loads(STAGE_PLAN.read_text())
    fit = json.loads(TRAINING_FIT.read_text())
    if (
        stage.get("schema") != "umi-studio-visual-lora-plan/1"
        or fit.get("schema") != "umi-studio-visual-training-fit-probe/1"
        or fit.get("split") != "training"
        or fit.get("stage_plan_sha256") != sha256(STAGE_PLAN)
        or fit.get("diagnostic_only") is not True
        or len(fit.get("records", [])) != 32
    ):
        raise ValueError("retrieval probe source diagnostic differs")
    fit_rows = {
        (row["source"], row["example_id"]): row
        for row in fit["records"] if row["source"] in SOURCES
    }
    if len(fit_rows) != 28:
        raise ValueError("retrieval probe needs 28 distinct How2Sign training queries")
    revision = miner_revision(args.miner_health_url, stage["miner_model_revision"])
    pool = []
    index_hashes = {}
    for source in SOURCES:
        layout = SOURCE_LAYOUT[source]
        index = ROOT / layout["index"]
        index_hashes[source] = sha256(index)
        if index_hashes[source] != stage["sources"][source]["index_sha256"]:
            raise ValueError(f"retrieval probe source index changed: {source}")
        rows = [json.loads(line) for line in index.read_text().splitlines()]
        if len(rows) != layout["expected"]:
            raise ValueError(f"retrieval probe source index count changed: {source}")
        for row in rows:
            example_id = row["sentence_name"]
            if row["split"] != "training" or row["schema"] != "umi-studio-how2sign-encoder-cache-example/1":
                raise ValueError(f"retrieval probe source row differs: {example_id}")
            pool.append({
                "source": source,
                "example_id": example_id,
                "video_id": source_video_id(example_id),
                "reference": row["reference"],
                "signer": row["signer"],
                "frames": row["encoder_shape"][0],
                "cache": str(ROOT / layout["cache_dir"] / f"{example_id}.npz"),
                "cache_sha256": row["encoder_cache_sha256"],
            })
    if len(pool) != 4241 or len({(row["source"], row["example_id"]) for row in pool}) != len(pool):
        raise ValueError("retrieval probe pool is incomplete or repeated")
    vectors = np.empty((len(pool), 1536), dtype=np.float32)
    for i, row in enumerate(pool):
        cache = Path(row["cache"])
        if sha256(cache) != row["cache_sha256"]:
            raise ValueError(f"retrieval probe cache changed: {row['example_id']}")
        with np.load(cache, allow_pickle=False) as payload:
            vectors[i] = normalized_mean(payload["encoder_hidden_state"], row["frames"])
        if (i + 1) % 64 == 0:
            revision = miner_revision(args.miner_health_url, revision)
            print(f"encoded {i+1}/{len(pool)} training embeddings", flush=True)
    revision = miner_revision(args.miner_health_url, revision)
    video_ids = [row["video_id"] for row in pool]
    records = []
    for i, query in enumerate(pool):
        key = (query["source"], query["example_id"])
        if key not in fit_rows:
            continue
        observed = fit_rows[key]
        if observed["cache_sha256"] != query["cache_sha256"] or observed["reference"] != query["reference"]:
            raise ValueError(f"retrieval probe query changed: {query['example_id']}")
        nearest_index, cosine = nearest_neighbor(i, vectors, video_ids)
        neighbor = pool[nearest_index]
        reference = normalize(query["reference"])
        retrieved_wer = distance(reference, normalize(neighbor["reference"])) / len(reference)
        records.append({
            "source": query["source"],
            "signer": query["signer"],
            "query_id": query["example_id"],
            "query_video_id": query["video_id"],
            "reference": query["reference"],
            "baseline_hypothesis": observed["baseline_hypothesis"],
            "baseline_wer": observed["baseline_wer"],
            "neighbor_source": neighbor["source"],
            "neighbor_id": neighbor["example_id"],
            "neighbor_video_id": neighbor["video_id"],
            "neighbor_reference": neighbor["reference"],
            "cosine": cosine,
            "retrieved_wer": retrieved_wer,
        })
    if len(records) != 28:
        raise ValueError("retrieval probe result is incomplete")
    result = {
        "schema": "umi-studio-visual-retrieval-probe/1",
        "split": "training",
        "diagnostic_only": True,
        "retrieval_rule": "L2-normalized temporal mean of frozen SHuBERT encoder states, cosine top 1, exclude every clip from the query source video, fixed source-index order for ties",
        "stage_plan_sha256": sha256(STAGE_PLAN),
        "training_fit_sha256": sha256(TRAINING_FIT),
        "probe_source_sha256": sha256(Path(__file__)),
        "source_index_sha256": index_hashes,
        "pool_rows": len(pool),
        "miner_model_revision": revision,
        "summary": {
            "examples": len(records),
            "baseline_mean_wer": statistics.fmean(row["baseline_wer"] for row in records),
            "retrieval_mean_wer": statistics.fmean(row["retrieved_wer"] for row in records),
            "retrieval_better_count": sum(row["retrieved_wer"] < row["baseline_wer"] for row in records),
            "retrieval_worse_count": sum(row["retrieved_wer"] > row["baseline_wer"] for row in records),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    if temporary.exists():
        parser.error("unfinished retrieval probe needs inspection")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
