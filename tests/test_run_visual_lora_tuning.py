"""The visual tuning runner uses only frozen tuning caches and verifies scores."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_fleurs_dev import sha256
from run_visual_lora_tuning import checked_results, evaluation_command, result_path


def test_command_uses_visual_streams_and_tuning_split(tmp_path):
    command = evaluation_command(
        tmp_path / "index.jsonl",
        tmp_path / "cache",
        "tuning",
        tmp_path / "adapter.safetensors",
        tmp_path / "results.jsonl",
        "http://127.0.0.1:8787/healthz",
        7,
    )
    assert command[:3] == ["nice", "-n", "20"]
    assert "--use-stream-features" in command
    assert command[command.index("--split") + 1] == "tuning"
    assert command[-2:] == ["--limit", "7"]
    assert "confirmation" not in command
    assert result_path(tmp_path, "how2sign", 512).name == "tuning-how2sign-000512.jsonl"


def test_results_require_source_prefix_matching_adapter_and_recomputed_wer(tmp_path):
    index = tmp_path / "index.jsonl"
    source = [
        {
            "schema": "umi-studio-fleurs-dev-encoder-cache-example/1",
            "split": "tuning",
            "example_id": f"example-{number}",
            "selection_sha256": "s" * 64,
            "signer": 3,
            "encoder_cache_sha256": f"{number}" * 64,
            "reference": "hello world",
        }
        for number in (1, 2)
    ]
    index.write_text("".join(json.dumps(row) + "\n" for row in source))
    output = tmp_path / "results.jsonl"
    assert checked_results(output, index, "tuning", "c" * 64, "a" * 64) == 0
    row = {
        "schema": "umi-studio-fleurs-cached-evaluation-example/1",
        "split": "tuning",
        "example_id": "example-1",
        "selection_sha256": "s" * 64,
        "signer": 3,
        "index_sha256": sha256(index),
        "encoder_cache_sha256": "1" * 64,
        "checkpoint_sha256": "c" * 64,
        "adapter_sha256": "a" * 64,
        "evaluation_input": "visual_streams",
        "reference": "hello world",
        "hypothesis": "hello",
        "single_reference_wer": 0.5,
        "single_reference_word_edits": 1,
        "reference_word_count": 2,
    }
    output.write_text(json.dumps(row) + "\n")
    assert checked_results(output, index, "tuning", "c" * 64, "a" * 64) == 1
    row["single_reference_wer"] = 0.0
    output.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="source or score"):
        checked_results(output, index, "tuning", "c" * 64, "a" * 64)
    row["single_reference_wer"] = 0.5
    output.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="source or score"):
        checked_results(output, index, "tuning", "c" * 64, "b" * 64)
