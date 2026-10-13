"""The confirmation runner exposes only the two frozen configurations."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_temporal_stride_confirmation import ADAPTER, evaluation_command


def test_baseline_is_stride_one_without_adapter():
    command, results = evaluation_command("how2sign", False)
    assert command[command.index("--temporal-stride") + 1] == "1"
    assert "--adapter" not in command
    assert command[command.index("--limit") + 1] == "42"
    assert results.name == "confirmation-how2sign-baseline-stride1.jsonl"


def test_candidate_is_stride_two_with_only_frozen_adapter():
    command, results = evaluation_command("fleurs", True)
    assert command[command.index("--temporal-stride") + 1] == "2"
    assert command[command.index("--adapter") + 1] == str(ADAPTER)
    assert command[command.index("--limit") + 1] == "46"
    assert results.name == "confirmation-fleurs-candidate-stride2.jsonl"
