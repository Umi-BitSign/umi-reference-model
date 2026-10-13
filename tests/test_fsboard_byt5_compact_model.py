from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from fsboard_byt5_compact_model import FSboardByT5Compact


def test_compact_model_accepts_83_point_features() -> None:
    language_model = T5ForConditionalGeneration(T5Config(
        d_model=32, d_ff=64, num_layers=1, num_decoder_layers=1, num_heads=2,
        vocab_size=300, decoder_start_token_id=0, pad_token_id=0,
    ))
    model = FSboardByT5Compact(language_model)
    output = model(torch.zeros(1, 5, 249), torch.ones(1, 5, dtype=torch.long),
                   torch.tensor([[4, 5, 1]]))
    assert torch.isfinite(output.loss)
    with pytest.raises(ValueError, match="249 columns"):
        model(torch.zeros(1, 5, 255), torch.ones(1, 5, dtype=torch.long),
              torch.tensor([[4, 5, 1]]))
