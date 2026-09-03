from __future__ import annotations

import inspect
from collections.abc import Callable

import pytest
import torch
from torch import nn

import bitsign_motion.portable_model as portable_model
from bitsign_motion.portable_model import (
    BOS_TOKEN_ID,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_VOCABULARY_SIZE,
    EOS_TOKEN_ID,
    MOTION_FEATURE_DIM,
    PAD_TOKEN_ID,
    UNK_TOKEN_ID,
    PortableS0,
    PortableS0Config,
    PortableS1,
    PortableS1Config,
)


def _small_s0() -> PortableS0:
    return PortableS0(
        PortableS0Config(
            vocabulary_size=32,
            max_frames=12,
            max_output_tokens=8,
            hidden_size=32,
            dropout=0.0,
        )
    )


def _small_s1() -> PortableS1:
    return PortableS1(
        PortableS1Config(
            vocabulary_size=32,
            max_frames=12,
            max_output_tokens=8,
            hidden_size=32,
            num_heads=4,
            feed_forward_size=64,
            encoder_layers=2,
            decoder_layers=2,
            dropout=0.0,
        )
    )


def _motion_inputs(batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    motion = torch.randn(batch_size, 12, MOTION_FEATURE_DIM)
    frame_mask = torch.ones(batch_size, 12, dtype=torch.int32)
    frame_mask[-1, 8:] = 0
    return motion, frame_mask


def _decoder_inputs(batch_size: int = 2) -> torch.Tensor:
    values = torch.randint(4, 32, (batch_size, 8), dtype=torch.int64)
    values[:, 0] = BOS_TOKEN_ID
    return values


def test_reserved_ids_and_default_contract_are_fixed() -> None:
    assert (PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID, UNK_TOKEN_ID) == (0, 1, 2, 3)
    assert DEFAULT_MAX_FRAMES == 120
    assert DEFAULT_MAX_OUTPUT_TOKENS == 128
    assert DEFAULT_VOCABULARY_SIZE == 4_096
    assert MOTION_FEATURE_DIM == 1_184


def test_s0_default_architecture_and_parameter_bound() -> None:
    model = PortableS0()

    assert model.stem[0].in_features == MOTION_FEATURE_DIM
    assert model.stem[0].out_features == 128
    assert len(model.temporal_blocks) == 4
    assert len(model.output_mixer) == 2
    assert model.output_queries.shape == (128, 128)
    assert model.output_projection.out_features == 46
    assert model.parameter_count == 608_430
    assert 0 < model.parameter_count < 2_000_000


def test_s1_default_architecture_and_parameter_bound() -> None:
    model = PortableS1()

    assert model.motion_stem[0].in_features == MOTION_FEATURE_DIM
    assert model.motion_stem[0].out_features == 256
    assert isinstance(model.motion_stem[1], nn.GELU)
    assert model.motion_stem[1].approximate == "tanh"
    assert model.motion_stem[2].in_features == 256
    assert model.motion_stem[2].out_features == 256
    assert model.encoder_positions.shape == (120, 256)
    assert model.decoder_positions.shape == (128, 256)
    assert len(model.encoder_layers) == 4
    assert len(model.decoder_layers) == 4
    assert model.encoder_layers[0].attention.num_heads == 4
    assert model.encoder_layers[0].attention.head_size == 64
    assert model.encoder_layers[0].feed_forward.input_projection.out_features == 1_024
    assert model.token_embedding.weight.shape == (4_096, 256)
    assert model.output_projection.weight is model.token_embedding.weight
    assert 8_000_000 < model.parameter_count <= 9_000_000

    prohibited = (nn.MultiheadAttention, nn.TransformerEncoder, nn.TransformerDecoder)
    assert not any(isinstance(module, prohibited) for module in model.modules())
    source = inspect.getsource(portable_model)
    assert "scaled_dot_product_attention" not in source


def test_s0_forward_backward_shape_and_float16_inference() -> None:
    model = _small_s0()
    motion, frame_mask = _motion_inputs()

    logits = model(motion, frame_mask)
    logits.square().mean().backward()

    assert logits.shape == (2, 8, 32)
    assert all(parameter.grad is not None for parameter in model.parameters())

    half_model = _small_s0().half().eval()
    with torch.no_grad():
        half_logits = half_model(motion[:1].half(), frame_mask[:1])
    assert half_logits.shape == (1, 8, 32)
    assert half_logits.dtype == torch.float16


def test_s1_encode_decode_forward_backward_and_greedy_shapes() -> None:
    model = _small_s1()
    motion, frame_mask = _motion_inputs()
    decoder_input_ids = _decoder_inputs()

    memory = model.encode(motion, frame_mask)
    decoded_logits = model.decode(decoder_input_ids, memory, frame_mask)
    forward_logits = model(motion, frame_mask, decoder_input_ids)
    forward_logits.square().mean().backward()

    assert memory.shape == (2, 12, 32)
    assert decoded_logits.shape == (2, 8, 32)
    assert forward_logits.shape == (2, 8, 32)
    torch.testing.assert_close(decoded_logits, forward_logits)
    assert all(parameter.grad is not None for parameter in model.parameters())

    model.eval()
    with torch.no_grad():
        model.token_embedding.weight.zero_()
        model.output_projection.bias.fill_(-10.0)
        model.output_projection.bias[EOS_TOKEN_ID] = 10.0
    generated = model.greedy_decode(motion[:1], frame_mask[:1], max_new_tokens=4)
    assert generated.shape == (1, 4)
    assert generated.tolist() == [[EOS_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID]]

    half_model = _small_s1().half().eval()
    with torch.no_grad():
        half_logits = half_model(motion[:1].half(), frame_mask[:1], decoder_input_ids[:1])
    assert half_logits.shape == (1, 8, 32)
    assert half_logits.dtype == torch.float16

    with pytest.raises(ValueError, match="max_new_tokens"):
        model.greedy_decode(motion[:1], frame_mask[:1], max_new_tokens=0)


def test_s1_greedy_decode_stops_once_the_batch_is_finished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _small_s1().eval()
    motion, frame_mask = _motion_inputs()
    decode_calls = 0

    def eos_decode(
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        observed_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal decode_calls
        decode_calls += 1
        assert memory.shape[0] == decoder_input_ids.shape[0]
        assert observed_frame_mask.shape[0] == decoder_input_ids.shape[0]
        logits = torch.full(
            (
                decoder_input_ids.shape[0],
                model.config.max_output_tokens,
                model.config.vocabulary_size,
            ),
            -1.0,
            device=decoder_input_ids.device,
        )
        logits[:, :, EOS_TOKEN_ID] = 1.0
        return logits

    monkeypatch.setattr(model, "decode", eos_decode)
    generated = model.greedy_decode(motion, frame_mask, max_new_tokens=8)

    assert decode_calls == 1
    assert generated[:, 0].tolist() == [EOS_TOKEN_ID, EOS_TOKEN_ID]
    assert torch.count_nonzero(generated[:, 1:]).item() == 0


def test_s1_beam_decode_uses_cumulative_log_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _small_s1().eval()
    motion, frame_mask = _motion_inputs(batch_size=1)

    def scripted_decode(
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        observed_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        del memory, observed_frame_mask
        logits = torch.full(
            (
                decoder_input_ids.shape[0],
                model.config.max_output_tokens,
                model.config.vocabulary_size,
            ),
            -10.0,
            device=decoder_input_ids.device,
        )
        for row in range(decoder_input_ids.shape[0]):
            generated_count = int(decoder_input_ids[row].ne(PAD_TOKEN_ID).sum().item()) - 1
            if generated_count == 0:
                # Greedy chooses 4, but token 4 then has a diffuse continuation.
                logits[row, 0, 4] = 3.0
                logits[row, 0, 5] = 2.9
            elif int(decoder_input_ids[row, 1].item()) == 4:
                logits[row, 1, :] = 0.0
            else:
                logits[row, 1, :] = 0.0
                logits[row, 1, EOS_TOKEN_ID] = 10.0
        return logits

    monkeypatch.setattr(model, "decode", scripted_decode)
    generated = model.beam_decode(
        motion,
        frame_mask,
        max_new_tokens=4,
        beam_width=2,
        no_repeat_ngram_size=3,
    )

    assert generated.tolist() == [[5, EOS_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID]]


def test_s1_beam_decode_suppresses_reserved_tokens_and_breaks_ties_by_token_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _small_s1().eval()
    motion, frame_mask = _motion_inputs(batch_size=1)

    def tied_decode(
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        observed_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        del memory, observed_frame_mask
        logits = torch.full(
            (
                decoder_input_ids.shape[0],
                model.config.max_output_tokens,
                model.config.vocabulary_size,
            ),
            -10.0,
            device=decoder_input_ids.device,
        )
        logits[:, 0, PAD_TOKEN_ID] = 100.0
        logits[:, 0, BOS_TOKEN_ID] = 100.0
        logits[:, 0, 4] = 10.0
        logits[:, 0, 5] = 10.0
        return logits

    monkeypatch.setattr(model, "decode", tied_decode)
    generated = model.beam_decode(motion, frame_mask, max_new_tokens=1, beam_width=2)

    assert generated.tolist() == [[4]]


def test_s1_beam_decode_blocks_repeated_generated_token_trigrams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _small_s1().eval()
    motion, frame_mask = _motion_inputs(batch_size=1)

    def repetitive_decode(
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        observed_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        del memory, observed_frame_mask
        logits = torch.full(
            (
                decoder_input_ids.shape[0],
                model.config.max_output_tokens,
                model.config.vocabulary_size,
            ),
            -10.0,
            device=decoder_input_ids.device,
        )
        logits[:, :, 4] = 10.0
        logits[:, :, 5] = 9.0
        return logits

    monkeypatch.setattr(model, "decode", repetitive_decode)
    generated = model.beam_decode(
        motion,
        frame_mask,
        max_new_tokens=4,
        beam_width=1,
        no_repeat_ngram_size=3,
    )

    assert generated.tolist() == [[4, 4, 4, 5]]


def test_s1_beam_decode_batches_heterogeneous_live_beams_without_cross_sample_competition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _small_s1().eval()
    torch.manual_seed(41)
    motion = torch.randn(3, model.config.max_frames, MOTION_FEATURE_DIM)
    frame_mask = torch.zeros(3, model.config.max_frames, dtype=torch.int32)
    frame_mask[0, :2] = 1
    frame_mask[1, :3] = 1
    frame_mask[2, :4] = 1
    decode_batch_sizes: list[int] = []

    def heterogeneous_decode(
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        observed_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        del memory
        decode_batch_sizes.append(decoder_input_ids.shape[0])
        logits = torch.full(
            (
                decoder_input_ids.shape[0],
                model.config.max_output_tokens,
                model.config.vocabulary_size,
            ),
            -torch.inf,
            device=decoder_input_ids.device,
        )
        logits[:, :, PAD_TOKEN_ID] = 100.0
        logits[:, :, BOS_TOKEN_ID] = 100.0
        for row in range(decoder_input_ids.shape[0]):
            generated_count = int(decoder_input_ids[row].ne(PAD_TOKEN_ID).sum().item()) - 1
            sample_kind = int(observed_frame_mask[row].sum().item())
            if sample_kind == 2:
                logits[row, generated_count, EOS_TOKEN_ID] = 2.0
            elif sample_kind == 3:
                if generated_count == 0:
                    logits[row, 0, 4] = 2.0
                    logits[row, 0, 5] = 2.0
                elif int(decoder_input_ids[row, 1].item()) == 4:
                    logits[row, generated_count, EOS_TOKEN_ID] = 2.0
                else:
                    logits[row, generated_count, 6] = 2.0
            else:
                logits[row, generated_count, 4] = 2.0
                logits[row, generated_count, 5] = 1.0
        return logits

    monkeypatch.setattr(model, "decode", heterogeneous_decode)
    batched = model.beam_decode(
        motion,
        frame_mask,
        max_new_tokens=4,
        beam_width=2,
        no_repeat_ngram_size=3,
    )
    batched_call_sizes = list(decode_batch_sizes)
    decode_batch_sizes.clear()
    independent = torch.cat(
        [
            model.beam_decode(
                motion[index : index + 1],
                frame_mask[index : index + 1],
                max_new_tokens=4,
                beam_width=2,
                no_repeat_ngram_size=3,
            )
            for index in range(motion.shape[0])
        ],
        dim=0,
    )

    assert torch.equal(batched, independent)
    assert batched.tolist() == [
        [EOS_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID],
        [4, EOS_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID],
        [4, 4, 4, 5],
    ]
    assert batched_call_sizes == [3, 4, 3, 3]
    assert len(decode_batch_sizes) == 9


def test_s1_beam_decode_real_model_batch_matches_independent_samples_exactly() -> None:
    torch.manual_seed(73)
    model = _small_s1().eval()
    motion = torch.randn(3, model.config.max_frames, MOTION_FEATURE_DIM)
    frame_mask = torch.zeros(3, model.config.max_frames, dtype=torch.int32)
    frame_mask[0, :5] = 1
    frame_mask[1, :8] = 1
    frame_mask[2, :] = 1

    batched = model.beam_decode(
        motion,
        frame_mask,
        max_new_tokens=6,
        beam_width=2,
        no_repeat_ngram_size=3,
    )
    independent = torch.cat(
        [
            model.beam_decode(
                motion[index : index + 1],
                frame_mask[index : index + 1],
                max_new_tokens=6,
                beam_width=2,
                no_repeat_ngram_size=3,
            )
            for index in range(motion.shape[0])
        ],
        dim=0,
    )

    assert torch.equal(batched, independent)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"max_new_tokens": True}, "max_new_tokens"),
        ({"beam_width": 0}, "beam_width"),
        ({"beam_width": True}, "beam_width"),
        ({"no_repeat_ngram_size": 1}, "no_repeat_ngram_size"),
        ({"no_repeat_ngram_size": True}, "no_repeat_ngram_size"),
    ],
)
def test_s1_beam_decode_rejects_invalid_controls(arguments: dict[str, int], message: str) -> None:
    model = _small_s1().eval()
    motion, frame_mask = _motion_inputs(batch_size=1)

    with pytest.raises(ValueError, match=message):
        model.beam_decode(motion, frame_mask, **arguments)


@pytest.mark.parametrize("model_factory", [_small_s0, _small_s1])
def test_masked_motion_values_cannot_change_logits(
    model_factory: Callable[[], PortableS0 | PortableS1],
) -> None:
    torch.manual_seed(7)
    model = model_factory().eval()
    motion, frame_mask = _motion_inputs(batch_size=1)
    frame_mask[:, 7:] = 0
    changed = motion.clone()
    changed[:, 7:] = torch.randn_like(changed[:, 7:]) * 1_000.0

    with torch.no_grad():
        if isinstance(model, PortableS0):
            original_logits = model(motion, frame_mask)
            changed_logits = model(changed, frame_mask)
        else:
            decoder_input_ids = _decoder_inputs(batch_size=1)
            original_logits = model(motion, frame_mask, decoder_input_ids)
            changed_logits = model(changed, frame_mask, decoder_input_ids)

    torch.testing.assert_close(original_logits, changed_logits, atol=1e-6, rtol=1e-6)


def test_s1_decoder_is_causal() -> None:
    torch.manual_seed(11)
    model = _small_s1().eval()
    motion, frame_mask = _motion_inputs(batch_size=1)
    decoder_input_ids = _decoder_inputs(batch_size=1)
    changed = decoder_input_ids.clone()
    changed[:, 4:] = torch.randint(4, 32, changed[:, 4:].shape)

    with torch.no_grad():
        memory = model.encode(motion, frame_mask)
        original_logits = model.decode(decoder_input_ids, memory, frame_mask)
        changed_logits = model.decode(changed, memory, frame_mask)

    torch.testing.assert_close(
        original_logits[:, :4],
        changed_logits[:, :4],
        atol=1e-6,
        rtol=1e-6,
    )


def test_fixed_shape_and_dtype_validation_happens_in_eager_mode() -> None:
    s0 = _small_s0()
    motion, frame_mask = _motion_inputs(batch_size=1)

    with pytest.raises(ValueError, match="motion must have shape"):
        s0(motion[:, :-1], frame_mask[:, :-1])
    with pytest.raises(TypeError, match="frame_mask must use torch.int32"):
        s0(motion, frame_mask.to(torch.float32))

    s1 = _small_s1()
    memory = s1.encode(motion, frame_mask)
    with pytest.raises(ValueError, match="decoder_input_ids must have shape"):
        s1.decode(torch.ones(1, 7, dtype=torch.int64), memory, frame_mask)
    with pytest.raises(TypeError, match="decoder_input_ids must use torch.int64"):
        s1.decode(torch.ones(1, 8, dtype=torch.int32), memory, frame_mask)


def test_s0_exports_with_fixed_shapes() -> None:
    model = _small_s0().eval()
    motion, frame_mask = _motion_inputs(batch_size=1)

    exported = torch.export.export(model, (motion, frame_mask), strict=True)
    logits = exported.module()(motion, frame_mask)

    assert logits.shape == (1, 8, 32)


def test_s1_exports_with_fixed_shapes() -> None:
    model = _small_s1().eval()
    motion, frame_mask = _motion_inputs(batch_size=1)
    decoder_input_ids = _decoder_inputs(batch_size=1)

    exported = torch.export.export(
        model,
        (motion, frame_mask, decoder_input_ids),
        strict=True,
    )
    logits = exported.module()(motion, frame_mask, decoder_input_ids)

    assert logits.shape == (1, 8, 32)
