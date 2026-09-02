from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Final, cast

from .canonical import canonical_json_bytes
from .portable_model import BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID, UNK_TOKEN_ID

TOKENIZER_SCHEMA: Final = "umi-unigram-tokenizer/1"
S1_TOKENIZER_VOCABULARY_SIZE: Final = 4_096
S1_TOKENIZER_MAXIMUM_OUTPUT_TOKENS: Final = 128
S1_TOKENIZER_MAXIMUM_RECORD_BYTES: Final = 2 * 1024 * 1024
S1_TOKENIZER_MAXIMUM_MODEL_BYTES: Final = 1024 * 1024
S1_MAXIMUM_HYPOTHESIS_UTF8_BYTES: Final = 4 * 1024

_ROOT_FIELDS: Final = {
    "schema",
    "configuration",
    "corpus_sha256",
    "model_sha256",
    "normalization_revision",
    "pieces",
    "reserved_ids",
    "sentence_count",
    "training_partition_sha256",
}
_CONFIGURATION_FIELDS: Final = {"vocabulary_size", "max_output_tokens", "byte_fallback"}
_RESERVED_FIELDS: Final = {"bos", "eos", "pad", "unk"}
_PIECE_FIELDS: Final = {"byte", "control", "id", "piece", "score", "unknown", "unused"}
_LOWER_SHA256: Final = frozenset("0123456789abcdef")
_WHITESPACE_MARKER: Final = "▁"
_UNKNOWN_SURFACE: Final = " ⁇ "


class S1DecodeTokenizerError(ValueError):
    """Raised when a decode-only tokenizer artifact violates the S1 contract."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and set(value) <= _LOWER_SHA256
    )


def _strict_json_object(payload: bytes) -> dict[str, Any]:
    if not 1 <= len(payload) <= S1_TOKENIZER_MAXIMUM_RECORD_BYTES:
        raise S1DecodeTokenizerError("tokenizer record violates its byte ceiling")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise S1DecodeTokenizerError("tokenizer record contains a duplicate member")
            value[key] = item
        return value

    try:
        value = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                S1DecodeTokenizerError(f"tokenizer record contains {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise S1DecodeTokenizerError("tokenizer record is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise S1DecodeTokenizerError("tokenizer record is not canonical JSON")
    return value


def _exact_dict(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise S1DecodeTokenizerError(f"tokenizer {label} field set is invalid")
    return value


def _strict_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise S1DecodeTokenizerError(f"tokenizer {label} is invalid")
    return value


@dataclass(frozen=True, slots=True)
class S1DecodePiece:
    identifier: int
    text: str
    is_byte: bool
    is_control: bool
    is_unknown: bool


@dataclass(frozen=True, slots=True)
class S1DecodeTokenizer:
    """Strict SentencePiece-compatible decoder with no training or encoding surface."""

    pieces: tuple[S1DecodePiece, ...]
    record_sha256: str
    model_sha256: str
    normalization_revision: str

    @classmethod
    def from_bytes(
        cls,
        *,
        record_bytes: bytes,
        model_bytes: bytes,
        expected_record_sha256: str,
        expected_model_sha256: str,
    ) -> S1DecodeTokenizer:
        if not _is_sha256(expected_record_sha256) or not _is_sha256(expected_model_sha256):
            raise S1DecodeTokenizerError("expected tokenizer digests are invalid")
        if not 1 <= len(model_bytes) <= S1_TOKENIZER_MAXIMUM_MODEL_BYTES:
            raise S1DecodeTokenizerError("tokenizer model violates its byte ceiling")
        if _sha256(record_bytes) != expected_record_sha256:
            raise S1DecodeTokenizerError("tokenizer record digest differs")
        if _sha256(model_bytes) != expected_model_sha256:
            raise S1DecodeTokenizerError("tokenizer model digest differs")

        record = _strict_json_object(record_bytes)
        _exact_dict(record, _ROOT_FIELDS, "record")
        if record["schema"] != TOKENIZER_SCHEMA:
            raise S1DecodeTokenizerError("tokenizer schema is unsupported")
        configuration = _exact_dict(record["configuration"], _CONFIGURATION_FIELDS, "configuration")
        if configuration != {
            "vocabulary_size": S1_TOKENIZER_VOCABULARY_SIZE,
            "max_output_tokens": S1_TOKENIZER_MAXIMUM_OUTPUT_TOKENS,
            "byte_fallback": True,
        }:
            raise S1DecodeTokenizerError("tokenizer configuration is not frozen S1")
        reserved = _exact_dict(record["reserved_ids"], _RESERVED_FIELDS, "reserved IDs")
        if reserved != {
            "bos": BOS_TOKEN_ID,
            "eos": EOS_TOKEN_ID,
            "pad": PAD_TOKEN_ID,
            "unk": UNK_TOKEN_ID,
        }:
            raise S1DecodeTokenizerError("tokenizer reserved IDs differ from S1")
        for label in ("corpus_sha256", "model_sha256", "training_partition_sha256"):
            if not _is_sha256(record[label]):
                raise S1DecodeTokenizerError(f"tokenizer {label} is not SHA-256")
        if record["model_sha256"] != expected_model_sha256:
            raise S1DecodeTokenizerError("tokenizer record names another model")
        normalization_revision = record["normalization_revision"]
        if (
            not isinstance(normalization_revision, str)
            or not normalization_revision
            or len(normalization_revision.encode("utf-8")) > 1024
        ):
            raise S1DecodeTokenizerError("tokenizer normalization revision is invalid")
        _strict_int(record["sentence_count"], "sentence count", minimum=1)
        raw_pieces = record["pieces"]
        if not isinstance(raw_pieces, list) or len(raw_pieces) != S1_TOKENIZER_VOCABULARY_SIZE:
            raise S1DecodeTokenizerError("tokenizer piece count differs from S1")

        pieces: list[S1DecodePiece] = []
        observed_text: set[str] = set()
        for index, raw_piece in enumerate(raw_pieces):
            piece = _exact_dict(raw_piece, _PIECE_FIELDS, f"piece {index}")
            identifier = _strict_int(piece["id"], f"piece {index} ID")
            text = piece["piece"]
            score = piece["score"]
            flags = tuple(piece[name] for name in ("byte", "control", "unknown", "unused"))
            if (
                identifier != index
                or not isinstance(text, str)
                or not text
                or text in observed_text
                or type(score) not in (int, float)
                or isinstance(score, bool)
                or not math.isfinite(float(score))
                or any(type(flag) is not bool for flag in flags)
            ):
                raise S1DecodeTokenizerError(f"tokenizer piece {index} is invalid")
            is_byte, is_control, is_unknown, is_unused = cast(tuple[bool, bool, bool, bool], flags)
            if index == PAD_TOKEN_ID:
                valid = (text, is_byte, is_control, is_unknown, is_unused) == (
                    "<pad>",
                    False,
                    True,
                    False,
                    False,
                )
            elif index == BOS_TOKEN_ID:
                valid = (text, is_byte, is_control, is_unknown, is_unused) == (
                    "<bos>",
                    False,
                    True,
                    False,
                    False,
                )
            elif index == EOS_TOKEN_ID:
                valid = (text, is_byte, is_control, is_unknown, is_unused) == (
                    "<eos>",
                    False,
                    True,
                    False,
                    False,
                )
            elif index == UNK_TOKEN_ID:
                valid = (text, is_byte, is_control, is_unknown, is_unused) == (
                    "<unk>",
                    False,
                    False,
                    True,
                    False,
                )
            elif 4 <= index < 260:
                valid = (text, is_byte, is_control, is_unknown, is_unused) == (
                    f"<0x{index - 4:02X}>",
                    True,
                    False,
                    False,
                    False,
                )
            else:
                valid = not (is_byte or is_control or is_unknown or is_unused)
            if not valid:
                raise S1DecodeTokenizerError(f"tokenizer piece {index} violates S1")
            observed_text.add(text)
            pieces.append(
                S1DecodePiece(
                    identifier=index,
                    text=text,
                    is_byte=is_byte,
                    is_control=is_control,
                    is_unknown=is_unknown,
                )
            )
        return cls(
            pieces=tuple(pieces),
            record_sha256=expected_record_sha256,
            model_sha256=expected_model_sha256,
            normalization_revision=normalization_revision,
        )

    @staticmethod
    def _decode_bytes(values: bytes) -> str:
        output: list[str] = []
        index = 0
        while index < len(values):
            first = values[index]
            width = 0
            scalar = 0
            if first <= 0x7F:
                width, scalar = 1, first
            elif 0xC2 <= first <= 0xDF and index + 1 < len(values):
                second = values[index + 1]
                if 0x80 <= second <= 0xBF:
                    width, scalar = 2, ((first & 0x1F) << 6) | (second & 0x3F)
            elif 0xE0 <= first <= 0xEF and index + 2 < len(values):
                second, third = values[index + 1 : index + 3]
                valid_second = (
                    0x80 <= second <= 0xBF
                    and (first != 0xE0 or second >= 0xA0)
                    and (first != 0xED or second <= 0x9F)
                )
                if valid_second and 0x80 <= third <= 0xBF:
                    width = 3
                    scalar = ((first & 0x0F) << 12) | ((second & 0x3F) << 6) | (third & 0x3F)
            elif 0xF0 <= first <= 0xF4 and index + 3 < len(values):
                second, third, fourth = values[index + 1 : index + 4]
                valid_second = (
                    0x80 <= second <= 0xBF
                    and (first != 0xF0 or second >= 0x90)
                    and (first != 0xF4 or second <= 0x8F)
                )
                if valid_second and 0x80 <= third <= 0xBF and 0x80 <= fourth <= 0xBF:
                    width = 4
                    scalar = (
                        ((first & 0x07) << 18)
                        | ((second & 0x3F) << 12)
                        | ((third & 0x3F) << 6)
                        | (fourth & 0x3F)
                    )
            if width:
                output.append(chr(scalar))
                index += width
            else:
                output.append("�")
                index += 1
        return "".join(output)

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        if len(token_ids) > S1_TOKENIZER_MAXIMUM_OUTPUT_TOKENS:
            raise S1DecodeTokenizerError("token sequence exceeds the S1 output ceiling")
        output: list[str] = []
        pending_bytes = bytearray()
        dummy_prefix_eligible = True

        def flush_bytes() -> None:
            nonlocal dummy_prefix_eligible
            if pending_bytes:
                output.append(self._decode_bytes(bytes(pending_bytes)))
                pending_bytes.clear()
                dummy_prefix_eligible = False

        for raw_identifier in token_ids:
            if type(raw_identifier) is not int:
                raise S1DecodeTokenizerError("token ID must be an integer")
            identifier = raw_identifier if 0 <= raw_identifier < len(self.pieces) else UNK_TOKEN_ID
            if identifier == EOS_TOKEN_ID:
                break
            piece = self.pieces[identifier]
            if identifier in (PAD_TOKEN_ID, BOS_TOKEN_ID) or piece.is_control:
                continue
            if piece.is_byte:
                pending_bytes.append(identifier - 4)
                continue
            flush_bytes()
            if identifier == UNK_TOKEN_ID or piece.is_unknown:
                output.append(_UNKNOWN_SURFACE)
                dummy_prefix_eligible = False
                continue
            text = piece.text.replace(_WHITESPACE_MARKER, " ")
            if dummy_prefix_eligible and piece.text.startswith(_WHITESPACE_MARKER):
                text = text[1:]
            output.append(text)
            dummy_prefix_eligible = False
        flush_bytes()
        result = "".join(output)
        if len(result.encode("utf-8")) > S1_MAXIMUM_HYPOTHESIS_UTF8_BYTES:
            raise S1DecodeTokenizerError("decoded text exceeds the S1 hypothesis ceiling")
        return result
