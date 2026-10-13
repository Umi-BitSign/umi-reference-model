"""Development-data split and single-reference diagnostic invariants."""

import pytest

from tools.benchmark_fleurs_dev import distance, normalize
from tools.fetch_fleurs_dev import _selected_files
from tools.prepare_fleurs_dev import _check_cross_split, _selected_row
from tools.select_fleurs_pilot import select


def _row(signer: int, document: int, sentence: int) -> dict:
    return {
        "zs_split": "val",
        "sd_split": "train",
        "signer": signer,
        "start_time_sec": 1.0,
        "end_time_sec": 5.0,
        "video_name": f"dev_document_{document}_signer_{signer}.mp4",
        "example_id": f"dev_document_{document}_sentence_{sentence}_signer_{signer}",
        "document_id": f"dev_document_{document}",
        "sentence_id": f"dev_sentence_{sentence}",
        "eng": f"Sentence {sentence}.",
    }


def test_selected_development_rows_exclude_test_and_devtest() -> None:
    row = _row(3, 10, 20)
    selected = _selected_row(row)
    assert selected is not None
    assert selected["source_video"] == "fleurs-asl-v1/videos/dev_document_10_signer_3.mp4"
    assert _selected_row({**row, "zs_split": "test"}) is None
    assert _selected_row({**row, "sd_split": "test"}) is None
    assert _selected_row({**row, "signer": 2}) is None


def test_cross_split_rejects_shared_document_or_reference() -> None:
    train = _selected_row(_row(3, 10, 20))
    val = _selected_row(_row(4, 10, 21))
    assert train is not None and val is not None
    with pytest.raises(ValueError, match="document_id"):
        _check_cross_split([train], [val])
    independent = _selected_row(_row(4, 11, 21))
    assert independent is not None
    _check_cross_split([train], [independent])
    with pytest.raises(ValueError, match="reference"):
        _check_cross_split([train], [{**independent, "reference": train["reference"]}])


def test_fetch_only_allows_expected_dev_video_paths() -> None:
    selection = {
        "schema": "umi-studio-fleurs-dev-split/1",
        "training": [{"source_video": "fleurs-asl-v1/videos/dev_document_10_signer_3.mp4"}],
        "validation": [{"source_video": "fleurs-asl-v1/videos/dev_document_11_signer_4.mp4"}],
    }
    assert len(_selected_files(selection, "both")) == 2
    selection["validation"][0]["source_video"] = "fleurs-asl-v1/videos/dev_document_../x"
    with pytest.raises(ValueError, match="non-development"):
        _selected_files(selection, "validation")


def test_diagnostic_normalization_and_edit_distance() -> None:
    assert normalize("  It\u2019s, ready!  ") == ("it\u2019s", "ready")
    assert distance(("we", "agree"), ("we", "do", "not", "agree")) == 2


def test_pilot_spreads_new_examples_across_unused_documents() -> None:
    training = [_selected_row(_row(3, document, 100 + document)) for document in range(1, 5)]
    assert all(row is not None for row in training)
    selection = {
        "schema": "umi-studio-fleurs-dev-split/1",
        "training": training,
        "validation": [_selected_row(_row(4, 10, 200))],
    }
    existing = [{"example_id": training[0]["example_id"], "split": "training", "signer": 3}]
    chosen = select(selection, existing, target=3)
    assert len(chosen["requested_example_ids"]) == 2
    assert chosen["source_documents"] == 3
    assert len(chosen["all_example_ids"]) == len(set(chosen["all_example_ids"]))
