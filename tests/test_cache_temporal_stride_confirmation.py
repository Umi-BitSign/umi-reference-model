"""Confirmation stream caching binds references only after candidate freezing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from cache_temporal_stride_confirmation import confirmation_references


def test_confirmation_references_match_all_fixed_records():
    plan = {
        "records": [
            {"domain": "how2sign", "example_id": f"h-{i}", "signer": 3}
            for i in range(42)
        ]
        + [
            {"domain": "fleurs", "example_id": f"f-{i}", "signer": 4}
            for i in range(46)
        ]
    }
    h_selection = {
        "rows": [
            {"sentence_name": f"h-{i}", "signer": 3, "reference": f"H {i}"}
            for i in range(42)
        ]
    }
    f_selection = {
        "validation": [
            {"example_id": f"f-{i}", "signer": 4, "reference": f"F {i}"}
            for i in range(46)
        ]
    }
    result = confirmation_references(plan, h_selection, f_selection)
    assert len(result) == 88
    assert result["h-0"]["reference"] == "H 0"
    assert result["f-45"]["reference"] == "F 45"


def test_confirmation_references_reject_changed_signer():
    plan = {
        "records": [
            {"domain": "how2sign", "example_id": f"h-{i}", "signer": 3}
            for i in range(88)
        ]
    }
    selection = {
        "rows": [
            {"sentence_name": f"h-{i}", "signer": 3, "reference": "Reference"}
            for i in range(88)
        ]
    }
    selection["rows"][0]["signer"] = 8
    with pytest.raises(ValueError, match="metadata"):
        confirmation_references(plan, selection, {"validation": []})
