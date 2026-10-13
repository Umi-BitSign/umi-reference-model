import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from audit_c5_validation_capacity import candidate_rows


def test_requires_recropping_shifted_unused_validation_and_rejects_training_overlap():
    original=[{"VIDEO_ID":"source", "VIDEO_NAME":"source-2-rgb_front", "SENTENCE_ID":"s", "SENTENCE_NAME":"source_0-2-rgb_front", "SENTENCE":"Heat up the hair", "START":"0", "END":"4"}]
    aligned=[{key:value for key,value in original[0].items() if key not in {"START","END"}} | {"START_REALIGNED":"5","END_REALIGNED":"9"}]
    rows,counts=candidate_rows(original,aligned,set(),set(),set())
    assert len(rows)==1 and rows[0]["start_sec"]==5
    assert counts["precut_alignment_eligible"]==0
    assert candidate_rows(original,aligned,{"source"},set(),set())[0]==[]
    assert candidate_rows(original,aligned,set(),set(),{("heat","up","the","hair")})[0]==[]
    with pytest.raises(ValueError,match="overlaps training"):
        candidate_rows(original,aligned,set(),{"source"},set())
