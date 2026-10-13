import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from freeze_c5_openasl_scale_protocol import RECIPE, select_schedule


def test_schedule_has_frozen_source_mix():
    replay = [
        {
            "id": f"h-{signer}-{index}",
            "source": "how2sign_new",
            "signer": signer,
        }
        for signer in (1, 2, 3, 4, 5, 8, 9, 11)
        for index in range(40)
    ] + [
        {"id": f"f-{index}", "source": "fleurs", "signer": 3}
        for index in range(150)
    ]
    stem = [
        {
            "id": f"{source}-{index}",
            "source": source,
            "signer": f"p-{index % 20}",
            "participant": f"p-{index % 20}",
        }
        for source in ("asl_stem_original", "asl_stem_supplemental")
        for index in range(100)
    ]
    scale = [
        {
            "id": f"open-{source}-{clip}",
            "source": "openasl_scale",
            "signer": f"source-{source}",
            "source_group": f"source-{source}",
        }
        for source in range(120)
        for clip in range(2)
    ]
    schedule = select_schedule(replay, stem, scale)
    assert len(schedule) == RECIPE["train_examples"] == 768
    assert Counter(row["source"] for row in schedule) == {
        "openasl_scale": 240,
        "how2sign_new": 256,
        "fleurs": 136,
        "asl_stem_original": 68,
        "asl_stem_supplemental": 68,
    }
