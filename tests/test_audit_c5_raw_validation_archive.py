import sys
from pathlib import Path
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_c5_raw_validation_archive import recording_sources


def test_same_content_different_signers_remain_separate_recordings():
    candidates=[];members={}
    for signer in (3,5):
        recording=f'_5CV2fIG7qY-{signer}-rgb_front'
        candidates.append({'video_id':'_5CV2fIG7qY','video_name':recording,'signer':signer,'sentence_name':f'_5CV2fIG7qY_7-{signer}-rgb_front'})
        info=zipfile.ZipInfo(f'raw_videos/{recording}.mp4');info.compress_size=100;info.file_size=100;info.CRC=1;info.header_offset=0
        members[info.filename]=info
    rows,missing=recording_sources(candidates,members)
    assert not missing and len(rows)==2
    assert all(len(row['candidate_clips'])==1 for row in rows)
    assert {row['video_name'] for row in rows}=={r['video_name'] for r in candidates}
