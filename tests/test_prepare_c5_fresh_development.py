import io
import sys
from pathlib import Path
import zipfile

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_c5_fresh_development import iter_member,choose_sources


@pytest.mark.parametrize('method',[zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED])
def test_bounded_zip_member_stream_preserves_bytes_and_checks_crc(method):
    archive=io.BytesIO();payload=b'public validation video bytes'*100
    with zipfile.ZipFile(archive,'w',compression=method) as source:
        source.writestr('raw_videos/example.mp4',payload)
    with zipfile.ZipFile(archive) as source:
        row=source.infolist()[0]
    entry={'header_offset':row.header_offset,'archive_member':row.filename,'compressed_size':row.compress_size,
           'compression_method':row.compress_type,'file_size':row.file_size,'crc32':row.CRC}
    assert b''.join(iter_member(archive,entry,chunk_bytes=17))==payload
    with pytest.raises(ValueError,match='CRC'):
        b''.join(iter_member(archive,{**entry,'crc32':0},chunk_bytes=17))


def test_source_selection_rejects_annotations_for_a_different_recording():
    sources=[];clips=[]
    for signer,target in ((8,1),(1,4),(2,4),(3,4),(5,4)):
        for position in range(target):
            video=f'video{signer}{position}'
            name=f'{video}-{signer}-rgb_front'
            count=7 if signer==8 else 4
            identifiers=[]
            for index in range(count):
                key=f'{video}_{index}-{signer}-rgb_front';identifiers.append(key)
                clips.append({'sentence_name':key,'video_name':name,'video_id':video,'signer':signer})
            sources.append({'video_id':video,'video_name':name,'signer':signer,'compressed_size':100,'candidate_clips':identifiers})
    selected,chosen=choose_sources(sources,clips)
    assert len(selected)==17 and len(chosen)==71
    clips[-1]['signer']=3
    with pytest.raises(ValueError,match='mixes recording'):
        choose_sources(sources,clips)
