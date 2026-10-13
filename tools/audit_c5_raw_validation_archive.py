"""Inspect publisher ZIP metadata for unused How2Sign validation sources only."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import zipfile

from benchmark_fleurs_dev import sha256
from fetch_how2sign_pilot import archive_fingerprint
from how2sign_remote_zip import How2SignRemoteZip

ROOT = Path(__file__).resolve().parents[1]
FILE_ID = "1fCkyuKSsc7gauljuL9sx_jBomf3N6i0g"
ARCHIVE_SIZE = 17_528_664_599


def recording_sources(candidates: list[dict], by_name: dict) -> tuple[list[dict],list[str]]:
    source_rows={};missing=[]
    for row in candidates:
        expected=f"{row['video_id']}-{row['signer']}-rgb_front"
        if row['video_name']!=expected: raise ValueError("candidate recording and signer identity differ")
        member=f"raw_videos/{row['video_name']}.mp4"
        info=by_name.get(member)
        if info is None: missing.append(row['sentence_name']);continue
        if info.is_dir() or info.flag_bits&1 or info.compress_type not in {zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED}:
            raise ValueError("raw validation member has an unsupported encoding")
        source=source_rows.setdefault(row['video_name'],{
            'video_id':row['video_id'],'video_name':row['video_name'],'signer':row['signer'],
            'archive_member':member,'compressed_size':info.compress_size,'file_size':info.file_size,
            'crc32':info.CRC,'header_offset':info.header_offset,'compression_method':info.compress_type,'candidate_clips':[],
        })
        if source['signer']!=row['signer']: raise ValueError("recording group mixes signers")
        source['candidate_clips'].append(row['sentence_name'])
    return list(source_rows.values()),missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT/'runs'):
        parser.error("raw validation archive audit needs a new runs/ output")
    candidates_path = ROOT/'runs/c5-reassessment-v1/validation-capacity.json'
    candidates = json.loads(candidates_path.read_text())['records']
    with How2SignRemoteZip(file_id=FILE_ID,archive_name='val_raw_videos.zip',archive_size=ARCHIVE_SIZE) as remote:
        with zipfile.ZipFile(remote) as archive:
            entries = archive.infolist()
            by_name = {entry.filename:entry for entry in entries}
            sources,missing=recording_sources(candidates,by_name)
            transferred = remote.bytes_downloaded
            requests = remote.requests
    sizes = [row['compressed_size'] for row in sources]
    report = {
        'schema':'umi-studio-c5-raw-validation-archive-audit/1',
        'source_url':'https://how2sign.github.io/', 'file_id':FILE_ID,'archive_name':'val_raw_videos.zip',
        'archive_size_bytes':ARCHIVE_SIZE,'archive_fingerprint_sha256':archive_fingerprint(entries),
        'candidate_inventory_sha256':sha256(candidates_path),'audit_source_sha256':sha256(Path(__file__)),
        'available_source_videos':len(sources),'missing_candidate_clips':missing,
        'available_content_groups':len({row['video_id'] for row in sources}),
        'grouping':'Source bytes are keyed by recording VIDEO_NAME; split exclusion remains content VIDEO_ID based.',
        'available_candidate_clips':sum(len(row['candidate_clips']) for row in sources),
        'total_candidate_source_compressed_bytes':sum(sizes),
        'median_candidate_source_compressed_bytes':statistics.median(sizes) if sizes else None,
        'candidate_sources_under_128_mib_by_signer':dict(sorted(Counter(row['signer'] for row in sources if row['compressed_size']<=128*1024**2).items())),
        'sources':sources,'metadata_bytes_transferred':transferred,'metadata_requests':requests,
        'video_downloaded':False,'predictions_inspected':False,'confirmation_decoded':False,'training_allowed':False,
        'qualification':'Exact HTTP ranges verified archive metadata only. Source-video CRC, actual timestamps and crop quality must be checked during bounded acquisition.',
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({key:report[key] for key in ('available_source_videos','available_candidate_clips','total_candidate_source_compressed_bytes','median_candidate_source_compressed_bytes','candidate_sources_under_128_mib_by_signer','metadata_bytes_transferred')}))


if __name__=='__main__':
    main()
