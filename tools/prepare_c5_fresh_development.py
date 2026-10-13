"""Freeze and acquire a bounded, evaluation-only raw How2Sign development set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import zipfile
import zlib

from benchmark_fleurs_dev import sha256
from c5_experiment_sources import archive_sources
from fetch_how2sign_pilot import archive_fingerprint
from how2sign_remote_zip import How2SignRemoteZip, MAX_RANGE_BYTES

ROOT = Path(__file__).resolve().parents[1]
RECIPE = {"id":"c5-fresh-how2sign-development-v1","sources_per_signer":{1:4,2:4,3:4,5:4,8:1},
          "clips_per_source":4,"rare_signer8_clips":7,"max_source_compressed_bytes":128*1024**2,
          "minimum_free_bytes":80*1024**3,"data_budget_bytes":4*1024**3,"frame_rate":25,
          "training_allowed":False,"source_selection":"sha256_order_among_bounded_sources_with_required_clips"}


def choose_sources(sources: list[dict], candidates: list[dict]) -> tuple[list[dict],list[dict]]:
    by_id={row['sentence_name']:row for row in candidates}
    chosen=[];clips=[];content_used=set()
    for signer in (8,1,2,3,5):
        target=RECIPE['sources_per_signer'][signer]
        count=RECIPE['rare_signer8_clips'] if signer==8 else RECIPE['clips_per_source']
        eligible=[row for row in sources if row['signer']==signer and row['compressed_size']<=RECIPE['max_source_compressed_bytes'] and len(row['candidate_clips'])>=count]
        eligible.sort(key=lambda row:hashlib.sha256(f"fresh-development-v1\0{row['video_id']}".encode()).hexdigest())
        if len(eligible)<target: raise ValueError("fresh development lacks a required bounded signer group")
        selected=[]
        for row in eligible:
            if row['video_id'] in content_used: continue
            if len(selected)==target: break
            selected.append(row);content_used.add(row['video_id'])
        if len(selected)!=target: raise ValueError("fresh cohort lacks distinct content groups across signers")
        for row in selected:
            names=sorted(row['candidate_clips'],key=lambda name:hashlib.sha256(f"fresh-development-clip-v1\0{name}".encode()).hexdigest())[:count]
            if any(by_id[name]['video_name']!=row['video_name'] or by_id[name]['signer']!=row['signer'] for name in names):
                raise ValueError("fresh source mixes recording or signer annotations")
            chosen.append({**row,'candidate_clips':names})
            clips.extend(by_id[name] for name in names)
    if len({row['sentence_name'] for row in clips})!=len(clips) or len({row['video_id'] for row in chosen})!=len(chosen):
        raise ValueError("fresh development repeats a clip or source")
    return chosen,clips


def iter_member(remote, info: dict, chunk_bytes: int = MAX_RANGE_BYTES):
    """Stream one CRC-checked ZIP member without reading the entire archive."""
    remote.seek(info['header_offset'])
    header=remote.read(30)
    if len(header)!=30: raise ValueError("ZIP local header is truncated")
    fields=struct.unpack('<4s5H3I2H',header)
    if fields[0]!=b'PK\x03\x04' or fields[2]&1 or fields[3]!=info['compression_method']:
        raise ValueError("ZIP local header disagrees with frozen directory")
    filename=remote.read(fields[9]).decode('utf-8' if fields[2]&0x800 else 'cp437')
    if filename!=info['archive_member']: raise ValueError("ZIP local member name differs")
    if fields[10]: remote.read(fields[10])
    remaining=info['compressed_size'];size=0;crc=0
    decoder=zlib.decompressobj(-15) if fields[3]==zipfile.ZIP_DEFLATED else None
    if decoder is None and fields[3]!=zipfile.ZIP_STORED: raise ValueError("ZIP compression unsupported")
    while remaining:
        block=remote.read(min(chunk_bytes,remaining))
        if not block: raise ValueError("ZIP member compressed bytes are truncated")
        remaining-=len(block)
        data=decoder.decompress(block,info['file_size']-size+1) if decoder else block
        size+=len(data)
        if size>info['file_size']: raise ValueError("ZIP member exceeds frozen decoded size")
        crc=zlib.crc32(data,crc)
        yield data
    if decoder:
        extra=decoder.flush();size+=len(extra);crc=zlib.crc32(extra,crc)
        if extra: yield extra
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail: raise ValueError("ZIP deflate stream is incomplete")
    if size!=info['file_size'] or crc!=info['crc32']: raise ValueError("ZIP member size or CRC differs")


def freeze(output: Path) -> None:
    output=output.resolve()
    if output.exists() or not output.is_relative_to(ROOT/'runs'): raise ValueError("fresh development needs a new runs/ directory")
    inventory_path=ROOT/'runs/c5-reassessment-v1/validation-capacity.json'
    archive_path=ROOT/'runs/c5-reassessment-v1/raw-validation-archive-v2.json'
    inventory=json.loads(inventory_path.read_text());archive=json.loads(archive_path.read_text())
    if archive['candidate_inventory_sha256']!=sha256(inventory_path): raise ValueError("fresh archive candidate inventory changed")
    sources,clips=choose_sources(archive['sources'],inventory['records'])
    reuse={}
    previous=ROOT/'runs/c5-fresh-development-v1/source-receipts.jsonl'
    if previous.is_file():
        by_name={row['video_name']:row for row in sources}
        for line in previous.read_text().splitlines():
            record=json.loads(line);name=record['video_name']
            if name not in by_name: continue
            path=ROOT/'data/c5-fresh-development-v1/sources'/f'{name}.mp4'
            row=by_name[name]
            if record['archive_crc32']!=row['crc32'] or record['file_size']!=row['file_size'] or sha256(path)!=record['sha256']:
                raise ValueError("prior verified source bytes differ from corrected recording inventory")
            reuse[name]={'path':str(path.relative_to(ROOT)),'sha256':record['sha256'],'probe':record['probe']}
    plan={"schema":"umi-studio-c5-fresh-development-plan/1","recipe":RECIPE,"sources":sources,"clips":clips,
          "archive":{key:archive[key] for key in ('file_id','archive_name','archive_size_bytes','archive_fingerprint_sha256','source_url')},
          "inventory_sha256":sha256(inventory_path),"archive_audit_sha256":sha256(archive_path),
          "preparation_source_sha256":sha256(Path(__file__)),"license_sha256":sha256(ROOT/'data/how2sign/source/CC-BY-NC-4.0.txt'),
          "data_dir":str((ROOT/'data'/output.name).relative_to(ROOT)),"planned_raw_source_bytes":sum(row['file_size'] for row in sources),
          "reuse_verified_sources":reuse,
          "purpose":"Fresh open development evidence for fixed-weight face recovery, never training or final confirmation.",
          "prediction_inspected":False,"confirmation_decoded":False,"miner_changed":False,
          "qualification":"Source-disjoint from previous local validation and official training. The baseline's complete pretraining inventory is unknown. Signer 8 has only one remaining eligible source and seven clips."}
    output.mkdir(parents=True)
    path=output/'plan.json';path.write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    archive_sources(path,output/'source-snapshot')
    print(json.dumps({'plan_sha256':sha256(path),'sources':len(sources),'clips':len(clips),'raw_source_bytes':plan['planned_raw_source_bytes'],'reuse_sources':len(reuse)}))


def fetch(plan_path: Path) -> None:
    plan=json.loads(plan_path.read_text())
    # JSON object keys serialize integer signer labels as strings.
    normalized=json.loads(json.dumps(RECIPE))
    if plan['recipe']!=normalized or plan['preparation_source_sha256']!=sha256(Path(__file__)):
        raise ValueError("fresh development acquisition recipe changed")
    data=ROOT/plan['data_dir']
    if data.exists() or not data.resolve().is_relative_to(ROOT/'data'): raise ValueError("fresh data output already exists")
    sources_dir=data/'sources';sources_dir.mkdir(parents=True)
    arc=plan['archive']
    with How2SignRemoteZip(file_id=arc['file_id'],archive_name=arc['archive_name'],archive_size=arc['archive_size_bytes']) as remote:
        with zipfile.ZipFile(remote) as archive:
            if archive_fingerprint(archive.infolist())!=arc['archive_fingerprint_sha256']:
                raise ValueError("fresh raw validation archive directory changed")
        for row in plan['sources']:
            current=sum(path.stat().st_size for path in data.rglob('*') if path.is_file())
            if current+row['file_size']+128*1024**2>plan['recipe']['data_budget_bytes'] or shutil.disk_usage(ROOT).free<plan['recipe']['minimum_free_bytes']+row['file_size']+128*1024**2:
                raise RuntimeError("fresh development storage budget exhausted")
            name=row['video_name']
            if Path(name).name!=name: raise ValueError("fresh source name contains path separators")
            target=sources_dir/f'{name}.mp4';temporary=target.with_suffix('.partial.mp4')
            digest=hashlib.sha256()
            reuse=plan.get('reuse_verified_sources',{}).get(name)
            if reuse:
                prior=(ROOT/reuse['path']).resolve()
                if not prior.is_relative_to(ROOT/'data') or sha256(prior)!=reuse['sha256']:
                    raise ValueError("source bytes changed before reuse")
                os.link(prior,target)
                payload=reuse['probe'];source_hash=reuse['sha256']
            else:
                try:
                    with temporary.open('xb') as output:
                        for chunk in iter_member(remote,row): output.write(chunk);digest.update(chunk)
                        output.flush();os.fsync(output.fileno())
                    probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,duration:format=duration','-of','json',str(temporary)],capture_output=True,text=True,check=True)
                    payload=json.loads(probe.stdout)
                    if len(payload.get('streams',[]))!=1: raise ValueError("raw validation source lacks one video stream")
                    temporary.rename(target)
                finally:
                    temporary.unlink(missing_ok=True)
                source_hash=digest.hexdigest()
            duration=float(payload['streams'][0].get('duration',payload['format']['duration']))
            selected=[clip for clip in plan['clips'] if clip['video_name']==name]
            if any(clip['signer']!=row['signer'] for clip in selected): raise ValueError("source recording mixes signer labels")
            invalid=[clip['sentence_name'] for clip in selected if clip['end_sec']>duration+.05]
            record={"schema":"umi-studio-c5-fresh-development-source/1","plan_sha256":sha256(plan_path),
                    "video_id":row['video_id'],"video_name":name,"signer":row['signer'],'sha256':source_hash,
                    "file_size":row['file_size'],"archive_crc32":row['crc32'],"archive_member":row['archive_member'],
                    "probe":payload,"license_sha256":plan['license_sha256'],"training_allowed":False,"predictions_inspected":False,
                    "reused_source":reuse is not None,"invalid_selected_intervals":invalid}
            with (plan_path.parent/'source-receipts.jsonl').open('a') as output:
                output.write(json.dumps(record,sort_keys=True)+'\n');output.flush();os.fsync(output.fileno())
            print(f"downloaded signer={row['signer']} {name} bytes={row['file_size']}",flush=True)
            if duration<=0 or invalid: raise ValueError("realigned interval exceeds raw video; source/probe were preserved for diagnosis")
    print(json.dumps({'verified_sources':len(plan['sources']),'planned_clips':len(plan['clips']),'prediction_inspected':False}))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',type=Path);group.add_argument('--plan',type=Path)
    args=parser.parse_args()
    freeze(args.freeze) if args.freeze else fetch(args.plan)


if __name__=='__main__': main()
