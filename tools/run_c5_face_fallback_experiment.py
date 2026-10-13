"""Freeze/run a paired face-recovery diagnostic, with no training or confirmation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
from benchmark_fleurs_dev import distance, normalize, sha256
from c5_face_fallback import RECIPE, ROOT, overlay_faces
from c5_experiment_sources import archive_sources
from prepare_asl_stem_wiki_landmarks import DOCKER_SOCKET, IMAGE, docker_image_id
from run_c5_stem_tuning import LAYOUT

BASELINE = ROOT / "artifacts/baseline-6fe8df59"
PINNED_IMAGE = "sha256:5cad3554c60a6b0bf7d0b059dde0fed09c5a357cd0142c6dfa0b6fa9cfe63378"


def freeze(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists() or not output_dir.is_relative_to(ROOT / "runs"):
        raise ValueError("face-fallback plan requires a new runs/ directory")
    control_path = ROOT / "runs/c5-all-source-retention-v1/plan.json"
    control = json.loads(control_path.read_text())
    rows = []
    index_hashes = {}
    for domain,layout in LAYOUT.items():
        index_hashes[domain] = sha256(layout["index"])
        expected = {row["example_id"]:row for row in read_jsonl(ROOT / control["open_tuning"][domain]["results"])}
        if sha256(ROOT / control["open_tuning"][domain]["results"]) != control["open_tuning"][domain]["results_sha256"]:
            raise ValueError("face-fallback baseline outputs changed")
        for row in read_jsonl(layout["index"]):
            if row["split"] != layout["split"]:
                continue
            key = row.get("sentence_name",row.get("example_id"))
            clip_dir = ROOT / ("data/how2sign/validation-stream-clips-25fps" if domain=="how2sign" else "data/fleurs-asl/tuning-stream-clips-25fps")
            clip = clip_dir / f"{key}.mp4"
            cache = layout["cache"] / f"{key}.npz"
            expected_clip = row.get("resampled_clip_sha256",row.get("clip_sha256"))
            if sha256(clip)!=expected_clip or sha256(cache)!=row["encoder_cache_sha256"] or key not in expected:
                raise ValueError("face-fallback source clip/cache differs")
            if row["reference"] != expected[key]["reference"]:
                raise ValueError("face-fallback source reference differs")
            rows.append({"domain":domain,"example_id":key,"signer":row["signer"],"reference":row["reference"],
                         "clip":str(clip.relative_to(ROOT)),"clip_sha256":expected_clip,
                         "cache":str(cache.relative_to(ROOT)),"cache_sha256":row["encoder_cache_sha256"],
                         "source_frames":row["encoder_shape"][0],"expected_control_hypothesis":expected[key]["hypothesis"],
                         "expected_control_wer":expected[key]["single_reference_wer"]})
        if sum(r['domain']==domain for r in rows)!=layout["examples"]:
            raise ValueError("face-fallback open cohort count differs")
    image = docker_image_id(DOCKER_SOCKET)
    if image!=PINNED_IMAGE:
        raise ValueError("face-fallback MediaPipe image changed")
    landmark_asset = BASELINE / "models/face_landmarker_v2_with_blendshapes.task"
    dino_asset = BASELINE / "models/dinov2face.safetensors"
    plan = {
        "schema":"umi-studio-c5-face-fallback-plan/1","recipe":RECIPE,"examples":rows,
        "starting_adapter":control["starting_adapter"],"starting_adapter_sha256":control["starting_adapter_sha256"],
        "baseline_checkpoint_sha256":control["baseline_checkpoint_sha256"],
        "open_tuning":control["open_tuning"],"index_sha256":index_hashes,"screen":control["screen"],
        "face_landmarker":str(landmark_asset.relative_to(ROOT)),"face_landmarker_sha256":sha256(landmark_asset),
        "face_dino_sha256":sha256(dino_asset),"mediapipe_image_id":image,
        "recovery_source_sha256":sha256(ROOT / "tools/c5_face_fallback.py"),
        "head_roi_source_sha256":sha256(ROOT / "tools/probe_c5_face_detection.py"),
        "experiment_source_sha256":sha256(Path(__file__)),"adapter_implementation_sha256":sha256(ROOT / "tools/shubert_lora.py"),
        "face_extractor_source_sha256":sha256(BASELINE / "vendor/demo/crop_face.py"),
        "data_dir":str((ROOT / "data" / output_dir.name).relative_to(ROOT)),
        "data_budget_bytes":1024**3,"minimum_free_bytes":80*1024**3,
        "confirmation_policy":"Do not decode confirmation, reserve, devtest, test or private cases. This is only the existing open tuning cohort.",
        "qualified_for_promotion":False,"miner_health_policy":"user_authorized_maintenance",
    }
    output_dir.mkdir(parents=True)
    path = output_dir / "plan.json"
    path.write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    archive_sources(path,output_dir / "source-snapshot")
    print(json.dumps({"plan_sha256":sha256(path),"examples":len(rows),"data_dir":plan["data_dir"]}))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def summarize(records: list[dict], screen: dict) -> dict:
    domains = {}
    for domain in ("how2sign","fleurs"):
        selected = [r for r in records if r["domain"]==domain]
        signers = defaultdict(list)
        for row in selected:
            signers[str(row["signer"])].append(row)
        domains[domain] = {
            "examples":len(selected),"control_mean_wer":statistics.fmean(r["control_wer"] for r in selected),
            "candidate_mean_wer":statistics.fmean(r["candidate_wer"] for r in selected),
            "signer_means":{key:{"control":statistics.fmean(r['control_wer'] for r in rows),"candidate":statistics.fmean(r['candidate_wer'] for r in rows)} for key,rows in sorted(signers.items())},
            "improved":sum(r['candidate_wer']<r['control_wer'] for r in selected),
            "regressed":sum(r['candidate_wer']>r['control_wer'] for r in selected),
        }
    h = domains['how2sign']['candidate_mean_wer']-domains['how2sign']['control_mean_wer']
    f = domains['fleurs']['candidate_mean_wer']-domains['fleurs']['control_mean_wer']
    improved = sum(row['candidate']<row['control'] for row in domains['how2sign']['signer_means'].values())
    passes = h<=screen['how2sign_max_mean_wer_change'] and f<=screen['fleurs_max_mean_wer_change'] and (h+f)/2<=screen['equal_domain_max_mean_wer_change'] and improved>=screen['minimum_improved_how2sign_signers']
    return {"domains":domains,"how2sign_signers_improved":improved,"equal_domain_mean_wer_change":(h+f)/2,"passes_open_screen":passes,
            "decision":"collect_larger_development_evidence" if passes else "retain_confirmed_candidate"}


def run(plan_path: Path) -> None:
    plan = json.loads(plan_path.read_text())
    if plan.get("recipe")!=RECIPE or plan["experiment_source_sha256"]!=sha256(Path(__file__)) or plan["adapter_implementation_sha256"]!=sha256(ROOT / "tools/shubert_lora.py"):
        raise ValueError("face-fallback experiment implementation changed")
    if plan["head_roi_source_sha256"]!=sha256(ROOT / "tools/probe_c5_face_detection.py") or plan["face_extractor_source_sha256"]!=sha256(BASELINE / "vendor/demo/crop_face.py"):
        raise ValueError("face-fallback head/crop implementation changed")
    data_dir = ROOT / plan['data_dir']
    if data_dir.exists() or not data_dir.resolve().is_relative_to(ROOT / "data") or (plan_path.parent / 'decision.json').exists():
        raise ValueError("face-fallback output already exists")
    if shutil.disk_usage(ROOT).free < plan['minimum_free_bytes']+128*1024**2:
        raise RuntimeError("face-fallback free-space reserve is unavailable")
    landmarks_dir = data_dir / "landmarks"
    landmarks_dir.mkdir(parents=True)
    environment = {**os.environ,"DOCKER_HOST":DOCKER_SOCKET}
    command = ['docker','run','--rm','--network','none','--cpus','2','--memory','3g',
               '--volume',f'{ROOT}:/workspace:ro','--volume',f'{landmarks_dir}:/output:rw',plan['mediapipe_image_id'],
               'python','/workspace/tools/c5_face_fallback.py','--plan',f'/workspace/{plan_path.resolve().relative_to(ROOT)}','--output-dir','/output']
    subprocess.run(command,env=environment,check=True)
    for key,value in {"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","HF_HOME":str(BASELINE/'run/hf'),
                      "TORCH_HOME":str(BASELINE/'run/torch'),"MPLCONFIGDIR":str(BASELINE/'run/matplotlib'),
                      "SHUBERT_DINOV2_SOURCE":str(BASELINE/'vendor/dinov2-source'),"OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2"}.items():
        os.environ[key]=value
    sys.path.insert(0,str(BASELINE/'vendor/demo'))
    sys.path.insert(0,str(BASELINE/'vendor'))
    import cv2
    import torch
    from crop_face import FaceExtractor
    from dinov2_features import DINOEmbedder
    from inference import SignLanguageByT5ForConditionalGeneration
    from safetensors import safe_open
    from safetensors.torch import load_file
    from shubert_visual_lora import install_visual_lora,load_visual_adapter
    from transformers import ByT5Tokenizer
    from transformers.generation.utils import GenerationMixin
    from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

    torch.set_num_threads(2)
    checkpoint = BASELINE/'models/checkpoint-11625'
    adapter = ROOT/plan['starting_adapter']
    if sha256(checkpoint/'model.safetensors')!=plan['baseline_checkpoint_sha256'] or sha256(adapter)!=plan['starting_adapter_sha256'] or sha256(BASELINE/'models/dinov2face.safetensors')!=plan['face_dino_sha256']:
        raise ValueError("face-fallback model weights changed")
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(str(checkpoint),local_files_only=True)
    with safe_open(str(adapter),framework='pt',device='cpu') as source:
        metadata = source.metadata()
    install_visual_lora(model,int(metadata['rank']),float(metadata['alpha']),int(metadata['visual_last_layers']))
    load_visual_adapter(model,load_file(str(adapter),device='cpu'))
    model.requires_grad_(False);model.to('cpu');model.eval()
    tokenizer = ByT5Tokenizer.from_pretrained(str(BASELINE/'models/byt5_base'),local_files_only=True)
    embedder = DINOEmbedder(str(BASELINE/'models/dinov2face.safetensors'),batch_size=32,device='cpu')
    cropper = FaceExtractor()
    def decode(streams):
        with torch.no_grad():
            hidden=model.encoder(**{key:torch.from_numpy(values).unsqueeze(0) for key,values in streams.items()},return_dict=True).last_hidden_state
            ids=GenerationMixin.generate(model,encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden),
                                         max_length=2048,num_beams=5,early_stopping=True,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(ids[0],skip_special_tokens=True)
    records=[]
    for row in plan['examples']:
        started=time.perf_counter()
        cache=ROOT/row['cache'];clip=ROOT/row['clip']
        if sha256(cache)!=row['cache_sha256'] or sha256(clip)!=row['clip_sha256']:
            raise ValueError("face-fallback input bytes changed")
        with np.load(cache,allow_pickle=False) as payload:
            control={key:payload[key][::2].copy() for key in ('face_features','left_hand_features','right_hand_features','pose_features')}
        landmarks=landmarks_dir/f"{row['example_id']}.npz"
        with np.load(landmarks,allow_pickle=False) as payload:
            recovery=payload['recovery_mask'];points=payload['recovered_face_landmarks'];full=payload['full_frame_face_counts']
            source_metadata=json.loads(str(payload['metadata'].item()))
        if source_metadata['plan_sha256']!=sha256(plan_path) or source_metadata['clip_sha256']!=row['clip_sha256'] or len(recovery)!=len(control['face_features']) or np.any(full[recovery]):
            raise ValueError("face-fallback recovered landmarks differ from their source")
        indices=np.flatnonzero(recovery)
        crops=[]
        capture=cv2.VideoCapture(str(clip))
        try:
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES,int(index)*2)
                available,bgr=capture.read()
                if not available:
                    raise RuntimeError("face-fallback crop frame unavailable")
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                image=cropper.cues_on_grey_background(rgb,points[index].tolist())
                crops.append(cropper.resize_frame(image,(224,224)))
        finally:
            capture.release()
        replacement=control['face_features'].copy()
        if crops:
            replacement[recovery]=embedder.extract_embeddings_from_frames(crops)
        candidate=overlay_faces(control,replacement,recovery)
        feature_path=data_dir/f"{row['example_id']}.npz"
        with feature_path.open('xb') as output:
            np.savez_compressed(output,face_features=candidate['face_features'],recovery_mask=recovery)
        control_text=decode(control)
        if control_text!=row['expected_control_hypothesis']:
            raise RuntimeError(f"face-fallback control did not reproduce frozen output: {row['example_id']}")
        candidate_text=decode(candidate)
        words=normalize(row['reference'])
        record={"domain":row['domain'],"example_id":row['example_id'],"signer":row['signer'],"reference":row['reference'],
                "control_hypothesis":control_text,"candidate_hypothesis":candidate_text,
                "control_wer":distance(words,normalize(control_text))/len(words),"candidate_wer":distance(words,normalize(candidate_text))/len(words),
                "recovered_frames":int(recovery.sum()),"evaluated_frames":len(recovery),"nonface_features_preserved":True,
                "original_cache_sha256":row['cache_sha256'],"candidate_face_sha256":sha256(feature_path),"recovered_landmarks_sha256":sha256(landmarks),
                "seconds_face_dino_and_paired_decode":time.perf_counter()-started}
        if record['control_wer']!=row['expected_control_wer']:
            raise RuntimeError("face-fallback control metric differs")
        records.append(record)
        with (plan_path.parent/'paired-results.jsonl').open('a') as output:
            output.write(json.dumps(record,sort_keys=True)+'\n');output.flush();os.fsync(output.fileno())
        if sum(p.stat().st_size for p in data_dir.rglob('*') if p.is_file())>plan['data_budget_bytes']:
            raise RuntimeError("face-fallback data budget exceeded")
        print(f"{row['domain']} {row['example_id']} WER={record['control_wer']:.4f}/{record['candidate_wer']:.4f}",flush=True)
    result={"schema":"umi-studio-c5-face-fallback-decision/1","plan_sha256":sha256(plan_path),**summarize(records,plan['screen']),
            "all_frozen_control_hypotheses_reproduced":True,"nonface_features_preserved":True,"weights_changed":False,
            "confirmation_decoded":False,"device":"cpu","cpu_threads":2,"miner_changed":False,
            "qualification":"Single-reference open-development diagnostic only. This does not establish official UMI score, larger-suite generalization or full runtime latency."}
    (plan_path.parent/'decision.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',type=Path)
    group.add_argument('--plan',type=Path)
    args=parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    else:
        run(args.plan)


if __name__=='__main__':
    main()
