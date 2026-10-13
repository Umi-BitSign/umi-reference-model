"""Freeze, prepare and evaluate all fixed face/crop policies on fresh development."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from benchmark_fleurs_dev import distance,normalize,sha256
from c5_experiment_sources import archive_sources
from c5_fresh_face_benchmark import ARMS,GATES,ROOT,calibrate,summarize
from prepare_asl_stem_wiki_landmarks import DOCKER_SOCKET,docker_image_id

BUNDLE=ROOT/'artifacts/c5-corrective-stride2-crop168-confirmed'
BASELINE=ROOT/'artifacts/baseline-6fe8df59'
IMAGE_ID='sha256:5cad3554c60a6b0bf7d0b059dde0fed09c5a357cd0142c6dfa0b6fa9cfe63378'


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def source_hashes() -> dict:
    return {name:sha256(ROOT/'tools'/name) for name in ('c5_fresh_face_benchmark.py','extract_c5_fresh_face_landmarks.py',
            'run_c5_fresh_face_benchmark.py','probe_c5_face_detection.py','extract_mediapipe_landmarks.py','c5_stride2_runtime.py','c5_crop_calibrator.py','resolution_policy.py')}


def freeze(output: Path) -> None:
    output=output.resolve()
    if output.exists() or not output.is_relative_to(ROOT/'runs'): raise ValueError('fresh benchmark needs a new runs/ output')
    acquisition_path=ROOT/'runs/c5-fresh-development-v2/plan.json'
    acquisition=json.loads(acquisition_path.read_text())
    receipt_path=acquisition_path.with_name('source-receipts.jsonl')
    receipts=read_jsonl(receipt_path)
    by_name={row['video_name']:row for row in receipts}
    if len(by_name)!=17 or len(receipts)!=17 or len(acquisition['clips'])!=71: raise ValueError('fresh acquisition is incomplete')
    examples=[]
    for row in acquisition['clips']:
        receipt=by_name[row['video_name']]
        source=ROOT/acquisition['data_dir']/'sources'/f"{row['video_name']}.mp4"
        if receipt['plan_sha256']!=sha256(acquisition_path) or receipt['signer']!=row['signer'] or sha256(source)!=receipt['sha256'] or receipt['invalid_selected_intervals']:
            raise ValueError('fresh clip/source recording identity or integrity differs')
        examples.append({**row,'source_video':str(source.relative_to(ROOT)),'source_video_sha256':receipt['sha256']})
    if docker_image_id(DOCKER_SOCKET)!=IMAGE_ID: raise ValueError('fresh CPU landmark image changed')
    expected_assets={BASELINE/'models/checkpoint-11625/model.safetensors':'92c6c77ed7f2343f346c2a0cc7c025d733d47fcb1adc55150939f934d5f3b3b9',
                     BUNDLE/'models/checkpoint-11625/model.safetensors':'419315ded8438bb59fe1b1a7e3c774012e098067db21335621789b56e9f98210',
                     BUNDLE/'MODEL_IDENTITY.json':'8c813666d4edadbf1e658b9edc747a2f6426950c90f44e712d48bec22b1cebf6',
                     BUNDLE/'crop_calibrator.safetensors':'f2f0b23839c7de012d5aaf1c6bd9d10f156f28eb8f8bcf6f2ce4240a98de68c6'}
    if any(sha256(path)!=expected for path,expected in expected_assets.items()):raise ValueError('frozen baseline/selected assets changed')
    plan={'schema':'umi-studio-c5-fresh-face-benchmark-plan/1','arms':ARMS,'gates':GATES,'examples':examples,
          'acquisition_plan_sha256':sha256(acquisition_path),'source_receipts_sha256':sha256(receipt_path),
          'data_dir':acquisition['data_dir'],'data_budget_bytes':acquisition['recipe']['data_budget_bytes'],
          'minimum_free_bytes':acquisition['recipe']['minimum_free_bytes'],'image_id':IMAGE_ID,'source_sha256':source_hashes(),
          'original_checkpoint_sha256':sha256(BASELINE/'models/checkpoint-11625/model.safetensors'),
          'selected_checkpoint_sha256':sha256(BUNDLE/'models/checkpoint-11625/model.safetensors'),
          'selected_identity_sha256':sha256(BUNDLE/'MODEL_IDENTITY.json'),'calibrator_sha256':sha256(BUNDLE/'crop_calibrator.safetensors'),
          'dino_hand_sha256':sha256(BUNDLE/'models/dinov2hand.safetensors'),'dino_face_sha256':sha256(BUNDLE/'models/dinov2face.safetensors'),
          'face_asset_sha256':sha256(BUNDLE/'models/face_landmarker_v2_with_blendshapes.task'),
          'hand_asset_sha256':sha256(BUNDLE/'models/hand_landmarker.task'),
          'post_mediapipe_temporal_stride':2,'crop_frame_rate':25,'cpu_threads':2,'device':'cpu',
          'generation':{'num_beams':5,'max_length':2048,'early_stopping':True},'training_allowed':False,
          'confirmation_policy':'No sealed confirmation, reserve, test, devtest or private cases. Score only this fixed fresh public-development cohort.',
          'timing_qualification':'Active feature/encoder/decode timings exclude shared offline media/landmark preparation; they do not establish complete standalone runtime latency.',
          'original_control_qualification':'The original checkpoint uses common post-MediaPipe stride-2 inputs, not its unmodified full-frame-rate runtime.',
          'all_policy_choices_frozen_before_predictions':True,'miner_health_policy':'user_authorized_maintenance'}
    output.mkdir(parents=True)
    path=output/'plan.json';path.write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    archive_sources(path,output/'source-snapshot')
    print(json.dumps({'plan_sha256':sha256(path),'clips':len(examples),'policies':list(ARMS)}))


def check_storage(plan: dict) -> None:
    data=ROOT/plan['data_dir']
    if shutil.disk_usage(ROOT).free<plan['minimum_free_bytes']+128*1024**2: raise RuntimeError('fresh benchmark disk floor reached')
    if sum(path.stat().st_size for path in data.rglob('*') if path.is_file())>plan['data_budget_bytes']-128*1024**2:
        raise RuntimeError('fresh benchmark full-data budget reached')


def prepare(plan_path: Path) -> None:
    plan=json.loads(plan_path.read_text())
    if plan['source_sha256']!=source_hashes(): raise ValueError('frozen preparation implementation changed')
    data=ROOT/plan['data_dir'];clips=data/'clips-25fps';landmarks=data/'landmarks'
    if clips.exists() or landmarks.exists(): raise ValueError('fresh prepared outputs already exist')
    clips.mkdir();landmarks.mkdir()
    environment={**os.environ,'DOCKER_HOST':DOCKER_SOCKET}
    for row in plan['examples']:
        check_storage(plan)
        source=ROOT/row['source_video'];key=row['sentence_name'];clip=clips/f'{key}.mp4'
        if sha256(source)!=row['source_video_sha256']: raise ValueError('fresh source video changed')
        command=['ffmpeg','-nostdin','-v','error','-ss',str(row['start_sec']),'-i',str(source),'-t',str(row['duration_sec']),
                 '-an','-vf','setpts=PTS-STARTPTS,fps=25','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p',
                 '-threads','2','-filter_threads','1',str(clip)]
        subprocess.run(command,check=True)
        command=['docker','run','--rm','--network','none','--cpus','2','--memory','3g','--volume',f'{ROOT}:/workspace:ro',
                 '--volume',f'{landmarks}:/output:rw',plan['image_id'],'python','/workspace/tools/extract_c5_fresh_face_landmarks.py',
                 '--input',f'/workspace/{clip.relative_to(ROOT)}','--output-dir','/output',
                 '--face-model','/workspace/artifacts/c5-corrective-stride2-crop168-confirmed/models/face_landmarker_v2_with_blendshapes.task',
                 '--hand-model','/workspace/artifacts/c5-corrective-stride2-crop168-confirmed/models/hand_landmarker.task']
        subprocess.run(command,env=environment,check=True)
        from extract_mediapipe_landmarks import unpack_landmarks
        _,metadata=unpack_landmarks(landmarks/f'{key}.npz')
        if metadata['source_sha256']!=sha256(clip) or metadata['fps']!=25 or abs(metadata['frames']-round(row['duration_sec']*25))>1:
            raise ValueError('fresh normalized clip duration/fps differs')
        record={'schema':'umi-studio-c5-fresh-face-prepared/1','plan_sha256':sha256(plan_path),'example_id':key,
                'video_name':row['video_name'],'signer':row['signer'],'source_sha256':row['source_video_sha256'],
                'clip_sha256':sha256(clip),'landmarks_sha256':sha256(landmarks/f'{key}.npz'),
                'recovery_sha256':sha256(landmarks/f'{key}-recovery.npz'),'frames':metadata['frames'],
                'training_allowed':False,'predictions_inspected':False}
        with (plan_path.parent/'prepared.jsonl').open('a') as output:
            output.write(json.dumps(record,sort_keys=True)+'\n');output.flush();os.fsync(output.fileno())
        print(f'prepared {key} frames={metadata["frames"]}',flush=True)


def evaluate(plan_path: Path) -> None:
    plan=json.loads(plan_path.read_text())
    if plan['arms']!=ARMS or plan['gates']!=GATES or plan['source_sha256']!=source_hashes(): raise ValueError('frozen benchmark policies changed')
    rows=read_jsonl(plan_path.parent/'prepared.jsonl')
    if len(rows)!=71 or {r['example_id'] for r in rows}!={r['sentence_name'] for r in plan['examples']}: raise ValueError('fresh media preparation is incomplete')
    prepared={r['example_id']:r for r in rows};data=ROOT/plan['data_dir'];feature_dir=data/'features'
    if feature_dir.exists() or (plan_path.parent/'results.jsonl').exists(): raise ValueError('fresh predictions or features already exist')
    feature_dir.mkdir()
    for key,value in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','SHUBERT_DINOV2_SOURCE':str(BUNDLE/'vendor/dinov2-source'),
                      'HF_HOME':str(BASELINE/'run/hf'),'TORCH_HOME':str(BASELINE/'run/torch'),'MPLCONFIGDIR':str(BASELINE/'run/matplotlib'),
                      'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}.items():os.environ[key]=value
    sys.path.insert(0,str(BUNDLE/'vendor/demo'));sys.path.insert(0,str(BUNDLE/'vendor'))
    import cv2
    import torch
    from body_features import process_pose_landmarks
    from crop_face import FaceExtractor
    from crop_hands import HandExtractor
    from dinov2_features import DINOEmbedder
    from inference import SignLanguageByT5ForConditionalGeneration
    from safetensors.numpy import load_file
    from transformers import ByT5Tokenizer
    from transformers.generation.utils import GenerationMixin
    from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
    from torchvision import transforms
    from c5_stride2_runtime import temporal_subsample_landmarks
    from extract_mediapipe_landmarks import unpack_landmarks

    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    for path,key in [(BASELINE/'models/checkpoint-11625/model.safetensors','original_checkpoint_sha256'),(BUNDLE/'models/checkpoint-11625/model.safetensors','selected_checkpoint_sha256'),
                     (BUNDLE/'crop_calibrator.safetensors','calibrator_sha256'),(BUNDLE/'models/dinov2hand.safetensors','dino_hand_sha256'),(BUNDLE/'models/dinov2face.safetensors','dino_face_sha256')]:
        if sha256(path)!=plan[key]:raise ValueError('fresh benchmark model asset changed')
    models={name:SignLanguageByT5ForConditionalGeneration.from_pretrained(str(root/'models/checkpoint-11625'),local_files_only=True) for name,root in [('original',BASELINE),('selected',BUNDLE)]}
    for model in models.values():model.requires_grad_(False);model.to('cpu');model.eval()
    tokenizer=ByT5Tokenizer.from_pretrained(str(BUNDLE/'models/byt5_base'),local_files_only=True)
    hand_model=DINOEmbedder(str(BUNDLE/'models/dinov2hand.safetensors'),batch_size=32,device='cpu')
    face_model=DINOEmbedder(str(BUNDLE/'models/dinov2face.safetensors'),batch_size=32,device='cpu')
    hand_cropper=HandExtractor();face_cropper=FaceExtractor()
    calibrator=load_file(str(BUNDLE/'crop_calibrator.safetensors'))
    def set_pixels(size):
        for embedder in (hand_model,face_model):
            embedder.transform=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])])
        hand_cropper.output_size=(size,size);face_cropper.output_size=(size,size)
    def decode(model,streams):
        with torch.no_grad():
            hidden=model.encoder(**{key:torch.from_numpy(values).unsqueeze(0) for key,values in streams.items()},return_dict=True).last_hidden_state
            ids=GenerationMixin.generate(model,encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden),max_length=2048,num_beams=5,
                                         early_stopping=True,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(ids[0],skip_special_tokens=True)
    results=[]
    for row in plan['examples']:
        check_storage(plan)
        key=row['sentence_name'];receipt=prepared[key];clip=data/'clips-25fps'/f'{key}.mp4'
        landmark_path=data/'landmarks'/f'{key}.npz';recovery_path=data/'landmarks'/f'{key}-recovery.npz'
        if receipt['plan_sha256']!=sha256(plan_path) or sha256(clip)!=receipt['clip_sha256'] or sha256(landmark_path)!=receipt['landmarks_sha256'] or sha256(recovery_path)!=receipt['recovery_sha256']:
            raise ValueError('fresh prepared inputs changed')
        landmarks,metadata=unpack_landmarks(landmark_path)
        with np.load(recovery_path,allow_pickle=False) as payload:
            recovery=payload['recovery_mask'][::2];faces=payload['recovered_faces'][::2]
            recovery_metadata=json.loads(str(payload['metadata'].item()))
        if recovery_metadata['source_sha256']!=receipt['clip_sha256']:raise ValueError('face-recovery source differs')
        capture=cv2.VideoCapture(str(clip));frames=[]
        try:
            while True:
                valid,bgr=capture.read()
                if not valid:break
                frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        finally:capture.release()
        if len(frames)!=metadata['frames']:raise ValueError('fresh decoded frame count differs')
        frames=np.ascontiguousarray(np.array(frames)[::2]);landmarks=temporal_subsample_landmarks(landmarks,2)
        pose=np.asarray(process_pose_landmarks(landmarks),dtype=np.float32)
        policies={};feature_seconds={}
        for size in (224,168):
            started=time.perf_counter();set_pixels(size)
            left,right=hand_cropper.extract_hand_frames(frames,landmarks)
            base_faces=face_cropper.extract_face_frames(frames,landmarks)
            control={'left_hand_features':hand_model.extract_embeddings_from_frames([f[...,::-1].copy() for f in left]).astype(np.float32),
                     'right_hand_features':hand_model.extract_embeddings_from_frames([f[...,::-1].copy() for f in right]).astype(np.float32),
                     'face_features':face_model.extract_embeddings_from_frames([f[...,::-1].copy() for f in base_faces]).astype(np.float32),'pose_features':pose.copy()}
            base_feature_seconds=time.perf_counter()-started
            started=time.perf_counter();replacement=control['face_features'].copy();crops=[]
            for index in np.flatnonzero(recovery):
                image=face_cropper.cues_on_grey_background(frames[index],faces[index].tolist())
                crops.append(face_cropper.resize_frame(image,(size,size)))
            if crops:replacement[recovery]=face_model.extract_embeddings_from_frames(crops)
            extra_seconds=time.perf_counter()-started
            if size==168:
                for name in ('left_hand_features','right_hand_features','face_features'):
                    control[name]=calibrate(control[name],calibrator,'face' if name=='face_features' else 'hand')
                if recovery.any():replacement[recovery]=calibrate(replacement[recovery],calibrator,'face')
            candidate={name:value.copy() for name,value in control.items()}
            candidate['face_features'][recovery]=replacement[recovery]
            if any(not np.array_equal(candidate[name],control[name]) for name in ('left_hand_features','right_hand_features','pose_features')) or not np.array_equal(candidate['face_features'][~recovery],control['face_features'][~recovery]):
                raise RuntimeError('fresh face recovery changed an unrelated input')
            policies[(size,False)]=control;policies[(size,True)]=candidate
            feature_seconds[(size,False)]=base_feature_seconds;feature_seconds[(size,True)]=base_feature_seconds+extra_seconds
        record={'example_id':key,'video_id':row['video_id'],'video_name':row['video_name'],'signer':row['signer'],'reference':row['reference'],
                'reference_words':len(normalize(row['reference'])),'clip_sha256':receipt['clip_sha256'],'recovered_frames':int(recovery.sum()),'evaluated_frames':len(recovery),
                'nonface_inputs_preserved':True,'arms':{}}
        for arm,policy in ARMS.items():
            features=policies[(policy['crop'],policy['recovery'])]
            feature_path=feature_dir/f'{key}--{arm}.npz'
            if arm!='original_native224':
                with feature_path.open('xb') as output:np.savez_compressed(output,**features)
            started=time.perf_counter();hypothesis=decode(models[policy['weights']],features);seconds=time.perf_counter()-started
            edits=distance(normalize(row['reference']),normalize(hypothesis))
            record['arms'][arm]={'hypothesis':hypothesis,'edits':edits,'wer':edits/record['reference_words'],
                                 'active_seconds':seconds+feature_seconds[(policy['crop'],policy['recovery'])]}
        results.append(record)
        with (plan_path.parent/'results.jsonl').open('a') as output:
            output.write(json.dumps(record,sort_keys=True)+'\n');output.flush();os.fsync(output.fileno())
        print(f"scored {len(results)}/71 {key}",flush=True)
    decision={'schema':'umi-studio-c5-fresh-face-benchmark-decision/1','plan_sha256':sha256(plan_path),**summarize(results),
              'all_choices_frozen_before_predictions':True,'training_allowed':False,'weights_changed':False,'confirmation_decoded':False,'miner_changed':False,
              'timing_qualification':plan['timing_qualification'],'original_control_qualification':plan['original_control_qualification']}
    (plan_path.parent/'decision.json').write_text(json.dumps(decision,indent=2,sort_keys=True)+'\n')
    print(json.dumps(decision))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',type=Path);group.add_argument('--prepare',type=Path);group.add_argument('--evaluate',type=Path)
    args=parser.parse_args()
    if args.freeze:freeze(args.freeze)
    elif args.prepare:prepare(args.prepare)
    else:evaluate(args.evaluate)


if __name__=='__main__':main()
