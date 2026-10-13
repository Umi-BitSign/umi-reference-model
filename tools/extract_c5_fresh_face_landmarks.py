"""Prepare baseline landmarks plus isolated ROI face detections in pinned CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from extract_mediapipe_landmarks import write_landmarks
from probe_c5_face_detection import head_roi,remap_face,sha256

ROOT=Path(__file__).resolve().parents[1]


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--face-model',type=Path,required=True);parser.add_argument('--hand-model',type=Path,required=True)
    args=parser.parse_args()
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    options=vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(args.face_model),delegate=python.BaseOptions.Delegate.CPU),
                                         output_face_blendshapes=True,output_facial_transformation_matrixes=True,num_faces=6)
    face=vision.FaceLandmarker.create_from_options(options)
    recovery_detector=vision.FaceLandmarker.create_from_options(options)
    hand=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.hand_model),delegate=python.BaseOptions.Delegate.CPU),num_hands=6,min_hand_detection_confidence=.05))
    pose=mp.solutions.pose.Pose(min_detection_confidence=.1)
    capture=cv2.VideoCapture(str(args.input));fps=capture.get(cv2.CAP_PROP_FPS)
    width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps!=25 or min(width,height)<1: raise ValueError("fresh landmarks need normalized 25fps video")
    landmarks={};recovered=[];mask=[];base_seconds=0.;recovery_seconds=0.
    try:
        while True:
            available,bgr=capture.read()
            if not available: break
            rgb=np.ascontiguousarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
            started=time.perf_counter()
            faces=face.detect(image);hands=hand.detect(image);body=pose.process(rgb)
            points=np.array([[p.x,p.y,p.z] for p in body.pose_landmarks.landmark]) if body.pose_landmarks else None
            row={'face_landmarks':[[[p.x,p.y,p.z] for p in item] for item in faces.face_landmarks] or None,
                 'hand_landmarks':[[[p.x,p.y,p.z] for p in item] for item in hands.hand_landmarks] or None,
                 'pose_landmarks':[points.tolist()] if points is not None else None}
            index=len(landmarks);landmarks[index]=row;base_seconds+=time.perf_counter()-started
            chosen=None
            if index%2==0 and not faces.face_landmarks and points is not None:
                started=time.perf_counter();box=head_roi(points,width,height);x0,y0,x1,y1=box
                region=np.ascontiguousarray(rgb[y0:y1,x0:x1])
                results=recovery_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=region))
                mapped=[remap_face(np.array([[p.x,p.y,p.z] for p in item]),box,width,height) for item in results.face_landmarks]
                if mapped: chosen=min(mapped,key=lambda item:np.linalg.norm(points[0]-item[0]))
                recovery_seconds+=time.perf_counter()-started
            recovered.append(chosen if chosen is not None else np.full((478,3),np.nan))
            mask.append(chosen is not None)
    finally:
        capture.release();face.close();recovery_detector.close();hand.close();pose.close()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    name=args.input.stem
    metadata=write_landmarks(args.output_dir/f'{name}.npz',landmarks,source=args.input,width=width,height=height,fps=fps,mediapipe_version=mp.__version__)
    metadata.update({'extractor_sha256':sha256(Path(__file__)),'head_roi_source_sha256':sha256(ROOT/'tools/probe_c5_face_detection.py'),
                     'baseline_landmarks_seconds':base_seconds,'face_recovery_seconds':recovery_seconds,
                     'legacy_landmarks_modified':False,'separate_recovery_detector':True})
    with (args.output_dir/f'{name}-recovery.npz').open('xb') as output:
        np.savez_compressed(output,metadata=np.asarray(json.dumps(metadata,sort_keys=True)),recovery_mask=np.array(mask,dtype=np.bool_),recovered_faces=np.array(recovered,dtype=np.float64))
    print(json.dumps({'frames':len(landmarks),'recovered_frames':sum(mask),'baseline_seconds':base_seconds,'recovery_seconds':recovery_seconds}))


if __name__=='__main__':main()
