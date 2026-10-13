import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from c5_fresh_face_benchmark import ARMS,calibrate,summarize


def test_dense_and_lowrank_calibration_match_known_affine_maps():
    values=np.ones((2,3),dtype=np.float32)
    dense={'face_weight':np.eye(3,dtype=np.float32)*2,'face_bias':np.ones(3,dtype=np.float32)}
    low={'face_left':np.eye(3,dtype=np.float32),'face_right':np.eye(3,dtype=np.float32),'face_bias':np.ones(3,dtype=np.float32)}
    assert np.array_equal(calibrate(values,dense,'face'),values*3)
    assert np.array_equal(calibrate(values,low,'face'),values*3)


def test_source_group_bootstrap_and_selected_control_gate():
    rows=[]
    for i in range(20):
        values={arm:{'wer':.8,'edits':8,'hypothesis':f'{arm} {i}','active_seconds':1.} for arm in ARMS}
        for arm in ('corrective_native224_recovery','confirmed_crop168_recovery'):
            values[arm].update(wer=.6,edits=6)
        rows.append({'video_id':str(i),'signer':i%5,'reference_words':10,'arms':values})
    report=summarize(rows)
    assert report['selected_research_arm']=='confirmed_crop168_recovery'
    assert report['comparisons']['confirmed_crop168_recovery']['source_clustered_95_interval_vs_selected'][1]<0
    for row in rows:row['arms']['confirmed_crop168_recovery'].update(wer=.8,edits=8)
    assert not summarize(rows)['comparisons']['confirmed_crop168_recovery']['passes']
