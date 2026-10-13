import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from c5_face_fallback import overlay_faces
from run_c5_face_fallback_experiment import summarize


def test_only_recovered_faces_change_without_mutating_control():
    control={name:np.zeros((3,4),dtype=np.float32) for name in ('face_features','left_hand_features','right_hand_features','pose_features')}
    replacements=np.ones((3,4),dtype=np.float32)
    mask=np.array([False,True,False])
    candidate=overlay_faces(control,replacements,mask)
    assert np.array_equal(candidate['face_features'][1],np.ones(4))
    assert not np.count_nonzero(candidate['face_features'][[0,2]])
    assert all(not np.count_nonzero(control[name]) for name in control)
    assert all(np.array_equal(candidate[name],control[name]) for name in control if name!='face_features')
    with pytest.raises(ValueError,match='shapes'):
        overlay_faces(control,replacements[:2],mask)


def test_three_signer_screen_is_not_relaxed_for_average_improvement():
    screen={'how2sign_max_mean_wer_change':0,'fleurs_max_mean_wer_change':0,'equal_domain_max_mean_wer_change':-.01,'minimum_improved_how2sign_signers':3}
    rows=[{'domain':'how2sign','signer':s,'control_wer':.8,'candidate_wer':.7 if s<3 else .8} for s in (1,2,3)]
    rows.append({'domain':'fleurs','signer':4,'control_wer':.8,'candidate_wer':.7})
    assert not summarize(rows,screen)['passes_open_screen']
    rows[2]['candidate_wer']=.7
    result=summarize(rows,screen)
    assert result['passes_open_screen'] and result['decision']=='collect_larger_development_evidence'
