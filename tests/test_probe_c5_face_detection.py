import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from probe_c5_face_detection import head_roi, remap_face


def test_head_box_is_square_bounded_and_contains_head():
    pose=np.zeros((33,3)); pose[:11,:2]=[.5,.3]
    box=head_roi(pose,1280,720)
    x0,y0,x1,y1=box
    assert 0<=x0<x1<=1280 and 0<=y0<y1<=720
    assert x1-x0==y1-y0==64
    assert x0<=640<=x1 and y0<=216<=y1


def test_remapping_preserves_original_normalized_coordinates():
    points=np.array([[.5,.5,.1],[0,0,0],[1,1,-.2]])
    mapped=remap_face(points,(100,50,300,250),1000,500)
    assert mapped.tolist()==pytest.approx(np.array([[.2,.3,.02],[.1,.1,0],[.3,.5,-.04]]))
