from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_parallel_face_recovery_runtime import detect_legacy_frame


def test_independent_tasks_overlap_pose_and_are_joined_before_return():
    started = [threading.Event(), threading.Event()]; release = threading.Event(); finished = []
    class Task:
        def __init__(self, index): self.index = index
        def detect(self, image):
            started[self.index].set()
            assert release.wait(2)
            finished.append(self.index)
            return f"task-{self.index}"
    class Pose:
        def process(self, frame):
            assert all(event.wait(2) for event in started)
            release.set(); return "pose"
    with ThreadPoolExecutor(max_workers=2) as executor:
        result = detect_legacy_frame(None, None, Task(0), Task(1), Pose(), executor)
    assert result == ("task-0", "task-1", "pose") and sorted(finished) == [0, 1]


def test_both_tasks_join_even_when_pose_raises():
    finished = []; release = threading.Event()
    class Task:
        def __init__(self, index): self.index = index
        def detect(self, image):
            assert release.wait(2); finished.append(self.index); return self.index
    class Pose:
        def process(self, frame):
            release.set(); raise ValueError("pose failed")
    with ThreadPoolExecutor(max_workers=2) as executor:
        with pytest.raises(ValueError, match="pose failed"):
            detect_legacy_frame(None, None, Task(0), Task(1), Pose(), executor)
        assert sorted(finished) == [0, 1]
