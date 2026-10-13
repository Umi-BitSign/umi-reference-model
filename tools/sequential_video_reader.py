"""Sequential full-video decode matching the baseline VideoReader's RGB frames."""

from __future__ import annotations


def make_sequential_reader(base_reader_type):
    import cv2
    import numpy as np
    from video_reader import FrameArray

    class SequentialVideoReader(base_reader_type):
        def get_batch(self, indices):
            full_range = isinstance(indices, range) and indices == range(len(self))
            at_start = self._capture.get(cv2.CAP_PROP_POS_FRAMES) == 0
            if not full_range or not at_start:
                return super().get_batch(indices)
            frames = []
            for frame_index in indices:
                success, bgr = self._capture.read()
                if not success or bgr is None:
                    raise RuntimeError(f"failed to decode frame {frame_index}: {self.filename}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if self.width is not None or self.height is not None:
                    if self.width is None or self.height is None:
                        raise ValueError("width and height must be supplied together")
                    rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                frames.append(np.ascontiguousarray(rgb, dtype=np.uint8))
            return FrameArray(np.stack(frames, axis=0))

    return SequentialVideoReader
