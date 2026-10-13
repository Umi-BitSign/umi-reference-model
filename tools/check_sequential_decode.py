"""Compare sequential OpenCV decoding with the frozen baseline VideoReader."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("video", nargs="+", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.baseline.resolve() / "vendor/demo"))
    import numpy as np
    from sequential_video_reader import make_sequential_reader
    from video_reader import VideoReader

    sequential_reader = make_sequential_reader(VideoReader)
    for video in args.video:
        reader = VideoReader(video)
        started = time.perf_counter()
        expected = reader.get_batch(range(len(reader))).asnumpy()
        baseline_seconds = time.perf_counter() - started
        optimized = sequential_reader(video)
        started = time.perf_counter()
        actual = optimized.get_batch(range(len(optimized))).asnumpy()
        sequential_seconds = time.perf_counter() - started
        if not np.array_equal(expected, actual):
            raise RuntimeError(f"sequential decode differs from baseline: {video}")
        digest = hashlib.sha256(actual.tobytes())
        print(
            f"{video.name}: {len(reader)} frames byte-identical "
            f"baseline={baseline_seconds:.3f}s sequential={sequential_seconds:.3f}s "
            f"frames_sha256={digest.hexdigest()}",
            flush=True,
        )


if __name__ == "__main__":
    main()
