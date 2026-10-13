"""One-shot entrypoint for the confirmed C5 corrective stride-2 model."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    device = os.environ.get("UMI_EVALUATION_DEVICE", "cpu")
    if device not in {"cpu", "mps"}:
        raise RuntimeError("confirmed C5 bundle supports CPU or MPS")
    root = Path(__file__).resolve().parent
    video = Path(sys.argv[1]).resolve(strict=True)
    os.environ["SHUBERT_DINOV2_SOURCE"] = str(root / "vendor/dinov2-source")
    sys.stdout.flush()
    descriptor = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        import torch

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        from c5_stride2_runtime import install_post_mediapipe_stride
        from runtime import SHuBERTInferenceRuntime
        from umi_landmarks import video_holistic
        from umi_video_reader import VideoReader

        model = SHuBERTInferenceRuntime(
            root,
            device=device,
            generation_num_beams=5,
            generation_max_length=2048,
            dino_batch_size=128,
            model_execution_concurrency=1,
            verify_assets=True,
        )
        model._video_holistic = video_holistic
        model._video_reader = VideoReader
        install_post_mediapipe_stride(model)
        text = model.translate_path(video)
        if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 4096:
            raise ValueError("confirmed C5 output is outside the protocol bound")
        output.write(text)


if __name__ == "__main__":
    main()
