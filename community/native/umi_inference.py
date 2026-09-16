"""One-shot MPS entrypoint for the preserved, unrewarded community baseline."""
import os
import sys
from pathlib import Path


def main():
    if os.environ.get('UMI_EVALUATION_DEVICE') != 'mps':
        raise RuntimeError('this baseline entrypoint requires the native MPS profile')
    root = Path(__file__).resolve().parent
    video = Path(sys.argv[1]).resolve(strict=True)
    os.environ['SHUBERT_DINOV2_SOURCE'] = str(root / 'vendor/dinov2-source')
    # Keep Python and native-library diagnostics out of the scored output pipe.
    sys.stdout.flush()
    descriptor = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    with os.fdopen(descriptor, 'w', encoding='utf-8') as output:
        from runtime import SHuBERTInferenceRuntime
        model = SHuBERTInferenceRuntime(root, device='mps', generation_num_beams=5,
            generation_max_length=2048, dino_batch_size=128,
            model_execution_concurrency=1, verify_assets=True)
        from umi_landmarks import video_holistic
        from umi_video_reader import VideoReader
        model._video_holistic = video_holistic
        model._video_reader = VideoReader
        text = model.translate_path(video)
        if not isinstance(text, str) or not text.strip() or len(text.encode('utf-8')) > 4096:
            raise ValueError('baseline output is outside the protocol bound')
        output.write(text)


if __name__ == '__main__':
    main()
