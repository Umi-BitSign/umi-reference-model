# Third-party components

The model lineage and inference stack include third-party material. Release operators
must preserve upstream licenses, notices, and source obligations.

## Training lineage

| Source | License | Distribution in this project |
|---|---|---|
| FLEURS-ASL | CC BY-SA 4.0 | Dependent weights and attribution only |
| FSboard | CC BY 4.0 | Dependent weights and attribution only |

Source videos, annotations, and FSboard records are excluded from the release.

## Python runtime

The locked runtime includes PyTorch, NumPy, Safetensors, RFC 8785, and their pinned
dependencies. Their package archives and license metadata are named in `uv.lock`.
Installing the environment does not change the license applied to UMI-authored code or
the model weights.

## Locally built extractor image

The checked-in Docker source builds from a digest-pinned Debian Python base and
installs pinned MediaPipe, NumPy, OpenCV, FFmpeg, and system packages. The build keeps
the license files supplied by Debian and the Python packages inside the local image.

The installed FFmpeg binary reports `--enable-gpl`. This project does not publish or
transfer the built image. Operators build it locally from the checked-in recipe. An
operator who redistributes that image must independently satisfy the licenses and
source-offer obligations of every included component. The project makes no claim that
images built on separate machines are byte-equivalent.

The MediaPipe Holistic task model is a separate external download. The public release
does not bundle it because separate redistribution terms have not been established.
