# syntax=docker/dockerfile:1.7
FROM --platform=linux/arm64 python@sha256:9bb659dc6d5218917236f3711e866a5634bb4c2f208de9d4533aa4863f57c1d3

ARG TARGETARCH
RUN test "${TARGETARCH}" = "arm64"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ffmpeg=7:5.1.9-0+deb12u1 \
        libegl1=1.6.0-1 \
        libgl1=1.6.0-1 \
        libgles2=1.6.0-1 \
        libglib2.0-0=2.74.6-2+deb12u9 \
        libgomp1=12.2.0-14+deb12u1 \
        libice6=2:1.0.10-1 \
        libportaudio2=19.6.0-1.2 \
        libsm6=2:1.2.3-1 \
        libx11-6=2:1.8.4-2+deb12u2 \
        libxcb-render0=1.15-1 \
        libxcb-shm0=1.15-1 \
        libxcb-xfixes0=1.15-1 \
        libxcb-xkb1=1.15-1 \
        libxcb1=1.15-1 \
        libxext6=2:1.3.4-1+b1 \
        libxfixes3=1:6.0.0-2 \
        libxkbcommon-x11-0=1.5.0-1 \
        libxkbcommon0=1.5.0-1 \
        libxrender1=1:0.9.10-1.1 \
    && apt-get clean \
    && find /var/lib/apt/lists -mindepth 1 -delete

COPY requirements.lock /opt/bitsign/requirements.lock
RUN python -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --only-binary=:all: \
        --require-hashes \
        --requirement /opt/bitsign/requirements.lock

COPY worker.py /opt/bitsign/worker.py

RUN groupadd --gid 65532 bitsign \
    && useradd --uid 65532 --gid 65532 --no-create-home --home-dir /tmp \
        --shell /usr/sbin/nologin bitsign \
    && chmod 0555 /opt/bitsign/worker.py \
    && python -m compileall -q /opt/bitsign/worker.py

LABEL org.opencontainers.image.title="bitsign-motion MediaPipe Holistic isolation worker" \
      org.opencontainers.image.base.digest="sha256:9bb659dc6d5218917236f3711e866a5634bb4c2f208de9d4533aa4863f57c1d3" \
      org.bitsign.mediapipe.version="1.0.1" \
      org.bitsign.ffmpeg.version="5.1.9-0+deb12u1" \
      org.bitsign.holistic-model.sha256="e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8" \
      org.bitsign.worker.contract="umi-raw-holistic-landmarks/1" \
      org.bitsign.worker.receipt="umi-raw-holistic-extraction-receipt/3"

ENV HOME=/tmp \
    XDG_CACHE_HOME=/tmp/cache \
    MEDIAPIPE_DISABLE_GPU=1 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    QT_QPA_PLATFORM=offscreen \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1

USER 65532:65532
ENTRYPOINT ["python", "/opt/bitsign/worker.py"]
