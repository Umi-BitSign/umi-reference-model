FROM docker.io/library/python:3.10-slim-bookworm@sha256:68d914ec641a0b69267ce65184d000a2bc3a9ee2590ab702b82250ab2385735a

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir pip==24.0 \
    && python -m pip install --no-cache-dir \
        mediapipe==0.10.14 \
        numpy==1.26.4 \
        opencv-contrib-python==4.10.0.84 \
    && python -m pip check

ENV PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2
WORKDIR /workspace
