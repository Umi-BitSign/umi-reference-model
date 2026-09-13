# Build from the verified staging directory; weights stay outside the image.
FROM docker.io/library/python:3.10-slim-bookworm@sha256:68d914ec641a0b69267ce65184d000a2bc3a9ee2590ab702b82250ab2385735a

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY model/requirements.txt /opt/community-requirements.txt
# Older OmegaConf metadata needs pip 24.0. Preserve the supplied model pins;
# this environment is separate from UMI's Bittensor/Python dependency set.
RUN python -m pip install --no-cache-dir pip==24.0 \
    && python -m pip install --no-cache-dir -r /opt/community-requirements.txt \
    && python -m pip check
ENV PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/hf \
    TORCH_HOME=/tmp/torch \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache \
    SHUBERT_DEVICE=cpu
USER 65532:65532
WORKDIR /model
ENTRYPOINT ["/usr/local/bin/python3", "/model/umi_inference.py"]
