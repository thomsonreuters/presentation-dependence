# SFT-only training image.
#
# BASE_IMAGE must be a CUDA-matched PyTorch image. This image was built
# against PyTorch 2.8 / Python 3.12 / CUDA 12.9; see docs/HARDWARE.md.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=180 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=/workspace/src

WORKDIR /workspace

COPY scripts/train-sft/requirements.txt /tmp/slm_sft_requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/slm_sft_requirements.txt && \
    rm /tmp/slm_sft_requirements.txt
