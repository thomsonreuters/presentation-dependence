# vLLM eval/teacher image.
#
# BASE_IMAGE must be a CUDA-matched PyTorch image whose torch matches the vLLM
# wheel pinned in scripts/train/requirements.txt. The Gemma-4-capable stack is
# PyTorch 2.10 / Python 3.13 / CUDA 13.0; see docs/HARDWARE.md for why the
# pairing is load-bearing.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=180 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    PYTHONPATH=/workspace/src

WORKDIR /workspace

COPY scripts/train/requirements.txt /tmp/slm_train_requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/slm_train_requirements.txt && \
    rm /tmp/slm_train_requirements.txt
