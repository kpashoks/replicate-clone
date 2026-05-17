# Wan 2.2 Animate (Character Replace) inference server.
#
# Why a dedicated image: Replicate's run logs for wan-video/wan-2.2-animate-replace
# show they don't use ComfyUI - they call the official Wan-Video repo's Python API
# directly (wan.WanAnimate) with Meta's facebookresearch/sam2 for video masking.
# Five rounds of debugging the ComfyUI WanAnimateToVideo node failed to produce
# correct Mix-mode replacement, because the ComfyUI node either doesn't do the
# required preprocessing (background inpainting via mask, face cropping) or
# expects pre-processed inputs we never built.
#
# This image bypasses ComfyUI entirely. Same model weights, same SAM2, same
# preprocessing logic - but using the upstream Wan-Video code path that Replicate
# is using under the hood.

# Base: official Python 3.11 image. We used to base on nvidia/cuda but that
# image's apt python3.11 package is an old pre-release (~3.11.0rc1) that's
# missing `sys.get_int_max_str_digits` - which PyTorch's _dynamo module calls
# at import time, crashing with `AttributeError: module 'sys' has no attribute
# 'get_int_max_str_digits'`.
#
# CUDA libraries are provided via torch's cu124 wheel which bundles
# nvidia-cuda-runtime-cu12, nvidia-cudnn-cu12, etc. as PyPI dependencies.
# RunPod injects the NVIDIA driver into the container at start time via the
# nvidia container runtime, regardless of the base image. So a slim Python
# base is sufficient for GPU work.
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # NVIDIA container-runtime labels. Required so RunPod's nvidia container
    # runtime exposes the host GPU into our container. The nvidia/cuda base
    # images set these automatically; python:3.11-slim doesn't. Without them
    # torch.cuda.is_available() returns False even on a GPU pod, and the
    # first call to torch.cuda.current_device() throws:
    #   RuntimeError: CUDA unknown error ... Setting the available devices
    #   to be zero.
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ---------------------------------------------------------------------------
# System deps. ffmpeg for mp4 I/O via imageio; git for cloning Wan2.2;
# build-essential covers a few pip wheels (sam2 has some C extensions but
# they build optionally - we tolerate failures, just like Replicate does per
# their log line "cannot import name '_C' from 'sam2'"). libsm6/libxext6/libgl1
# are runtime deps for opencv-python.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates \
        ffmpeg libsm6 libxext6 libgl1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip \
    && echo "Python: $(python --version) | Pip: $(pip --version)"

# App files live at /app, NOT /workspace, because RunPod's default Pod
# template mounts the Network Volume at /workspace and would overlay
# (hide) our files. /app is conventional for application code and
# RunPod doesn't touch it.
WORKDIR /app

# ---------------------------------------------------------------------------
# Clone the official Wan-Video repo at a pinned commit. This is where
# wan.WanAnimate and wan.modules.animate.preprocess.process_pipepline live.
#
# Pinning is critical - the repo evolves and we don't want a future commit
# to break our server.py without warning. Update WAN_REVISION deliberately
# after testing.
# ---------------------------------------------------------------------------
ARG WAN_REVISION=main
RUN git clone --depth 1 --branch ${WAN_REVISION} \
        https://github.com/Wan-Video/Wan2.2.git /opt/wan22 \
    && echo "Pinned Wan-Video commit: $(cd /opt/wan22 && git rev-parse HEAD)"

# Install torch FIRST with an explicit CUDA 12.1 index URL.
#
# Why cu121 instead of cu124: PyTorch's cu124 wheel for 2.4.1 has a
# stricter runtime driver check - it needs NVIDIA driver build
# >= 550.54.15 (reports as CUDA 12.4.1 or 12050). Many RunPod Community
# Cloud hosts still run driver 550.54.14 (reports 12.4.0 / 12040), and
# torch refuses to initialize CUDA with the message "The NVIDIA driver
# on your system is too old (found version 12040)".
#
# The cu121 wheel is compiled against CUDA 12.1, requires driver >= 525,
# and works fine on any host with CUDA 12.x support. Performance loss for
# inference is <5%.
#
# Wan's requirements.txt says `torch>=2.4.0` without specifying the
# index, which would pull the CPU-only build from PyPI. Pre-installing
# the CUDA-built wheel satisfies the constraint and the subsequent
# `pip install -r /opt/wan22/requirements.txt` is a no-op for torch.
RUN pip install --no-cache-dir \
        torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu121

# Install Wan-Video's own deps, but strip flash_attn first.
#
# flash_attn compiles from source by default and requires the CUDA
# toolkit (nvcc + headers), which lives in the `-devel` base images but
# NOT in our `-runtime` base. Building from source on a GHA runner also
# typically exceeds the runner's time/disk budget. We strip it from the
# install list and accept the perf hit of standard attention - the model
# still runs, just slower. Production deploys can install a pre-built
# wheel later (https://github.com/Dao-AILab/flash-attention/releases) if
# the speed matters.
RUN sed -i -E '/^flash[_-]?attn/d' /opt/wan22/requirements.txt \
    && echo "[Dockerfile] Wan requirements.txt after stripping flash_attn:" \
    && cat /opt/wan22/requirements.txt \
    && pip install --no-cache-dir -r /opt/wan22/requirements.txt

# Then our extra runtime deps (FastAPI + Meta's SAM2).
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Wan2.2 is not installed via pip - import works because the repo lives on
# sys.path. Add it explicitly to avoid PYTHONPATH gotchas at runtime.
ENV PYTHONPATH=/opt/wan22:${PYTHONPATH}

# ---------------------------------------------------------------------------
# Application files. Located at /app to keep them outside any volume
# mount point a host might set up.
# ---------------------------------------------------------------------------
COPY server.py /app/server.py
COPY setup_model.py /app/setup_model.py
COPY entrypoint.sh /app/entrypoint.sh
# Normalize line endings in case the file was edited on Windows and
# committed with CRLF; the shebang `#!/usr/bin/env bash\r` would otherwise
# cause `exec: bash\r: not found` at container start.
RUN sed -i 's/\r$//' /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

# ---------------------------------------------------------------------------
# Runtime config defaults. These are overridable at `docker run` / RunPod
# endpoint creation time.
#
# WAN_CKPT_DIR defaults to the same Network Volume layout the main replicate-
# local app uses, so the same volume can mount at /runpod-volume and this
# server picks up the existing Wan 2.2 Animate weights at
# /runpod-volume/ComfyUI/models/diffusion_models/wan2.2_animate_14B_bf16.safetensors
# without re-downloading.
#
# OUTPUT_DIR is where generated mp4s land. Pointing at /runpod-volume/output
# (the same path the main worker uses) means the existing downloader endpoint
# can fetch them too, no extra plumbing.
# ---------------------------------------------------------------------------
ENV WAN_CKPT_DIR=/runpod-volume/ComfyUI/models \
    OUTPUT_DIR=/runpod-volume/output/wan-animate \
    SAM2_CKPT_DIR=/runpod-volume/ComfyUI/models/sam2 \
    PORT=8000

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
