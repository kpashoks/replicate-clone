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

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# System deps. ffmpeg is needed by imageio for mp4 I/O; git for cloning Wan2.2;
# build-essential covers a few pip wheels (sam2 has some C extensions but they
# build optionally - we tolerate failures, just like Replicate does per their
# log line "cannot import name '_C' from 'sam2'").
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3-pip \
        git wget curl ca-certificates \
        ffmpeg libsm6 libxext6 libgl1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && python -m pip install --upgrade pip

WORKDIR /workspace

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

# Install Wan-Video's own deps first (torch, diffusers, transformers, etc.).
# If any single line fails, we want to know - so no `|| true` here.
RUN pip install --no-cache-dir -r /opt/wan22/requirements.txt

# Then our extra runtime deps (FastAPI + Meta's SAM2).
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Wan2.2 is not installed via pip - import works because the repo lives on
# sys.path. Add it explicitly to avoid PYTHONPATH gotchas at runtime.
ENV PYTHONPATH=/opt/wan22:${PYTHONPATH}

# ---------------------------------------------------------------------------
# Application files.
# ---------------------------------------------------------------------------
COPY server.py /workspace/server.py
COPY setup_model.py /workspace/setup_model.py
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

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

ENTRYPOINT ["/workspace/entrypoint.sh"]
