#!/usr/bin/env bash
# One-time download of FLUX.1 [dev] weights onto a RunPod Network Volume.
#
# Run from a temporary RunPod Pod (a cheap CPU pod is fine, ~$0.04/hr) with
# the Network Volume mounted at /runpod-volume.
#
# Prerequisites:
#   1. Accept the FLUX.1-dev license on the model page:
#      https://huggingface.co/black-forest-labs/FLUX.1-dev
#   2. Create a HuggingFace token (read scope) at:
#      https://huggingface.co/settings/tokens
#   3. SSH/web-terminal into the pod, then:
#        export HF_TOKEN=hf_xxxxxxxxxx
#        bash bootstrap_volume.sh
#
# Expected runtime: ~30-60 minutes (depends on RunPod region bandwidth).
# Final disk usage: ~59 GB
#   flux1-dev fp16          ~24 GB
#   flux1-kontext-dev fp16  ~24 GB
#   t5xxl_fp16              ~9.5 GB
#   clip_l                  ~250 MB
#   ae (FLUX VAE)           ~335 MB

set -euo pipefail

# Mount path:
#   - Serverless endpoints mount Network Volumes at /runpod-volume (locked).
#   - Pods default to /workspace, but you can override the mount path at deploy time.
# This script writes to $VOLUME_ROOT (default /runpod-volume/ComfyUI). If your
# Pod mounted the volume at /workspace instead, run with:
#   export VOLUME_ROOT=/workspace/ComfyUI
VOLUME_ROOT="${VOLUME_ROOT:-/runpod-volume/ComfyUI}"
VOLUME_MOUNT="$(dirname "$VOLUME_ROOT")"

if [ ! -d "$VOLUME_MOUNT" ]; then
  echo "ERROR: $VOLUME_MOUNT does not exist."
  echo "Either:"
  echo "  - the Network Volume isn't attached to this Pod, or"
  echo "  - it's mounted at a different path."
  echo ""
  echo "Check what's mounted: 'mount | grep -i volume' or 'ls /workspace /runpod-volume 2>/dev/null'"
  echo "Then either re-deploy the Pod with mount path /runpod-volume, or:"
  echo "  export VOLUME_ROOT=<your-mount>/ComfyUI"
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN environment variable must be set."
  echo "  1. Get a token (read scope) at https://huggingface.co/settings/tokens"
  echo "  2. Accept the FLUX.1-dev license at https://huggingface.co/black-forest-labs/FLUX.1-dev"
  echo "  3. export HF_TOKEN=hf_xxxxxxxxxx"
  exit 1
fi

echo "=== Creating directory structure under $VOLUME_ROOT ==="
mkdir -p \
  "$VOLUME_ROOT/models/diffusion_models" \
  "$VOLUME_ROOT/models/text_encoders" \
  "$VOLUME_ROOT/models/vae" \
  "$VOLUME_ROOT/models/loras" \
  "$VOLUME_ROOT/models/upscale_models" \
  "$VOLUME_ROOT/models/clip_vision" \
  "$VOLUME_ROOT/models/controlnet" \
  "$VOLUME_ROOT/custom_nodes"

echo "=== Installing huggingface_hub CLI ==="
pip install -q "huggingface_hub[cli]"

echo "=== Authenticating with HuggingFace ==="
# Note: the CLI was renamed from `huggingface-cli` to `hf` in late 2025.
# `huggingface_hub[cli]>=0.26` ships `hf`. If you're on an older Pod image
# with only `huggingface-cli`, swap `hf auth login` -> `huggingface-cli login`
# and `hf download` -> `huggingface-cli download` below.
hf auth login --token "$HF_TOKEN" --add-to-git-credential

echo ""
echo "=== [1/5] Downloading FLUX.1 [dev] checkpoint (~24 GB) ==="
hf download black-forest-labs/FLUX.1-dev flux1-dev.safetensors \
  --local-dir "$VOLUME_ROOT/models/diffusion_models"

echo ""
echo "=== [2/5] Downloading FLUX.1 Kontext [dev] checkpoint (~24 GB) ==="
# Also gated under FLUX.1 Non-Commercial. Accept the license at
# https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev before this runs.
hf download black-forest-labs/FLUX.1-Kontext-dev flux1-kontext-dev.safetensors \
  --local-dir "$VOLUME_ROOT/models/diffusion_models"

echo ""
echo "=== [3/5] Downloading FLUX VAE (~335 MB) ==="
# Note: the VAE is identical between FLUX.1-dev and FLUX.1-schnell. The
# schnell repo is ungated, so we pull from there to avoid the gated download.
hf download black-forest-labs/FLUX.1-schnell ae.safetensors \
  --local-dir "$VOLUME_ROOT/models/vae"

echo ""
echo "=== [4/5] Downloading T5-XXL fp16 text encoder (~9.5 GB) ==="
hf download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors \
  --local-dir "$VOLUME_ROOT/models/text_encoders"

echo ""
echo "=== [5/5] Downloading CLIP-L text encoder (~250 MB) ==="
hf download comfyanonymous/flux_text_encoders clip_l.safetensors \
  --local-dir "$VOLUME_ROOT/models/text_encoders"

echo ""
echo "=== Bootstrap complete. Volume contents: ==="
du -sh "$VOLUME_ROOT"/models/* 2>/dev/null || true
echo ""
echo "Top-level structure:"
ls -la "$VOLUME_ROOT/models/"
echo ""
echo "Done. You can now terminate this pod (the volume persists)."
