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
# Expected runtime (full bootstrap): ~70-100 minutes.
# Final disk usage: ~105 GB
#   flux1-dev fp16                       ~24 GB
#   flux1-kontext-dev fp16               ~24 GB
#   wan2.2_animate_14B bf16              ~28 GB
#   juggernaut-XL v9 (SDXL all-in-one)   ~6.6 GB
#   umt5_xxl fp8 (Wan)                   ~5 GB
#   t5xxl_fp16 (FLUX)                    ~9.5 GB
#   wan_2.1_vae                          ~500 MB
#   clip_vision_h                        ~1.3 GB
#   clip_l                               ~250 MB
#   ae (FLUX VAE)                        ~335 MB
#
# The script is idempotent: re-running skips files that already exist on disk.
# If your volume already has FLUX assets (M2/M3 bootstrap done), this run only
# adds the new Wan-related downloads (~36 GB).

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
  "$VOLUME_ROOT/models/checkpoints" \
  "$VOLUME_ROOT/models/diffusion_models" \
  "$VOLUME_ROOT/models/text_encoders" \
  "$VOLUME_ROOT/models/vae" \
  "$VOLUME_ROOT/models/loras" \
  "$VOLUME_ROOT/models/upscale_models" \
  "$VOLUME_ROOT/models/clip_vision" \
  "$VOLUME_ROOT/models/controlnet" \
  "$VOLUME_ROOT/custom_nodes" \
  "$VOLUME_ROOT/input" \
  "$VOLUME_ROOT/output"

# Route huggingface_hub's cache to the Network Volume. Otherwise it writes to
# the container disk (typically <60 GB) and a 28 GB Wan checkpoint can fill
# it before the download even finishes ("No space left on device, os error 28").
VOLUME_MOUNT="$(dirname "$VOLUME_ROOT")"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$VOLUME_MOUNT/hf_cache}"
mkdir -p "$HF_HUB_CACHE"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"

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
echo "=== [5/9] Downloading CLIP-L text encoder (~250 MB) ==="
hf download comfyanonymous/flux_text_encoders clip_l.safetensors \
  --local-dir "$VOLUME_ROOT/models/text_encoders"

echo ""
echo "=== [6/9] Downloading Wan 2.2 Animate 14B bf16 checkpoint (~28 GB) ==="
# Comfy-Org's repackaged variant - ungated, sized for native ComfyUI workflows.
# If you want a smaller quantized version instead, swap this for a GGUF from
# QuantStack/Wan2.2-Animate-14B-GGUF (e.g. Wan2.2-Animate-14B-Q6_K.gguf at ~12 GB).
hf download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
  split_files/diffusion_models/wan2.2_animate_14B_bf16.safetensors \
  --local-dir "$VOLUME_ROOT/models/diffusion_models"
# Move it out of the nested split_files/diffusion_models/ subdir into the
# top-level diffusion_models/ dir, where the workflow expects it.
if [ -f "$VOLUME_ROOT/models/diffusion_models/split_files/diffusion_models/wan2.2_animate_14B_bf16.safetensors" ]; then
  mv "$VOLUME_ROOT/models/diffusion_models/split_files/diffusion_models/wan2.2_animate_14B_bf16.safetensors" \
     "$VOLUME_ROOT/models/diffusion_models/"
  rm -rf "$VOLUME_ROOT/models/diffusion_models/split_files"
fi

echo ""
echo "=== [7/9] Downloading Wan 2.1 VAE (~500 MB) ==="
hf download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
  split_files/vae/wan_2.1_vae.safetensors \
  --local-dir "$VOLUME_ROOT/models/vae"
if [ -f "$VOLUME_ROOT/models/vae/split_files/vae/wan_2.1_vae.safetensors" ]; then
  mv "$VOLUME_ROOT/models/vae/split_files/vae/wan_2.1_vae.safetensors" \
     "$VOLUME_ROOT/models/vae/"
  rm -rf "$VOLUME_ROOT/models/vae/split_files"
fi

echo ""
echo "=== [8/9] Downloading UMT5-XXL fp8 text encoder for Wan (~5 GB) ==="
hf download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
  split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
  --local-dir "$VOLUME_ROOT/models/text_encoders"
if [ -f "$VOLUME_ROOT/models/text_encoders/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" ]; then
  mv "$VOLUME_ROOT/models/text_encoders/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
     "$VOLUME_ROOT/models/text_encoders/"
  rm -rf "$VOLUME_ROOT/models/text_encoders/split_files"
fi

echo ""
echo "=== [9/10] Downloading CLIP Vision H for Wan (~1.3 GB) ==="
# CLIP Vision lives in the Wan 2.1 repackage, not 2.2 (the 2.2 repackage has
# no clip_vision subdir; the file is shared across Wan versions).
hf download Comfy-Org/Wan_2.1_ComfyUI_repackaged \
  split_files/clip_vision/clip_vision_h.safetensors \
  --local-dir "$VOLUME_ROOT/models/clip_vision"
if [ -f "$VOLUME_ROOT/models/clip_vision/split_files/clip_vision/clip_vision_h.safetensors" ]; then
  mv "$VOLUME_ROOT/models/clip_vision/split_files/clip_vision/clip_vision_h.safetensors" \
     "$VOLUME_ROOT/models/clip_vision/"
  rm -rf "$VOLUME_ROOT/models/clip_vision/split_files"
fi

echo ""
echo "=== [10/10] Downloading Juggernaut XL v9 SDXL checkpoint (~6.6 GB) ==="
# Single-file SDXL checkpoint (includes UNet + dual CLIP + VAE). Ungated.
hf download RunDiffusion/Juggernaut-XL-v9 Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors \
  --local-dir "$VOLUME_ROOT/models/checkpoints"

echo ""
echo "=== Bootstrap complete. Volume contents: ==="
du -sh "$VOLUME_ROOT"/models/* 2>/dev/null || true
echo ""
echo "Top-level structure:"
ls -la "$VOLUME_ROOT/models/"
echo ""
echo "Done. You can now terminate this pod (the volume persists)."
