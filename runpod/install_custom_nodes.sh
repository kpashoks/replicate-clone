#!/usr/bin/env bash
# Custom-node installer (runs at Docker build time).
#
# Currently tracks the default branch of each repo. Once we've verified a
# working M4 baseline, pin each clone to a specific commit SHA so workflow
# JSONs don't break when upstream nodes change. To pin: replace
#   git clone <url>
# with
#   git clone <url> && (cd <dir> && git checkout <sha>)

set -euo pipefail

# -----------------------------------------------------------------------------
# Redirect ComfyUI's output directory to the Network Volume
# -----------------------------------------------------------------------------
# By default ComfyUI writes outputs to /comfyui/output, which is on the
# ephemeral container disk and gets reaped with the worker. Files that
# worker-comfyui doesn't bubble back via the response (e.g., mp4 from
# VHS_VideoCombine) would be lost. By symlinking /comfyui/output to a
# directory on the Network Volume, those files persist after the worker
# exits and can be fetched by a separate downloader endpoint.
# /runpod-volume isn't mounted at build time, but the symlink target doesn't
# need to exist at creation time; it resolves at runtime when the volume
# mounts. ComfyUI's first SaveImage call creates the target dir on demand.
echo "[install_custom_nodes] symlinking /comfyui/output -> /runpod-volume/output"
rm -rf /comfyui/output
ln -s /runpod-volume/output /comfyui/output

CUSTOM_NODES_DIR="/comfyui/custom_nodes"
mkdir -p "$CUSTOM_NODES_DIR"
cd "$CUSTOM_NODES_DIR"

# -----------------------------------------------------------------------------
# M4 - Wan 2.2 Animate dependencies
# -----------------------------------------------------------------------------

# Kijai's utility nodes (Points Editor, image manipulation helpers used by the
# native Wan 2.2 Animate workflow).
echo "[install_custom_nodes] cloning ComfyUI-KJNodes..."
git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git

# Video I/O: VHS_LoadVideo (source video input) and VHS_VideoCombine
# (assemble output frames into mp4).
echo "[install_custom_nodes] cloning ComfyUI-VideoHelperSuite..."
git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git

# DWPose Estimator for extracting pose skeletons from the source video. Wan
# Animate uses these to transfer motion to the reference character.
echo "[install_custom_nodes] cloning comfyui_controlnet_aux..."
git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux.git

# GGUF loading - used if you swap the bf16 Wan checkpoint for a GGUF quant
# from QuantStack/Wan2.2-Animate-14B-GGUF (smaller, fits in less VRAM).
echo "[install_custom_nodes] cloning ComfyUI-GGUF..."
git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git

# SAM2 (Segment Anything 2) — produces the character_mask input required for
# Wan 2.2 Animate's "Mix mode" (character replacement). Without this, the
# swap reverts to the source character after the first frame because the
# model has no mask telling it which region to overwrite. Required SAM2
# weights live on the Network Volume at
# /runpod-volume/ComfyUI/models/sam2/sam2_hiera_base_plus.safetensors (see
# bootstrap_volume.sh / README for download instructions).
echo "[install_custom_nodes] cloning ComfyUI-segment-anything-2..."
git clone --depth 1 https://github.com/kijai/ComfyUI-segment-anything-2.git

# -----------------------------------------------------------------------------
# M3.5 - Image Character Swap (Juggernaut + IPAdapter + ControlNet OpenPose)
# -----------------------------------------------------------------------------

# cubiq's IPAdapter Plus suite - SDXL identity transfer via image conditioning.
echo "[install_custom_nodes] cloning ComfyUI_IPAdapter_plus..."
git clone --depth 1 https://github.com/cubiq/ComfyUI_IPAdapter_plus.git

# -----------------------------------------------------------------------------
# Install any Python deps the nodes need.
# Pip failures are non-fatal so an optional dep doesn't kill the whole build.
# -----------------------------------------------------------------------------

for d in ComfyUI-KJNodes ComfyUI-VideoHelperSuite comfyui_controlnet_aux ComfyUI-GGUF ComfyUI_IPAdapter_plus ComfyUI-segment-anything-2; do
  if [ -f "$CUSTOM_NODES_DIR/$d/requirements.txt" ]; then
    echo "[install_custom_nodes] pip install for $d..."
    pip install --no-cache-dir -r "$CUSTOM_NODES_DIR/$d/requirements.txt" || \
      echo "[install_custom_nodes] WARNING: $d requirements partial; continuing"
  fi
done

echo "[install_custom_nodes] done. Installed:"
ls -d "$CUSTOM_NODES_DIR"/*/ 2>/dev/null
