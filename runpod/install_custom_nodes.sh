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

# -----------------------------------------------------------------------------
# Install any Python deps the nodes need.
# Pip failures are non-fatal so an optional dep doesn't kill the whole build.
# -----------------------------------------------------------------------------

for d in ComfyUI-KJNodes ComfyUI-VideoHelperSuite comfyui_controlnet_aux ComfyUI-GGUF; do
  if [ -f "$CUSTOM_NODES_DIR/$d/requirements.txt" ]; then
    echo "[install_custom_nodes] pip install for $d..."
    pip install --no-cache-dir -r "$CUSTOM_NODES_DIR/$d/requirements.txt" || \
      echo "[install_custom_nodes] WARNING: $d requirements partial; continuing"
  fi
done

echo "[install_custom_nodes] done. Installed:"
ls -d "$CUSTOM_NODES_DIR"/*/ 2>/dev/null
