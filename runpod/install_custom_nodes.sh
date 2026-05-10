#!/usr/bin/env bash
# Custom-node installer (runs at Docker build time).
#
# For M2 (FLUX.1 [dev] text-to-image) no custom nodes are needed.
# Add nodes for later milestones below, pinning each clone to a specific
# git SHA so workflow JSONs don't break when upstream nodes change.
#
# Example for M4 (Wan 2.2 Animate):
#
#   set -euo pipefail
#   cd /comfyui/custom_nodes
#
#   git clone https://github.com/city96/ComfyUI-GGUF
#   (cd ComfyUI-GGUF && git checkout <pinned-sha>)
#
#   git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
#   (cd ComfyUI-VideoHelperSuite && git checkout <pinned-sha>)
#
#   git clone https://github.com/kijai/ComfyUI-KJNodes
#   (cd ComfyUI-KJNodes && git checkout <pinned-sha>)
#
#   # Wan 2.2 Animate community implementation:
#   git clone https://github.com/<...>/ComfyUI-WanAnimate
#   (cd ComfyUI-WanAnimate && git checkout <pinned-sha>)

set -euo pipefail

echo "install_custom_nodes.sh: nothing to install for M2 (FLUX text-to-image)"
