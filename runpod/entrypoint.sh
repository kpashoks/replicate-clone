#!/usr/bin/env bash
# Runtime wrapper that ensures /runpod-volume/output exists before the
# worker-comfyui handler starts.
#
# Background: install_custom_nodes.sh symlinks /comfyui/output to
# /runpod-volume/output at image build time. At build time the volume isn't
# mounted, so the symlink target doesn't exist (it's a dangling link). At
# runtime, /runpod-volume gets mounted but the /output subdir on the volume
# may not exist yet. When ComfyUI's SaveImage / VHS_VideoCombine calls
# os.makedirs('/comfyui/output', exist_ok=True), it follows the symlink and
# tries to mkdir the (nonexistent) target. Python raises:
#
#   [Errno 17] File exists: '/comfyui/output/'
#
# ...because the symlink LINK file exists, but path.isdir() returns False
# for a dangling link, so the exist_ok shortcut doesn't apply.
#
# Fix: ensure the target directory exists before ComfyUI runs.

set -e

mkdir -p /runpod-volume/output 2>/dev/null || \
  echo "[entrypoint] WARNING: could not mkdir /runpod-volume/output (volume not mounted?)"

# Defer to the base image's startup. If Docker passed a CMD (i.e. the base
# image declared one), use it. Otherwise fall back to the worker-comfyui
# convention (/start.sh) so we don't exec with empty args and silently exit 0
# (which causes RunPod to crash-loop the worker).
if [ "$#" -gt 0 ]; then
  echo "[entrypoint] exec'ing CMD: $*"
  exec "$@"
elif [ -x /start.sh ]; then
  echo "[entrypoint] no CMD passed; falling back to /start.sh"
  exec /start.sh
else
  echo "[entrypoint] ERROR: no CMD args and no /start.sh - cannot proceed"
  exit 1
fi
