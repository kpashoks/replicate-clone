#!/usr/bin/env bash
# Entrypoint for the wan-animate server.
#
# Two duties:
#   1. Make sure the output directory exists (RunPod mounts /runpod-volume at
#      container start, but won't auto-create subdirs).
#   2. Hand off to uvicorn, which loads server.py.
#
# We don't preload the model here - server.py does it lazily on the first
# request so the container starts fast (good for autoscaling).

set -euo pipefail

# Move into the app dir so `uvicorn server:app` finds server.py without
# having to qualify the module path. The Dockerfile WORKDIR also sets
# this, but a `cd` here is robust even if a host overrides the working
# directory (e.g. via docker run -w).
cd /app

OUTPUT_DIR="${OUTPUT_DIR:-/runpod-volume/output/wan-animate}"
mkdir -p "${OUTPUT_DIR}" 2>/dev/null || \
    echo "[entrypoint] WARNING: could not mkdir ${OUTPUT_DIR} (volume not mounted?)"

# Print Python / uvicorn versions on startup so we can verify the right
# interpreter is in use without exec'ing into the container.
echo "[entrypoint] $(python --version 2>&1) | $(uvicorn --version 2>&1)"

# RunPod Serverless workers don't expose a port - they communicate over the
# RunPod queue. If we're deployed as a Pod (long-running, exposes a port)
# we run uvicorn. The two modes are distinguished by RUNPOD_REALTIME_TASK or
# by an explicit MODE env var.
#
# For now we always start uvicorn. To run on Serverless later, add a
# runpod-python handler in a separate entrypoint.

exec uvicorn server:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --log-level info \
    --workers 1
