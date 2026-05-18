#!/usr/bin/env bash
#
# One-time setup for running Wan 2.2 Animate (character-replace mode)
# inference inside a RunPod Pod, without building our own Docker image.
#
# This is the PRIMARY path. We tried the Docker route (see Dockerfile in
# this directory) but pip's transitive resolver kept upgrading torch to
# versions whose CUDA needs newer drivers than any RunPod host provides.
# Using RunPod's stock pytorch template sidesteps that fight entirely:
# torch is already there at the right version for the host driver, and
# we install only the missing packages on top.
#
# ============================================================================
# Prereqs
# ============================================================================
# - A RunPod Pod deployed with this template:
#     runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# - GPU with 24 GB+ VRAM (RTX 4090 / A6000 / L40 / 6000 Ada / etc.)
# - Network Volume attached at /workspace with these files already
#   downloaded (see main README step 6):
#     /workspace/ComfyUI/models/diffusion_models/wan2.2_animate_14B_bf16.safetensors  (~33 GB)
#     /workspace/ComfyUI/models/vae/wan_2.1_vae.safetensors                            (~243 MB)
#     /workspace/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors   (~6.3 GB)
#     /workspace/ComfyUI/models/sam2/sam2_hiera_base_plus.safetensors                  (~310 MB)
# - HTTP port 8000 exposed
#
# ============================================================================
# Usage
# ============================================================================
# After SSH-ing or opening the Web Terminal into the Pod:
#
#   curl -fsSL https://raw.githubusercontent.com/kpashoks/replicate-clone/main/runpod-wan-animate/setup_wan_pod.sh | bash
#
# Or to inspect before running:
#
#   curl -O https://raw.githubusercontent.com/kpashoks/replicate-clone/main/runpod-wan-animate/setup_wan_pod.sh
#   bash setup_wan_pod.sh
#
# After this completes, start the server with:
#
#   /opt/wan-animate/start.sh
#
# Or in the background (survives terminal close):
#
#   nohup /opt/wan-animate/start.sh > /var/log/wan-animate.log 2>&1 &
#   tail -f /var/log/wan-animate.log
#
# ============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

log()  { echo -e "${GREEN}[setup]${RESET} $*"; }
warn() { echo -e "${YELLOW}[setup]${RESET} $*"; }
err()  { echo -e "${RED}[setup]${RESET} $*"; }

# ============================================================================
# 0. Sanity checks — GPU + torch + weights all present
# ============================================================================
log "Checking pre-installed environment..."
python3 - <<'PY'
import sys
import torch
assert torch.cuda.is_available(), \
    "torch.cuda.is_available() is False — is this a GPU pod with the right template?"
print(f"  Python: {sys.version.split()[0]}")
print(f"  Torch:  {torch.__version__}")
print(f"  CUDA:   {torch.version.cuda}")
print(f"  Device: {torch.cuda.get_device_name(0)}")
PY

TORCH_BEFORE=$(python3 -c "import torch; print(torch.__version__)")
log "Will preserve torch at: ${TORCH_BEFORE}"
log ""

log "Verifying model weights on Network Volume..."
WEIGHTS=(
    "/workspace/ComfyUI/models/diffusion_models/wan2.2_animate_14B_bf16.safetensors"
    "/workspace/ComfyUI/models/vae/wan_2.1_vae.safetensors"
    "/workspace/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    "/workspace/ComfyUI/models/sam2/sam2_hiera_base_plus.safetensors"
)
missing=0
for f in "${WEIGHTS[@]}"; do
    if [ -f "$f" ]; then
        size=$(du -h "$f" | cut -f1)
        log "  OK: $f ($size)"
    else
        err "  MISSING: $f"
        missing=$((missing + 1))
    fi
done
if [ "$missing" -gt 0 ]; then
    err "$missing required weight file(s) missing. Run the bootstrap from main README first."
    exit 1
fi
log ""

# ============================================================================
# 1. Clone Wan-Video/Wan2.2 (idempotent)
# ============================================================================
if [ ! -d /opt/wan22 ]; then
    log "Cloning Wan-Video/Wan2.2 to /opt/wan22..."
    git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git /opt/wan22
    log "  Pinned commit: $(cd /opt/wan22 && git rev-parse HEAD | cut -c1-12)"
else
    log "/opt/wan22 already exists, skipping clone."
fi
log ""

# ============================================================================
# 2. Install Python dependencies — pinned to versions compatible with
#    the pre-installed torch. This is the part that the Docker approach
#    kept getting wrong (pip kept upgrading torch transitively).
# ============================================================================
log "Installing FastAPI server deps + common Python libs..."
# These have no torch dependency, safe to install with deps. We install
# common transitive deps (pyyaml, regex, requests, Pillow) explicitly so
# the later --no-deps installs of diffusers/transformers/accelerate don't
# fail at import time looking for them.
pip install --no-cache-dir --quiet \
    'fastapi>=0.115' \
    'uvicorn[standard]>=0.32' \
    'pydantic>=2.9' \
    'python-multipart>=0.0.12' \
    'httpx>=0.27' \
    'imageio>=2.36' \
    'imageio-ffmpeg>=0.5' \
    'opencv-python-headless>=4.9' \
    'easydict' 'ftfy' 'tqdm' \
    'safetensors>=0.4.5' \
    'huggingface_hub>=0.30,<1.0' \
    'numpy<2' \
    'pyyaml>=6' 'regex' 'requests>=2.31' 'Pillow>=10' \
    'filelock' 'psutil' 'packaging' \
    'einops' 'omegaconf' \
    'decord' 'moviepy' 'librosa' 'soundfile' 'scipy'
log "  Done."
log ""

log "Installing Wan-Video model deps (pinned versions, no torch upgrade)..."
# Versions chosen at the MINIMUM end of what Wan accepts. These are the
# versions Wan tested against and which don't pull a transitive torch
# upgrade. Using --no-deps as belt-and-suspenders.
#
# tokenizers: transformers 4.51.3 requires tokenizers>=0.21,<0.22 (NOT
# the older 0.20.3 we tried previously). We install tokenizers in the
# correct range; the package has no torch dependency so no upgrade risk.
pip install --no-cache-dir --no-deps --quiet \
    'diffusers==0.31.0' \
    'transformers==4.51.3' \
    'accelerate==1.1.1' \
    'tokenizers>=0.21,<0.22'
log "  Done."
log ""

log "Installing Meta SAM2 (pinned commit known to work with torch 2.4)..."
# SAM2's main branch now requires torch>=2.5.1, which would force a
# torch upgrade. Pin to a commit from Oct 2024 (still torch>=2.3.1).
# This commit hash is what Build #3 in the Docker pipeline successfully
# installed before SAM2 bumped its torch requirement.
SAM2_COMMIT=2b90b9f5ceec907a1c18123530e92e794ad901a4
pip install --no-cache-dir --no-deps --quiet \
    "sam-2 @ git+https://github.com/facebookresearch/sam2.git@${SAM2_COMMIT}" \
    || warn "SAM2 install failed - inference will fail unless this is resolved"

# SAM2's --no-deps install skips hydra-core + iopath which it imports at
# runtime. Both are pure-Python with no torch dep - safe to install with
# normal resolution.
pip install --no-cache-dir --quiet hydra-core iopath
log "  Done."
log ""

# Optional / non-critical
pip install --no-cache-dir --quiet dashscope 2>/dev/null || warn "dashscope skipped (not critical)"

# ============================================================================
# 3. Verify torch wasn't upgraded
# ============================================================================
TORCH_AFTER=$(python3 -c "import torch; print(torch.__version__)")
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
    err "============================================================="
    err "TORCH WAS UPGRADED from $TORCH_BEFORE to $TORCH_AFTER!"
    err "One of the pip install steps pulled torch transitively despite"
    err "the --no-deps flags. Edit this script to add more explicit"
    err "version pins, or use --no-deps on the offending package."
    err "============================================================="
    exit 1
fi
log "Torch preserved at: ${TORCH_AFTER}"
log ""

# Quick smoke test - can we import the things we need?
log "Smoke-testing imports..."
python3 - <<'PY'
import sys

problems = []

def try_import(name, attr=None):
    try:
        m = __import__(name, fromlist=[attr] if attr else [])
        if attr:
            getattr(m, attr)
        print(f"  OK: {name}" + (f".{attr}" if attr else ""))
        return True
    except Exception as e:
        print(f"  FAIL: {name}: {type(e).__name__}: {e}")
        problems.append(name)
        return False

# Add Wan-Video to path
sys.path.insert(0, "/opt/wan22")

try_import("torch")
try_import("torch.cuda")
try_import("diffusers")
try_import("transformers")
try_import("accelerate")
try_import("sam2")
try_import("sam2.sam2_video_predictor")
try_import("fastapi")
try_import("uvicorn")
try_import("imageio")
try_import("wan")
try_import("wan.configs")

if problems:
    print(f"\n  Problems: {problems}")
    sys.exit(1)
PY
log "  All imports OK."
log ""

# ============================================================================
# 4. Drop server.py into /opt/wan-animate
# ============================================================================
log "Installing server.py from GitHub..."
mkdir -p /opt/wan-animate
curl -fsSL \
    "https://raw.githubusercontent.com/kpashoks/replicate-clone/main/runpod-wan-animate/server.py" \
    -o /opt/wan-animate/server.py
log "  /opt/wan-animate/server.py installed."
log ""

# ============================================================================
# 5. Write a start.sh with env vars + uvicorn invocation
# ============================================================================
log "Writing /opt/wan-animate/start.sh..."
cat > /opt/wan-animate/start.sh <<'SCRIPT'
#!/usr/bin/env bash
# Start the wan-animate FastAPI server.
# Generated by setup_wan_pod.sh.

set -e

# Point server.py at the Network Volume layout (Pod template mounts the
# volume at /workspace, NOT /runpod-volume).
export WAN_REPO=/opt/wan22
export WAN_CKPT_DIR=/workspace/ComfyUI/models
export SAM2_CKPT_DIR=/workspace/ComfyUI/models/sam2
export OUTPUT_DIR=/workspace/output/wan-animate
export TMP_DIR=/tmp/wan-animate-tmp
export JOBS_DIR=/tmp/wan-animate-jobs
export PYTHONPATH=/opt/wan22:${PYTHONPATH:-}

# Make sure runtime directories exist
mkdir -p "${OUTPUT_DIR}" "${TMP_DIR}" "${JOBS_DIR}"

# Diagnostic print so we can confirm the version + paths at startup
python3 -c "
import torch, sys
print(f'[start.sh] Python {sys.version.split()[0]} | torch {torch.__version__} | CUDA {torch.version.cuda} | CUDA OK: {torch.cuda.is_available()}')
"

cd /opt/wan-animate
exec uvicorn server:app --host 0.0.0.0 --port 8000 --log-level info --workers 1
SCRIPT
chmod +x /opt/wan-animate/start.sh
log "  /opt/wan-animate/start.sh installed (executable)."
log ""

# Write update_server.sh — pulls a fresh server.py from main and restarts
# uvicorn. Use this when iterating on server.py without re-running the
# full setup.
log "Writing /opt/wan-animate/update_server.sh..."
cat > /opt/wan-animate/update_server.sh <<'SCRIPT'
#!/usr/bin/env bash
# Pull a fresh server.py from this repo's main branch and restart uvicorn.
# Useful for fast iteration during debugging - no need to re-run the
# whole setup script.

set -e

echo "[update] Fetching latest server.py from GitHub..."
curl -fsSL \
    "https://raw.githubusercontent.com/kpashoks/replicate-clone/main/runpod-wan-animate/server.py" \
    -o /opt/wan-animate/server.py.new
mv /opt/wan-animate/server.py.new /opt/wan-animate/server.py
echo "[update] server.py updated."

echo "[update] Stopping any running uvicorn..."
pkill -f 'uvicorn.*server:app' || echo "[update]   (no uvicorn was running)"

# Brief wait so the port is released
sleep 2

echo "[update] Restarting in background..."
nohup /opt/wan-animate/start.sh > /var/log/wan-animate.log 2>&1 &
sleep 3
echo "[update] Done. Tail logs with: tail -f /var/log/wan-animate.log"
SCRIPT
chmod +x /opt/wan-animate/update_server.sh
log "  /opt/wan-animate/update_server.sh installed (executable)."
log ""

# ============================================================================
# 6. Done — show next steps
# ============================================================================
cat <<EOF

==========================================================
${GREEN}Setup complete.${RESET}
==========================================================

To START the wan-animate server in the foreground (you'll see logs
directly in the terminal, but it dies when you close the SSH session):

  /opt/wan-animate/start.sh

To run in the BACKGROUND (survives terminal close):

  nohup /opt/wan-animate/start.sh > /var/log/wan-animate.log 2>&1 &
  tail -f /var/log/wan-animate.log

To verify the server is up, hit these URLs in your browser
(replace <pod-id> with this Pod's ID):

  https://<pod-id>-8000.proxy.runpod.net/health
  https://<pod-id>-8000.proxy.runpod.net/debug/info

In particular /debug/info should show:
  "torch": "${TORCH_AFTER}",
  "cuda_available": true,
  "wan_repo_exists": true,
  "sam2_path": "..."

THEN: copy that base URL into your local app's .env as:
  WAN_ANIMATE_ENDPOINT=https://<pod-id>-8000.proxy.runpod.net

Restart your local uvicorn and submit a character-swap test.

If you later need to update server.py from main without re-running
the full setup:

  /opt/wan-animate/update_server.sh

==========================================================
EOF
