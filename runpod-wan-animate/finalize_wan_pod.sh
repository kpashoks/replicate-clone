#!/usr/bin/env bash
#
# Finalize a partially-completed Wan-animate Pod setup.
#
# Run this AFTER setup_wan_pod.sh has been attempted but failed at the
# smoke-test stage due to missing deps. This script:
#
#   1. Installs ALL packages Wan-Video might need (comprehensive
#      whack-a-mole prevention)
#   2. Re-verifies imports
#   3. Downloads server.py if missing
#   4. Writes start.sh + update_server.sh if missing
#   5. Starts uvicorn in the background
#   6. Prints how to verify
#
# Idempotent - safe to run multiple times. Won't disturb torch (verifies
# at the end and fails loudly if torch was upgraded).
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/kpashoks/replicate-clone/main/runpod-wan-animate/finalize_wan_pod.sh | bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

log()  { echo -e "${GREEN}[finalize]${RESET} $*"; }
warn() { echo -e "${YELLOW}[finalize]${RESET} $*"; }
err()  { echo -e "${RED}[finalize]${RESET} $*"; }

# ============================================================================
# 0. Sanity: torch state pre-install
# ============================================================================
TORCH_BEFORE=$(python3 -c "import torch; print(torch.__version__)")
log "Torch BEFORE install: ${TORCH_BEFORE}"

# ============================================================================
# 1. Install every dep we know Wan-Video, SAM2, transformers, or
#    diffusers might import. Pure-Python and pre-built wheels only -
#    nothing here has a torch dep.
# ============================================================================
log "Installing comprehensive dep bundle..."

# Core (no-deps, with strict version pins to avoid transitive torch
# upgrades from latest accelerate/transformers/diffusers)
pip install --no-cache-dir --no-deps --quiet \
    'diffusers==0.31.0' \
    'transformers==4.51.3' \
    'accelerate==1.1.1' \
    'tokenizers>=0.21,<0.22' \
    'huggingface_hub>=0.30,<1.0'

# Second-tier deps (these have NO torch dep, install with normal resolution)
pip install --no-cache-dir --quiet \
    'fastapi>=0.115' \
    'uvicorn[standard]>=0.32' \
    'pydantic>=2.9' \
    'python-multipart>=0.0.12' \
    'httpx>=0.27' \
    'imageio>=2.36' 'imageio-ffmpeg>=0.5' \
    'opencv-python-headless>=4.9' \
    'easydict' 'ftfy' 'tqdm' \
    'safetensors>=0.4.5' \
    'numpy<2' \
    'pyyaml>=6' 'regex' 'requests>=2.31' 'Pillow>=10' \
    'filelock' 'psutil' 'packaging' \
    'einops' 'omegaconf' \
    'hydra-core' 'iopath' \
    'decord' 'moviepy' 'librosa' 'soundfile' 'scipy' \
    'matplotlib' 'pandas' 'scikit-image' 'av' \
    'protobuf' 'sentencepiece' \
    'peft' 'loguru' \
    'onnxruntime-gpu'

# Optional - dashscope (Alibaba SDK) can fail; not critical
pip install --no-cache-dir --quiet dashscope 2>/dev/null || warn "dashscope skipped"

# SAM2 (already installed if setup_wan_pod.sh ran; idempotent here)
if ! python3 -c "import sam2" 2>/dev/null; then
    log "Installing Meta SAM2 (pinned Oct 2024 commit)..."
    SAM2_COMMIT=2b90b9f5ceec907a1c18123530e92e794ad901a4
    pip install --no-cache-dir --no-deps --quiet \
        "sam-2 @ git+https://github.com/facebookresearch/sam2.git@${SAM2_COMMIT}"
else
    log "SAM2 already installed, skipping."
fi

log "  Done."
log ""

# Patch Wan-Video's preprocess module to make the FluxKontextPipeline
# import optional. We use `use_flux=False` so we never need it, but
# diffusers==0.31.0 doesn't have it and the unconditional `from diffusers
# import FluxKontextPipeline` at module load time crashes the pipeline
# load otherwise.
log "Patching FluxKontextPipeline import to be optional..."
python3 - <<'PYPATCH' || warn "Patch failed (maybe Wan moved that import)"
import os
path = '/opt/wan22/wan/modules/animate/preprocess/process_pipepline.py'
if not os.path.exists(path):
    print(f"  SKIP: {path} not found")
else:
    with open(path) as f:
        content = f.read()
    old = 'from diffusers import FluxKontextPipeline'
    new = ('try:\n'
           '    from diffusers import FluxKontextPipeline\n'
           'except ImportError:\n'
           '    FluxKontextPipeline = None  # optional - needed only when use_flux=True')
    if old in content:
        with open(path, 'w') as f:
            f.write(content.replace(old, new))
        print("  OK: patched")
    elif 'except ImportError' in content and 'FluxKontextPipeline' in content:
        print("  SKIP: already patched")
    else:
        print("  WARN: FluxKontextPipeline import line not found - file may have changed upstream")
PYPATCH
log ""

# ============================================================================
# 2. Verify torch wasn't upgraded
# ============================================================================
TORCH_AFTER=$(python3 -c "import torch; print(torch.__version__)")
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
    err "TORCH WAS UPGRADED from $TORCH_BEFORE to $TORCH_AFTER"
    err "One of the pip installs pulled torch transitively. Edit script."
    exit 1
fi
log "Torch preserved at: ${TORCH_AFTER}"
log ""

# ============================================================================
# 3. Smoke-test imports - if any fail, list them but continue (start.sh
#    will show the precise error)
# ============================================================================
log "Smoke-testing Wan imports..."
python3 - <<'PY' || warn "Smoke test had failures - server may fail on import. See above and add the missing pkg."
import sys
sys.path.insert(0, "/opt/wan22")
problems = []
for name in [
    'torch', 'torch.cuda',
    'transformers', 'diffusers', 'accelerate',
    'sam2', 'sam2.sam2_video_predictor',
    'fastapi', 'uvicorn', 'imageio',
    'wan', 'wan.configs',
]:
    try:
        __import__(name)
        print(f"  OK: {name}")
    except Exception as e:
        print(f"  FAIL: {name}: {type(e).__name__}: {e}")
        problems.append(name)
if problems:
    print(f"\n  Still failing: {problems}")
    print("  (continuing anyway - start.sh will surface the real error)")
    sys.exit(1)
PY
log ""

# ============================================================================
# 4. Wan-Video repo present? Clone if not.
# ============================================================================
if [ ! -d /opt/wan22 ]; then
    log "Cloning Wan-Video/Wan2.2..."
    git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git /opt/wan22
fi

# ============================================================================
# 5. Download server.py + write start.sh + write update_server.sh
#    (idempotent - always pulls latest from main)
# ============================================================================
log "Downloading server.py from GitHub..."
mkdir -p /opt/wan-animate
curl -fsSL \
    "https://raw.githubusercontent.com/kpashoks/replicate-clone/main/runpod-wan-animate/server.py" \
    -o /opt/wan-animate/server.py

log "Writing start.sh..."
cat > /opt/wan-animate/start.sh <<'SCRIPT'
#!/usr/bin/env bash
set -e
export WAN_REPO=/opt/wan22
export WAN_CKPT_DIR=/workspace/ComfyUI/models
export SAM2_CKPT_DIR=/workspace/ComfyUI/models/sam2
export OUTPUT_DIR=/workspace/output/wan-animate
export TMP_DIR=/tmp/wan-animate-tmp
export JOBS_DIR=/tmp/wan-animate-jobs
export PYTHONPATH=/opt/wan22:${PYTHONPATH:-}
mkdir -p "${OUTPUT_DIR}" "${TMP_DIR}" "${JOBS_DIR}"
python3 -c "import torch, sys; print(f'[start.sh] Python {sys.version.split()[0]} | torch {torch.__version__} | CUDA {torch.version.cuda} | CUDA OK: {torch.cuda.is_available()}')"
cd /opt/wan-animate
exec uvicorn server:app --host 0.0.0.0 --port 8000 --log-level info --workers 1
SCRIPT
chmod +x /opt/wan-animate/start.sh

log "Writing update_server.sh..."
cat > /opt/wan-animate/update_server.sh <<'SCRIPT'
#!/usr/bin/env bash
set -e
curl -fsSL \
    "https://raw.githubusercontent.com/kpashoks/replicate-clone/main/runpod-wan-animate/server.py" \
    -o /opt/wan-animate/server.py.new
mv /opt/wan-animate/server.py.new /opt/wan-animate/server.py
pkill -f 'uvicorn.*server:app' || true
sleep 2
nohup /opt/wan-animate/start.sh > /var/log/wan-animate.log 2>&1 &
sleep 3
echo "[update] server restarted. Tail logs with: tail -f /var/log/wan-animate.log"
SCRIPT
chmod +x /opt/wan-animate/update_server.sh
log ""

# ============================================================================
# 6. Stop any previously-running uvicorn and start fresh in background
# ============================================================================
log "Stopping any previous uvicorn..."
pkill -f 'uvicorn.*server:app' 2>/dev/null && sleep 2 || log "  (none running)"

log "Starting uvicorn in background..."
nohup /opt/wan-animate/start.sh > /var/log/wan-animate.log 2>&1 &
sleep 5  # give uvicorn a moment to boot or crash

# Check if uvicorn actually started
if pgrep -f 'uvicorn.*server:app' > /dev/null; then
    log "  uvicorn running."
else
    err "uvicorn FAILED to start. Last 30 lines of log:"
    tail -30 /var/log/wan-animate.log
    exit 1
fi

# ============================================================================
# 7. Verify the server responds locally
# ============================================================================
log ""
log "Verifying /health endpoint..."
sleep 2
if curl -fsS http://localhost:8000/health > /dev/null 2>&1; then
    log "  Server responding on http://localhost:8000"
else
    warn "  Server did not respond yet - may still be starting. Check /var/log/wan-animate.log"
fi

log ""
log "Verifying /debug/info..."
DEBUG_INFO=$(curl -fsS http://localhost:8000/debug/info 2>/dev/null || echo '{}')
echo "$DEBUG_INFO" | python3 -m json.tool 2>/dev/null || echo "$DEBUG_INFO"

# ============================================================================
# 8. Done
# ============================================================================
cat <<EOF

==========================================================
${GREEN}Finalize complete.${RESET}
==========================================================

To watch the server logs in real time:
  tail -f /var/log/wan-animate.log

To restart the server after editing /opt/wan-animate/server.py:
  pkill -f 'uvicorn.*server:app'
  nohup /opt/wan-animate/start.sh > /var/log/wan-animate.log 2>&1 &

To pull a fresh server.py from GitHub and restart:
  /opt/wan-animate/update_server.sh

Get the Pod's public URL from the Connect tab (port 8000 HTTP service).
Then in your LOCAL .env, set:
  WAN_ANIMATE_ENDPOINT=https://<pod-id>-8000.proxy.runpod.net

Restart local uvicorn and submit a character-swap test.

==========================================================
EOF
