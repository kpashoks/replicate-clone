# replicate-local

A local-first Replicate-style multimodal AI playground. The web UI and FastAPI orchestrator run on your laptop; heavy image / video models run on a RunPod Serverless ComfyUI endpoint backed by a single Network Volume. Prompt enhancement runs locally on your GPU via Qwen3-4B (Transformers + bitsandbytes 4-bit).

## Capabilities

| Slug | Model | Type |
|---|---|---|
| `text-to-image` | FLUX.1 [dev] | text → image |
| `juggernaut-xl` | Juggernaut XL v9 (SDXL fine-tune) | text → image |
| `image-edit` | FLUX.1 Kontext [dev] | image + text → edited image |
| `image-char-swap` | Juggernaut XL + IP-Adapter + ControlNet OpenPose | source image + reference identity → identity-swapped image |
| `character-swap` | Wan 2.2 Animate | source video + reference identity → video with swapped character |

Plus a local **prompt-enhancement** feature (Qwen3-4B) that rewrites short prompts into more detailed diffusion-friendly versions, accessible via an "Enhance ✨" button on every model page.

**License caveat:** FLUX.1 [dev] and FLUX.1 Kontext [dev] use the FLUX.1 Non-Commercial License. Fine for personal use; do not expose this service publicly without a commercial license from Black Forest Labs.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Your laptop                                │
│                                                                  │
│   Browser ──► Next.js 15 (3000) ──► FastAPI (8000)              │
│                                          │                       │
│                                          ├─► Qwen3-4B (local GPU)│
│                                          │                       │
│                                          ├─► data/inputs/        │
│                                          ├─► data/outputs/       │
│                                          ├─► data/jobs/          │
│                                          │                       │
└──────────────────────────────────────────┼───────────────────────┘
                                           │
                                           │  HTTPS
                                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                         RunPod                                   │
│                                                                  │
│   Serverless Endpoint A:                                         │
│   ┌──────────────────────────────────────┐                       │
│   │  worker-comfyui :5.8.5-base          │                       │
│   │  + custom nodes (IPAdapter,          │                       │
│   │    KJNodes, VideoHelperSuite,        │                       │
│   │    controlnet_aux, GGUF)             │                       │
│   │  + extra_model_paths.yaml            │◄─┐                    │
│   │  + entrypoint.sh (mkdir output)      │  │                    │
│   └──────────────────────────────────────┘  │                    │
│                                              │ mounts             │
│   Serverless Endpoint B (downloader):        │                    │
│   ┌──────────────────────────────────────┐  │                    │
│   │  Tiny CPU Python handler             │◄─┤                    │
│   │  Reads /runpod-volume/output/        │  │                    │
│   │  Returns base64 in response          │  │                    │
│   └──────────────────────────────────────┘  │                    │
│                                              │                    │
│   ┌──────────────────────────────────────┐  │                    │
│   │  Network Volume (100 GB):            │──┘                    │
│   │  - ComfyUI/models/{checkpoints,…}    │                       │
│   │  - output/<prefix>_*.mp4 (videos)    │                       │
│   └──────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

Two RunPod Serverless endpoints:
- **Main** (GPU): runs ComfyUI. Built from `runpod/Dockerfile` via GitHub Actions, pushed to ghcr.io.
- **Downloader** (CPU): fetches non-image outputs (mp4) from the Network Volume. worker-comfyui doesn't return video files inline, so we pull them back via this companion endpoint. Built from `runpod-downloader/Dockerfile`.

---

## Prerequisites

- **Windows / macOS / Linux** with **Python 3.12** (we use bitsandbytes which has best wheel support on 3.12).
- **NVIDIA GPU** with ≥ 6 GB VRAM and a recent CUDA driver. Tested on RTX 4060 Laptop 8 GB + CUDA 12.7 driver (uses cu124 PyTorch wheels via forward compat).
- **Node.js 20+** for the frontend.
- **Docker Desktop** *(optional — only if you want to build worker images locally instead of letting GitHub Actions do it).*
- **A GitHub account.** The worker images are built by GitHub Actions and pushed to GHCR.
- **A RunPod account** with billing enabled. Cost expectation for personal use: ~$0.05–0.30 per image gen on RTX A6000/L40 48GB; ~$0.30–1 per video gen.
- **A HuggingFace account** with an access token (Read scope) and acceptance of the FLUX.1-dev + FLUX.1-Kontext-dev licenses (gated).

---

## From-scratch setup

End-to-end you should budget ~2–3 hours, mostly waiting for model downloads.

### 1. Clone the repo

```
git clone https://github.com/<your-fork>/replicate-clone.git
cd replicate-clone
```

### 2. Set up the local backend

```
cd backend
py -3.12 -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux
python -m pip install --upgrade pip
```

Install PyTorch with CUDA wheels (PyPI's default torch is CPU-only):

```
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

(Adjust `cu124` to match your driver — `cu121`/`cu125` etc. The cu124 wheels work with any 12.x driver via forward compatibility.)

Then the rest:

```
pip install -r requirements.txt
```

Qwen3-4B-Instruct weights (~8 GB) download automatically into `%USERPROFILE%\.cache\huggingface\hub` the first time you hit `/api/prompts/enhance`. Don't worry about it yet.

### 3. Set up the frontend

```
cd ../frontend
npm install
```

### 4. Push the repo to GitHub & enable GitHub Actions

The worker images are built by GitHub Actions. Push your local repo to a private GitHub repository, then visit the **Actions** tab and enable workflows if prompted.

After push, two workflows run automatically on every push to `main` that touches `runpod/**` or `runpod-downloader/**`:

- **Build and push worker image** → `ghcr.io/<owner>/replicate-clone:latest` (the heavy ComfyUI worker, ~12 GB)
- **Build and push downloader image** → `ghcr.io/<owner>/replicate-clone-downloader:latest` (tiny CPU helper)

**Make both packages public** so RunPod can pull them without credentials:

1. Go to `https://github.com/<owner>?tab=packages`
2. Click each package → **Package settings** → bottom of page → **Change visibility** → **Public**

### 5. Create the RunPod Network Volume

RunPod console → Storage → **+ Network Volume**.

- **Size:** 200 GB (the full M1–M4 model set is ~108 GB; leave headroom)
- **Region:** anywhere with good GPU supply; **this is permanent** — volumes can't move regions. Tested in US-KS-2.
- **Name:** anything, e.g. `replicate-local-models`

### 6. Bootstrap the volume with model weights

The volume needs to be pre-populated with all the model files. This is a one-time process that uses a temporary CPU pod (cost ≪ $1).

Visit https://huggingface.co/settings/tokens and create a Read-scope token. Then accept the licenses (browse each repo while signed in and click "Agree"):

- https://huggingface.co/black-forest-labs/FLUX.1-dev
- https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev

Spin up a temp pod:

- RunPod → Pods → Deploy → template `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **CPU pool** if available (cheapest); otherwise any cheap GPU
- **Attach** the Network Volume at `/workspace`
- **Container disk:** 50 GB minimum (the HF cache grows during downloads)
- Deploy → open the **Web Terminal**

In the web terminal:

```
pip install -q "huggingface_hub[cli]"
export HF_TOKEN=hf_xxxxxxxxxx
export VOLUME_ROOT=/workspace/ComfyUI
export HF_HUB_CACHE=/workspace/hf_cache
hf auth login --token "$HF_TOKEN" --add-to-git-credential
mkdir -p $VOLUME_ROOT/models/{checkpoints,diffusion_models,text_encoders,vae,clip_vision,controlnet,ipadapter,loras,upscale_models}
mkdir -p /workspace/output
```

Then download all the models. Each `hf download` block is independent — run them one at a time and verify each completes:

```bash
# FLUX.1 [dev] (~24 GB)
hf download black-forest-labs/FLUX.1-dev flux1-dev.safetensors --local-dir $VOLUME_ROOT/models/diffusion_models

# FLUX.1 Kontext [dev] (~24 GB)
hf download black-forest-labs/FLUX.1-Kontext-dev flux1-kontext-dev.safetensors --local-dir $VOLUME_ROOT/models/diffusion_models

# FLUX VAE (~335 MB, same file across FLUX models, sourced from ungated schnell repo)
hf download black-forest-labs/FLUX.1-schnell ae.safetensors --local-dir $VOLUME_ROOT/models/vae

# T5-XXL text encoder for FLUX (~9.5 GB)
hf download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --local-dir $VOLUME_ROOT/models/text_encoders

# CLIP-L text encoder for FLUX (~250 MB)
hf download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir $VOLUME_ROOT/models/text_encoders

# Wan 2.2 Animate 14B bf16 (~28 GB)
hf download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/diffusion_models/wan2.2_animate_14B_bf16.safetensors --local-dir $VOLUME_ROOT/models/diffusion_models
mv $VOLUME_ROOT/models/diffusion_models/split_files/diffusion_models/wan2.2_animate_14B_bf16.safetensors $VOLUME_ROOT/models/diffusion_models/
rm -rf $VOLUME_ROOT/models/diffusion_models/split_files

# Wan 2.1 VAE (~500 MB)
hf download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/vae/wan_2.1_vae.safetensors --local-dir $VOLUME_ROOT/models/vae
mv $VOLUME_ROOT/models/vae/split_files/vae/wan_2.1_vae.safetensors $VOLUME_ROOT/models/vae/
rm -rf $VOLUME_ROOT/models/vae/split_files

# UMT5-XXL fp8 text encoder for Wan (~5 GB)
hf download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors --local-dir $VOLUME_ROOT/models/text_encoders
mv $VOLUME_ROOT/models/text_encoders/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors $VOLUME_ROOT/models/text_encoders/
rm -rf $VOLUME_ROOT/models/text_encoders/split_files

# CLIP Vision H (~1.3 GB — lives in the Wan 2.1 repackage, not 2.2)
hf download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/clip_vision/clip_vision_h.safetensors --local-dir $VOLUME_ROOT/models/clip_vision
mv $VOLUME_ROOT/models/clip_vision/split_files/clip_vision/clip_vision_h.safetensors $VOLUME_ROOT/models/clip_vision/
rm -rf $VOLUME_ROOT/models/clip_vision/split_files

# Juggernaut XL v9 (~6.6 GB SDXL all-in-one)
hf download RunDiffusion/Juggernaut-XL-v9 Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors --local-dir $VOLUME_ROOT/models/checkpoints

# IP-Adapter Plus SDXL (~700 MB)
hf download h94/IP-Adapter sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors --local-dir $VOLUME_ROOT/models/ipadapter
mv $VOLUME_ROOT/models/ipadapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors $VOLUME_ROOT/models/ipadapter/
rm -rf $VOLUME_ROOT/models/ipadapter/sdxl_models

# ControlNet OpenPose SDXL (~2.5 GB, renamed for clarity)
hf download xinsir/controlnet-openpose-sdxl-1.0 diffusion_pytorch_model.safetensors --local-dir $VOLUME_ROOT/models/controlnet/_tmp_openpose
mv $VOLUME_ROOT/models/controlnet/_tmp_openpose/diffusion_pytorch_model.safetensors $VOLUME_ROOT/models/controlnet/controlnet-openpose-sdxl-xinsir.safetensors
rm -rf $VOLUME_ROOT/models/controlnet/_tmp_openpose
```

Verify the final state (~108 GB total):

```
du -sh $VOLUME_ROOT/models/* /workspace/output
```

You should see filled `checkpoints/`, `diffusion_models/`, `text_encoders/`, `vae/`, `clip_vision/`, `ipadapter/`, `controlnet/`. **Terminate the temp pod** when done.

### 7. Create the main Serverless endpoint

RunPod console → Serverless → **+ New Endpoint**:

- **Source:** Deploy from Docker registry → image `ghcr.io/<owner>/replicate-clone:latest`
- **Name:** anything, e.g. `replicate-main`
- **Endpoint type:** Queue
- **Worker type:** GPU
- **GPU configuration:** **48 GB** as 1st priority (RTX A6000 or L40 — both work for FLUX, Kontext, Wan). Add 80 GB as 2nd fallback for resilience.
- **Max workers:** 1, **Active workers:** 0
- **Idle timeout:** 60 s, **Execution timeout:** **1500 s** ⚠️ (Wan video gens can run 10+ min)
- **FlashBoot:** ON
- **Container disk:** 20 GB
- **Advanced → Network Volume:** attach the volume created in step 5. Mount path locks to `/runpod-volume`.
- Deploy. Copy the **Endpoint ID**.

### 8. Create the downloader Serverless endpoint

Same flow, much smaller:

- **Image:** `ghcr.io/<owner>/replicate-clone-downloader:latest`
- **Name:** `replicate-downloader`
- **Worker type:** **CPU** (cheapest tier — no GPU needed)
- **Max workers:** 1, **Active workers:** 0
- **Idle timeout:** 30 s, **Execution timeout:** 120 s
- **Container disk:** 5 GB
- **Attach the same Network Volume** (same datacenter as main endpoint — that's why region matters in step 5)
- Deploy. Copy the **Endpoint ID**.

### 9. Create your RunPod API key

Top-right account menu → Settings → API Keys → **+ Create API Key**. User role is fine. Copy the `rpa_…` token.

### 10. Configure `.env`

Copy the example and fill in:

```
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
```

Edit `.env`:

```
RUNPOD_API_KEY=rpa_xxxxxxxxxx
RUNPOD_ENDPOINT_ID=<main endpoint ID from step 7>
RUNPOD_DOWNLOADER_ENDPOINT_ID=<downloader endpoint ID from step 8>
RUNPOD_TIMEOUT_SECONDS=600

QWEN_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507
DATA_DIR=./data
FRONTEND_URL=http://localhost:3000
```

### 11. Start the dev servers

In two separate terminals:

**Terminal A — backend:**
```
cd backend
.venv\Scripts\activate
uvicorn main:app --port 8000 --reload
```

**Terminal B — frontend:**
```
cd frontend
npm run dev
```

Open **http://localhost:3000** in your browser. You'll see the model gallery.

### 12. Smoke-test each capability

| Model | Steps | Expected first-run time |
|---|---|---|
| **Text to Image (FLUX)** | Type a prompt, click Run | 60–90 s warm; ~3–5 min cold |
| **Juggernaut XL** | Type a prompt, click Run | 30–60 s warm; ~2–3 min cold |
| **Image Edit** | Upload an image, type edit instruction, click Run | 60–90 s warm; ~3–5 min cold |
| **Image Character Swap** | Upload source + reference character, click Run | 60–120 s warm; ~3–4 min cold |
| **Character Swap (Video)** | Upload video + character image, click Run | 8–15 min |
| **✨ Enhance** button (any page) | Type a short prompt, click Enhance | ~10–15 s first call (Qwen3 model load), ~7–9 s subsequent |

If everything works on first try, congratulations — that's unprecedented. If something fails, see **Troubleshooting** below.

---

## Day-to-day usage

After the one-time setup:

1. Open two terminals
2. `cd backend && .venv\Scripts\activate && uvicorn main:app --port 8000 --reload`
3. `cd frontend && npm run dev`
4. Browse to http://localhost:3000

When you're done, `Ctrl+C` in each terminal. RunPod endpoints with `Active workers = 0` automatically idle to $0/s after the idle timeout.

---

## Repo layout

```
backend/                 FastAPI orchestration
  main.py                  app entrypoint, CORS, route registration
  routes/
    generate.py            POST /api/generate/{slug}, GET /api/jobs/...
    files.py               GET /api/files/... (serves data/outputs/)
    uploads.py             POST /api/uploads (multipart, content-addressed)
    prompts.py             POST /api/prompts/enhance (Qwen3 local)
    models.py              GET /api/models
  prompt_enhance.py        local Qwen3-4B via Transformers + bitsandbytes
  runpod_client.py         async httpx wrapper for RunPod Serverless
  workflows.py             load + parameterize ComfyUI workflow JSONs
  jobs.py                  in-memory job registry + async runner
  storage.py               data/ dir mgmt
  models_registry.py       static metadata for each model slug
  requirements.txt
  .venv/                   gitignored

frontend/                Next.js 15 (App Router) + Tailwind + shadcn/ui
  app/
    page.tsx                 gallery (data-driven from /api/models)
    models/
      text-to-image/page.tsx
      juggernaut-xl/page.tsx
      image-edit/page.tsx
      image-char-swap/page.tsx
      character-swap/page.tsx
  components/
    EnhanceDiff.tsx
    ImageDropzone.tsx
    VideoDropzone.tsx
    ModelCard.tsx
    ui/                      shadcn primitives (base-ui based)
  lib/
    api.ts                   fetch wrappers
    types.ts                 Job, ModelEntry types

runpod/                  Main worker (ComfyUI + custom nodes)
  Dockerfile               FROM runpod/worker-comfyui:5.8.5-base
  install_custom_nodes.sh  KJNodes, VideoHelperSuite, controlnet_aux, GGUF,
                           IPAdapter_plus + output-dir symlink
  entrypoint.sh            mkdir /runpod-volume/output then exec base startup
  extra_model_paths.yaml   point ComfyUI at /runpod-volume/ComfyUI/models/
  bootstrap_volume.sh      reference download script (not used directly;
                           README step 6 is the source of truth)
  workflows/               ComfyUI API-format JSON per slug
    text2img_flux.json
    text2img_juggernaut.json
    imgedit_flux_kontext.json
    charswap_juggernaut.json
    video_swap_wan22_animate.json

runpod-downloader/       Downloader Serverless worker (CPU)
  Dockerfile               FROM python:3.11-slim + runpod SDK
  handler.py               reads /runpod-volume/output, returns base64
  README.md

.github/workflows/       GHA pipelines
  build-worker.yml         builds runpod/Dockerfile
  build-downloader.yml     builds runpod-downloader/Dockerfile

data/                    gitignored
  inputs/                  uploaded images/videos (content-addressed)
  outputs/<job_id>/        generated PNGs/MP4s
  jobs/<job_id>.json       job snapshots

.env                     gitignored - RunPod credentials, etc.
.env.example             template
```

---

## Troubleshooting

### Worker crash-loop (`worker exited with exit code 0` every ~17 s)

The image's ENTRYPOINT is failing immediately. Most often happens after a Dockerfile change. Check the build logs in the GHA workflow for that commit; verify `runpod/entrypoint.sh` falls back to `/start.sh` if `$@` is empty.

### Job stays `IN_QUEUE` for >5 min

GPU pool likely has no supply in your datacenter. Endpoint → Manage → GPU configuration → add a fallback (e.g. enable 80 GB as well as 48 GB).

### `Workflow validation failed: 'X.safetensors' not in []`

A model file isn't on the Network Volume at the path ComfyUI expects. Re-check step 6 — run the specific `hf download` command for the missing file. The `extra_model_paths.yaml` controls which subdirectories ComfyUI scans (`checkpoints/`, `diffusion_models/`, etc.).

### `No decodable outputs in RunPod response, output.status = 'success_no_images'`

The model produced a non-image output (video / gif) that worker-comfyui doesn't return in the response. Confirm:
- `RUNPOD_DOWNLOADER_ENDPOINT_ID` is set in `.env`
- The downloader endpoint exists and has the **same Network Volume attached**
- The main worker is on the **latest image** (with the `/comfyui/output → /runpod-volume/output` symlink). Terminate cached workers if you suspect they're stale.

### `[Errno 17] File exists: '/comfyui/output/'`

The symlink target `/runpod-volume/output/` doesn't exist on the volume. The `entrypoint.sh` should `mkdir -p` it at startup, but if you want a belt-and-suspenders fix, spin up a temp pod and run `mkdir -p /workspace/output` directly on the volume.

### Frontend sliders don't drag / form values won't update

The shadcn-generated Slider needs `index={i}` on each Thumb and the wrapper must normalize Base UI's scalar `onValueChange` callback to an array. If you've upgraded shadcn or regenerated `components/ui/slider.tsx`, re-apply the patches we made (see `git log frontend/components/ui/slider.tsx`).

### `TypeError: Failed to fetch` on upload

Browser-side network failure. Check:
- Backend running on port 8000
- File size under the 32 MB cap (frontend `VideoDropzone` enforces this client-side)
- DevTools Network tab for the actual `/api/uploads` request status

### GHA didn't rebuild after a worker-source change

Check `.github/workflows/build-worker.yml`'s `paths:` filter. Only these files trigger a rebuild:
- `runpod/Dockerfile`
- `runpod/install_custom_nodes.sh`
- `runpod/extra_model_paths.yaml`
- `runpod/entrypoint.sh`
- `.github/workflows/build-worker.yml` itself

If you edited a different file you care about, add it to the filter.

### Where to find logs

- **Backend (local):** stdout of the `uvicorn` terminal
- **Frontend (local):** stdout of `npm run dev` terminal + browser DevTools Console
- **RunPod worker:** Endpoint detail page → **Logs** tab (filter by request ID for a specific job)
- **GHA builds:** https://github.com/`<owner>`/replicate-clone/actions

---

## Cost expectations

Approximate per-generation cost on a 48 GB RTX A6000/L40 endpoint at ~$0.00034/s:

| Capability | Wall time | Cost per gen |
|---|---|---|
| Text to Image (FLUX) | 60–90 s | ~$0.02–0.03 |
| Juggernaut XL | 30–60 s | ~$0.01–0.02 |
| Image Edit (Kontext) | 60–90 s | ~$0.02–0.03 |
| Image Character Swap | 60–120 s | ~$0.02–0.04 |
| Character Swap (Video) | 8–15 min | ~$0.16–0.30 |
| Downloader fetch (per video) | 5–15 s | ~$0.00005 (CPU pricing) |

Plus storage: the 100–200 GB Network Volume runs ~$0.07/GB/month → $7–14/month if you keep all models. Endpoints with `Active = 0` cost $0 when idle.

---

## Architecture decision log

If you're curious why various things are the way they are, see `.claude/plans/` for the original implementation plan and individual commit messages for in-flight decisions (e.g. why we switched from llama-cpp-python to Transformers, why GHA→ghcr.io instead of RunPod's GitHub builder, why a separate downloader endpoint instead of S3).
