# RunPod runbook (M2: FLUX.1 [dev] text-to-image)

This directory holds everything that runs on RunPod: the worker `Dockerfile`,
the volume bootstrap script, ComfyUI workflow JSONs, and this runbook.

The architecture is one Serverless ComfyUI endpoint backed by one Network
Volume that holds all model weights. The image is small and rebuilds quickly;
weights live on the volume so cold starts don't redownload them.

---

## One-time setup

You need a RunPod account with API key and a HuggingFace account that has
accepted the FLUX.1-dev license.

### 1. Accept the FLUX.1-dev license on HuggingFace

Open https://huggingface.co/black-forest-labs/FLUX.1-dev while logged in and
click "Agree and access repository". (FLUX.1 [dev] is gated behind a click-
through license; you must accept it before the bootstrap script can download.)

### 2. Create a HuggingFace token

https://huggingface.co/settings/tokens → "Create new token" with **read**
scope. Save the `hf_xxxxxxxxxx` value somewhere — you'll need it during
bootstrap.

### 3. Create a Network Volume on RunPod

RunPod console → Storage → New Network Volume.

- **Size: 100 GB** (M2 uses ~35 GB; the rest is headroom for M3 Kontext
  and M4 Wan 2.2 Animate.)
- **Region:** pick the one closest to you. **Important: this is locked.**
  Migrating a populated volume to another region means redownloading
  ~35–75 GB. Pick once and stick with it.
- **Name:** something like `replicate-local-models`.

Note the volume ID — you'll attach it to a Pod next, then to the Serverless
endpoint.

---

## Bootstrap the volume (download FLUX weights once)

### 4. Spin up a temporary Pod with the volume attached

RunPod console → Pods → Deploy. Cheapest option works:

- Template: **RunPod PyTorch** (any recent version, just need `pip` and
  `huggingface-cli`)
- GPU: **none** (CPU pod). If a CPU-only option isn't visible in your account,
  pick the cheapest GPU and ignore it — bootstrap is just downloads.
- **Network Volume: select the one you just created**, mounted at
  `/runpod-volume`.
- Container disk: 10 GB is plenty.

Wait ~30 s for the pod to be Running, then click "Connect" → "Web Terminal".

### 5. Run the bootstrap script

In the web terminal:

```bash
# Pull the bootstrap script from your repo. Easiest: paste it directly.
# (Or git clone the repo if you've pushed it.)
cat > /tmp/bootstrap_volume.sh <<'EOF'
# ... paste the contents of replicate-local/runpod/bootstrap_volume.sh ...
EOF

export HF_TOKEN=hf_xxxxxxxxxx   # the token from step 2
bash /tmp/bootstrap_volume.sh
```

Or git clone if you've pushed the repo:

```bash
cd /tmp && git clone https://github.com/<you>/replicate-local
export HF_TOKEN=hf_xxxxxxxxxx
bash replicate-local/runpod/bootstrap_volume.sh
```

Expect 20–40 minutes of downloading. When you see `Bootstrap complete`,
**terminate the pod** (the volume persists). The pod itself only cost a few
cents.

---

## Build & deploy the Serverless endpoint

### 6. Build the worker image

You have two options. Pick one.

**Option A — let RunPod build from GitHub (easiest):**

1. Push your local `replicate-local` repo to GitHub (private is fine).
2. RunPod console → Serverless → New Endpoint → "Build from GitHub".
3. Repo: yours. Branch: `main` (or wherever the `runpod/Dockerfile` lives).
4. Build context path: `runpod/`.
5. Dockerfile path: `runpod/Dockerfile`.

**Option B — build & push locally:**

```bash
cd replicate-local/runpod
docker build -t <youruser>/replicate-local-worker:0.1 .
docker push <youruser>/replicate-local-worker:0.1
```

Then in the RunPod console, point the endpoint at that image tag.

### 7. Create the Serverless endpoint

RunPod console → Serverless → New Endpoint.

- **Name:** `replicate-local`
- **Image:** the one you built in step 6
- **Network Volume:** attach the bootstrapped one. Mount path:
  `/runpod-volume` (this is the default; do not change).
- **GPU pool:** **RTX 4090 (24 GB)** to start. Cheapest 24 GB SKU; fits
  FLUX.1 [dev] fp16 with headroom. If quality demands more later
  (especially for Wan 2.2 Animate at fp16 in M4), switch the pool to
  **L40 48 GB** — same volume, same image, no other changes.
- **Active workers: 0** (scale to zero)
- **Max workers: 1** (personal use; bump later if needed)
- **Idle timeout:** 60 seconds
- **FlashBoot:** **on** (keeps a warm container snapshot for fast warm starts)
- **Execution timeout:** 600 seconds (FLUX gens take ~20–60 s; long videos
  later need more headroom)

Save → copy the **endpoint ID** from the endpoint detail page.

### 8. Wire the FastAPI backend

Back on your laptop:

```bash
cd C:\Users\kpash\projects\replicate-local
# .env  (copy from .env.example if you haven't)
RUNPOD_API_KEY=<your runpod api key>
RUNPOD_ENDPOINT_ID=<endpoint id from step 7>
RUNPOD_TIMEOUT_SECONDS=600
# (the rest stays as in .env.example)
```

Restart `uvicorn` so the new env is picked up:

```bash
cd backend
.venv\Scripts\activate
uvicorn main:app --reload --port 8000
```

---

## End-to-end smoke test

```bash
# 1. Submit a generation
curl -X POST http://localhost:8000/api/generate/text-to-image ^
  -H "Content-Type: application/json" ^
  -d "{\"params\":{\"prompt\":\"a red panda eating ramen, cinematic, photorealistic\",\"steps\":20,\"width\":1024,\"height\":1024}}"
# -> {"job_id":"abc123def456","status":"queued"}

# 2. Poll the job
curl http://localhost:8000/api/jobs/abc123def456
# -> {"id":"...","status":"running","runpod_status":"IN_PROGRESS",...}

# 3. After ~30-90 s the status becomes "succeeded" with output_files populated:
# {
#   "id":"abc123def456",
#   "status":"succeeded",
#   "output_files":["/api/files/outputs/abc123def456/text_to_image_abc12_00001_.png"],
#   ...
# }

# 4. View the image
# Open http://localhost:8000<output_file_url> in your browser.
```

If it works, M2 is done — you have an end-to-end FLUX text-to-image pipeline
running through your local FastAPI orchestrator. The frontend (Next.js
gallery + model page) comes in M2.5 / next session.

---

## Troubleshooting

- **`RunPod submit failed [401]`** — `RUNPOD_API_KEY` is wrong or unset.
- **`RunPod submit failed [404]`** — `RUNPOD_ENDPOINT_ID` is wrong, or the
  endpoint doesn't exist yet.
- **Job sits at `queued` for >2 min then fails** — the worker can't pull the
  image. Check the endpoint's "Logs" tab in the RunPod console.
- **Job runs but fails with "model not found"** — the bootstrap script
  didn't finish or the volume isn't mounted at `/runpod-volume`. Check
  the endpoint's volume attachment and the worker logs.
- **`No decodable outputs in RunPod response`** — open the `Logs` tab on
  the RunPod endpoint dashboard and inspect the worker's response. The
  workflow may have errored mid-execution (e.g., OOM on a too-large
  resolution); check ComfyUI logs for the real error.
- **Worker keeps OOMing on FLUX.1 [dev] fp16** — switch the endpoint's GPU
  pool to L40 48 GB, or swap the workflow's `weight_dtype` to
  `"fp8_e4m3fn"` for the FLUX UNet (saves ~50% VRAM).

## Cost expectations

- Bootstrap pod: ~$0.05 (CPU pod for ~30 min).
- Serverless endpoint: 4090 is roughly $0.00040/sec while running. A 30-second
  FLUX generation costs around $0.012. Idle costs $0 (active workers = 0).
- Soft cap: set a monthly budget alert in the RunPod billing dashboard.
