# Wan 2.2 Animate (Character Replace) inference server

A standalone HTTP server that runs Wan 2.2 Animate's **character replace** task
(Mix mode) using the official [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
Python API plus [Meta's SAM2](https://github.com/facebookresearch/sam2). This
is the same pipeline Replicate's `wan-video/wan-2.2-animate-replace` runs
under the hood (confirmed by their inference logs).

Why a dedicated server: we spent five rounds trying to make ComfyUI's
`WanAnimateToVideo` node do Mix-mode replacement and could not. Replicate's
prediction logs revealed they don't use ComfyUI — they call the Wan-Video
Python API directly. This server mirrors that pattern.

## Architecture

```
                                                ┌─ /workspace/server.py
Frontend ─▶ Backend (FastAPI on laptop)         │    POST /character-swap
              │                                  │    GET  /jobs/{id}
              │  HTTP (multipart upload)         │    GET  /jobs/{id}/output
              ▼                                  │
        RunPod Pod ──── wan-animate server ──────┘
              │
              ├─ Wan-Video/Wan2.2 (cloned at /opt/wan22)
              │     └─ wan.WanAnimate.generate(replace_flag=True)
              │
              ├─ facebookresearch/sam2 (pip install)
              │     └─ SAM2VideoPredictor.propagate_in_video()
              │
              └─ Network Volume (/runpod-volume)
                    └─ models/diffusion_models/wan2.2_animate_14B_bf16.safetensors
                    └─ models/sam2/sam2_hiera_base_plus.safetensors
                    └─ output/wan-animate/{job_id}.mp4  ◀── output
```

## What the pipeline does (matches Replicate's logs verbatim)

1. **Load source video** — `imageio` decodes the input into a frame batch
   (`video frame: 0%|..` to `100%|..` progress bar in Replicate logs).
2. **Build per-frame masks with SAM2 video predictor** —
   `SAM2VideoPredictor.propagate_in_video()` produces a binary mask per frame
   tracking the character (`propagate in video: 0%|..` progress bar).
3. **Run the Wan-Video preprocessing pipeline** — the upstream
   `wan.modules.animate.preprocess.process_pipepline` produces the five
   inputs Wan Animate actually needs:
     - `src_ref.png` — reference character image (resized to model resolution)
     - `src_face.mp4` — 512×512 face crop per frame (locks facial identity)
     - `src_pose.mp4` — pose sequence (DWPose under the hood)
     - `src_bg.mp4` — **background WITH the character region INPAINTED OUT** (this
       is the input we never had in our ComfyUI attempts — the model expects a
       scene with the character pixels removed, not the raw source video)
     - `src_mask.mp4` — binary character mask
4. **Run Wan Animate's two-stage diffusion** — `wan.WanAnimate.generate(replace_flag=True)`
   with `refert_num=77` and chunked sampling (~32 frames per chunk, two passes
   for low-noise + high-noise refinement).
5. **Encode mp4** and store at `/runpod-volume/output/wan-animate/<job_id>.mp4`.

## HTTP API

### `POST /character-swap`

Multipart form upload. Submit a job, get a job_id back.

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `character_image` | file | yes | — | PNG/JPG of the reference character |
| `source_video` | file | yes | — | MP4 of the source motion |
| `prompt` | string | no | `""` | Optional scene/style direction |
| `seed` | int | no | `-1` | `-1` = random |
| `resolution` | string | no | `"832x480"` | One of: `832x480`, `1280x720`, `1408x640`, `480x832`, `720x1280` |
| `replace_flag` | bool | no | `true` | `true` = Mix mode (preserve scene), `false` = Move mode |
| `sampling_steps` | int | no | `20` | Per-stage. Total = `2 * sampling_steps` |
| `frame_num` | int | no | `81` | ~5 s at 16 fps |
| `refert_num` | int | no | `77` | Temporal guidance frames per chunk |
| `guide_scale` | float | no | `5.0` | CFG |

Returns: `{ "job_id": "abc123", "status": "queued" }`

### `GET /jobs/{job_id}`

Returns current status:

```json
{
  "job_id": "abc123",
  "status": "queued" | "running" | "completed" | "failed",
  "progress_step": "preprocess" | "sample" | "encode",
  "output_path": "/runpod-volume/output/wan-animate/abc123.mp4",
  "error": null
}
```

### `GET /jobs/{job_id}/output`

Streams the output mp4 (only when `status == "completed"`).
Returns `409 Conflict` otherwise.

### `GET /health`

Liveness + readiness:

```json
{ "status": "ready" | "loading", "model_loaded": true }
```

## Building the image

The image is built and pushed by GitHub Actions, **manually triggered** (no
automatic build on push, to avoid spinning up expensive GHA runners every
time you tweak a comment). To build:

```
gh workflow run "Build wan-animate image" --ref main
```

Or via the GitHub UI: Actions tab → **Build wan-animate image** → **Run workflow**.

The image lands at `ghcr.io/<owner>/replicate-clone-wan-animate:latest`.

## Local build (smoke test only)

The image is large (~12 GB) and requires CUDA, but you can build it locally
to validate the Dockerfile + ensure deps resolve:

```
cd runpod-wan-animate
docker build -t wan-animate-local .
```

Running locally without a GPU is not supported — Wan 2.2 Animate 14B needs
~24 GB VRAM. Use the RunPod deploy instructions below for actual inference.

## Deploying to RunPod

### As a Pod (simplest, recommended for first deploy)

1. RunPod console → Pods → Deploy
2. Custom container:
   - Image: `ghcr.io/<owner>/replicate-clone-wan-animate:latest`
   - GPU: **24 GB minimum** (RTX 4090, RTX A6000, L40, or larger)
   - Network volume: attach the existing `replicate-local-models` volume at
     `/runpod-volume` (same volume that hosts FLUX, Juggernaut, Wan weights,
     and now also `sam2/sam2_hiera_base_plus.safetensors`).
   - Expose HTTP port `8000`.
   - Container disk: 30 GB (the image is ~12 GB, plus runtime tempfiles).
3. Once running, RunPod will give you a public URL like
   `https://<pod-id>-8000.proxy.runpod.net`.
4. Put that URL in the main app's `.env`:
   ```
   WAN_ANIMATE_ENDPOINT=https://<pod-id>-8000.proxy.runpod.net
   ```

### As a Serverless endpoint (later, when scale matters)

The current `server.py` exposes an HTTP API. To run on Serverless, wrap it
with a `runpod-python` handler. Out of scope for this initial deploy.

## Model weights required on the volume

Most of the weights are already there from the main app's bootstrap. The
new addition for this server is **just SAM2** if you don't already have it:

```bash
# From a temp Pod with the volume mounted at /workspace:
export VOLUME_ROOT=/workspace/ComfyUI
mkdir -p $VOLUME_ROOT/models/sam2
hf download Kijai/sam2-safetensors sam2_hiera_base_plus.safetensors \
    --local-dir $VOLUME_ROOT/models/sam2
```

Wan 2.2 Animate weights should already be at:
```
/runpod-volume/ComfyUI/models/diffusion_models/wan2.2_animate_14B_bf16.safetensors
```

If not, see the main app's README for the `hf download` command.

## Integration with the main app

The main app's backend treats this server as a new `provider` type called
`wan-animate-http`. The character-swap model card now dispatches HTTP
requests to this server instead of submitting a ComfyUI workflow to the
worker-comfyui endpoint.

Setup:

1. Build & deploy this server (steps above).
2. Set `WAN_ANIMATE_ENDPOINT` in the main app's `.env`.
3. Restart the main app's uvicorn dev server.
4. The character-swap card in the UI now hits this server end-to-end.

The legacy ComfyUI Wan workflow JSON (`runpod/workflows/video_swap_wan22_animate.json`)
is no longer used. It's kept in the repo as a record of the failed experiment.

## Cost expectations

- Pod with RTX 4090 24 GB: ~$0.69/hr while running. Idle time still bills,
  so stop the Pod between sessions or use Serverless.
- Per-generation: ~30–60 seconds wall clock (matches Replicate's logs of
  ~40 s for a 95-frame clip). At $0.69/hr that's about $0.01 per generation.
- Storage: the existing Network Volume already covers all weights. No new
  ongoing storage cost.

## Known gotchas

- **`cannot import name '_C' from 'sam2'` warning**: harmless. SAM2's optional
  CUDA extensions don't compile in our slim image. The library falls back to
  pure-Python post-processing. Replicate's logs show this same warning — they
  ship without the extensions too.
- **First request after cold start is slow** (~60 s extra) because the Wan
  Animate 14B model loads into VRAM. Subsequent requests are fast.
- **Single in-flight job**: the server serializes requests with an asyncio
  lock since a single 24 GB GPU can only run one Wan generation at a time.
  For higher throughput, scale horizontally (more Pods) rather than batching
  in one process.
