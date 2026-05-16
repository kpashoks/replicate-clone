# Downloader Serverless endpoint

Tiny companion service that exposes files on the RunPod Network Volume by
base64-encoding them in the response. Used when ComfyUI generates outputs
(currently: videos from `VHS_VideoCombine`) that the main `worker-comfyui`
endpoint doesn't bubble back via its `images` response array.

## Architecture

```
[Local backend]
  POST /api/generate/character-swap
   │
   ├─→ [Main endpoint: replicate-clone2]      [Volume: /runpod-volume]
   │     Wan workflow runs                          │
   │     VHS_VideoCombine writes mp4  ─────────────►  /runpod-volume/output/<prefix>_*.mp4
   │     Returns success_no_images
   │
   └─→ [Downloader endpoint: replicate-clone-downloader]
         handler.py reads /runpod-volume/output/<prefix>* ◄────  (same volume attached)
         Returns base64 in {files: [...]}
   │
   ▼
[Local backend]
  Decode base64, save to data/outputs/<job_id>/<file>.mp4
```

The symlink trick in the main worker image (`install_custom_nodes.sh`) makes
ComfyUI's default output directory (`/comfyui/output`) point at
`/runpod-volume/output`, so no workflow JSON changes are needed.

## Deployment (one-time)

After GHA pushes `ghcr.io/kpashoks/replicate-clone-downloader:latest`:

1. **RunPod console → Serverless → New Endpoint**
2. **Source:** Deploy from Docker registry
3. **Image:** `ghcr.io/kpashoks/replicate-clone-downloader:latest`
4. **Endpoint name:** `replicate-clone-downloader` (or anything)
5. **Endpoint type:** Queue
6. **Worker type:** **CPU** (the cheapest tier — no GPU needed; this is pure file I/O + base64)
7. **Max workers:** 1 (or 2 if you might run multiple downloads concurrently)
8. **Active workers:** 0 (cold start ~20–40s acceptable for the once-per-video usage)
9. **Idle timeout:** 30 seconds
10. **Execution timeout:** 120 seconds (plenty for reading + encoding tens of MB)
11. **Container disk:** 5 GB
12. **Advanced → Network Volume:** attach the same `runpod-volume` as the main endpoint (must be in the same datacenter — US-KS-2). Mount path locked to `/runpod-volume`.
13. **Deploy.** Copy the endpoint ID.

## Wiring it up locally

Add to `.env`:

```
RUNPOD_DOWNLOADER_ENDPOINT_ID=<id from step 13>
```

Restart `uvicorn`. The backend now calls the downloader whenever a
character-swap job returns `success_no_images`.

## Cost

CPU Serverless on RunPod is roughly $0.000005/s while running. A typical
fetch takes 5–15 s of compute (cold start + read + base64). At ~10 s per
fetch that's $0.00005/video — fractions of a penny. With Active=0, idle
cost is $0.

## Manual test

```bash
curl -X POST https://api.runpod.ai/v2/<DOWNLOADER_ENDPOINT_ID>/runsync \
  -H 'Authorization: Bearer <RUNPOD_API_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{"input": {"prefix": "character-swap_"}}'
```

Should return `{"files": [...]}` if any matching files exist on the volume.
