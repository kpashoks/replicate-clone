# M4 runbook: Wan 2.2 Animate single-character video swap

Before you can use the **Character Swap** page in the UI, you need to do three things on RunPod's side:

1. **Wait for the worker image to rebuild** (GHA, ~12–15 min after the push)
2. **Bump the endpoint Execution Timeout** to 1500 s
3. **Bootstrap the Wan 2.2 Animate weights** onto the Network Volume (~25–35 min)

Total: ~50–70 min including the GHA build that runs automatically.

---

## 1. Watch the GHA build

After the M4 commit lands on `main`, GHA detects changes to `runpod/Dockerfile`, `runpod/install_custom_nodes.sh`, or `runpod/extra_model_paths.yaml` and rebuilds the worker image. This time the install_custom_nodes.sh actually installs 4 packages (ComfyUI-KJNodes, ComfyUI-VideoHelperSuite, comfyui_controlnet_aux, ComfyUI-GGUF), so the build will be slightly longer than M2/M3 rebuilds.

- Watch at **https://github.com/kpashoks/replicate-clone/actions**
- Expected: ~12–15 min total
- Done when the new image is pushed: `ghcr.io/kpashoks/replicate-clone:latest`

**Important:** RunPod doesn't auto-pull new images. Force the next worker to use the latest tag by either:
- Terminating any active worker on the endpoint's **Workers** tab (next request pulls fresh), OR
- Editing the endpoint to re-save the image (same tag is fine; the save triggers a re-pull)

---

## 2. Bump the endpoint Execution Timeout

Wan video inference can take 8–15 minutes per clip. Our backend's per-job poll timeout is already bumped to 1500 s for `character-swap`, but RunPod also has its own per-request execution timeout.

- RunPod console → Serverless → `replicate-clone2` → **Manage** → **Advanced** (or wherever timeouts live in your UI version)
- **Execution timeout:** `600` → `1500` (25 min)
- **Save**

If you skip this, RunPod will kill long Wan runs at 10 min with a TIMED_OUT status regardless of our backend's patience.

---

## 3. Bootstrap Wan 2.2 Animate weights onto the volume

Same flow as the FLUX Kontext bootstrap, but with new files. The `bootstrap_volume.sh` in the repo now includes Wan downloads (steps 6–9). `hf download` is idempotent so the existing FLUX files won't redownload.

### Spin up a temp pod

- RunPod console → Pods → Deploy → `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Attach `runpod-volume` mounted at **`/workspace`**
- Container disk: **60 GB** (Wan bf16 is 28 GB and the HF cache doubles it briefly during download)

### In the pod's web terminal, run these (one at a time)

```bash
export HF_TOKEN=hf_PASTE_YOUR_TOKEN_HERE
export VOLUME_ROOT=/workspace/ComfyUI
```

```bash
pip install -q "huggingface_hub[cli]"
```

```bash
hf auth login --token "$HF_TOKEN" --add-to-git-credential
```

Now the four new downloads (the other files from M2/M3 stay in place — `hf download` skips files that already exist):

```bash
# Wan 2.2 Animate 14B bf16 (~28 GB, the big one — 10-20 min)
hf download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
  split_files/diffusion_models/wan2.2_animate_14B_bf16.safetensors \
  --local-dir $VOLUME_ROOT/models/diffusion_models
mv $VOLUME_ROOT/models/diffusion_models/split_files/diffusion_models/wan2.2_animate_14B_bf16.safetensors \
   $VOLUME_ROOT/models/diffusion_models/
rm -rf $VOLUME_ROOT/models/diffusion_models/split_files
```

```bash
# Wan 2.1 VAE (~500 MB)
hf download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
  split_files/vae/wan_2.1_vae.safetensors \
  --local-dir $VOLUME_ROOT/models/vae
mv $VOLUME_ROOT/models/vae/split_files/vae/wan_2.1_vae.safetensors \
   $VOLUME_ROOT/models/vae/
rm -rf $VOLUME_ROOT/models/vae/split_files
```

```bash
# UMT5-XXL text encoder for Wan (~5 GB)
hf download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
  split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
  --local-dir $VOLUME_ROOT/models/text_encoders
mv $VOLUME_ROOT/models/text_encoders/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
   $VOLUME_ROOT/models/text_encoders/
rm -rf $VOLUME_ROOT/models/text_encoders/split_files
```

```bash
# CLIP Vision H (~2.5 GB)
hf download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
  split_files/clip_vision/clip_vision_h.safetensors \
  --local-dir $VOLUME_ROOT/models/clip_vision
mv $VOLUME_ROOT/models/clip_vision/split_files/clip_vision/clip_vision_h.safetensors \
   $VOLUME_ROOT/models/clip_vision/
rm -rf $VOLUME_ROOT/models/clip_vision/split_files
```

### Verify

```bash
ls -lh $VOLUME_ROOT/models/diffusion_models/  # 3 files: flux1-dev, flux1-kontext-dev, wan2.2_animate
ls -lh $VOLUME_ROOT/models/vae/               # 2 files: ae.safetensors, wan_2.1_vae.safetensors
ls -lh $VOLUME_ROOT/models/text_encoders/     # 3 files: t5xxl_fp16, clip_l, umt5_xxl
ls -lh $VOLUME_ROOT/models/clip_vision/       # 1 file: clip_vision_h
du -sh $VOLUME_ROOT/models/*
```

Expected total: ~95 GB on the volume.

Terminate the pod when done.

---

## 4. Test from the UI

Once GHA finished + timeout bumped + bootstrap done:

1. Open http://localhost:3000 → click **Character Swap (Video)**
2. **Source video:** drop a short (5–10 s) clip of a person moving. mp4 < 16 MB.
3. **Reference character:** drop an image of who you want to insert (clean front-facing photo works best).
4. Leave defaults: FPS 16, Frames 81 (≈5 s), Steps 20.
5. Click **Run**.

Expected timing:
- Cold worker start + new-image pull: 2–5 min (first run after GHA rebuild)
- Wan inference: 8–12 min
- **Total: ~10–17 min**

The status panel updates with `RunPod: IN_QUEUE → IN_PROGRESS`. The page polls every 3 s.

## When Wan returns the video

The workflow ends with VHS_VideoCombine which writes an mp4 to `/comfyui/output/`. Our backend listens for outputs under the `images`, `gifs`, `videos`, and `files` response buckets and decodes any base64 payload it finds. The output viewer detects video URLs by extension (`.mp4` / `.mov` / `.webm`) and renders a `<video controls>` player.

---

## Likely failure modes (and how to fix)

### "No decodable outputs in RunPod response"

worker-comfyui (5.8.x) might not bubble VHS_VideoCombine output through to the response. Two workarounds:

- **Quickest:** add a `SaveImage` node at the end of the workflow that saves the last decoded frame. We at least get a still as confirmation the job ran, then iterate on video output handling.
- **Proper fix:** add a small custom encoder node that base64-encodes the mp4 file and emits it via the SaveImage-compatible output bucket. Or: switch to S3 output via the worker's S3 env vars.

### "Node `WanAnimateToVideo` not found" or "Node `DWPreprocessor` not found"

A custom node didn't install correctly during the GHA build. Check the build logs in GHA for which `git clone` failed.

### "model not found: wan2.2_animate_14B_bf16.safetensors"

Bootstrap didn't write the file to the right path. Check `ls $VOLUME_ROOT/models/diffusion_models/`. Re-run the move step from section 3.

### Worker is using an old image

RunPod cached the previous image and didn't pull the new one. Terminate the worker on the Workers tab and submit a new request — the platform will pull the latest `:latest` tag.

### Job stays IN_QUEUE for >5 min after the worker is ready

Same as M2 — RunPod GPU supply issue. Check the GPU configuration on the endpoint and add a fallback GPU (e.g., 48 GB if 48 GB PRO is unavailable).

---

## Cost expectations

- ~$0.00034/s × 600–900 s of Wan inference = **$0.20–0.30 per video generation** on A6000
- Wan is the most compute-expensive of our models. Plan ~$0.50/day if you do 2–3 gens for testing.
