# replicate-local

A local-first Replicate-style multimodal AI playground. The web UI runs on your laptop; heavy image/video models run on a RunPod Serverless ComfyUI endpoint. Prompt enhancement runs locally on your GPU via Qwen3-4B.

**Status:** Stage 1, M1 (backend skeleton + local prompt enhance).

## Stage 1 features

1. Text-to-image (FLUX.1 [dev])
2. Image transformation (FLUX.1 Kontext [dev])
3. Single-character video replacement (Wan 2.2 Animate)
4. Prompt enhancement (Qwen3-4B-Instruct, local)
5. Replicate-style web UI

License caveat: FLUX.1 [dev] and FLUX.1 Kontext [dev] are released under the **FLUX.1 Non-Commercial License**. This project is fine for personal use; do not expose it publicly without obtaining a commercial license from Black Forest Labs.

## Setup

### 1. Local prompt enhancement (M1 — what works today)

Verify CUDA driver:

```
nvidia-smi
```

This project uses **Python 3.12** (Transformers + bitsandbytes have well-tested wheels for it). Create the venv with Python 3.12 explicitly:

```
cd backend
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Install **PyTorch with CUDA support** first (separate index URL — PyPI's torch is CPU-only):

```
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

(CUDA 12.7 driver → cu124 wheel via forward compatibility. Adjust if you have a different toolkit.)

Then install the rest:

```
pip install -r requirements.txt
```

Copy `.env.example` → `.env` (defaults are fine).

The Qwen3-4B safetensors weights (~8 GB) are downloaded automatically by HuggingFace on the first prompt-enhance call and cached under `%USERPROFILE%\.cache\huggingface\hub`. bitsandbytes 4-bit quantizes at load time, so on-GPU footprint is ~3 GB.

Run the dev server:

```
cd backend
uvicorn main:app --reload --port 8000
```

Test:

```
curl -X POST http://localhost:8000/api/prompts/enhance ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\": \"a fox at sunset\"}"
```

First call takes 10–20 s (model load); subsequent calls are sub-second.

### 2. RunPod ComfyUI endpoint (M2+ — not yet implemented)

See `runpod/README.md` (TBD).

### 3. Frontend (M2+ — not yet implemented)

```
cd frontend
npm install
npm run dev
```

## Repo layout

See [the implementation plan](../../.claude/plans/i-would-like-to-peppy-gem.md) for the full architecture and milestone roadmap.

```
backend/      FastAPI orchestration
frontend/     Next.js 15 + App Router
runpod/       Dockerfile + ComfyUI workflow JSONs + bootstrap script
data/         Local uploads & outputs (gitignored)
models/       Local GGUF weights (gitignored)
```
