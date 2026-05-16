# Tiny CPU-only Serverless worker that exposes files on the attached
# RunPod Network Volume by base64-encoding them in the response.
#
# Companion to the main worker-comfyui endpoint. Use when ComfyUI generates
# outputs (videos, .glb, etc.) that worker-comfyui doesn't bubble back via
# its response.

FROM python:3.11-slim

# RunPod's serverless SDK. Tiny, no GPU deps.
RUN pip install --no-cache-dir runpod==1.7.13

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
