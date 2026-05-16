# replicate-local custom ComfyUI worker
#
# Base image is RunPod's official worker-comfyui — already cached on RunPod's
# infrastructure, so cold starts skip the multi-GB image pull. Model weights
# live entirely on the attached Network Volume (NOT in this image), keeping
# the image small and rebuild-cheap.
#
# Tag pinned to 5.8.5 (Mar 2026 stable). Bump deliberately when validating
# workflow compatibility.
#
# Cache-bust marker: rebuild after RunPod builder crashed on previous attempt.

FROM runpod/worker-comfyui:5.8.5-base

# Install custom nodes.
# For M2 (FLUX text-to-image) none are needed. M4 (Wan 2.2 Animate) will add
# ComfyUI-GGUF, ComfyUI-VideoHelperSuite, ComfyUI-WanAnimate, ComfyUI-KJNodes.
COPY install_custom_nodes.sh /install_custom_nodes.sh
RUN chmod +x /install_custom_nodes.sh && /install_custom_nodes.sh

# Tell ComfyUI to read model weights from the Network Volume.
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml

# Wrap the base image's entrypoint so /runpod-volume/output exists at runtime
# (install_custom_nodes.sh symlinks /comfyui/output -> /runpod-volume/output
# but the target subdir isn't created at build time since the volume isn't
# mounted then; without this, ComfyUI's first SaveImage call raises EEXIST).
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
