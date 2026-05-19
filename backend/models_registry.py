from typing import Literal

from pydantic import BaseModel


OutputKind = Literal["image", "video"]

# How a model entry is invoked at job time.
#   "runpod"            -> existing ComfyUI workflow JSON path, submitted to
#                          settings.RUNPOD_ENDPOINT_ID via the queue API
#   "wan-animate-http"  -> HTTP multipart upload to the dedicated wan-animate
#                          server (see runpod-wan-animate/); URL configured
#                          via settings.WAN_ANIMATE_ENDPOINT
#   "atlas"             -> Atlas Cloud unified inference API (see
#                          backend/providers/atlas_client.py). Async submit +
#                          poll. Model identifier carried in `atlas_model_id`.
Provider = Literal["runpod", "wan-animate-http", "atlas"]

# Task buckets surfaced in the UI. The frontend groups the gallery by task and
# the per-task page shows a Model picker over every entry sharing the same
# task value.
Task = Literal["t2i", "i2i", "video-swap"]

# Coarse speed bucket for the picker. fast = <5s, medium = 5-15s, slow = >15s
# (warm). Local RunPod entries are always slow because of cold-start + GPU
# scheduling overhead even when the model itself is fast.
SpeedBucket = Literal["fast", "medium", "slow"]


class ModelEntry(BaseModel):
    slug: str
    label: str
    description: str
    output_kind: OutputKind
    provider: Provider = "runpod"
    # RunPod-only: workflow JSON filename under runpod/workflows/.
    workflow_file: str | None = None
    # Atlas-only: the model identifier passed to AtlasClient.submit_image /
    # submit_video. Verify against Atlas's playground for each slug — the
    # public URL slug usually matches the API id but xAI's didn't, so don't
    # assume.
    atlas_model_id: str | None = None
    # Atlas I2I only: the body key under which the uploaded reference image
    # URLs are sent. Atlas is mostly consistent ("images" for OpenAI, Alibaba,
    # Google) but a few vendors diverge — xAI's Grok uses "image_urls". Override
    # per-slug only when the playground example shows something other than
    # "images".
    atlas_images_param: str = "images"
    accepts_image: bool = False
    accepts_video: bool = False
    stage: int = 1
    available: bool = False

    # ---- Picker metadata (Phase 1) ---------------------------------------
    task: Task = "t2i"
    nsfw: bool = False
    speed: SpeedBucket = "slow"
    # Short tag rendered as a chip in the picker row: "general", "typography",
    # "photoreal", "precise edits", "character swap", etc.
    best_for: str = "general"
    # Effective per-image USD price at default settings. None = self-hosted
    # (user pays GPU time, not per-image), so the picker shows "Self-hosted".
    price_per_image_usd: float | None = None
    # I2I-only: max number of reference images the model accepts in one call.
    # None means "n/a" (T2I) or "unlimited / not a meaningful constraint".
    max_ref_images: int | None = None
    # Human label for the picker row, e.g. "OpenAI · Atlas",
    # "Black Forest Labs · Atlas", "Local · FLUX.1 dev".
    provider_label: str = ""


REGISTRY: dict[str, ModelEntry] = {
    # ---- RunPod ComfyUI workflows (existing, unchanged behavior) ---------
    "text-to-image": ModelEntry(
        slug="text-to-image",
        label="Text to Image (FLUX)",
        description="Generate an image from a text prompt using FLUX.1 [dev].",
        workflow_file="text2img_flux.json",
        output_kind="image",
        stage=1,
        available=True,
        task="t2i",
        nsfw=True,  # FLUX.1 dev itself is uncensored; we run it locally.
        speed="slow",
        best_for="general",
        price_per_image_usd=None,
        provider_label="Local · FLUX.1 dev",
    ),
    "juggernaut-xl": ModelEntry(
        slug="juggernaut-xl",
        label="Juggernaut XL (Photorealistic)",
        description="SDXL photorealism workhorse. Uses dual CLIP + negative prompts. Slower but often more cinematic for portraits than FLUX.",
        workflow_file="text2img_juggernaut.json",
        output_kind="image",
        stage=1,
        available=True,
        task="t2i",
        nsfw=True,
        speed="slow",
        best_for="photoreal",
        price_per_image_usd=None,
        provider_label="Local · Juggernaut XL",
    ),
    "image-edit": ModelEntry(
        slug="image-edit",
        label="FLUX Kontext Dev",
        description="Edit an uploaded image with a text prompt using FLUX.1 Kontext [dev]. Runs locally on your GPU - no per-image cost, but slower and queued behind any other local job.",
        workflow_file="imgedit_flux_kontext.json",
        output_kind="image",
        accepts_image=True,
        stage=1,
        available=True,
        task="i2i",
        nsfw=True,
        speed="slow",
        best_for="precise edits",
        price_per_image_usd=None,
        max_ref_images=1,
        provider_label="Local · FLUX Kontext Dev",
    ),
    "atlas-flux-kontext-dev": ModelEntry(
        slug="atlas-flux-kontext-dev",
        label="FLUX Kontext Dev (Atlas)",
        description="Same FLUX.1 Kontext [dev] weights as the local entry, hosted on Atlas Cloud's GPUs. Use this when your local GPU is busy or you want faster turnaround. Atlas does not content-moderate BFL models.",
        provider="atlas",
        atlas_model_id="black-forest-labs/flux-kontext-dev",
        output_kind="image",
        accepts_image=True,
        stage=1,
        available=True,
        task="i2i",
        nsfw=True,
        speed="fast",
        best_for="precise edits (hosted)",
        price_per_image_usd=0.025,
        max_ref_images=1,
        provider_label="Black Forest Labs · Atlas",
    ),
    # ---- Wan 2.2 Animate via dedicated HTTP server -----------------------
    # Bypasses ComfyUI entirely. See runpod-wan-animate/ for the inference
    # server that mirrors Replicate's wan-2.2-animate-replace pipeline.
    "character-swap": ModelEntry(
        slug="character-swap",
        label="Character Swap (Video)",
        description="Replace one character in a 5-10s video with a reference character image, using Wan 2.2 Animate. Runs on a dedicated inference server (not the main ComfyUI worker).",
        provider="wan-animate-http",
        workflow_file=None,
        output_kind="video",
        accepts_image=True,
        accepts_video=True,
        stage=1,
        available=True,
        task="video-swap",
        nsfw=True,
        speed="slow",
        best_for="video character swap",
        price_per_image_usd=None,
        max_ref_images=1,
        provider_label="Local · Wan 2.2 Animate",
    ),

    # ---- Atlas Cloud: SFW text-to-image ----------------------------------
    # All Atlas entries are available=False until Phase 1b wires the "atlas"
    # provider into jobs.py.
    "atlas-flux-2-pro": ModelEntry(
        slug="atlas-flux-2-pro",
        label="FLUX.2 Pro",
        description="Black Forest Labs' versatile production T2I model on Atlas Cloud. Fast, broad style range, decent text rendering.",
        provider="atlas",
        atlas_model_id="black-forest-labs/flux-2-pro",
        output_kind="image",
        stage=1,
        available=True,
        task="t2i",
        nsfw=False,
        speed="fast",
        best_for="general",
        price_per_image_usd=0.03,
        provider_label="Black Forest Labs · Atlas",
    ),
    "atlas-ideogram-v3": ModelEntry(
        slug="atlas-ideogram-v3",
        label="Ideogram v3",
        description="Best-in-class typography and text rendering. Use for posters, logos, branding materials, infographics.",
        provider="atlas",
        atlas_model_id="ideogram/ideogram-v3",
        output_kind="image",
        stage=1,
        available=True,
        task="t2i",
        nsfw=False,
        speed="fast",
        best_for="typography",
        price_per_image_usd=0.03,
        provider_label="Ideogram · Atlas",
    ),
    "atlas-imagen-4-ultra": ModelEntry(
        slug="atlas-imagen-4-ultra",
        label="Imagen 4 Ultra",
        description="Google DeepMind's premium photoreal model. Exceptional skin / fabric / lighting detail. ~8s per image.",
        provider="atlas",
        atlas_model_id="google/imagen-4-ultra",
        output_kind="image",
        stage=1,
        available=True,
        task="t2i",
        nsfw=False,
        speed="medium",
        best_for="photoreal",
        price_per_image_usd=0.04,
        provider_label="Google · Atlas",
    ),

    # ---- Atlas Cloud: NSFW text-to-image ---------------------------------
    "atlas-flux-dev": ModelEntry(
        slug="atlas-flux-dev",
        label="FLUX.1 Dev (Atlas)",
        description="Top NSFW T2I pick per Atlas's uncensored ranking. Same FLUX.1 dev weights as the local entry, hosted on Atlas's GPUs.",
        provider="atlas",
        atlas_model_id="black-forest-labs/flux-dev",
        output_kind="image",
        stage=1,
        available=True,
        task="t2i",
        nsfw=True,
        speed="medium",
        best_for="general NSFW",
        price_per_image_usd=0.012,
        provider_label="Black Forest Labs · Atlas",
    ),
    "atlas-flux-schnell": ModelEntry(
        slug="atlas-flux-schnell",
        label="FLUX.1 Schnell (Atlas)",
        description="Fastest NSFW T2I option. 4-step distilled FLUX. Use for drafts and batch generation; quality below FLUX.1 dev.",
        provider="atlas",
        atlas_model_id="wavespeed-ai/flux-schnell",
        output_kind="image",
        stage=1,
        available=True,
        task="t2i",
        nsfw=True,
        speed="fast",
        best_for="fast / draft NSFW",
        price_per_image_usd=0.003,
        provider_label="Black Forest Labs · Atlas",
    ),

    # ---- Atlas Cloud: SFW image-to-image ---------------------------------
    "atlas-gpt-image-2-edit": ModelEntry(
        slug="atlas-gpt-image-2-edit",
        label="GPT Image 2 Edit",
        description="OpenAI's edit model: text prompt + one or more reference images. Cheapest SFW edit option on Atlas; price varies slightly with size/quality.",
        provider="atlas",
        atlas_model_id="openai/gpt-image-2/edit",
        output_kind="image",
        accepts_image=True,
        stage=1,
        available=True,
        task="i2i",
        nsfw=False,
        speed="medium",
        best_for="budget edits",
        price_per_image_usd=0.01,
        max_ref_images=10,
        provider_label="OpenAI · Atlas",
    ),
    "atlas-nano-banana-2-edit": ModelEntry(
        slug="atlas-nano-banana-2-edit",
        label="Nano Banana 2 Edit",
        description="Google's premium multi-reference edit model. Up to 14 reference images per call. 2K and 4K output tiers available; 1K price quoted.",
        provider="atlas",
        atlas_model_id="google/nano-banana-2/edit",
        output_kind="image",
        accepts_image=True,
        stage=1,
        available=True,
        task="i2i",
        nsfw=False,
        speed="medium",
        best_for="multi-ref premium edits",
        price_per_image_usd=0.08,
        max_ref_images=14,
        provider_label="Google · Atlas",
    ),

    # ---- Atlas Cloud: NSFW image-to-image --------------------------------
    "atlas-qwen-edit-plus": ModelEntry(
        slug="atlas-qwen-edit-plus",
        label="Qwen-Image Edit Plus",
        description="Alibaba's Qwen image edit (20251215 build). 1-3 reference images, prompt up to 800 chars. Strong at object add/remove, text-in-image edits, style transfer.",
        provider="atlas",
        atlas_model_id="alibaba/qwen-image/edit-plus-20251215",
        output_kind="image",
        accepts_image=True,
        stage=1,
        available=True,
        task="i2i",
        nsfw=True,
        speed="medium",
        best_for="object edits, style transfer",
        price_per_image_usd=0.021,
        max_ref_images=3,
        provider_label="Alibaba · Atlas",
    ),
    "atlas-wan-2-6-edit": ModelEntry(
        slug="atlas-wan-2-6-edit",
        label="Wan 2.6 Image Edit",
        description="Alibaba's Wan 2.6 image-to-image. Single reference image, broad edit support. Same price tier as Qwen Edit Plus.",
        provider="atlas",
        atlas_model_id="alibaba/wan-2.6/image-edit",
        output_kind="image",
        accepts_image=True,
        stage=1,
        available=True,
        task="i2i",
        nsfw=True,
        speed="medium",
        best_for="general edits",
        price_per_image_usd=0.021,
        max_ref_images=1,
        provider_label="Alibaba · Atlas",
    ),
    "atlas-grok-imagine-edit": ModelEntry(
        slug="atlas-grok-imagine-edit",
        label="Grok Imagine Edit",
        description="xAI's Grok Imagine Image Quality edit model. 2K studio-grade output, strong prompt adherence. Atlas does not surface a public price on the catalog page; xAI's published rate is ~$0.022/img — confirm in-console before relying on it.",
        provider="atlas",
        atlas_model_id="xai/grok-imagine-image-quality/edit",
        atlas_images_param="image_urls",
        output_kind="image",
        accepts_image=True,
        stage=1,
        available=True,
        task="i2i",
        nsfw=True,
        speed="medium",
        best_for="2K studio edits",
        price_per_image_usd=0.022,
        max_ref_images=8,
        provider_label="xAI · Atlas",
    ),

    # ---- Future / placeholder --------------------------------------------
    "scail-2char": ModelEntry(
        slug="scail-2char",
        label="Two-Character Swap (Stage 2)",
        description="Replace two characters in a video using the SCAIL workflow on Wan 2.1.",
        workflow_file="stage2_scail_wan21.json",
        output_kind="video",
        accepts_image=True,
        accepts_video=True,
        stage=2,
        available=False,
        task="video-swap",
        nsfw=True,
        speed="slow",
        best_for="two-character video swap",
        price_per_image_usd=None,
        max_ref_images=2,
        provider_label="Local · SCAIL on Wan 2.1",
    ),
}


def list_models() -> list[ModelEntry]:
    return list(REGISTRY.values())


def get_model(slug: str) -> ModelEntry | None:
    return REGISTRY.get(slug)
