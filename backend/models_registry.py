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
    "atlas-flux-kontext-dev-lora": ModelEntry(
        slug="atlas-flux-kontext-dev-lora",
        label="FLUX Kontext Dev LoRA (Atlas)",
        description="FLUX.1 Kontext [dev] with LoRA passthrough, hosted on Atlas. Same Kontext editing capabilities as the local entry, plus you can supply a LoRA HF slug to apply a style/character adapter to the edit. Useful when you want Kontext's precise control over edits AND a custom aesthetic on top.",
        provider="atlas",
        atlas_model_id="black-forest-labs/flux-kontext-dev-lora",
        output_kind="image",
        accepts_image=True,
        stage=1,
        available=True,
        task="i2i",
        nsfw=True,
        speed="fast",
        best_for="LoRA-styled edits",
        price_per_image_usd=0.03,
        max_ref_images=1,
        provider_label="Black Forest Labs · Atlas",
    ),
    "atlas-wan-2-7-edit": ModelEntry(
        slug="atlas-wan-2-7-edit",
        label="Wan 2.7 Image-to-Image (Atlas)",
        description="Alibaba's Wan 2.7 i2i editor. Successor to Wan 2.6 - improved instruction following, multi-image composition (multiple references in a single call), source-scene preservation, and multi-variation output. Hosted on Atlas Cloud's GPUs.",
        provider="atlas",
        atlas_model_id="alibaba/wan-2.7/image-edit",
        output_kind="image",
        accepts_image=True,
        stage=1,
        available=True,
        task="i2i",
        nsfw=True,
        speed="medium",
        best_for="instruction edits, multi-ref composition",
        price_per_image_usd=0.03,
        max_ref_images=3,
        provider_label="Alibaba · Atlas",
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
    "atlas-nano-banana-2": ModelEntry(
        slug="atlas-nano-banana-2",
        label="Nano Banana 2",
        description="Google's flagship T2I model (sibling of the Nano Banana 2 i2i edit endpoint). Same model family that powers their premium image tools. Note: Atlas's catalog shows tiered pricing - $0.048 at 1K is the base rate; 2K and 4K outputs cost more.",
        provider="atlas",
        atlas_model_id="google/nano-banana-2/text-to-image",
        output_kind="image",
        stage=1,
        available=True,
        task="t2i",
        nsfw=False,
        speed="medium",
        best_for="SOTA quality",
        price_per_image_usd=0.048,
        provider_label="Google · Atlas",
    ),
    "atlas-seedream-v5-lite": ModelEntry(
        slug="atlas-seedream-v5-lite",
        label="Seedream v5.0 Lite",
        description="ByteDance's lightweight Seedream variant. Strong at typography, posters, and stylized illustration; faster than the full Seedream tier.",
        provider="atlas",
        atlas_model_id="bytedance/seedream-v5.0-lite",
        output_kind="image",
        stage=1,
        available=True,
        task="t2i",
        nsfw=False,
        speed="medium",
        best_for="typography / illustration",
        price_per_image_usd=0.032,
        provider_label="ByteDance · Atlas",
    ),
    "atlas-gpt-image-2": ModelEntry(
        slug="atlas-gpt-image-2",
        label="GPT Image 2",
        description="OpenAI's text-to-image variant (sibling of GPT Image 2 Edit). Cheapest SFW option in the picker - $0.009/img. Good general-purpose photoreal and illustration.",
        provider="atlas",
        atlas_model_id="openai/gpt-image-2/text-to-image",
        output_kind="image",
        stage=1,
        available=True,
        task="t2i",
        nsfw=False,
        speed="fast",
        best_for="budget / general",
        price_per_image_usd=0.009,
        provider_label="OpenAI · Atlas",
    ),
    "atlas-wan-2-7": ModelEntry(
        slug="atlas-wan-2-7",
        label="Wan 2.7 Text-to-Image",
        description="Alibaba's Wan 2.7 T2I. Successor to Wan 2.6 - improved prompt fidelity and composition. A Pro tier (alibaba/wan-2.7-pro/text-to-image) also exists at $0.075/img if higher quality is needed.",
        provider="atlas",
        atlas_model_id="alibaba/wan-2.7/text-to-image",
        output_kind="image",
        stage=1,
        available=True,
        task="t2i",
        nsfw=False,
        speed="medium",
        best_for="prompt fidelity",
        price_per_image_usd=0.03,
        provider_label="Alibaba · Atlas",
    ),
    "atlas-wan-2-6": ModelEntry(
        slug="atlas-wan-2-6",
        label="Wan 2.6 Text-to-Image",
        description="Alibaba's Wan 2.6 T2I. Solid all-rounder. Use the 2.7 variant for new work; keep 2.6 here for parity with reference outputs and reproducibility of older jobs.",
        provider="atlas",
        atlas_model_id="alibaba/wan-2.6/text-to-image",
        output_kind="image",
        stage=1,
        available=True,
        task="t2i",
        nsfw=False,
        speed="medium",
        best_for="all-rounder",
        price_per_image_usd=0.03,
        provider_label="Alibaba · Atlas",
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
    "atlas-flux-dev-lora": ModelEntry(
        slug="atlas-flux-dev-lora",
        label="FLUX.1 [dev] LoRA",
        description="FLUX.1 dev variant on Atlas that accepts a LoRA URL parameter for custom style adapters at inference time. Use this when you have specific style/character LoRAs hosted somewhere and want to apply them on top of dev weights.",
        provider="atlas",
        atlas_model_id="black-forest-labs/flux-dev-lora",
        output_kind="image",
        stage=1,
        available=True,
        task="t2i",
        nsfw=True,
        speed="medium",
        best_for="LoRA styles",
        price_per_image_usd=0.015,
        provider_label="Black Forest Labs · Atlas",
    ),
    "atlas-z-image-turbo": ModelEntry(
        slug="atlas-z-image-turbo",
        label="Z-Image Turbo",
        description="Alibaba Tongyi-MAI's fast photoreal model. Sub-1s generation, lower cost than FLUX dev. Note: $0.005 is the base rate without prompt-rewrite; enabling the rewrite adds cost (~$0.015-0.03). NSFW handling is permissive per Atlas's uncensored guide.",
        provider="atlas",
        atlas_model_id="z-image/turbo",
        output_kind="image",
        stage=1,
        available=True,
        task="t2i",
        nsfw=True,
        speed="fast",
        best_for="sub-1s photoreal",
        price_per_image_usd=0.005,
        provider_label="Tongyi (Alibaba) · Atlas",
    ),
    "atlas-flux-schnell": ModelEntry(
        slug="atlas-flux-schnell",
        label="FLUX.1 Schnell (Atlas)",
        description="Fastest NSFW T2I option. 4-step distilled FLUX from Black Forest Labs. Use for drafts and batch generation; quality below FLUX.1 dev.",
        provider="atlas",
        atlas_model_id="black-forest-labs/flux-schnell",
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
    "atlas-wan-2-5-edit": ModelEntry(
        slug="atlas-wan-2-5-edit",
        label="Wan 2.5 Image Edit",
        description="Alibaba's Wan 2.5 image-to-image. Older sibling of Wan 2.6 / 2.7 at the same price tier. Useful for reproducibility of older jobs or comparison against the newer Wan variants. Supports multi-image reference composition.",
        provider="atlas",
        atlas_model_id="alibaba/wan-2.5/image-edit",
        output_kind="image",
        accepts_image=True,
        stage=1,
        available=True,
        task="i2i",
        nsfw=True,
        speed="medium",
        best_for="reference / older",
        price_per_image_usd=0.021,
        max_ref_images=4,
        provider_label="Alibaba · Atlas",
    ),
    "atlas-wan-2-6-edit": ModelEntry(
        slug="atlas-wan-2-6-edit",
        label="Wan 2.6 Image Edit",
        description="Alibaba's Wan 2.6 image-to-image. Accepts up to 4 reference images for multi-ref composition (verified in the Atlas dashboard UI; the public docs don't surface the limit). Broad edit support, same price tier as Qwen Edit Plus.",
        provider="atlas",
        atlas_model_id="alibaba/wan-2.6/image-edit",
        output_kind="image",
        accepts_image=True,
        stage=1,
        available=True,
        task="i2i",
        nsfw=True,
        speed="medium",
        best_for="general edits, multi-ref",
        price_per_image_usd=0.021,
        max_ref_images=4,
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
