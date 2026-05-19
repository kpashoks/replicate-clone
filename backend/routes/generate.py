import secrets
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import jobs
import models_registry
from runpod_client import RunPodClient, RunPodError


router = APIRouter(prefix="/api", tags=["generate"])


# ---- per-model input schemas -----------------------------------------------


class TextToImageParams(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    steps: int = Field(20, ge=1, le=50)
    guidance: float = Field(3.5, ge=0.0, le=20.0)
    seed: int = Field(-1, description="-1 means random")


# SDXL-tuned defaults: cfg ~7 (vs FLUX 1.0), explicit negative prompt, 25 steps.
_SDXL_DEFAULT_NEGATIVE = (
    "low quality, worst quality, blurry, jpeg artifacts, watermark, signature, "
    "text, deformed, distorted, mutated, extra fingers, missing fingers, "
    "out of frame, cropped"
)


class JuggernautParams(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = Field(_SDXL_DEFAULT_NEGATIVE, max_length=2000)
    width: int = Field(1024, ge=512, le=2048)
    height: int = Field(1024, ge=512, le=2048)
    steps: int = Field(25, ge=1, le=60)
    cfg: float = Field(7.0, ge=1.0, le=15.0)
    seed: int = Field(-1, description="-1 means random")


class ImageEditParams(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    steps: int = Field(20, ge=1, le=50)
    # FLUX Kontext typically uses lower guidance than FLUX dev (2.5 vs 3.5).
    guidance: float = Field(2.5, ge=0.0, le=20.0)
    seed: int = Field(-1, description="-1 means random")


class CharacterSwapParams(BaseModel):
    """Wan 2.2 Animate via the dedicated wan-animate inference server.

    Dispatched to a separate HTTP service (NOT the main ComfyUI worker) -
    see runpod-wan-animate/. That server runs the official Wan-Video Python
    API plus Meta sam2, matching Replicate's wan-2.2-animate-replace pipeline.
    """
    prompt: str = Field(
        "",
        max_length=2000,
        description=(
            "Optional scene/style context. With replace_flag=True (Mix mode), "
            "the source video's background is preserved and the prompt only "
            "guides minor stylistic choices. With replace_flag=False (Move "
            "mode), the prompt drives the entire generated scene."
        ),
    )
    resolution: Literal[
        "832x480", "1280x720", "1408x640", "480x832", "720x1280"
    ] = Field(
        "832x480",
        description=(
            "Output dimensions. 832x480 is fastest (~30s). 1280x720 and "
            "1408x640 are higher quality but slower. 480x832 / 720x1280 are "
            "vertical (phone) orientation."
        ),
    )
    replace_flag: bool = Field(
        True,
        description=(
            "True = Mix mode (character_mask + background_video; preserves "
            "the source video's scene). False = Move mode (no mask; character "
            "performs the source's motion in a freshly-generated scene)."
        ),
    )
    sampling_steps: int = Field(
        20, ge=1, le=60,
        description="Diffusion steps per stage. Total compute = 2 * this (dual-stage refinement).",
    )
    frame_num: int = Field(
        81, ge=17, le=161,
        description="Number of frames to generate (~5-10 s at 16 fps).",
    )
    refert_num: Literal[1, 5] = Field(
        5,
        description=(
            "Number of reference frames overlapped between Wan animation "
            "clips. Wan only accepts 1 or 5 (their codebase asserts this). "
            "5 = smoother multi-clip transitions; 1 = each clip independent."
        ),
    )
    guide_scale: float = Field(
        5.0, ge=1.0, le=15.0,
        description="Classifier-free guidance strength. 5.0 is the Wan default.",
    )
    seed: int = Field(-1, description="-1 means random")


class AtlasT2IParams(BaseModel):
    """Permissive shared schema for Atlas-hosted T2I models. Atlas's per-vendor
    models accept different subsets of these; the backend forwards what's
    set and Atlas ignores unknowns. Fields here mirror what the consolidated
    task page sends.
    """
    prompt: str = Field(..., min_length=1, max_length=2000)
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    steps: int = Field(20, ge=1, le=50)
    guidance: float = Field(3.5, ge=0.0, le=20.0)
    seed: int = Field(-1, description="-1 means random")


class AtlasI2IParams(BaseModel):
    """Permissive shared schema for Atlas-hosted I2I models. Width/height are
    typically derived from the reference image, so they're omitted here.
    """
    prompt: str = Field(..., min_length=1, max_length=2000)
    steps: int = Field(20, ge=1, le=50)
    guidance: float = Field(3.5, ge=0.0, le=20.0)
    seed: int = Field(-1, description="-1 means random")


_PARAMS_SCHEMA: dict[str, type[BaseModel]] = {
    "text-to-image": TextToImageParams,
    "juggernaut-xl": JuggernautParams,
    "image-edit": ImageEditParams,
    "character-swap": CharacterSwapParams,
    # Atlas T2I
    "atlas-flux-2-pro": AtlasT2IParams,
    "atlas-ideogram-v3": AtlasT2IParams,
    "atlas-imagen-4-ultra": AtlasT2IParams,
    "atlas-flux-dev": AtlasT2IParams,
    "atlas-flux-schnell": AtlasT2IParams,
    # Atlas I2I
    "atlas-gpt-image-2-edit": AtlasI2IParams,
    "atlas-nano-banana-2-edit": AtlasI2IParams,
    "atlas-qwen-edit-plus": AtlasI2IParams,
    "atlas-wan-2-6-edit": AtlasI2IParams,
    "atlas-grok-imagine-edit": AtlasI2IParams,
    "atlas-wan-2-7-edit": AtlasI2IParams,
}

# Per-slug minimum-input requirements (number of uploaded files needed).
_MIN_INPUT_IDS: dict[str, int] = {
    "text-to-image": 0,
    "juggernaut-xl": 0,
    "image-edit": 1,
    "character-swap": 2,  # [source_video, reference_character_image]
    # Atlas T2I: no inputs
    "atlas-flux-2-pro": 0,
    "atlas-ideogram-v3": 0,
    "atlas-imagen-4-ultra": 0,
    "atlas-flux-dev": 0,
    "atlas-flux-schnell": 0,
    # Atlas I2I: at least one reference image
    "atlas-gpt-image-2-edit": 1,
    "atlas-nano-banana-2-edit": 1,
    "atlas-qwen-edit-plus": 1,
    "atlas-wan-2-6-edit": 1,
    "atlas-grok-imagine-edit": 1,
    "atlas-wan-2-7-edit": 1,
}


# ---- request/response models -----------------------------------------------


class GenerateRequest(BaseModel):
    params: dict = Field(default_factory=dict)
    input_ids: list[str] = Field(default_factory=list)


class GenerateResponse(BaseModel):
    job_id: str
    status: str


class JobView(BaseModel):
    id: str
    slug: str
    status: str
    params: dict
    runpod_request_id: str | None = None
    runpod_status: str | None = None
    output_files: list[str] = []
    error: str | None = None
    created_at: float
    updated_at: float


def _to_view(j: jobs.Job) -> JobView:
    return JobView(
        id=j.id,
        slug=j.slug,
        status=j.status,
        params=j.params,
        runpod_request_id=j.runpod_request_id,
        runpod_status=j.runpod_status,
        output_files=j.output_files,
        error=j.error,
        created_at=j.created_at,
        updated_at=j.updated_at,
    )


# ---- routes ----------------------------------------------------------------


@router.post("/generate/{slug}", response_model=GenerateResponse)
async def submit_generate(slug: str, req: GenerateRequest) -> GenerateResponse:
    model = models_registry.get_model(slug)
    if not model:
        raise HTTPException(status_code=404, detail=f"Unknown model slug: {slug}")
    if not model.available:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{slug}' is not yet available in this milestone.",
        )

    schema_cls = _PARAMS_SCHEMA.get(slug)
    if not schema_cls:
        raise HTTPException(status_code=400, detail=f"No input schema registered for {slug}")

    min_inputs = _MIN_INPUT_IDS.get(slug, 0)
    if len(req.input_ids) < min_inputs:
        raise HTTPException(
            status_code=422,
            detail=f"{slug} requires {min_inputs} uploaded input(s); got {len(req.input_ids)}",
        )

    try:
        validated = schema_cls.model_validate(req.params)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    params = validated.model_dump()
    if params.get("seed", -1) == -1:
        # randbits(31) gives [0, 2**31 - 1] = signed-INT32 positive range.
        # randbits(32) would exceed Atlas Cloud's seed limit (Atlas Pydantic
        # validates seed <= 2147483647). All other providers (Wan-Animate,
        # FLUX, etc.) accept any non-negative int up to INT32_MAX, so 31
        # bits is the lowest common denominator that works everywhere.
        params["seed"] = secrets.randbits(31)

    job = jobs.registry.create(slug, params)
    jobs.schedule(
        jobs.run_job(job.id, model, params, req.input_ids),
    )
    return GenerateResponse(job_id=job.id, status=job.status)


@router.get("/jobs/{job_id}", response_model=JobView)
async def get_job(job_id: str) -> JobView:
    j = jobs.registry.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")
    return _to_view(j)


@router.get("/jobs", response_model=list[JobView])
async def list_jobs() -> list[JobView]:
    return [_to_view(j) for j in jobs.registry.list()]


@router.post("/jobs/{job_id}/cancel", response_model=JobView)
async def cancel_job(job_id: str) -> JobView:
    j = jobs.registry.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")
    if j.status in {"succeeded", "failed", "cancelled"}:
        return _to_view(j)
    if j.runpod_request_id:
        try:
            client = RunPodClient()
            await client.cancel(j.runpod_request_id)
        except RunPodError as e:
            raise HTTPException(status_code=502, detail=f"RunPod cancel failed: {e}")
    jobs.registry.update(job_id, status="cancelled")
    j2 = jobs.registry.get(job_id)
    assert j2 is not None
    return _to_view(j2)
