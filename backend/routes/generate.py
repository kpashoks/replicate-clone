import secrets

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


class ImageCharSwapParams(BaseModel):
    """SDXL + IP-Adapter + ControlNet OpenPose character replacement in a still image.

    Output dimensions are derived automatically from the source image (via the
    workflow's GetImageSize node) so the extracted pose skeleton, the latent,
    and the output all stay at the same aspect ratio. No width/height knobs.
    """
    prompt: str = Field(
        "",
        max_length=2000,
        description="Optional scene/style direction. Identity comes from the reference image.",
    )
    negative_prompt: str = Field(_SDXL_DEFAULT_NEGATIVE, max_length=2000)
    steps: int = Field(25, ge=1, le=60)
    cfg: float = Field(7.0, ge=1.0, le=15.0)
    identity_strength: float = Field(
        0.8, ge=0.0, le=2.0,
        description=(
            "IP-Adapter weight (style-transfer mode). 0.8 default keeps identity "
            "without crowding out the pose. Lower if pose still doesn't show; "
            "higher if face/body doesn't look like the reference."
        ),
    )
    pose_strength: float = Field(
        1.0, ge=0.0, le=2.0,
        description=(
            "ControlNet OpenPose strength. 1.0 default (xinsir's recommended max). "
            "Above 1.0 starts over-constraining: pose lands but anatomy distorts."
        ),
    )
    pose_end_percent: float = Field(
        0.8, ge=0.0, le=1.0,
        description=(
            "Fraction of denoising during which ControlNet is active. 0.8 default "
            "= enforce pose for the first 80%% of steps, let the model self-correct "
            "anatomy in the last 20%%. Lower if anatomy still looks broken."
        ),
    )
    seed: int = Field(-1, description="-1 means random")


class CharacterSwapParams(BaseModel):
    prompt: str = Field(
        "",
        max_length=2000,
        description="Optional scene/style context for the swap (Wan focuses on motion + identity from the inputs).",
    )
    steps: int = Field(20, ge=1, le=50)
    fps: int = Field(16, ge=8, le=30)
    frames: int = Field(81, ge=17, le=161, description="Number of frames to generate (~5-10 s at 16 fps).")
    # SAM2 seed point: pixel coordinates in the 832x480 resized source frame
    # that tell the segmenter "this is the character to track". Default is
    # center-of-frame, which works whenever the character is roughly centered
    # in frame 1. Override if your subject is off-center (e.g. (250, 240) for
    # a left-third subject).
    seed_x: int = Field(416, ge=0, le=832, description="SAM2 seed point X (0-832).")
    seed_y: int = Field(240, ge=0, le=480, description="SAM2 seed point Y (0-480).")
    seed: int = Field(-1, description="-1 means random")


_PARAMS_SCHEMA: dict[str, type[BaseModel]] = {
    "text-to-image": TextToImageParams,
    "juggernaut-xl": JuggernautParams,
    "image-edit": ImageEditParams,
    "image-char-swap": ImageCharSwapParams,
    "character-swap": CharacterSwapParams,
}

# Per-slug minimum-input requirements (number of uploaded files needed).
_MIN_INPUT_IDS: dict[str, int] = {
    "text-to-image": 0,
    "juggernaut-xl": 0,
    "image-edit": 1,
    "image-char-swap": 2,  # [source_image, reference_character_image]
    "character-swap": 2,  # [source_video, reference_character_image]
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
        params["seed"] = secrets.randbits(32)

    job = jobs.registry.create(slug, params)
    jobs.schedule(
        jobs.run_job(job.id, slug, model.workflow_file, params, req.input_ids),
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
