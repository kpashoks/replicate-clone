from typing import Literal

from pydantic import BaseModel


OutputKind = Literal["image", "video"]


class ModelEntry(BaseModel):
    slug: str
    label: str
    description: str
    workflow_file: str
    output_kind: OutputKind
    accepts_image: bool = False
    accepts_video: bool = False
    stage: int = 1
    available: bool = False


REGISTRY: dict[str, ModelEntry] = {
    "text-to-image": ModelEntry(
        slug="text-to-image",
        label="Text to Image (FLUX)",
        description="Generate an image from a text prompt using FLUX.1 [dev].",
        workflow_file="text2img_flux.json",
        output_kind="image",
        stage=1,
        available=True,
    ),
    "juggernaut-xl": ModelEntry(
        slug="juggernaut-xl",
        label="Juggernaut XL (Photorealistic)",
        description="SDXL photorealism workhorse. Uses dual CLIP + negative prompts. Slower but often more cinematic for portraits than FLUX.",
        workflow_file="text2img_juggernaut.json",
        output_kind="image",
        stage=1,
        available=True,
    ),
    "image-edit": ModelEntry(
        slug="image-edit",
        label="Image Edit",
        description="Edit an uploaded image with a text prompt using FLUX.1 Kontext [dev].",
        workflow_file="imgedit_flux_kontext.json",
        output_kind="image",
        accepts_image=True,
        stage=1,
        available=True,
    ),
    "character-swap": ModelEntry(
        slug="character-swap",
        label="Character Swap (Video)",
        description="Replace one character in a 5-10s video with a reference character image, using Wan 2.2 Animate.",
        workflow_file="video_swap_wan22_animate.json",
        output_kind="video",
        accepts_image=True,
        accepts_video=True,
        stage=1,
        available=True,
    ),
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
    ),
}


def list_models() -> list[ModelEntry]:
    return list(REGISTRY.values())


def get_model(slug: str) -> ModelEntry | None:
    return REGISTRY.get(slug)
