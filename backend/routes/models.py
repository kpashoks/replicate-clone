from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import atlas_schemas
from models_registry import REGISTRY, ModelEntry, list_models


router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=list[ModelEntry])
def get_models() -> list[ModelEntry]:
    return list_models()


class ParamSpec(BaseModel):
    """One renderable form field. Mirrors the dict shape produced by
    atlas_schemas._simplify() so the frontend has a typed contract."""
    name: str
    label: str
    description: str = ""
    type: str
    default: object | None = None
    enum: list | None = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    ui_component: str = ""
    required: bool = False


class ModelSchemaResponse(BaseModel):
    """Response shape of GET /api/models/{slug}/schema.

    provider: the model's provider, so the frontend knows whether to use
              the dynamic form (Atlas) or fall back to the hardcoded one.
    params:   the renderable parameter list, or None if no schema is
              available (Atlas doesn't host one for this model, OR the
              model isn't an Atlas model). Frontend should render its
              hardcoded form in that case.
    """
    provider: str
    params: list[ParamSpec] | None


@router.get("/{slug}/schema", response_model=ModelSchemaResponse)
async def get_model_schema(slug: str) -> ModelSchemaResponse:
    model = REGISTRY.get(slug)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Unknown model slug: {slug}")

    if model.provider != "atlas" or not model.atlas_model_id:
        # Non-Atlas models keep their existing hardcoded UI; the frontend
        # uses provider != "atlas" as the cue to skip the dynamic form.
        return ModelSchemaResponse(provider=model.provider, params=None)

    params = await atlas_schemas.get_params_schema(model.atlas_model_id)
    return ModelSchemaResponse(
        provider=model.provider,
        params=[ParamSpec(**p) for p in params] if params is not None else None,
    )
