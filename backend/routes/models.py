from fastapi import APIRouter

from models_registry import ModelEntry, list_models


router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=list[ModelEntry])
def get_models() -> list[ModelEntry]:
    return list_models()
