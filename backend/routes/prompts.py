from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import prompt_enhance


router = APIRouter(prefix="/api/prompts", tags=["prompts"])


class EnhanceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    target_model: str = "text-to-image"


class EnhanceResponse(BaseModel):
    original: str
    enhanced: str
    target_model: str


@router.post("/enhance", response_model=EnhanceResponse)
def enhance_prompt(req: EnhanceRequest) -> EnhanceResponse:
    try:
        enhanced = prompt_enhance.enhance(req.prompt, req.target_model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prompt enhancement failed: {e}")
    return EnhanceResponse(original=req.prompt, enhanced=enhanced, target_model=req.target_model)
