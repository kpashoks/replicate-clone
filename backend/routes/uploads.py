from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

import storage


router = APIRouter(prefix="/api/uploads", tags=["uploads"])


# Cap upload size at 8 MB (FLUX Kontext doesn't benefit from larger inputs, and
# base64-encoded payloads to RunPod balloon to ~10.6 MB at that size).
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


class UploadResponse(BaseModel):
    id: str
    url: str
    name: str
    size: int


@router.post("", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(raw)} bytes, max {MAX_UPLOAD_BYTES}",
        )
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        rec = storage.save_upload(raw, file.filename or "upload.bin")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return UploadResponse(**rec)
