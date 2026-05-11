from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

import storage


router = APIRouter(prefix="/api/uploads", tags=["uploads"])


# Cap upload size at 16 MB. Images are typically 2-5 MB; short video clips for
# Wan 2.2 Animate (5-10 s at 480p) usually weigh 3-15 MB. Base64 encoding adds
# ~33% overhead, so 16 MB raw becomes ~21 MB on the wire to RunPod.
MAX_UPLOAD_BYTES = 16 * 1024 * 1024


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
