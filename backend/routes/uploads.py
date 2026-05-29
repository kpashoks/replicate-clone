from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

import storage


router = APIRouter(prefix="/api/uploads", tags=["uploads"])


# Cap upload size at 32 MB. Images are typically 2-5 MB; short video clips for
# Wan 2.2 Animate (5-10 s) at 720p can be 10-25 MB. Base64 encoding adds ~33%
# overhead, so 32 MB raw becomes ~42 MB on the wire to RunPod /run (which
# permits up to ~50 MB per request).
MAX_UPLOAD_BYTES = 32 * 1024 * 1024


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


@router.get("/{input_id}", response_model=UploadResponse)
async def get_upload(input_id: str) -> UploadResponse:
    """Rehydrate an upload's metadata by id. Used when a recipe is loaded
    to restore the file preview in a dropzone without re-uploading. 404 if
    the file is no longer on disk (data/inputs/ was cleared)."""
    meta = storage.upload_meta(input_id)
    if meta is None:
        raise HTTPException(
            status_code=404,
            detail=f"Upload {input_id} not found (file may have been deleted)",
        )
    return UploadResponse(**meta)
