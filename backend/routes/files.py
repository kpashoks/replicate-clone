from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import settings


router = APIRouter(prefix="/api/files", tags=["files"])


_KIND_TO_SUBDIR = {
    "outputs": "outputs",
    "inputs": "inputs",
}


@router.get("/{kind}/{job_id}/{name}")
async def get_file(
    kind: str,
    job_id: str,
    name: str,
    download: bool = False,
) -> FileResponse:
    sub = _KIND_TO_SUBDIR.get(kind)
    if not sub:
        raise HTTPException(status_code=404, detail="Unknown file kind")

    data_root = settings.data_dir_abs.resolve()
    target = (data_root / sub / job_id / name).resolve()

    # Path-traversal guard: target must be under data_root
    try:
        target.relative_to(data_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not target.is_file():
        raise HTTPException(status_code=404, detail="Not found")

    # When download=1 is passed, set filename which forces
    # Content-Disposition: attachment so the browser saves it.
    # Otherwise serve inline (used by <img src> and "open in new tab").
    if download:
        return FileResponse(target, filename=name)
    return FileResponse(target)
