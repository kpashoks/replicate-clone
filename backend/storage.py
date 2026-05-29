import hashlib
import re
import uuid
from pathlib import Path

from config import settings


_SAFE_RE = re.compile(r"[^a-zA-Z0-9_.-]")
_ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
_ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
# Audio for the i2v driving-audio / reference-audio upload fields.
_ALLOWED_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
_ALLOWED_UPLOAD_EXTS = _ALLOWED_IMAGE_EXTS | _ALLOWED_VIDEO_EXTS | _ALLOWED_AUDIO_EXTS


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def output_dir(job_id: str) -> Path:
    p = settings.data_dir_abs / "outputs" / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def inputs_dir() -> Path:
    p = settings.data_dir_abs / "inputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def jobs_dir() -> Path:
    p = settings.data_dir_abs / "jobs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_output_prefix(s: str) -> str:
    """ComfyUI uses the prefix as part of the output filename; sanitize."""
    cleaned = _SAFE_RE.sub("_", s)[:64]
    return cleaned or "out"


def write_output(job_id: str, name: str, raw_bytes: bytes) -> str:
    """Write a binary output to data/outputs/<job_id>/<name>. Returns a relative URL served by /api/files."""
    d = output_dir(job_id)
    safe_name = _SAFE_RE.sub("_", Path(name).stem) + Path(name).suffix
    p = d / safe_name
    p.write_bytes(raw_bytes)
    return f"/api/files/outputs/{job_id}/{safe_name}"


def save_upload(file_bytes: bytes, filename: str) -> dict:
    """Save an uploaded file to data/inputs/<sha256>/<original_filename> (content-addressed).

    Returns a dict with id, url, name, size.
    """
    ext = Path(filename).suffix.lower() or ".bin"
    if ext not in _ALLOWED_UPLOAD_EXTS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Allowed: {sorted(_ALLOWED_UPLOAD_EXTS)}"
        )

    h = hashlib.sha256(file_bytes).hexdigest()[:16]
    safe_name = _SAFE_RE.sub("_", Path(filename).stem) + ext

    d = inputs_dir() / h
    d.mkdir(parents=True, exist_ok=True)
    p = d / safe_name
    p.write_bytes(file_bytes)

    return {
        "id": h,
        "url": f"/api/files/inputs/{h}/{safe_name}",
        "name": safe_name,
        "size": len(file_bytes),
    }


def resolve_input(input_id: str) -> Path:
    """Resolve a previously-uploaded input ID to its file path."""
    d = inputs_dir() / input_id
    if not d.is_dir():
        raise FileNotFoundError(f"Unknown input id: {input_id}")
    # There should be exactly one file in the directory.
    files = [p for p in d.iterdir() if p.is_file()]
    if not files:
        raise FileNotFoundError(f"Input id {input_id} has no file on disk")
    return files[0]
