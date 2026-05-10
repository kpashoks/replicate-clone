import re
import uuid
from pathlib import Path

from config import settings


_SAFE_RE = re.compile(r"[^a-zA-Z0-9_-]")


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def output_dir(job_id: str) -> Path:
    p = settings.data_dir_abs / "outputs" / job_id
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
