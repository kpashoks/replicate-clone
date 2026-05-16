"""Lightweight downloader Serverless worker.

Attaches the same Network Volume as the main worker-comfyui endpoint. Given a
filename prefix, finds matching files in /runpod-volume/output/, reads them,
and returns base64-encoded contents in the response.

Used because worker-comfyui (5.8.x) only returns image-type outputs in its
response. Video files (mp4 from VHS_VideoCombine) live on disk only; this
endpoint exposes them to our local backend.

Input schema (POST /run):
  {
    "input": {
      "prefix": "character-swap_8b34967c1dfa",
      "delete_after": false
    }
  }

Output schema:
  {
    "files": [
      {
        "filename": "character-swap_8b34967c1dfa_00001.mp4",
        "data": "<base64>",
        "size": 5234567
      },
      ...
    ]
  }

Or on no matches:
  {"files": [], "error": "No files found matching prefix"}
"""

import base64
import glob
import os
import time
from typing import Any

import runpod


OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/runpod-volume/output")

# Per-file size cap to avoid blowing up the response. RunPod's /run async
# endpoint accepts ~25 MB on the wire (base64-encoded). A 50% safety margin
# gives ~18 MB raw, generously enough for any 5-15 s Wan clip at 480-720p.
MAX_BYTES_PER_FILE = 18 * 1024 * 1024

# Total response cap across all matched files.
MAX_TOTAL_BYTES = 22 * 1024 * 1024


def _safe_prefix(prefix: str) -> bool:
    """Reject path-traversal attempts."""
    if not prefix or len(prefix) > 256:
        return False
    bad = ("/", "\\", "..", "\0")
    return not any(b in prefix for b in bad)


def handler(event: dict) -> dict:
    input_data = event.get("input", {}) or {}
    prefix = input_data.get("prefix")
    delete_after = bool(input_data.get("delete_after", False))

    if not prefix:
        return {"files": [], "warning": "Missing required input field 'prefix'"}

    if not _safe_prefix(prefix):
        return {
            "files": [],
            "warning": f"Invalid prefix '{prefix}' (path traversal blocked)",
        }

    if not os.path.isdir(OUTPUT_DIR):
        # Don't use top-level "error" key - RunPod's serverless wrapper strips
        # that to the response root, hiding it from our backend which only
        # reads response.output. Surface the diagnostic via the regular output
        # dict instead.
        return {
            "files": [],
            "warning": (
                f"Output directory does not exist yet: {OUTPUT_DIR}. "
                "This usually means the main worker is still on the OLD image "
                "(no /comfyui/output -> /runpod-volume/output symlink). "
                "Terminate cached workers on the main endpoint so the next "
                "request pulls the latest worker image."
            ),
            "output_dir": OUTPUT_DIR,
        }

    # Allow ComfyUI's filename suffixing (e.g., "_00001.mp4") and a brief
    # retry in case the worker-comfyui pod is still flushing its writes.
    matches: list[str] = []
    deadline = time.time() + 5.0
    while time.time() < deadline:
        matches = sorted(glob.glob(os.path.join(OUTPUT_DIR, f"{prefix}*")))
        if matches:
            break
        time.sleep(0.3)

    if not matches:
        # List sample directory contents to help diagnose prefix mismatches.
        try:
            all_entries = sorted(os.listdir(OUTPUT_DIR))[:30]
        except OSError as e:
            all_entries = [f"<listdir failed: {e}>"]
        return {
            "files": [],
            "warning": f"No files matched '{prefix}*' in {OUTPUT_DIR}",
            "sample_entries": all_entries,
            "output_dir": OUTPUT_DIR,
        }

    files_out: list[dict[str, Any]] = []
    total_bytes = 0
    for path in matches:
        try:
            size = os.path.getsize(path)
        except OSError as e:
            files_out.append({"filename": os.path.basename(path), "error": f"stat failed: {e}"})
            continue

        if size > MAX_BYTES_PER_FILE:
            files_out.append(
                {
                    "filename": os.path.basename(path),
                    "error": f"file too large: {size} bytes > MAX_BYTES_PER_FILE",
                }
            )
            continue

        if total_bytes + size > MAX_TOTAL_BYTES:
            files_out.append(
                {
                    "filename": os.path.basename(path),
                    "error": f"would exceed MAX_TOTAL_BYTES ({MAX_TOTAL_BYTES})",
                }
            )
            continue

        try:
            with open(path, "rb") as fp:
                raw = fp.read()
        except OSError as e:
            files_out.append({"filename": os.path.basename(path), "error": f"read failed: {e}"})
            continue

        files_out.append(
            {
                "filename": os.path.basename(path),
                "data": base64.b64encode(raw).decode("ascii"),
                "size": len(raw),
            }
        )
        total_bytes += len(raw)

    # Optional cleanup so the volume doesn't fill up over time. Only deletes
    # files we successfully returned (i.e., have "data" set).
    if delete_after:
        for path, item in zip(matches, files_out):
            if item.get("data") is not None:
                try:
                    os.remove(path)
                except OSError:
                    pass

    return {
        "files": files_out,
        "total_bytes": total_bytes,
        "output_dir": OUTPUT_DIR,
    }


runpod.serverless.start({"handler": handler})
