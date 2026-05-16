import asyncio
import base64
import json
import logging
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Literal

import storage
import workflows
from config import settings
from runpod_client import RunPodClient, RunPodError


log = logging.getLogger(__name__)


JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


@dataclass
class Job:
    id: str
    slug: str
    status: JobStatus = "queued"
    params: dict = field(default_factory=dict)
    runpod_request_id: str | None = None
    runpod_status: str | None = None
    output_files: list[str] = field(default_factory=list)
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, slug: str, params: dict) -> Job:
        job_id = storage.new_job_id()
        job = Job(id=job_id, slug=slug, params=params)
        with self._lock:
            self._jobs[job_id] = job
        self._persist(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields_) -> Job | None:
        with self._lock:
            j = self._jobs.get(job_id)
            if not j:
                return None
            for k, v in fields_.items():
                setattr(j, k, v)
            j.updated_at = time.time()
            self._persist(j)
            return j

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def _persist(self, job: Job) -> None:
        path = storage.jobs_dir() / f"{job.id}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(job), f, indent=2)
        except OSError as e:
            log.warning("Failed to persist job %s: %s", job.id, e)


registry = JobRegistry()


def _input_names_for_slug(slug: str, input_ids: list[str]) -> list[str]:
    """Map uploaded input IDs to the file names the workflow expects.

    - text-to-image: no inputs.
    - image-edit:    input[0] -> user_input.png  (FLUX Kontext LoadImage)
    - character-swap: input[0] -> user_input_video.mp4  (VHS_LoadVideo)
                      input[1] -> user_input_character.png  (LoadImage)
    """
    if slug == "character-swap":
        return ["user_input_video.mp4", "user_input_character.png"][: len(input_ids)]
    if slug == "image-char-swap":
        return ["user_input_source.png", "user_input_character.png"][: len(input_ids)]
    # default: first input gets the canonical image name, rest are numbered
    return [
        "user_input.png" if i == 0 else f"user_input_{i}.png"
        for i in range(len(input_ids))
    ]


# Per-slug max poll duration (seconds). Video jobs need more headroom.
_MAX_POLL_SECONDS: dict[str, int] = {
    "character-swap": 1500,  # 25 min for video generation
}


# Keep strong references to fire-and-forget tasks so they aren't GC'd mid-run.
_running_tasks: set[asyncio.Task] = set()


def schedule(coro) -> None:
    task = asyncio.create_task(coro)
    _running_tasks.add(task)
    task.add_done_callback(_running_tasks.discard)


async def run_job(
    job_id: str,
    slug: str,
    workflow_file: str,
    params: dict,
    input_ids: list[str] | None = None,
) -> None:
    """Background runner: build workflow -> submit to RunPod -> poll -> decode outputs."""
    try:
        client = RunPodClient()
    except RunPodError as e:
        registry.update(job_id, status="failed", error=str(e))
        return

    output_prefix = storage.safe_output_prefix(f"{slug}_{job_id}")
    full_params = {**params, "output_prefix": output_prefix}
    try:
        workflow = workflows.build_workflow(workflow_file, full_params)
    except Exception as e:
        registry.update(job_id, status="failed", error=f"workflow build: {e}")
        return

    # Resolve uploaded inputs to base64 payloads. Each slug has its own naming
    # convention so the workflow JSON can reference the inputs by a fixed name.
    images_payload: list[dict] = []
    if input_ids:
        names = _input_names_for_slug(slug, input_ids)
        for input_id, name in zip(input_ids, names):
            try:
                path = storage.resolve_input(input_id)
                raw = path.read_bytes()
            except (FileNotFoundError, OSError) as e:
                registry.update(
                    job_id,
                    status="failed",
                    error=f"upload {input_id}: {e}",
                )
                return
            images_payload.append({"name": name, "image": base64.b64encode(raw).decode("ascii")})

    try:
        request_id = await client.submit(workflow, images_payload or None)
    except RunPodError as e:
        registry.update(job_id, status="failed", error=f"submit: {e}")
        return

    registry.update(job_id, status="running", runpod_request_id=request_id)

    def _on_status(data: dict) -> None:
        registry.update(job_id, runpod_status=data.get("status"))

    try:
        max_seconds = _MAX_POLL_SECONDS.get(slug)
        result = await client.wait_for_completion(
            request_id, max_seconds=max_seconds, on_status=_on_status
        )
    except RunPodError as e:
        registry.update(job_id, status="failed", error=f"poll: {e}")
        return

    rp_status = result.get("status")
    if rp_status != "COMPLETED":
        registry.update(
            job_id,
            status="failed",
            error=f"runpod terminal status: {rp_status}: {result.get('error') or ''}",
        )
        return

    output = result.get("output") or {}

    # If the main worker returned 'success_no_images' (typical for video
    # workflows where VHS_VideoCombine writes an mp4 worker-comfyui doesn't
    # bubble back), try the companion downloader endpoint to fetch the
    # output(s) from the Network Volume.
    needs_downloader_fallback = (
        isinstance(output, dict)
        and output.get("status") == "success_no_images"
        and not output.get("images")
        and settings.RUNPOD_DOWNLOADER_ENDPOINT_ID
    )
    if needs_downloader_fallback:
        try:
            dl = RunPodClient(endpoint_id=settings.RUNPOD_DOWNLOADER_ENDPOINT_ID)
            dl_resp = await dl.run_sync(
                {"input": {"prefix": output_prefix, "delete_after": True}},
                max_seconds=120,
            )
        except RunPodError as e:
            registry.update(
                job_id,
                status="failed",
                error=f"downloader failed: {e}",
            )
            return
        dl_out = (dl_resp.get("output") or {}) if isinstance(dl_resp, dict) else {}
        dl_files = dl_out.get("files") or []
        if not dl_files:
            registry.update(
                job_id,
                status="failed",
                error=(
                    "Downloader returned no files. "
                    f"Looked for prefix '{output_prefix}*' on the volume. "
                    f"Response: {json.dumps(dl_out)[:500]}"
                ),
            )
            return
        # Replace the output dict with the downloader's so the unified
        # decoder below picks the files up.
        output = {"images": dl_files}

    # worker-comfyui returns image outputs under "images". Video / gif outputs
    # from VHS_VideoCombine end up under "gifs" (ComfyUI's UI convention) or
    # similar keys. We harvest all known output buckets.
    candidates: list = []
    if isinstance(output, dict):
        for key in ("images", "gifs", "videos", "files"):
            v = output.get(key)
            if isinstance(v, list):
                candidates.extend(v)
            elif v is not None:
                candidates.append(v)
    elif isinstance(output, list):
        candidates = output

    # Persist the raw response for post-mortem when nothing decodes cleanly.
    try:
        debug_path = storage.output_dir(job_id) / "_runpod_response.json"
        debug_path.write_text(
            json.dumps(result, default=str)[:200_000],  # cap at ~200 KB
            encoding="utf-8",
        )
    except OSError:
        pass

    output_files: list[str] = []
    skipped_reasons: list[str] = []
    for i, item in enumerate(candidates):
        data: str | None = None
        url_from_worker: str | None = None
        name = f"{i:04d}.png"  # fallback
        if isinstance(item, dict):
            data = item.get("data") or item.get("image") or item.get("video")
            url_from_worker = item.get("url")  # S3 / signed URL case
            if "filename" in item:
                name = item["filename"]
            elif "name" in item:
                name = item["name"]
        elif isinstance(item, str):
            data = item

        if not data and url_from_worker:
            # Worker is configured for S3 output and returned a URL instead
            # of inline base64. Fetch and save locally.
            try:
                with urllib.request.urlopen(url_from_worker, timeout=60) as resp:
                    raw = resp.read()
            except Exception as e:
                skipped_reasons.append(f"item[{i}] url fetch failed: {e}")
                continue
            saved = storage.write_output(job_id, name, raw)
            output_files.append(saved)
            continue

        if not data:
            if isinstance(item, dict):
                skipped_reasons.append(
                    f"item[{i}] keys={list(item.keys())} - no data/image/video/url"
                )
            else:
                skipped_reasons.append(f"item[{i}] type={type(item).__name__} unrecognized")
            continue
        if data.startswith("data:"):
            data = data.split(",", 1)[1]
        try:
            raw = base64.b64decode(data)
        except Exception as e:
            skipped_reasons.append(f"item[{i}] base64 decode failed: {e}")
            continue
        saved = storage.write_output(job_id, name, raw)
        output_files.append(saved)

    if not output_files:
        # Build a maximally useful diagnostic message.
        diag_parts: list[str] = []
        if isinstance(output, dict):
            diag_parts.append(f"output keys: {list(output.keys())}")
            imgs = output.get("images")
            if isinstance(imgs, list):
                diag_parts.append(f"images list length: {len(imgs)}")
                if imgs:
                    first = imgs[0]
                    if isinstance(first, dict):
                        diag_parts.append(
                            f"images[0] keys: {list(first.keys())} "
                            f"(types: {[type(v).__name__ for v in first.values()]})"
                        )
                    else:
                        diag_parts.append(f"images[0] type: {type(first).__name__}")
            for k in ("gifs", "videos", "files"):
                v = output.get(k)
                if v is not None:
                    diag_parts.append(f"{k}: {type(v).__name__} (len={len(v) if hasattr(v, '__len__') else 'n/a'})")
            status_field = output.get("status")
            if status_field is not None:
                diag_parts.append(f"output.status = {status_field!r}")
        else:
            diag_parts.append(f"output type: {type(output).__name__}")
        if skipped_reasons:
            diag_parts.append("skipped: " + "; ".join(skipped_reasons[:5]))

        diag_parts.append(f"full response saved to data/outputs/{job_id}/_runpod_response.json")

        registry.update(
            job_id,
            status="failed",
            error="No decodable outputs in RunPod response. " + " | ".join(diag_parts),
        )
        return

    registry.update(job_id, status="succeeded", output_files=output_files)
