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
from models_registry import ModelEntry
from runpod_client import RunPodClient, RunPodError
from wan_animate_client import WanAnimateClient, WanAnimateError


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
# Empty by default - WAN_ANIMATE_TIMEOUT_SECONDS in config.py (default 5400s
# = 90 min) covers character-swap. Add slug-specific overrides here only if
# a model genuinely needs a budget different from the global default.
_MAX_POLL_SECONDS: dict[str, int] = {}


# Keep strong references to fire-and-forget tasks so they aren't GC'd mid-run.
_running_tasks: set[asyncio.Task] = set()


def schedule(coro) -> None:
    task = asyncio.create_task(coro)
    _running_tasks.add(task)
    task.add_done_callback(_running_tasks.discard)


async def run_job(
    job_id: str,
    model: ModelEntry,
    params: dict,
    input_ids: list[str] | None = None,
) -> None:
    """Top-level background runner. Dispatches to a provider-specific helper
    based on model.provider.

    The signature changed in this refactor: callers used to pass
    (job_id, slug, workflow_file, params, input_ids). They now pass the full
    ModelEntry so the dispatcher can branch on provider without re-looking-up
    the model. Routes/generate.py was updated to match.
    """
    if model.provider == "runpod":
        await _run_runpod_job(job_id, model, params, input_ids)
    elif model.provider == "wan-animate-http":
        await _run_wan_animate_http_job(job_id, model, params, input_ids)
    else:
        registry.update(
            job_id,
            status="failed",
            error=f"Unknown provider for slug {model.slug!r}: {model.provider!r}",
        )


async def _run_runpod_job(
    job_id: str,
    model: ModelEntry,
    params: dict,
    input_ids: list[str] | None,
) -> None:
    """RunPod ComfyUI-worker path. Logic preserved verbatim from the pre-
    refactor run_job - the only change is reading slug/workflow_file from the
    model entry rather than receiving them as separate args.
    """
    slug = model.slug
    workflow_file = model.workflow_file
    if not workflow_file:
        registry.update(
            job_id,
            status="failed",
            error=f"runpod provider but no workflow_file set for slug {slug!r}",
        )
        return

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
            warning = dl_out.get("warning")
            sample = dl_out.get("sample_entries")
            top_error = dl_resp.get("error") if isinstance(dl_resp, dict) else None
            parts = [f"Downloader returned no files for prefix '{output_prefix}*'."]
            if warning:
                parts.append(warning)
            if top_error and top_error != warning:
                parts.append(f"runpod error: {top_error}")
            if sample is not None:
                parts.append(f"sample entries in dir: {sample}")
            registry.update(
                job_id,
                status="failed",
                error=" | ".join(parts),
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

    # When multiple files are returned (e.g., batch outputs or a workflow with
    # multiple SaveImage nodes), sort by URL so the alphabetically-first
    # filename appears as the primary output in the UI. Workflows that need a
    # specific ordering should give their main output a prefix that sorts
    # before any auxiliary outputs.
    output_files.sort()

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


async def _run_wan_animate_http_job(
    job_id: str,
    model: ModelEntry,
    params: dict,
    input_ids: list[str] | None,
) -> None:
    """Dedicated wan-animate inference server path.

    Expected inputs (from input_ids, in this order):
      [0] source video file (mp4)
      [1] character reference image (png/jpg)
    This matches the existing convention in _input_names_for_slug() for the
    "character-swap" slug.
    """
    slug = model.slug

    # ---- Validate inputs ------------------------------------------------
    if not input_ids or len(input_ids) < 2:
        registry.update(
            job_id,
            status="failed",
            error=f"{slug} requires 2 inputs: [source_video, character_image]; got {len(input_ids or [])}",
        )
        return

    try:
        source_video_path = storage.resolve_input(input_ids[0])
        character_image_path = storage.resolve_input(input_ids[1])
    except (FileNotFoundError, OSError) as e:
        registry.update(job_id, status="failed", error=f"resolve inputs: {e}")
        return

    # ---- Submit to remote -----------------------------------------------
    try:
        client = WanAnimateClient()
    except WanAnimateError as e:
        registry.update(job_id, status="failed", error=str(e))
        return

    try:
        remote_job_id = await client.submit(
            character_image_path=character_image_path,
            source_video_path=source_video_path,
            prompt=params.get("prompt", ""),
            seed=params.get("seed", -1),
            resolution=params.get("resolution", "832x480"),
            replace_flag=params.get("replace_flag", True),
            sampling_steps=params.get("sampling_steps", 20),
            frame_num=params.get("frame_num", 81),
            refert_num=params.get("refert_num", 5),
            guide_scale=params.get("guide_scale", 5.0),
        )
    except WanAnimateError as e:
        registry.update(job_id, status="failed", error=f"submit: {e}")
        return

    # Reuse the runpod_request_id field to surface the remote job id in the
    # JobView - the frontend already renders this and the name change isn't
    # worth a schema migration.
    registry.update(job_id, status="running", runpod_request_id=remote_job_id)

    def _on_status(data: dict) -> None:
        # data has {status, progress_step, started_at, ...}
        # Surface progress_step via runpod_status so the existing UI shows it.
        step = data.get("progress_step") or data.get("status")
        if step:
            registry.update(job_id, runpod_status=str(step))

    # ---- Poll until terminal --------------------------------------------
    try:
        final = await client.wait_for_completion(
            remote_job_id,
            max_seconds=_MAX_POLL_SECONDS.get(slug, settings.WAN_ANIMATE_TIMEOUT_SECONDS),
            on_status=_on_status,
        )
    except WanAnimateError as e:
        registry.update(job_id, status="failed", error=f"poll: {e}")
        return

    if final.get("status") != "completed":
        err = final.get("error") or f"terminal status: {final.get('status')}"
        registry.update(job_id, status="failed", error=err)
        return

    # ---- Download mp4 and write to data/outputs/<job_id>/ ---------------
    try:
        mp4_bytes = await client.download_output(remote_job_id)
    except WanAnimateError as e:
        registry.update(job_id, status="failed", error=f"output download: {e}")
        return

    output_name = f"{slug}_{job_id}.mp4"
    saved = storage.write_output(job_id, output_name, mp4_bytes)
    registry.update(job_id, status="succeeded", output_files=[saved])
