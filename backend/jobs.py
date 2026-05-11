import asyncio
import base64
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

import storage
import workflows
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

    # Resolve uploaded inputs to base64 payloads. The workflow JSON references
    # them by a fixed name ("user_input.png"); first input wins.
    images_payload: list[dict] = []
    if input_ids:
        for i, input_id in enumerate(input_ids):
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
            name = "user_input.png" if i == 0 else f"user_input_{i}.png"
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
        result = await client.wait_for_completion(request_id, on_status=_on_status)
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
    images = output.get("images")
    if images is None:
        # Some workers return {"output": [...]} directly
        images = output if isinstance(output, list) else []
    if not isinstance(images, list):
        images = [images]

    output_files: list[str] = []
    for i, item in enumerate(images):
        data: str | None = None
        name = f"{i:04d}.png"
        if isinstance(item, dict):
            data = item.get("data") or item.get("image")
            if "filename" in item:
                name = item["filename"]
        elif isinstance(item, str):
            data = item

        if not data:
            continue
        if data.startswith("data:"):
            data = data.split(",", 1)[1]
        try:
            raw = base64.b64decode(data)
        except Exception as e:
            log.warning("Could not base64-decode output %d: %s", i, e)
            continue
        url = storage.write_output(job_id, name, raw)
        output_files.append(url)

    if not output_files:
        registry.update(
            job_id,
            status="failed",
            error=f"No decodable outputs in RunPod response. Raw output keys: {list(output.keys()) if isinstance(output, dict) else type(output).__name__}",
        )
        return

    registry.update(job_id, status="succeeded", output_files=output_files)
