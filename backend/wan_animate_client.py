"""
Async HTTP client for the dedicated wan-animate inference server.

Same shape as runpod_client.RunPodClient: submit a job, poll until terminal,
fetch the output. Differences from RunPod:

  - submit takes MULTIPART form data (uploading two files: character image
    and source video) rather than a JSON workflow payload.
  - status polling hits the wan-animate server's own /jobs/{id} endpoint,
    not RunPod's queue.
  - output is a direct mp4 stream from /jobs/{id}/output rather than a
    base64-encoded payload buried in a status response.

This module is provider-specific glue. The dispatcher in jobs.py decides
which provider to invoke based on ModelEntry.provider.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import httpx

from config import settings

log = logging.getLogger(__name__)


class WanAnimateError(Exception):
    pass


_TERMINAL = {"completed", "failed"}


class WanAnimateClient:
    def __init__(self, base_url: Optional[str] = None) -> None:
        url = (base_url or settings.WAN_ANIMATE_ENDPOINT).rstrip("/")
        if not url:
            raise WanAnimateError(
                "WAN_ANIMATE_ENDPOINT is not set in .env. Deploy the wan-animate "
                "server (see runpod-wan-animate/README.md) and set this to the "
                "Pod's public HTTP URL."
            )
        self.base_url = url

    async def submit(
        self,
        character_image_path: Path,
        source_video_path: Path,
        *,
        prompt: str = "",
        seed: int = -1,
        resolution: str = "832x480",
        replace_flag: bool = True,
        sampling_steps: int = 20,
        frame_num: int = 81,
        refert_num: int = 5,
        guide_scale: float = 5.0,
        timeout: float = 60.0,
    ) -> str:
        """Multipart upload + form fields. Returns the server-assigned job_id."""
        for p in (character_image_path, source_video_path):
            if not p.exists():
                raise WanAnimateError(f"Input file not found: {p}")

        # httpx wants tuples of (filename, file_obj, content_type) for multipart.
        # Keep both files open during the request via a context manager.
        with character_image_path.open("rb") as fchar, source_video_path.open("rb") as fvid:
            files = {
                "character_image": (
                    character_image_path.name, fchar, "image/png",
                ),
                "source_video": (
                    source_video_path.name, fvid, "video/mp4",
                ),
            }
            data = {
                "prompt": prompt,
                "seed": str(seed),
                "resolution": resolution,
                "replace_flag": str(replace_flag).lower(),
                "sampling_steps": str(sampling_steps),
                "frame_num": str(frame_num),
                "refert_num": str(refert_num),
                "guide_scale": str(guide_scale),
            }
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(
                    f"{self.base_url}/character-swap", files=files, data=data,
                )
        if r.status_code != 200:
            raise WanAnimateError(
                f"wan-animate submit failed [{r.status_code}]: {r.text[:500]}"
            )
        body = r.json()
        job_id = body.get("job_id")
        if not job_id:
            raise WanAnimateError(f"submit returned no job_id: {body}")
        return str(job_id)

    async def status(self, job_id: str, *, timeout: float = 60.0) -> dict:
        # Separate the read timeout from the connect timeout - the Pod is
        # doing heavy GPU work and an event-loop stall under load can briefly
        # block FastAPI's /jobs/{id} response. Be generous on reads.
        timeout_cfg = httpx.Timeout(timeout, connect=10.0, read=timeout, write=10.0)
        async with httpx.AsyncClient(timeout=timeout_cfg) as client:
            r = await client.get(f"{self.base_url}/jobs/{job_id}")
        if r.status_code != 200:
            raise WanAnimateError(
                f"wan-animate status failed [{r.status_code}]: {r.text[:500]}"
            )
        return r.json()

    async def wait_for_completion(
        self,
        job_id: str,
        *,
        max_seconds: Optional[int] = None,
        on_status: Optional[Callable[[dict], Awaitable[None] | None]] = None,
    ) -> dict:
        """Poll until the remote job is completed or failed. Returns final status.

        Resilient to transient httpx errors (ReadTimeout, ConnectError,
        RemoteProtocolError, etc.). Wan inference runs for minutes and the
        Pod's HTTP server can briefly stall under GPU load -- one network
        blip should not kill a job that's still progressing on the Pod.

        Polling only gives up when the absolute max_seconds budget is
        exhausted, OR when consecutive failures exceed an internal threshold.
        """
        max_seconds = max_seconds or settings.WAN_ANIMATE_TIMEOUT_SECONDS
        start = time.monotonic()
        delay = 2.0
        last: dict = {}
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 10  # ~30-100s of unbroken failure at backoff cap
        while True:
            try:
                last = await self.status(job_id)
                consecutive_errors = 0
            except (
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
                httpx.ConnectError,
                httpx.RemoteProtocolError,
                httpx.ReadError,
                httpx.WriteError,
                httpx.NetworkError,
            ) as e:
                consecutive_errors += 1
                log.warning(
                    "wan-animate status poll transient error #%d for job %s: %s: %s",
                    consecutive_errors, job_id, type(e).__name__, e,
                )
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    raise WanAnimateError(
                        f"wan-animate status polling for job {job_id} failed "
                        f"{consecutive_errors} times in a row "
                        f"(last error: {type(e).__name__}: {e}). "
                        f"The Pod may be down -- check /var/log/wan-animate.log."
                    ) from e
                if time.monotonic() - start > max_seconds:
                    raise WanAnimateError(
                        f"wan-animate job {job_id} timed out after {max_seconds}s "
                        f"(last error: {type(e).__name__}: {e})"
                    ) from e
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, 10.0)
                continue

            if on_status is not None:
                res = on_status(last)
                if asyncio.iscoroutine(res):
                    await res
            s = (last.get("status") or "").lower()
            if s in _TERMINAL:
                return last
            if time.monotonic() - start > max_seconds:
                raise WanAnimateError(
                    f"wan-animate job {job_id} timed out after {max_seconds}s "
                    f"(last status={s})"
                )
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 10.0)

    async def download_output(self, job_id: str, *, timeout: float = 300.0) -> bytes:
        """Fetch the mp4 file for a completed job."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{self.base_url}/jobs/{job_id}/output")
        if r.status_code != 200:
            raise WanAnimateError(
                f"wan-animate output download failed [{r.status_code}]: "
                f"{r.text[:200] if r.text else '<empty>'}"
            )
        return r.content
