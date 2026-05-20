"""Atlas Cloud API client.

Atlas is a unified inference aggregator — 600+ models behind one API key,
NSFW-permissive (their stated policy: "removes content moderation at the
model level"). We use it for:

  - Image generation (Grok Imagine and friends)
    POST /api/v1/model/generateImage    -> {data: {id: prediction_id}}
    GET  /api/v1/model/prediction/{id}  -> poll until status == completed

  - Video generation (HappyHorse, Wan, Kling, Veo, etc.)
    POST /api/v1/model/generateVideo    -> same async pattern

Output URLs come back signed under outputs[].url (or similar). We download
them ourselves so the user-facing app keeps the same "files served from
data/outputs/{job_id}/" contract the RunPod path uses.
"""

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Optional

import httpx

from config import settings


log = logging.getLogger(__name__)


# Atlas uses ad-hoc status strings across model providers. We treat anything
# that contains one of these as terminal.
_TERMINAL_OK = {"completed", "succeeded", "success", "finished"}
_TERMINAL_BAD = {"failed", "error", "cancelled", "canceled", "timeout", "timed_out"}


class AtlasError(Exception):
    pass


class AtlasClient:
    def __init__(self) -> None:
        if not settings.ATLAS_CLOUD_API_KEY:
            raise AtlasError("ATLAS_CLOUD_API_KEY is not set in .env")
        self.base_url = settings.ATLAS_CLOUD_BASE_URL.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {settings.ATLAS_CLOUD_API_KEY}",
            "Content-Type": "application/json",
        }

    # ---- low-level submit + poll -----------------------------------------

    async def submit_image(self, model: str, prompt: str, **extra: Any) -> str:
        """Submit an image-gen request. Returns prediction_id."""
        body = {"model": model, "prompt": prompt, **extra}
        return await self._submit("/api/v1/model/generateImage", body)

    async def submit_video(self, model: str, prompt: str, **extra: Any) -> str:
        """Submit a video-gen request. Returns prediction_id."""
        body = {"model": model, "prompt": prompt, **extra}
        return await self._submit("/api/v1/model/generateVideo", body)

    async def _submit(self, path: str, body: dict) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{self.base_url}{path}", headers=self.headers, json=body)
        if r.status_code not in (200, 201, 202):
            raise AtlasError(f"Atlas submit {path} failed [{r.status_code}]: {r.text[:500]}")
        data = r.json()
        pid = (
            (data.get("data") or {}).get("id")
            or data.get("id")
            or data.get("prediction_id")
        )
        if not pid:
            raise AtlasError(f"Atlas submit returned no prediction id. body={data}")
        return str(pid)

    async def status(self, prediction_id: str) -> dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(
                f"{self.base_url}/api/v1/model/prediction/{prediction_id}",
                headers=self.headers,
            )
        if r.status_code != 200:
            raise AtlasError(f"Atlas status failed [{r.status_code}]: {r.text[:500]}")
        return r.json()

    async def wait_for_completion(
        self,
        prediction_id: str,
        *,
        max_seconds: int,
        on_status: Optional[Callable[[dict], Awaitable[None] | None]] = None,
    ) -> dict:
        """Poll until terminal. Returns the final status payload."""
        start = time.monotonic()
        delay = 2.0
        last: dict = {}
        while True:
            last = await self.status(prediction_id)
            if on_status is not None:
                res = on_status(last)
                if asyncio.iscoroutine(res):
                    await res
            status = _extract_status(last)
            if status in _TERMINAL_OK:
                return last
            if status in _TERMINAL_BAD:
                raise AtlasError(
                    f"Atlas prediction {prediction_id} terminated: status={status} "
                    f"body={str(last)[:500]}"
                )
            if time.monotonic() - start > max_seconds:
                raise AtlasError(
                    f"Atlas prediction {prediction_id} timed out after {max_seconds}s "
                    f"(last status={status})"
                )
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 10.0)

    # ---- input upload ----------------------------------------------------

    async def upload_media(self, file_path, *, content_type: str | None = None) -> str:
        """Upload a local file to Atlas's media bucket and return the public
        https URL. Atlas's I2I models reject data: URLs and require real
        https URLs (typically rooted at storage.atlascloud.ai), so any local
        upload must go through this endpoint before being referenced in the
        request body.

        Endpoint: POST /api/v1/model/uploadMedia (multipart/form-data)
        Response: {"data": {"download_url": "https://storage.atlascloud.ai/uploads/.../<name>"}}
        """
        from pathlib import Path

        p = Path(file_path)
        if not p.exists():
            raise AtlasError(f"upload_media: file not found: {p}")

        # Don't set Content-Type here — httpx will pick the multipart boundary.
        auth_headers = {"Authorization": self.headers["Authorization"]}
        files = {"file": (p.name, p.read_bytes(), content_type or "application/octet-stream")}

        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{self.base_url}/api/v1/model/uploadMedia",
                headers=auth_headers,
                files=files,
            )
        if r.status_code not in (200, 201):
            raise AtlasError(f"Atlas uploadMedia failed [{r.status_code}]: {r.text[:500]}")
        data = r.json()
        url = (
            (data.get("data") or {}).get("download_url")
            or (data.get("data") or {}).get("url")
            or data.get("download_url")
            or data.get("url")
        )
        if not url:
            raise AtlasError(f"Atlas uploadMedia returned no download_url. body={data}")
        return str(url)

    # ---- output extraction -----------------------------------------------

    async def download(self, url: str, *, timeout: float = 300.0) -> bytes:
        """Fetch a signed output URL and return raw bytes."""
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            r = await client.get(url)
        if r.status_code != 200:
            raise AtlasError(f"Atlas output download failed [{r.status_code}]: {url}")
        return r.content

    # ---- delete (console / dashboard API) --------------------------------

    async def delete_prediction(self, prediction_id: str) -> dict:
        """Delete a prediction from Atlas's dashboard history.

        Note: the dashboard's delete endpoint lives on a DIFFERENT subdomain
        (`console.atlascloud.ai`) from the public submit/poll API
        (`api.atlascloud.ai`). The user's API key works on both, but the
        URL is hardcoded here rather than derived from ATLAS_CLOUD_BASE_URL.

        On success, Atlas returns 200 with a small JSON body. We return it
        for logging. On 404 we treat it as already-deleted (idempotent).
        Any other status raises AtlasError.

        Endpoint reverse-engineered from the dashboard's DevTools Network
        tab; not part of the documented public API surface as of 2026-05.
        """
        # Hardcoded because the documented base is api.atlascloud.ai and the
        # dashboard surface is console.atlascloud.ai. If Atlas ever changes
        # this, override via ATLAS_CONSOLE_BASE_URL env var.
        console_base = "https://console.atlascloud.ai"
        url = f"{console_base}/api/v1/model/history/{prediction_id}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.delete(url, headers=self.headers)
        if r.status_code == 404:
            # Already gone (or never existed). Treat as a no-op rather than
            # an error - the user's intent ("remove this") is satisfied.
            log.info("Atlas prediction %s already absent (404)", prediction_id)
            return {"already_absent": True}
        if r.status_code != 200:
            raise AtlasError(
                f"Atlas delete failed [{r.status_code}]: "
                f"{r.text[:300] if r.text else '<empty body>'}"
            )
        try:
            return r.json()
        except ValueError:
            return {}


def _extract_status(payload: dict) -> str:
    """Atlas wraps the actual status inside data or surfaces it at the top.
    Normalize to a lowercase string we can compare."""
    if not isinstance(payload, dict):
        return ""
    d = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    s = d.get("status") or payload.get("status") or ""
    return str(s).lower().strip()


def extract_output_urls(payload: dict) -> list[str]:
    """Pull all media URLs out of a completed Atlas prediction payload.

    Atlas shapes vary across models:
      data.outputs: ["https://..."]                  (e.g. Nano Banana 2, Wan)
      data.outputs: [{url: "https://...", ...}]      (some video models)
      data.output_url / data.output: "https://..."   (single-result T2I)
      data.images[].url, data.videos[].url           (older shapes)
    Return URL strings (de-duplicated, preserving order).
    """
    urls: list[str] = []
    seen: set[str] = set()

    def _add(u: Any) -> None:
        if isinstance(u, str) and u.startswith(("http://", "https://")) and u not in seen:
            urls.append(u)
            seen.add(u)

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k in ("url", "output_url", "output", "video_url", "image_url"):
                v = obj.get(k)
                if isinstance(v, str):
                    _add(v)
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                if isinstance(v, str):
                    # Lists-of-URL-strings, e.g. data.outputs: ["https://..."]
                    _add(v)
                else:
                    _walk(v)

    _walk(payload)
    return urls
