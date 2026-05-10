import asyncio
import time
from typing import Awaitable, Callable, Optional

import httpx

from config import settings


TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


class RunPodError(Exception):
    pass


class RunPodClient:
    """Thin async wrapper over RunPod Serverless REST API."""

    def __init__(self) -> None:
        if not settings.RUNPOD_API_KEY:
            raise RunPodError("RUNPOD_API_KEY is not set in .env")
        if not settings.RUNPOD_ENDPOINT_ID:
            raise RunPodError("RUNPOD_ENDPOINT_ID is not set in .env")
        self.base_url = f"https://api.runpod.ai/v2/{settings.RUNPOD_ENDPOINT_ID}"
        self.headers = {
            "Authorization": f"Bearer {settings.RUNPOD_API_KEY}",
            "Content-Type": "application/json",
        }

    async def submit(self, workflow: dict, images: Optional[list[dict]] = None) -> str:
        """Submit a workflow. Returns the RunPod request_id."""
        payload: dict = {"input": {"workflow": workflow}}
        if images:
            payload["input"]["images"] = images
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{self.base_url}/run", headers=self.headers, json=payload)
            if r.status_code != 200:
                raise RunPodError(f"RunPod submit failed [{r.status_code}]: {r.text}")
            data = r.json()
            request_id = data.get("id")
            if not request_id:
                raise RunPodError(f"RunPod submit returned no id: {data}")
            return request_id

    async def status(self, request_id: str) -> dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(f"{self.base_url}/status/{request_id}", headers=self.headers)
            if r.status_code != 200:
                raise RunPodError(f"RunPod status failed [{r.status_code}]: {r.text}")
            return r.json()

    async def cancel(self, request_id: str) -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{self.base_url}/cancel/{request_id}", headers=self.headers)
            if r.status_code != 200:
                raise RunPodError(f"RunPod cancel failed [{r.status_code}]: {r.text}")

    async def wait_for_completion(
        self,
        request_id: str,
        max_seconds: Optional[int] = None,
        on_status: Optional[Callable[[dict], Awaitable[None] | None]] = None,
    ) -> dict:
        """Poll /status until terminal. Returns the full final status payload."""
        max_seconds = max_seconds or settings.RUNPOD_TIMEOUT_SECONDS
        start = time.monotonic()
        delay = 2.0
        while True:
            data = await self.status(request_id)
            if on_status is not None:
                result = on_status(data)
                if asyncio.iscoroutine(result):
                    await result
            if data.get("status") in TERMINAL_STATUSES:
                return data
            if time.monotonic() - start > max_seconds:
                raise RunPodError(
                    f"RunPod request {request_id} timed out after {max_seconds}s "
                    f"(last status: {data.get('status')})"
                )
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 10.0)
