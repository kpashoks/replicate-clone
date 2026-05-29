"""Atlas Cloud per-model parameter schemas.

Atlas hosts an OpenAPI 3.0 schema per model at:
    https://static.atlascloud.ai/model/schema/<slug-with-slashes-as-dashes>.json

This module fetches and caches those schemas, then simplifies them into a
flat list of parameter descriptors the frontend can render as form fields.

Cache lifetime: process. Atlas changes schemas rarely; restart uvicorn to
refresh. Cache misses (404 / network error) are also cached negatively so
we don't re-probe a missing schema on every page load.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx


log = logging.getLogger(__name__)

# atlas_model_id -> simplified param list (or None for no-schema-available)
_CACHE: dict[str, list[dict] | None] = {}
_CACHE_LOCK = asyncio.Lock()


# Property names we never want to expose in the dynamic form because they
# are either set server-side (model), handled by specialized UI controls
# (prompt/seed/lora_*), or supplied via the PRIMARY upload dropzone
# (image/video for i2i/i2v/v2v -- the task page's own dropzone).
_FILTERED_NAMES: set[str] = {
    "model",
    "prompt",
    "seed",
    "lora_url",
    "lora_scale",
    "loras",
    "image",        # primary first-frame / source image (task-page dropzone)
    "images",
    "image_urls",
    "video",        # primary source video (task-page dropzone)
    "videos",
    "mask_image",
    "reference_images",
    "reference_videos",
    "reference_audios",
    # Output-control / system-y knobs that don't belong in the form:
    "enable_base64_output",
    "enable_sync_mode",
}

# SECONDARY file-upload fields: these are optional extra files (end frame,
# driving audio) that we DO want to surface -- as upload dropzones, not
# URL text inputs. Maps field name -> upload kind for the frontend widget.
# A field is also treated as an uploader if its x-ui-component is
# "uploader"/"uploaders" (Atlas's own hint), with kind inferred by name.
_UPLOAD_FIELD_KINDS: dict[str, str] = {
    "last_image": "image",
    "end_image": "image",
    "tail_image": "image",
    "first_image": "image",
    "audio": "audio",
    "audio_url": "audio",
    "reference_audio": "audio",
    "driving_audio": "audio",
}


def _upload_kind_for(name: str, ui_component: str) -> str | None:
    """Return 'image' | 'audio' | 'video' if this field is a secondary
    file-upload field, else None. Combines an explicit name map with
    Atlas's x-ui-component=uploader hint (kind inferred from the name)."""
    if name in _UPLOAD_FIELD_KINDS:
        return _UPLOAD_FIELD_KINDS[name]
    if ui_component in ("uploader", "uploaders"):
        lname = name.lower()
        if "audio" in lname or "sound" in lname or "voice" in lname:
            return "audio"
        if "video" in lname or "clip" in lname:
            return "video"
        return "image"  # default for an uploader of unknown kind
    return None


def _humanize(name: str) -> str:
    """Convert a snake_case param name into a Title Case label.

    Examples:
        cfg_scale       -> "Cfg Scale"
        enhance_prompt  -> "Enhance Prompt"
        num_inference_steps -> "Num Inference Steps"
    """
    return name.replace("_", " ").title()


def _schema_url(atlas_model_id: str) -> str:
    """Atlas slug -> static schema URL. Slashes become dashes."""
    fname = atlas_model_id.replace("/", "-")
    return f"https://static.atlascloud.ai/model/schema/{fname}.json"


def _simplify(input_schema: dict) -> list[dict]:
    """Walk an OpenAPI Input schema and produce a flat param list.

    Each output param has:
        name        - the raw JSON key (sent to Atlas verbatim)
        label       - human-readable name for the form label
        description - help text for the tooltip
        type        - "string" | "integer" | "number" | "boolean" | "array"
        default     - suggested default (or None)
        enum        - list of allowed values (or None)
        minimum     - numeric lower bound (or None)
        maximum     - numeric upper bound (or None)
        step        - numeric step (or None)
        ui_component- Atlas's hint: "slider" | "select" | "textarea" | "uploaders" | ""
        required    - True if this property is in the schema's `required` list

    Filters out _FILTERED_NAMES so the form doesn't double up with
    specialized controls (prompt textarea + Enhance button, seed dice,
    upload dropzones, etc.).
    """
    props = input_schema.get("properties", {}) or {}
    required = set(input_schema.get("required", []) or [])
    out: list[dict] = []
    for name, p in props.items():
        if name in _FILTERED_NAMES:
            continue
        if not isinstance(p, dict):
            continue
        ui = p.get("x-ui-component") or ""
        upload_kind = _upload_kind_for(name, ui)
        out.append(
            {
                "name": name,
                "label": _humanize(name),
                "description": p.get("description") or "",
                "type": p.get("type") or "string",
                "default": p.get("default"),
                "enum": p.get("enum") if isinstance(p.get("enum"), list) else None,
                "minimum": p.get("minimum"),
                "maximum": p.get("maximum"),
                "step": p.get("step"),
                "ui_component": ui,
                "required": name in required,
                # Secondary file-upload field (end frame, driving audio).
                # When set, the frontend renders an upload dropzone of this
                # kind and the backend resolves an "upload://<id>" sentinel
                # value into an Atlas media URL at submit time.
                "is_upload": upload_kind is not None,
                "upload_kind": upload_kind or "",
            }
        )
    return out


async def get_params_schema(atlas_model_id: str) -> list[dict] | None:
    """Return the simplified param list for an Atlas model.

    Returns None if Atlas doesn't host a schema for this model (rare but
    real -- e.g. flux-2-pro, imagen-4-ultra, ideogram-v3 all 404'd in
    our survey). The frontend should render a minimal prompt-only form
    in that case.

    Cached after the first fetch (positive AND negative) until uvicorn
    restart.
    """
    async with _CACHE_LOCK:
        if atlas_model_id in _CACHE:
            return _CACHE[atlas_model_id]

    url = _schema_url(atlas_model_id)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url)
    except httpx.HTTPError as e:
        log.warning("Atlas schema fetch failed for %s: %s", atlas_model_id, e)
        async with _CACHE_LOCK:
            _CACHE[atlas_model_id] = None
        return None

    if r.status_code != 200:
        log.info(
            "Atlas schema not available for %s (HTTP %d)",
            atlas_model_id, r.status_code,
        )
        async with _CACHE_LOCK:
            _CACHE[atlas_model_id] = None
        return None

    try:
        spec = r.json()
    except ValueError as e:
        log.warning("Atlas schema JSON parse error for %s: %s", atlas_model_id, e)
        async with _CACHE_LOCK:
            _CACHE[atlas_model_id] = None
        return None

    inp = (spec.get("components", {}) or {}).get("schemas", {}).get("Input", {})
    if not isinstance(inp, dict) or not inp.get("properties"):
        log.warning(
            "Atlas schema for %s has no usable Input.properties", atlas_model_id,
        )
        async with _CACHE_LOCK:
            _CACHE[atlas_model_id] = None
        return None

    params = _simplify(inp)
    async with _CACHE_LOCK:
        _CACHE[atlas_model_id] = params
    log.info(
        "Atlas schema for %s: %d param(s) after filtering", atlas_model_id, len(params),
    )
    return params


def clear_cache() -> None:
    """Useful for tests / hot-reload during dev."""
    _CACHE.clear()
