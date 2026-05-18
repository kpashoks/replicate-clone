"""
FastAPI inference server for Wan 2.2 Animate (character replace).

Bypasses ComfyUI in favor of the upstream Wan-Video/Wan2.2 Python API plus
Meta's facebookresearch/sam2. Matches Replicate's `wan-2.2-animate-replace`
pipeline as observed in their prediction logs (sam2.sam2_video_predictor +
chunked dual-stage diffusion).

API shape mirrors the skyreels-story server pattern (already battle-tested
in a sibling project):
  POST /character-swap     -> multipart upload, returns job_id immediately
  GET  /jobs/{job_id}      -> polling status
  GET  /jobs/{job_id}/output -> stream the generated mp4
  GET  /health             -> liveness + model-loaded readiness

Implementation notes:
  - Lazy model loading on first request so container start is fast.
  - asyncio.Lock around the GPU inference call (single in-flight job).
  - Job registry is in-memory + per-job JSON files for post-mortem debugging.
  - Output mp4s land in /runpod-volume/output/wan-animate/ so the main app's
    downloader endpoint can fetch them with no extra plumbing.

VERIFY AT FIRST DEPLOY: the exact class name + constructor signature of the
Wan-Video preprocessing pipeline. Web search showed the file at
  /opt/wan22/wan/modules/animate/preprocess/process_pipepline.py
with a __call__ method, but did not surface the class name. The TRY block in
_load_model() will print a helpful error if the import is wrong; fix per the
runtime traceback rather than guessing here.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import imageio
import numpy as np
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, Field


# --- safetensors CUDA workaround ------------------------------------------
#
# safetensors_rust throws "device cuda:0 is invalid" when loading large
# shards over MooseFS/NFS-mounted Network Volumes - even though direct
# cuda loads of small local files work fine. accelerate.load_state_dict
# triggers this when device_map specifies a cuda device for the DiT
# shards in /workspace/Wan2.2-Animate-14B/.
#
# Workaround: force safetensors to always load to CPU. accelerate's
# load_checkpoint_and_dispatch flow handles the subsequent GPU placement
# separately. Slightly slower (CPU -> GPU transfer adds ~10s) but
# bypasses the bug entirely.
#
# This must be applied BEFORE any code imports accelerate, because
# accelerate caches a local reference: `from safetensors.torch import
# load_file as safe_load_file`. We patch both the source module's
# function AND accelerate's cached alias defensively.
def _patch_safetensors_cpu_only() -> None:
    import safetensors.torch  # type: ignore
    _orig_load_file = safetensors.torch.load_file

    def _patched_load_file(filename, device="cpu"):
        # Always load to CPU - ignore the requested device.
        return _orig_load_file(filename, device="cpu")

    safetensors.torch.load_file = _patched_load_file

    # If accelerate has already been imported (unlikely at module load
    # time but defensive), patch its cached reference too.
    try:
        import accelerate.utils.modeling as _accel_mod  # type: ignore
        _accel_mod.safe_load_file = _patched_load_file
    except ImportError:
        pass


_patch_safetensors_cpu_only()


# --- flash_attn -> torch SDPA fallback ------------------------------------
#
# Wan's CLIP visual encoder (wan/modules/animate/clip.py) hardcodes
# `flash_attention(..., version=2)` and the wrapper in wan/modules/attention.py
# asserts `FLASH_ATTN_2_AVAILABLE`. flash-attn doesn't have prebuilt wheels
# for torch 2.8 + py3.12 + cu128 (Blackwell), so we monkey-patch the call site
# to use torch.nn.functional.scaled_dot_product_attention instead.
#
# Wan's flash_attention signature:
#   def flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
#                       softmax_scale=None, q_scale=None, causal=False,
#                       window_size=(-1,-1), deterministic=False,
#                       dtype=torch.bfloat16, version=None)
# Shapes: q [B, Lq, Nq, C], k [B, Lk, Nk, C], v [B, Lk, Nk, C2] -> out [B, Lq, Nq, C2]
# SDPA wants [B, N, L, C], so we transpose in/out. GQA supported via repeat.
def _patch_wan_flash_attention_to_sdpa() -> None:
    """Replace wan.modules.attention.flash_attention with an SDPA shim.

    Must be called AFTER sys.path includes WAN_REPO but BEFORE any wan
    submodule that does `from ..attention import flash_attention` is imported
    (clip.py, model.py, etc. bind the name at import time).
    """
    import sys as _sys
    import torch
    import torch.nn.functional as F
    import wan.modules.attention as _wa  # type: ignore

    # IMPORTANT: importing wan.modules.attention may eagerly trigger
    # wan/__init__.py, which in turn imports WanAnimate -> animate.py ->
    # clip.py. clip.py does `from ..attention import flash_attention` at
    # module load time, which binds clip.py's local `flash_attention` name
    # to the ORIGINAL function object. Just setting
    # _wa.flash_attention = shim is therefore not enough -- we must also
    # sweep every loaded wan.* submodule and rebind any name still pointing
    # to the old function.
    _orig_flash_attention = _wa.flash_attention

    def _sdpa_shim(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.0,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        version=None,
    ):
        # q: [B, Lq, Nq, C], k/v: [B, Lk, Nk, C(2)]
        out_dtype = q.dtype

        # GQA: replicate K/V heads to match Q heads.
        Nq = q.shape[2]
        Nk = k.shape[2]
        if Nq != Nk:
            assert Nq % Nk == 0, f"Nq={Nq} must be divisible by Nk={Nk}"
            rep = Nq // Nk
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)

        # Optional pre-softmax Q scaling (Wan-specific).
        if q_scale is not None:
            q = q * q_scale

        # Transpose to SDPA layout: [B, N, L, C].
        q_ = q.transpose(1, 2)
        k_ = k.transpose(1, 2)
        v_ = v.transpose(1, 2)

        # Cast to half/bf16 for compute parity with FA (their assert chains
        # this dtype too). Skip if already half.
        compute_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        if q_.dtype not in (torch.float16, torch.bfloat16):
            q_ = q_.to(compute_dtype)
            k_ = k_.to(compute_dtype)
            v_ = v_.to(compute_dtype)

        # Build padding mask if variable-length sequences are supplied.
        attn_mask = None
        if q_lens is not None or k_lens is not None:
            B, _, Lq, _ = q_.shape
            Lk = k_.shape[2]
            device = q_.device
            if k_lens is not None:
                key_mask = (
                    torch.arange(Lk, device=device).unsqueeze(0)
                    < k_lens.to(device).unsqueeze(1)
                )  # [B, Lk] True=keep
                attn_mask = key_mask[:, None, None, :].expand(B, 1, Lq, Lk)
                attn_mask = attn_mask.to(dtype=torch.bool)

        # SDPA can't combine is_causal with an explicit attn_mask, so when
        # both are set we fold causality into the mask.
        if causal and attn_mask is not None:
            B, _, Lq, Lk = attn_mask.shape
            causal_mask = torch.tril(
                torch.ones(Lq, Lk, dtype=torch.bool, device=attn_mask.device)
            )
            attn_mask = attn_mask & causal_mask
            is_causal = False
        else:
            is_causal = bool(causal)

        out = F.scaled_dot_product_attention(
            q_, k_, v_,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=softmax_scale,
        )
        # Back to [B, L, N, C2] and restore original dtype.
        out = out.transpose(1, 2).contiguous().to(out_dtype)
        return out

    _wa.flash_attention = _sdpa_shim
    # Pretend FA2 is available so the `assert FLASH_ATTN_2_AVAILABLE` in
    # Wan's wrapper passes (it will then route to our shim).
    _wa.FLASH_ATTN_2_AVAILABLE = True
    # Leave FA3 flag alone (False) -- Wan's logic prefers FA3 only when
    # version=3 is explicitly requested; clip.py calls with version=2.

    # Sweep all loaded wan.* submodules and rebind any reference to the
    # original flash_attention function (e.g. clip.py binds it via
    # `from ..attention import flash_attention`).
    rebound = 0
    for mod_name, mod in list(_sys.modules.items()):
        if not mod_name.startswith("wan"):
            continue
        if mod is None or mod is _wa:
            continue
        try:
            mod_vars = vars(mod)
        except TypeError:
            continue
        for attr_name, attr_val in list(mod_vars.items()):
            if attr_val is _orig_flash_attention:
                setattr(mod, attr_name, _sdpa_shim)
                rebound += 1
                log.info("  rebound %s.%s -> SDPA shim", mod_name, attr_name)

    log.info(
        "Patched wan.modules.attention.flash_attention -> torch SDPA "
        "(also rebound %d existing references in loaded wan.* submodules)",
        rebound,
    )

    # ---- flash_attn_func shim ------------------------------------------
    #
    # In addition to Wan's own flash_attention wrapper, wan/modules/animate/
    # face_blocks.py imports the flash-attn package directly:
    #
    #     try:
    #         from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    #     except ImportError:
    #         flash_attn_func = None
    #
    # Then `attention(...)` defaults to mode="flash" and calls
    # `flash_attn_func(q, k, v)` -- which is None when flash-attn isn't
    # installed, so it raises `TypeError: 'NoneType' object is not callable`.
    # The face-adapter inside the DiT goes through this every diffusion step.
    #
    # Shape contract for flash_attn_func: q/k/v are [B, S, H, D]; output is
    # [B, S, H, D] with the same dtype as q. SDPA wants [B, H, S, D], so
    # transpose in/out.
    def _flash_attn_func_sdpa(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        **kwargs,
    ):
        out_dtype = q.dtype
        # [B, S, H, D] -> [B, H, S, D]
        q_ = q.transpose(1, 2)
        k_ = k.transpose(1, 2)
        v_ = v.transpose(1, 2)

        # GQA support if K/V have fewer heads than Q.
        if q_.shape[1] != k_.shape[1]:
            rep = q_.shape[1] // k_.shape[1]
            k_ = k_.repeat_interleave(rep, dim=1)
            v_ = v_.repeat_interleave(rep, dim=1)

        # SDPA prefers half/bf16. Match flash_attn's behavior.
        if q_.dtype not in (torch.float16, torch.bfloat16):
            q_ = q_.to(torch.bfloat16)
            k_ = k_.to(torch.bfloat16)
            v_ = v_.to(torch.bfloat16)

        out = F.scaled_dot_product_attention(
            q_, k_, v_,
            dropout_p=dropout_p,
            is_causal=bool(causal),
            scale=softmax_scale,
        )
        # [B, H, S, D] -> [B, S, H, D]
        return out.transpose(1, 2).contiguous().to(out_dtype)

    # Sweep all loaded wan.* modules for `flash_attn_func` attributes that
    # are None (or that point to a real flash_attn function we want to
    # replace anyway). face_blocks.py is the known callsite; others may
    # exist.
    _MISSING = object()
    fn_rebound = 0
    for mod_name, mod in list(_sys.modules.items()):
        if not mod_name.startswith("wan"):
            continue
        if mod is None:
            continue
        try:
            current = getattr(mod, "flash_attn_func", _MISSING)
        except Exception:
            continue
        if current is _MISSING:
            continue
        # Replace whether it's None (the import-failed case we expect) or a
        # real function (defensive -- we want SDPA everywhere).
        setattr(mod, "flash_attn_func", _flash_attn_func_sdpa)
        fn_rebound += 1
        log.info("  rebound %s.flash_attn_func -> SDPA shim (was %r)",
                 mod_name, type(current).__name__)

    log.info(
        "Patched flash_attn_func -> torch SDPA in %d wan.* submodules",
        fn_rebound,
    )


# --- logging --------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("wan-animate-server")

# --- paths ----------------------------------------------------------------

WAN_REPO = Path(os.environ.get("WAN_REPO", "/opt/wan22"))
WAN_CKPT_DIR = Path(os.environ.get("WAN_CKPT_DIR", "/runpod-volume/ComfyUI/models"))
SAM2_CKPT_DIR = Path(os.environ.get("SAM2_CKPT_DIR", "/runpod-volume/ComfyUI/models/sam2"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/runpod-volume/output/wan-animate"))
TMP_DIR = Path(os.environ.get("TMP_DIR", "/app/tmp"))
JOBS_DIR = Path(os.environ.get("JOBS_DIR", "/app/jobs"))

for d in (OUTPUT_DIR, TMP_DIR, JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Make sure the Wan-Video repo is on sys.path. The Dockerfile sets PYTHONPATH
# too, but in case someone runs server.py directly (e.g. during local dev) we
# add it here defensively.
if str(WAN_REPO) not in sys.path:
    sys.path.insert(0, str(WAN_REPO))

# --- app ------------------------------------------------------------------

app = FastAPI(title="Wan 2.2 Animate (Character Replace) server")

# --- job state ------------------------------------------------------------

JobStatus = Literal["queued", "running", "completed", "failed"]


@dataclass
class Job:
    id: str
    status: JobStatus = "queued"
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress_step: Optional[str] = None  # "preprocess" | "sample" | "encode"
    output_path: Optional[str] = None
    error: Optional[str] = None
    # echo back the params so the main app can persist them with the job
    params: dict = field(default_factory=dict)


_jobs: dict[str, Job] = {}
_jobs_lock = asyncio.Lock()  # protects the _jobs dict itself
_gpu_lock = asyncio.Lock()  # serializes GPU-bound inference (single in-flight)


def _persist_job(job: Job) -> None:
    """Write per-job JSON so a worker restart can still surface results.

    (We don't read these back on startup yet — they exist purely for post-
    mortem inspection. A future improvement: rebuild _jobs on startup.)
    """
    try:
        (JOBS_DIR / f"{job.id}.json").write_text(
            json.dumps(asdict(job), default=str, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        log.warning("Failed to persist job %s: %s", job.id, e)


async def _update_job(job_id: str, **fields_) -> Optional[Job]:
    async with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return None
        for k, v in fields_.items():
            setattr(job, k, v)
        _persist_job(job)
        return job


# --- model loading (lazy) ------------------------------------------------

_wan_animate = None  # type: ignore[var-annotated]
_preprocess_pipeline = None  # type: ignore[var-annotated]
_sam2_predictor = None  # type: ignore[var-annotated]
_model_load_error: Optional[str] = None


def _load_model() -> None:
    """Import Wan-Video + SAM2 and instantiate the pipeline objects.

    Called lazily on first inference request. Raises with a useful message
    if imports or instantiation fail.

    VERIFY at first deploy: the exact API of WanAnimate and ProcessPipeline.
    The web search that informed this code returned:

      - wan.WanAnimate(config, checkpoint_dir, device_id, rank, t5_fsdp,
                       dit_fsdp, use_sp, t5_cpu, convert_model_dtype,
                       use_relighting_lora)
      - wan.WanAnimate.generate(src_root_path, replace_flag, refert_num,
                                clip_len, shift, sample_solver, sampling_steps,
                                guide_scale, seed, offload_model)
      - wan.modules.animate.preprocess.process_pipepline.<???>(video_path,
                                                refer_image_path, output_path,
                                                resolution_area=[1280, 720],
                                                fps=30, iterations=3, k=7,
                                                w_len=1, h_len=1,
                                                retarget_flag=False,
                                                use_flux=False,
                                                replace_flag=False)

    The class name of the preprocessing pipeline wasn't surfaced. Common
    conventions would be ProcessPipeline or PreprocessPipeline. If neither
    works, `python -c "from wan.modules.animate.preprocess import process_pipepline; print(dir(process_pipepline))"`
    inside the running container will reveal the actual name.
    """
    global _wan_animate, _preprocess_pipeline, _sam2_predictor, _model_load_error
    if _wan_animate is not None:
        return

    try:
        log.info("Loading Wan 2.2 Animate ...")

        # Replace flash_attention with an SDPA fallback BEFORE importing
        # anything from `wan` that binds the name at import time (e.g. clip.py
        # does `from ..attention import flash_attention`).
        _patch_wan_flash_attention_to_sdpa()

        # Wan-Video's own config + WanAnimate class.
        # The config object resolves which checkpoint files to load.
        from wan.configs import WAN_CONFIGS  # type: ignore
        from wan import WanAnimate  # type: ignore

        cfg = WAN_CONFIGS["animate-14B"]
        _wan_animate = WanAnimate(
            config=cfg,
            checkpoint_dir=str(WAN_CKPT_DIR),
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            convert_model_dtype=True,
            use_relighting_lora=False,
        )
        log.info("Wan 2.2 Animate loaded.")

        log.info("Loading Wan-Video preprocessing pipeline ...")

        # The preprocess module uses BARE imports for its sibling files
        # (`from pose2d import Pose2d`, `from utils import ...`, etc.)
        # instead of relative imports. That only works if the preprocess
        # directory itself is on sys.path. Add it before importing.
        preprocess_dir = WAN_REPO / "wan" / "modules" / "animate" / "preprocess"
        if str(preprocess_dir) not in sys.path:
            sys.path.insert(0, str(preprocess_dir))
            log.info("  Added %s to sys.path", preprocess_dir)

        # Try the most likely class names. Fall back to introspection if
        # neither works.
        from wan.modules.animate.preprocess import process_pipepline as pp  # type: ignore

        ProcessClass = (
            getattr(pp, "ProcessPipeline", None)
            or getattr(pp, "PreprocessPipeline", None)
            or getattr(pp, "Pipeline", None)
        )
        if ProcessClass is None:
            # Last-ditch: find the first class in the module that has __call__.
            candidates = [
                v for v in vars(pp).values()
                if isinstance(v, type) and hasattr(v, "__call__")
                and v.__module__ == pp.__name__
            ]
            if not candidates:
                raise ImportError(
                    f"No callable class found in {pp.__name__}. "
                    f"Module contents: {sorted(vars(pp).keys())}"
                )
            ProcessClass = candidates[0]
            log.warning(
                "Falling back to %s for preprocessing pipeline (verify it's right)",
                ProcessClass.__name__,
            )

        # ProcessPipeline requires 4 explicit checkpoint paths. Layout
        # matches Wan-Video's own preprocess_data.py CLI script:
        #
        #     <ckpt_path>/det/yolov10m.onnx
        #     <ckpt_path>/pose2d/vitpose_h_wholebody.onnx
        #     <ckpt_path>/sam2/sam2_hiera_large.pt      (only when replace_flag=True)
        #     <ckpt_path>/FLUX.1-Kontext-dev/           (only when use_flux=True)
        #
        # where args.ckpt_path == /workspace/Wan2.2-Animate-14B/process_checkpoint/
        # (different from the main WAN_CKPT_DIR which holds the DiT+T5+CLIP+VAE).
        preprocess_ckpt_dir = WAN_CKPT_DIR / "process_checkpoint"
        det_checkpoint_path = preprocess_ckpt_dir / "det" / "yolov10m.onnx"
        pose2d_checkpoint_path = preprocess_ckpt_dir / "pose2d" / "vitpose_h_wholebody.onnx"
        sam_checkpoint_path = preprocess_ckpt_dir / "sam2" / "sam2_hiera_large.pt"
        flux_kontext_path = None  # not needed (use_flux=False)

        # Fail loudly with a clear list if any required path is missing
        missing = [
            f"  {label}={path}"
            for label, path in [
                ("det_checkpoint_path", det_checkpoint_path),
                ("pose2d_checkpoint_path", pose2d_checkpoint_path),
                ("sam_checkpoint_path", sam_checkpoint_path),
            ]
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing preprocess checkpoint(s):\n"
                + "\n".join(missing)
                + f"\nExpected base dir: {preprocess_ckpt_dir}"
            )

        log.info("Constructing ProcessPipeline with:")
        log.info("  det_checkpoint_path    = %s", det_checkpoint_path)
        log.info("  pose2d_checkpoint_path = %s", pose2d_checkpoint_path)
        log.info("  sam_checkpoint_path    = %s", sam_checkpoint_path)
        log.info("  flux_kontext_path      = %s", flux_kontext_path)

        _preprocess_pipeline = ProcessClass(
            det_checkpoint_path=str(det_checkpoint_path),
            pose2d_checkpoint_path=str(pose2d_checkpoint_path),
            sam_checkpoint_path=str(sam_checkpoint_path),
            flux_kontext_path=flux_kontext_path,
        )
        log.info("Preprocessing pipeline (%s) loaded.", ProcessClass.__name__)

    except Exception as e:
        _model_load_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        log.error("Model load failed:\n%s", _model_load_error)
        raise


# --- request/response models ---------------------------------------------


class SubmitResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobView(BaseModel):
    job_id: str
    status: JobStatus
    submitted_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress_step: Optional[str] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    params: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: Literal["ready", "loading", "error"]
    model_loaded: bool
    error: Optional[str] = None


# --- helpers --------------------------------------------------------------


def _parse_resolution(res: str) -> list[int]:
    """Parse '1280x720' -> [1280, 720]. Validates against known Wan resolutions."""
    SUPPORTED = {"832x480", "1280x720", "1408x640", "480x832", "720x1280"}
    if res not in SUPPORTED:
        raise HTTPException(
            status_code=422,
            detail=f"resolution must be one of {sorted(SUPPORTED)}; got {res!r}",
        )
    w, h = res.split("x")
    return [int(w), int(h)]


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    """Stream an UploadFile to disk so big videos don't hit memory."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        while True:
            chunk = await upload.read(1 << 20)  # 1 MB
            if not chunk:
                break
            f.write(chunk)


# --- main inference task -------------------------------------------------


async def _run_job(
    job_id: str,
    character_image_path: Path,
    source_video_path: Path,
    prompt: str,
    seed: int,
    resolution: str,
    replace_flag: bool,
    sampling_steps: int,
    frame_num: int,
    refert_num: int,
    guide_scale: float,
) -> None:
    """Background coroutine that actually runs the pipeline.

    Captures any exception, marks job failed, and writes the error message.
    """
    job_workspace = TMP_DIR / job_id
    job_workspace.mkdir(parents=True, exist_ok=True)
    src_root_path = job_workspace / "src"
    src_root_path.mkdir(parents=True, exist_ok=True)

    try:
        # Lazy-load on first request (or no-op if already loaded).
        _load_model()
        assert _wan_animate is not None and _preprocess_pipeline is not None

        async with _gpu_lock:
            await _update_job(job_id, status="running", started_at=time.time())

            resolution_area = _parse_resolution(resolution)
            actual_seed = seed if seed >= 0 else int(uuid.uuid4().int & 0xFFFFFFFF)

            # ---- Preprocessing -----------------------------------------------
            await _update_job(job_id, progress_step="preprocess")
            log.info("[%s] preprocessing video=%s ref=%s -> %s",
                     job_id, source_video_path, character_image_path, src_root_path)

            # Run on a thread because the preprocessing is sync + GPU-bound.
            await asyncio.to_thread(
                _preprocess_pipeline,
                video_path=str(source_video_path),
                refer_image_path=str(character_image_path),
                output_path=str(src_root_path),
                resolution_area=resolution_area,
                fps=16,
                iterations=3,
                k=7,
                w_len=1,
                h_len=1,
                retarget_flag=False,
                use_flux=False,
                replace_flag=replace_flag,
            )

            # ---- Sampling ----------------------------------------------------
            # Wan's animate.py asserts refert_num must be exactly 1 or 5 (it's
            # the number of reference frames overlapped between clips). Clamp
            # anything else to 5 (their default for multi-clip animation).
            if refert_num not in (1, 5):
                log.warning("[%s] refert_num=%d invalid (must be 1 or 5); clamping to 5",
                            job_id, refert_num)
                refert_num = 5

            await _update_job(job_id, progress_step="sample")
            log.info("[%s] sampling (seed=%d, steps=%d, frames=%d, refert_num=%d)",
                     job_id, actual_seed, sampling_steps, frame_num, refert_num)

            frames = await asyncio.to_thread(
                _wan_animate.generate,
                src_root_path=str(src_root_path),
                replace_flag=replace_flag,
                refert_num=refert_num,
                clip_len=frame_num,
                shift=8.0,                 # Wan default
                sample_solver="unipc",     # Wan default
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                seed=actual_seed,
                offload_model=False,
            )

            # ---- Encode mp4 --------------------------------------------------
            await _update_job(job_id, progress_step="encode")
            output_path = OUTPUT_DIR / f"{job_id}.mp4"

            # Wan's WanAnimate.generate returns video tensors in (C, T, H, W)
            # layout, float values in approximately [-1, 1] (some configs use
            # [0, 1]). imageio.mimwrite wants a list of (H, W, C) uint8 arrays
            # with C in {1, 2, 3, 4}.
            if hasattr(frames, "cpu"):
                frames = frames.cpu().numpy()

            log.info("[%s] raw frames shape=%s dtype=%s min=%.3f max=%.3f",
                     job_id, frames.shape, frames.dtype,
                     float(frames.min()), float(frames.max()))

            # Normalize layout to (T, H, W, C).
            if frames.ndim == 4:
                d0, d1, d2, d3 = frames.shape
                if d0 == 3 and d3 != 3:
                    # (C=3, T, H, W) -> (T, H, W, C)
                    frames = np.transpose(frames, (1, 2, 3, 0))
                elif d1 == 3 and d3 != 3:
                    # (T, C=3, H, W) -> (T, H, W, C)
                    frames = np.transpose(frames, (0, 2, 3, 1))
                # else: already (T, H, W, C); leave alone
            else:
                raise ValueError(
                    f"Unexpected Wan output shape {frames.shape}; "
                    f"expected 4D tensor."
                )

            # Normalize value range to uint8 [0, 255].
            if frames.dtype != np.uint8:
                mn, mx = float(frames.min()), float(frames.max())
                if mn < -0.01:
                    # Looks like [-1, 1]; rescale.
                    frames = (np.clip(frames, -1.0, 1.0) + 1.0) * 0.5
                # Now in [0, 1] (or close).
                frames = (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)

            log.info("[%s] encoding %d frames %s -> %s",
                     job_id, len(frames), frames.shape, output_path)
            await asyncio.to_thread(
                imageio.mimwrite,
                str(output_path),
                list(frames),
                fps=16,
                quality=8,
                codec="libx264",
                pixelformat="yuv420p",
            )

            await _update_job(
                job_id,
                status="completed",
                finished_at=time.time(),
                output_path=str(output_path),
                progress_step=None,
            )
            log.info("[%s] done (took %.1f s)",
                     job_id, time.time() - (_jobs[job_id].started_at or time.time()))

    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        log.exception("[%s] failed: %s", job_id, msg)
        await _update_job(
            job_id,
            status="failed",
            finished_at=time.time(),
            error=msg,
            progress_step=None,
        )
    finally:
        # Clean up the job's temp workspace (preserves output_path, which is
        # outside the workspace).
        try:
            shutil.rmtree(job_workspace, ignore_errors=True)
        except Exception:
            pass


# --- routes ---------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if _wan_animate is None and _model_load_error is None:
        return HealthResponse(status="loading", model_loaded=False)
    if _model_load_error is not None:
        return HealthResponse(status="error", model_loaded=False, error=_model_load_error[:1000])
    return HealthResponse(status="ready", model_loaded=True)


@app.get("/debug/info")
async def debug_info() -> dict:
    """Lightweight diagnostic endpoint - returns versions + CUDA state
    without needing terminal access to the container. Hit this when
    something's wrong and you can't `exec` in.
    """
    import platform
    import sys as _sys
    info: dict = {
        "python": platform.python_version(),
        "python_executable": _sys.executable,
        "platform": platform.platform(),
        "model_loaded": _wan_animate is not None,
        "model_load_error": _model_load_error[:500] if _model_load_error else None,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda_version"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["torch_error"] = f"{type(e).__name__}: {e}"

    # Wan-Video repo existence + git revision
    info["wan_repo_exists"] = WAN_REPO.exists()
    if WAN_REPO.exists():
        info["wan_repo_path"] = str(WAN_REPO)
        try:
            import subprocess
            r = subprocess.run(
                ["git", "-C", str(WAN_REPO), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            info["wan_repo_commit"] = r.stdout.strip()[:12] if r.returncode == 0 else None
        except Exception:
            pass

    # SAM2 import availability
    try:
        import sam2
        info["sam2_path"] = sam2.__file__
    except ImportError as e:
        info["sam2_error"] = str(e)

    return info


@app.post("/character-swap", response_model=SubmitResponse)
async def submit_character_swap(
    character_image: UploadFile,
    source_video: UploadFile,
    prompt: str = Form(""),
    seed: int = Form(-1),
    resolution: str = Form("832x480"),
    replace_flag: bool = Form(True),
    sampling_steps: int = Form(20),
    frame_num: int = Form(81),
    refert_num: int = Form(77),
    guide_scale: float = Form(5.0),
) -> SubmitResponse:
    if not (1 <= sampling_steps <= 60):
        raise HTTPException(422, "sampling_steps must be in [1, 60]")
    if not (17 <= frame_num <= 161):
        raise HTTPException(422, "frame_num must be in [17, 161]")

    job_id = uuid.uuid4().hex[:12]
    job_workspace = TMP_DIR / job_id
    job_workspace.mkdir(parents=True, exist_ok=True)

    char_path = job_workspace / "character.png"
    video_path = job_workspace / "source.mp4"
    await _save_upload(character_image, char_path)
    await _save_upload(source_video, video_path)

    params = {
        "prompt": prompt,
        "seed": seed,
        "resolution": resolution,
        "replace_flag": replace_flag,
        "sampling_steps": sampling_steps,
        "frame_num": frame_num,
        "refert_num": refert_num,
        "guide_scale": guide_scale,
    }
    async with _jobs_lock:
        _jobs[job_id] = Job(id=job_id, params=params)
        _persist_job(_jobs[job_id])

    asyncio.create_task(_run_job(
        job_id,
        char_path,
        video_path,
        prompt=prompt,
        seed=seed,
        resolution=resolution,
        replace_flag=replace_flag,
        sampling_steps=sampling_steps,
        frame_num=frame_num,
        refert_num=refert_num,
        guide_scale=guide_scale,
    ))

    return SubmitResponse(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobView)
async def get_job(job_id: str) -> JobView:
    async with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JobView(
        job_id=job.id,
        status=job.status,
        submitted_at=job.submitted_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        progress_step=job.progress_step,
        output_path=job.output_path,
        error=job.error,
        params=job.params,
    )


@app.get("/jobs/{job_id}/output")
async def get_job_output(job_id: str):
    async with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "completed":
        raise HTTPException(409, f"Job is {job.status}, not completed")
    if not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(500, "Output file missing from disk")
    return FileResponse(
        job.output_path,
        media_type="video/mp4",
        filename=f"{job_id}.mp4",
    )


@app.get("/jobs")
async def list_jobs() -> list[JobView]:
    async with _jobs_lock:
        return [
            JobView(
                job_id=j.id,
                status=j.status,
                submitted_at=j.submitted_at,
                started_at=j.started_at,
                finished_at=j.finished_at,
                progress_step=j.progress_step,
                output_path=j.output_path,
                error=j.error,
                params=j.params,
            )
            for j in sorted(_jobs.values(), key=lambda x: x.submitted_at, reverse=True)
        ]
