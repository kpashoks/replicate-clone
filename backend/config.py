from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    RUNPOD_API_KEY: str = ""
    RUNPOD_ENDPOINT_ID: str = ""
    RUNPOD_TIMEOUT_SECONDS: int = 600

    # Optional companion Serverless endpoint that reads files off the Network
    # Volume and returns them as base64. Used when worker-comfyui doesn't
    # bubble certain outputs (notably mp4 from VHS_VideoCombine) back in its
    # response. Leave empty if you haven't deployed it yet.
    RUNPOD_DOWNLOADER_ENDPOINT_ID: str = ""

    QWEN_MODEL_ID: str = "Qwen/Qwen3-4B-Instruct-2507"

    # HTTP base URL of the wan-animate inference server (the separate
    # FastAPI server built from runpod-wan-animate/). Leave empty if you
    # haven't deployed it yet - models with provider="wan-animate-http"
    # will fail with a clear error.
    # Example: https://abc123-8000.proxy.runpod.net
    WAN_ANIMATE_ENDPOINT: str = ""
    # Max wall-clock seconds to wait for the wan-animate server to finish a
    # job. Wan 2.2 Animate runs ~25s per diffusion step on Blackwell, plus
    # ~3 min model load on cold start. Observed budgets on a Blackwell GPU:
    #   2 s  clip (81 frames) : ~12 min end-to-end
    #   5 s  clip              : ~25 min
    #   10 s clip              : ~50 min
    #   15 s clip              : ~75 min
    # Default to 90 min so 10-15s clips don't trip the cap. Override via env.
    WAN_ANIMATE_TIMEOUT_SECONDS: int = 5400

    # Atlas Cloud (https://atlascloud.ai). Unified inference aggregator used
    # for hosted T2I / I2I models (FLUX.2 Pro, Imagen 4 Ultra, Qwen-Image
    # Edit Plus, etc.). Leave key empty if you haven't signed up — models
    # with provider="atlas" will fail at submit with a clear error.
    ATLAS_CLOUD_API_KEY: str = ""
    ATLAS_CLOUD_BASE_URL: str = "https://api.atlascloud.ai"
    # Max wall-clock seconds the local polling client will wait for an
    # Atlas job to complete. Observed durations:
    #   T2I / I2I:                  10-60 s   (default amply covers)
    #   T2V (Seedance, Wan 2.7):    30-120 s
    #   I2V:                        60-300 s  (Veo, HappyHorse longest)
    #   V2V (Wan video-edit):       180-900 s on longer source clips
    #   video-swap (Wan R2V):       60-180 s
    # 1800s (30 min) is the lowest-common-denominator that covers v2v
    # without timing out cheap image jobs. Atlas keeps processing
    # server-side regardless; this only governs how long WE wait.
    ATLAS_TIMEOUT_SECONDS: int = 1800

    DATA_DIR: str = "./data"
    FRONTEND_URL: str = "http://localhost:3000"

    # Optional default destination folder for the "Save & Rename" feature on
    # completed jobs. The /api/jobs/{id}/save endpoint copies output files
    # from data/outputs/<job_id>/ into this folder (with a user-chosen
    # filename). Empty = no default; the user must type a folder in the
    # rename modal each time. Use a Windows-style absolute path on Windows
    # (C:\Users\you\Pictures\replicate-out) or POSIX elsewhere.
    DOWNLOAD_DIR: str = ""

    @property
    def data_dir_abs(self) -> Path:
        p = Path(self.DATA_DIR)
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    @property
    def download_dir_abs(self) -> Path | None:
        """Resolved DOWNLOAD_DIR, or None if unset. Supports ~ expansion."""
        if not self.DOWNLOAD_DIR:
            return None
        p = Path(self.DOWNLOAD_DIR).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()


settings = Settings()


def ensure_data_dirs() -> None:
    base = settings.data_dir_abs
    for sub in ("inputs", "outputs", "jobs"):
        (base / sub).mkdir(parents=True, exist_ok=True)
