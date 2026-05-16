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

    DATA_DIR: str = "./data"
    FRONTEND_URL: str = "http://localhost:3000"

    @property
    def data_dir_abs(self) -> Path:
        p = Path(self.DATA_DIR)
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()


settings = Settings()


def ensure_data_dirs() -> None:
    base = settings.data_dir_abs
    for sub in ("inputs", "outputs", "jobs"):
        (base / sub).mkdir(parents=True, exist_ok=True)
