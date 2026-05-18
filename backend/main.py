import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import ensure_data_dirs, settings
from routes import files as files_routes
from routes import generate as generate_routes
from routes import models as models_routes
from routes import prompts as prompts_routes
from routes import uploads as uploads_routes


log = logging.getLogger(__name__)


def _log_provider_config() -> None:
    """Surface key provider URLs at startup so it's obvious when an .env
    edit didn't actually land. Common gotcha: change WAN_ANIMATE_ENDPOINT
    in .env, but uvicorn's --reload didn't re-import settings, so the new
    URL never took effect. This print makes the actual loaded value
    visible the moment uvicorn (re)starts.
    """
    def _redact_token(s: str) -> str:
        if not s:
            return "<unset>"
        return s[:6] + "..." + s[-4:] if len(s) > 14 else "<set>"

    log.info("=" * 60)
    log.info("replicate-local backend - provider config:")
    log.info("  RUNPOD_ENDPOINT_ID         = %s",
             settings.RUNPOD_ENDPOINT_ID or "<unset>")
    log.info("  RUNPOD_DOWNLOADER_ENDPOINT = %s",
             settings.RUNPOD_DOWNLOADER_ENDPOINT_ID or "<unset>")
    log.info("  RUNPOD_API_KEY             = %s",
             _redact_token(settings.RUNPOD_API_KEY))
    log.info("  WAN_ANIMATE_ENDPOINT       = %s",
             settings.WAN_ANIMATE_ENDPOINT or "<unset>")
    if settings.WAN_ANIMATE_ENDPOINT \
       and "REPLACE_WITH" in settings.WAN_ANIMATE_ENDPOINT:
        log.warning(
            "  ^^^ WAN_ANIMATE_ENDPOINT still has the placeholder string!"
            " Edit .env and restart uvicorn before testing character-swap."
        )
    log.info("=" * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_data_dirs()
    _log_provider_config()
    yield


app = FastAPI(title="replicate-local", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prompts_routes.router)
app.include_router(models_routes.router)
app.include_router(generate_routes.router)
app.include_router(files_routes.router)
app.include_router(uploads_routes.router)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}
