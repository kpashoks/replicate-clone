from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import ensure_data_dirs, settings
from routes import files as files_routes
from routes import generate as generate_routes
from routes import models as models_routes
from routes import prompts as prompts_routes
from routes import uploads as uploads_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_data_dirs()
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
