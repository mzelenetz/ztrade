from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.ideas_scan import start_warmer
from src.api.routers import auth, chain, dividends, ideas, meta, settings, spreads, vol_surface

app = FastAPI(title="ztrade")


@app.on_event("startup")
def _maybe_start_ideas_warmer() -> None:
    # Default on against cloud data (where cold scans are expensive), off for
    # local dev files; IDEAS_WARMER=1/0 overrides either way.
    default = "1" if os.getenv("DATA_SOURCE_TYPE", "local").lower().startswith("gcs") else "0"
    if os.getenv("IDEAS_WARMER", default) == "1":
        start_warmer()

app.include_router(auth.router)
app.include_router(meta.router)
app.include_router(chain.router)
app.include_router(spreads.router)
app.include_router(vol_surface.router)
app.include_router(dividends.router)
app.include_router(ideas.router)
app.include_router(settings.router)

WEB_DIST = Path(__file__).resolve().parent.parent.parent / "web" / "dist"

if WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str) -> FileResponse:
        candidate = WEB_DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(WEB_DIST / "index.html")
